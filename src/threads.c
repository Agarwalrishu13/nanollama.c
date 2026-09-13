/*
 * threads.c — a persistent thread pool + parallel-for, written from scratch.
 *
 * nanollama doesn't use OpenMP: the whole point of this repo is showing what
 * lives underneath the abstractions, so the threading primitive itself is
 * part of the lesson.
 *
 * Design notes (and why not the naive alternative):
 *
 *   v0 of this file spawned threads per call. It was 40% SLOWER than
 *   single-threaded — the model runs ~49 matmuls per token and each
 *   CreateThread costs tens of microseconds. Lesson: spawn overhead is why
 *   OpenMP runtimes keep a pool.
 *
 *   So this is a real pool: the workers are started once and park on a
 *   semaphore. `nl_parallel_for` posts the job (a [0,total) range split into
 *   contiguous chunks), wakes all workers, runs chunk 0 on the caller's own
 *   stack, then waits on a "done" semaphore until everyone reports back.
 *
 * Every worker participates in the barrier on every job; workers whose id
 * falls past the chunk split simply get an empty range, which keeps the
 * wake/done token counts perfectly balanced (no starvation, no deadlock).
 *
 * Windows: CreateSemaphore + CRITICAL_SECTION (available since forever).
 * POSIX:   pthread_mutex + two pthread_conds.
 */

#include "nanollama.h"

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <process.h>
#else
  #include <pthread.h>
  #include <unistd.h>
#endif

#define NL_MAX_THREADS 32

/* ---------------------------------------------------------------------------
 * pool state
 * ------------------------------------------------------------------------- */
typedef struct {
    void (*fn)(int start, int end, int id, void *ctx);
    void *ctx;
    int chunk_start[NL_MAX_THREADS];
    int chunk_end[NL_MAX_THREADS];
} NlJob;

static NlJob g_job;
static int g_n_workers = 0;          /* spawned workers (ids 1..g_n_workers) */
static int g_threads_wanted = 1;     /* from -th / nl_set_threads            */
static int g_pool_ready = 0;

#ifdef _WIN32
static HANDLE g_workers[NL_MAX_THREADS];
static HANDLE g_wake_sem;            /* worker: wait = "a job is posted"     */
static HANDLE g_done_sem;            /* main:   wait = "worker i finished"   */
static CRITICAL_SECTION g_cs;        /* guards g_job while it is written     */
#else
static pthread_t g_workers[NL_MAX_THREADS];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_wake_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER;
static unsigned long g_generation = 0;
static unsigned long g_done_count = 0;
#endif

/* ---------------------------------------------------------------------------
 * public API (thread count)
 * ------------------------------------------------------------------------- */
int nl_thread_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

void nl_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > NL_MAX_THREADS) n = NL_MAX_THREADS;
    g_threads_wanted = n;
}

int nl_get_threads(void) {
    return g_threads_wanted;
}

/* ---------------------------------------------------------------------------
 * worker loop
 * ------------------------------------------------------------------------- */
#ifdef _WIN32
static unsigned __stdcall nl_worker(void *arg)
#else
static void *nl_worker(void *arg)
#endif
{
    int id = (int)(intptr_t)arg;         /* 1..g_n_workers                      */
    (void)arg;

#ifdef _WIN32
    for (;;) {
        WaitForSingleObject(g_wake_sem, INFINITE);   /* one token = one job */

        EnterCriticalSection(&g_cs);
        void (*fn)(int, int, int, void *) = g_job.fn;
        void *ctx = g_job.ctx;
        int start = g_job.chunk_start[id];
        int end = g_job.chunk_end[id];
        LeaveCriticalSection(&g_cs);

        if (start < end) {
            fn(start, end, id, ctx);
        }
        ReleaseSemaphore(g_done_sem, 1, NULL);       /* check in at the barrier */
    }
    return 0;
#else
    unsigned long last_seen = 0;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (g_generation == last_seen) {
            pthread_cond_wait(&g_wake_cv, &g_mu);
        }
        last_seen = g_generation;
        void (*fn)(int, int, int, void *) = g_job.fn;
        void *ctx = g_job.ctx;
        int start = g_job.chunk_start[id];
        int end = g_job.chunk_end[id];
        pthread_mutex_unlock(&g_mu);

        if (start < end) {
            fn(start, end, id, ctx);
        }

        pthread_mutex_lock(&g_mu);
        g_done_count++;
        if (g_done_count >= (unsigned long)g_n_workers) {
            pthread_cond_signal(&g_done_cv);
        }
        pthread_mutex_unlock(&g_mu);
    }
    return NULL;
#endif
}

/* start the workers exactly once, from the main thread */
static void nl_pool_start(void) {
    if (g_pool_ready) return;
    g_pool_ready = 1;

    int hw = nl_thread_count();
    if (hw > NL_MAX_THREADS) hw = NL_MAX_THREADS;

#ifdef _WIN32
    InitializeCriticalSection(&g_cs);
    g_wake_sem = CreateSemaphoreA(NULL, 0, NL_MAX_THREADS, NULL);
    g_done_sem = CreateSemaphoreA(NULL, 0, NL_MAX_THREADS, NULL);
    if (!g_wake_sem || !g_done_sem) NL_ERROR("failed to create pool semaphores");
    for (int i = 1; i < hw; i++) {
        g_workers[i] = (HANDLE)_beginthreadex(NULL, 0, nl_worker,
                                              (void *)(intptr_t)i, 0, NULL);
        if (g_workers[i]) g_n_workers++;
    }
#else
    for (int i = 1; i < hw; i++) {
        if (pthread_create(&g_workers[i], NULL, nl_worker, (void *)(intptr_t)i) == 0) {
            g_n_workers++;
        }
    }
#endif
}

/* ---------------------------------------------------------------------------
 * nl_parallel_for: split [0, total) across the pool, chunk 0 runs on the
 * caller's own stack (it usually has plenty of other work to get back to).
 * ------------------------------------------------------------------------- */
void nl_parallel_for(int total, void (*fn)(int, int, int, void *), void *ctx, int threads) {
    if (threads <= 1 || total < 64) {
        fn(0, total, 0, ctx); /* coordination costs more than the work */
        return;
    }
    nl_pool_start();

    int used = threads;
    if (g_n_workers + 1 < used) used = g_n_workers + 1;
    if (used > total) used = total;
    if (used <= 1) {
        fn(0, total, 0, ctx);
        return;
    }

    /* split [0, total) into `used` contiguous chunks; ids past `used` get an
     * empty range so every worker still joins the barrier */
    int base = total / used, extra = total % used;
    int start = 0;
    for (int i = 0; i < NL_MAX_THREADS; i++) {
        if (i < used) {
            int count = base + (i < extra ? 1 : 0);
            g_job.chunk_start[i] = start;
            g_job.chunk_end[i] = start + count;
            start += count;
        } else {
            g_job.chunk_start[i] = 0;
            g_job.chunk_end[i] = 0;
        }
    }
    g_job.fn = fn;
    g_job.ctx = ctx;

#ifdef _WIN32
    EnterCriticalSection(&g_cs);
    LeaveCriticalSection(&g_cs);          /* params written above, main-only */
    ReleaseSemaphore(g_wake_sem, (LONG)g_n_workers, NULL);
#else
    pthread_mutex_lock(&g_mu);
    g_done_count = 0;
    g_generation++;
    pthread_cond_broadcast(&g_wake_cv);
    pthread_mutex_unlock(&g_mu);
#endif

    fn(g_job.chunk_start[0], g_job.chunk_end[0], 0, ctx); /* our own chunk */

#ifdef _WIN32
    for (int i = 0; i < g_n_workers; i++) {
        WaitForSingleObject(g_done_sem, INFINITE);
    }
#else
    pthread_mutex_lock(&g_mu);
    while (g_done_count < (unsigned long)g_n_workers) {
        pthread_cond_wait(&g_done_cv, &g_mu);
    }
    pthread_mutex_unlock(&g_mu);
#endif
}
