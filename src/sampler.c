/*
 * sampler.c — turn next-token logits into an actual next token.
 *
 * Pipeline (each stage optional):
 *
 *   logits /= temperature     sharper (->0) or flatter (->inf) distribution
 *   top-k                     zero out everything below the k-th best logit
 *   softmax                   logits -> probabilities
 *   top-p (nucleus)           keep the smallest set of tokens whose
 *                             cumulative probability exceeds `topp`
 *   sample                    draw from the remaining distribution
 *
 * temperature = 0 skips all of that and takes the argmax (greedy decoding),
 * which is also what the benchmark mode uses for deterministic timings.
 *
 * RNG: xorshift64* — tiny, fast, and reproducible from a seed.
 */

#include "nanollama.h"

static unsigned long long random_u64(unsigned long long *state) {
    /* xorshift64* */
    unsigned long long x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* uniform float in [0, 1) */
static float random_f32(unsigned long long *state) {
    return (float)((random_u64(state) >> 40) / 16777216.0f);
}

static void softmax_inplace(float *x, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) max = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void build_sampler(Sampler *s, int vocab_size, float temperature, float topp,
                   int topk, unsigned long long seed) {
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->topp = topp;
    s->topk = topk;
    s->rng_state = seed == 0 ? (unsigned long long)time(NULL) : seed;
    s->probindex = NULL;
    if (topp > 0.0f) {
        /* packed (probability, index) pairs */
        s->probindex = (float *)malloc((size_t)vocab_size * sizeof(float) * 2);
        if (!s->probindex) NL_ERROR("out of memory for sampler");
    }
}

void free_sampler(Sampler *s) {
    free(s->probindex);
}

/* pick an index proportional to the probabilities in x */
static int sample_mult(const float *probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; /* float paranoia: never reached with a valid distribution */
}

/* top-p (nucleus): keep the smallest prefix of tokens (sorted by probability,
 * descending) whose cumulative probability exceeds topp, then sample from it.
 * probindex is the packed scratch buffer from the Sampler. */
static int sample_topp(const float *probabilities, int n, float topp,
                       float *probindex, float coin) {
    /* quickfilter: any probability below this can never be inside the nucleus
     * (it would take more than `topp` cumulative mass just to reach it) */
    const float cutoff = (1.0f - topp) / (float)(n - 1);
    int n0 = 0;
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0 * 2]     = probabilities[i];
            probindex[n0 * 2 + 1] = (float)i;
            n0++;
        }
    }

    /* sort the survivors by probability, descending.
     * insertion sort: after the quickfilter only a handful survive. */
    for (int i = 1; i < n0; i++) {
        float p = probindex[i * 2];
        float idx = probindex[i * 2 + 1];
        int j = i - 1;
        while (j >= 0 && probindex[j * 2] < p) {
            probindex[(j + 1) * 2]     = probindex[j * 2];
            probindex[(j + 1) * 2 + 1] = probindex[j * 2 + 1];
            j--;
        }
        probindex[(j + 1) * 2]     = p;
        probindex[(j + 1) * 2 + 1] = idx;
    }

    /* find the smallest prefix that crosses the nucleus threshold */
    float cumulative = 0.0f;
    int last = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative += probindex[i * 2];
        if (cumulative > topp) {
            last = i;
            break;
        }
    }

    /* sample from the kept prefix (cdf over unnormalized probs works:
     * coin is uniform and the prefix covers > topp of all mass) */
    float cdf = 0.0f;
    for (int i = 0; i <= last; i++) {
        cdf += probindex[i * 2];
        if (coin < cdf) {
            return (int)probindex[i * 2 + 1];
        }
    }
    return (int)probindex[last * 2 + 1];
}

/* min-heap helpers for the top-k cutoff */
static void heap_sift_down(float *heap, int size, int pos) {
    while (1) {
        int left = pos * 2 + 1, right = left + 1, smallest = pos;
        if (left < size && heap[left] < heap[smallest]) smallest = left;
        if (right < size && heap[right] < heap[smallest]) smallest = right;
        if (smallest == pos) break;
        float tmp = heap[pos]; heap[pos] = heap[smallest]; heap[smallest] = tmp;
        pos = smallest;
    }
}

int sample(Sampler *s, float *logits) {
    int next;

    if (s->temperature <= 0.0f) {
        /* greedy: the argmax, deterministically */
        next = 0;
        float max = logits[0];
        for (int i = 1; i < s->vocab_size; i++) {
            if (logits[i] > max) {
                max = logits[i];
                next = i;
            }
        }
        return next;
    }

    /* apply temperature */
    for (int i = 0; i < s->vocab_size; i++) {
        logits[i] /= s->temperature;
    }

    /* top-k: find the k-th largest logit with a min-heap of size k,
     * then zero out everything below it. O(n log k), no sorting. */
    if (s->topk > 0 && s->topk < s->vocab_size) {
        int k = s->topk;
        float *heap = (float *)malloc((size_t)k * sizeof(float));
        if (!heap) NL_ERROR("out of memory for top-k");
        for (int i = 0; i < k; i++) {
            heap[i] = logits[i];
        }
        for (int i = k / 2 - 1; i >= 0; i--) {
            heap_sift_down(heap, k, i);
        }
        for (int i = k; i < s->vocab_size; i++) {
            if (logits[i] > heap[0]) {
                heap[0] = logits[i];
                heap_sift_down(heap, k, 0);
            }
        }
        float cutoff = heap[0];
        free(heap);
        for (int i = 0; i < s->vocab_size; i++) {
            if (logits[i] < cutoff) {
                logits[i] = -1e10f;
            }
        }
    }

    softmax_inplace(logits, s->vocab_size);

    float coin = random_f32(&s->rng_state);
    if (s->topp > 0.0f) {
        next = sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
    } else {
        next = sample_mult(logits, s->vocab_size, coin);
    }

    return next;
}
