/*
 * nanollama.h — shared types for the nanollama.c inference engine
 *
 * A from-scratch, zero-dependency inference engine for Llama-2-architecture
 * transformers. Every file in src/ includes this header; each .c file
 * implements one stage of the pipeline:
 *
 *   transformer.c  -> model loading, the full forward pass, quantization
 *   tokenizer.c    -> byte-pair encoding (BPE) from scratch
 *   sampler.c      -> temperature / top-k / top-p sampling
 *   main.c         -> CLI: generate | chat | bench
 *
 * Model format: the llama2.c export format (magic 0x616b3432), versions 0 and
 * >= 1, so every model in karpathy's HF repo `karpathy/tinyllamas` works.
 */

#ifndef NANOLLAMA_H
#define NANOLLAMA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>

/* ---------------------------------------------------------------------------
 * Portable wall-clock timer (ms). Windows lacks clock_gettime; POSIX lacks
 * QueryPerformanceCounter. Both give us sub-millisecond resolution.
 * ------------------------------------------------------------------------- */
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  static inline double now_ms(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
  }
#else
  #include <time.h>
  static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
  }
#endif

#define NL_ERROR(msg) do { fprintf(stderr, "[nanollama] %s\n", msg); exit(EXIT_FAILURE); } while (0)

/* threads.c — the hand-rolled parallel-for (no OpenMP needed) -------------- */
void nl_set_threads(int n);
int  nl_get_threads(void);
int  nl_thread_count(void);   /* hardware concurrency */
void nl_parallel_for(int total, void (*fn)(int start, int end, int id, void *ctx),
                     void *ctx, int threads);

/* ---------------------------------------------------------------------------
 * Config — the 7 integers that fully describe a Llama-2 model's shape.
 * ------------------------------------------------------------------------- */
typedef struct {
    int dim;              /* transformer width (embedding dimension)            */
    int hidden_dim;       /* width of the SwiGLU feed-forward layer             */
    int n_layers;         /* number of transformer blocks                       */
    int n_heads;          /* number of query heads                              */
    int n_kv_heads;       /* number of key/value heads (GQA: may be < n_heads)  */
    int vocab_size;       /* vocabulary size (32000 for the Llama-2 tokenizer)  */
    int seq_len;          /* maximum context length the KV cache is sized for   */
    int shared_classifier;/* 1 = wcls is the same tensor as token_embedding     */
} Config;

/* ---------------------------------------------------------------------------
 * Quantized weight matrix — int8 values grouped in chunks of GS floats.
 * Every group of GS values shares one fp32 scale, giving ~4x memory savings
 * at a small quality cost. (Same "q8" scheme as llama.cpp's q8_0.)
 * ------------------------------------------------------------------------- */
#define QGS 64

typedef struct {
    int8_t *q;    /* rows * cols int8 weights                                 */
    float  *s;    /* rows * groups_per_row scales, one per group of QGS       */
    int groups_per_row; /* ceil(cols / QGS) — scales restart on every row     */
} QMatrix;

/* Raw fp32 weight tensors, in the exact order they appear in the model file. */
typedef struct {
    float *token_embedding; /* (vocab_size, dim)                              */
    float *rms_att_weight;  /* (n_layers, dim)                                */
    float *wq;              /* (n_layers, dim, n_heads * head_size)           */
    float *wk;              /* (n_layers, dim, n_kv_heads * head_size)        */
    float *wv;              /* (n_layers, dim, n_kv_heads * head_size)        */
    float *wo;              /* (n_layers, n_heads * head_size, dim)           */
    float *rms_ffn_weight;  /* (n_layers, dim)                                */
    float *w1;              /* (n_layers, hidden_dim, dim)                    */
    float *w2;              /* (n_layers, dim, hidden_dim)                    */
    float *w3;              /* (n_layers, hidden_dim, dim)                    */
    float *rms_final_weight;/* (dim,)                                         */
    float *freq_cis_real;   /* (seq_len, head_size/2) — only in file v0       */
    float *freq_cis_imaj;   /* (seq_len, head_size/2) — only in file v0       */
    float *wcls;            /* (vocab_size, dim) classifier; may alias emb    */
} TransformerWeights;

/* The same six big matmuls, optionally held in int8. */
typedef struct {
    QMatrix *wq;  /* n_layers entries                                         */
    QMatrix *wk;
    QMatrix *wv;
    QMatrix *wo;
    QMatrix *w1;
    QMatrix *w2;
    QMatrix *w3;
    QMatrix *wcls; /* single entry; only used when not sharing embeddings     */
} QuantizedWeights;

/* ---------------------------------------------------------------------------
 * RunState — all the scratch buffers that exist while a forward pass runs.
 * The key_cache / value_cache persist across tokens: that IS the KV cache.
 * ------------------------------------------------------------------------- */
typedef struct {
    float *x;          /* activation at current time stamp (dim,)            */
    float *xb;         /* attention output (dim,)                            */
    float *xb2;        /* additional buffer (dim,)                           */
    float *hb;         /* hidden dim buffer for FFN (hidden_dim,)            */
    float *hb2;        /* hidden dim buffer for FFN (hidden_dim,)            */
    float *q;          /* query (dim,)                                       */
    float *k;          /* key (kv_dim,)                                      */
    float *v;          /* value (kv_dim,)                                    */
    float *att;        /* attention scores (n_heads, seq_len)                */
    float *logits;     /* output logits (vocab_size,)                        */
    float *key_cache;  /* (n_layers, seq_len, kv_dim)                        */
    float *value_cache;/* (n_layers, seq_len, kv_dim)                        */
} RunState;

typedef struct {
    Config cfg;
    float *data;              /* the raw weight blob read from disk           */
    size_t data_size;         /* bytes of `data`                              */
    TransformerWeights w;     /* fp32 view into `data`                        */
    QuantizedWeights qw;      /* valid only when `quantized` is 1             */
    int quantized;
    RunState s;
} Transformer;

/* BPE tokenizer ----------------------------------------------------------- */
typedef struct {
    const char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;            /* id -> utf-8 string                            */
    float *vocab_scores;     /* id -> BPE merge priority (lower = merged 1st) */
    TokenIndex *sorted_vocab;/* vocab sorted by string, for binary search     */
    int vocab_size;
    unsigned int max_token_length;
    char byte_pieces[512];   /* 256 raw-byte strings ("<0xXX>" tokens decode) */
} Tokenizer;

/* Sampling ---------------------------------------------------------------- */
typedef struct {
    int vocab_size;
    unsigned long long rng_state;
    float temperature;  /* 0 = greedy argmax                                  */
    float topp;         /* 0..1 nucleus threshold; <= 0 disables              */
    int topk;           /* keep k highest-prob tokens; <= 0 disables          */
    float *probindex;   /* scratch buffer for top-p filtering                 */
} Sampler;

/* transformer.c ------------------------------------------------------------ */
void   build_transformer(Transformer *t, const char *path, int quantize_flag);
void   free_transformer(Transformer *t);
float *forward(Transformer *t, int token, int pos);

/* tokenizer.c -------------------------------------------------------------- */
void   build_tokenizer(Tokenizer *t, const char *path, int vocab_size);
void   free_tokenizer(Tokenizer *t);
char  *decode(Tokenizer *t, int prev_token, int token);
void   encode(Tokenizer *t, const char *text, int bos, int eos, int *tokens, int *n_tokens);
void   safe_print(const char *piece);

/* sampler.c ---------------------------------------------------------------- */
void   build_sampler(Sampler *s, int vocab_size, float temperature, float topp, int topk, unsigned long long seed);
void   free_sampler(Sampler *s);
int    sample(Sampler *s, float *logits);

#endif /* NANOLLAMA_H */
