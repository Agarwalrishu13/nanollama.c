/*
 * transformer.c — model loading, the forward pass, and int8 quantization.
 *
 * nanollama reads the llama2.c model format. Files in the wild come in a few
 * flavors (the format evolved during 2023), so the loader auto-detects which
 * one it got by matching the file size against every legal layout:
 *
 *   header:  int dim, hidden_dim, n_layers, n_heads, n_kv_heads,
 *                vocab_size, seq_len
 *            (newer exports prepend a magic int 0x616b3432 + version)
 *
 *   weights: token_embedding, rms_att, wq, wk, wv, wo, rms_ffn,
 *            w1, w2, w3, rms_final,
 *            [2 x RoPE lookup tables — legacy files only; we skip them and
 *             compute RoPE on the fly instead],
 *            [classifier wcls — absent when it shares the embedding table]
 *
 * Everything else is the Llama-2 architecture, exactly as described in
 * docs/ARCHITECTURE.md.
 */

#include "nanollama.h"

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

/* ---------------------------------------------------------------------------
 * Small math kernels
 * ------------------------------------------------------------------------- */

/* y = x / rms(x) * weight   (Root Mean Square Layer Norm) */
static void rmsnorm(float *out, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= (float)size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        out[j] = x[j] * ss * weight[j];
    }
}

/* in-place softmax over the first `size` elements (numerically stable) */
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

/* Rotary position embeddings, applied in place to q or k.
 * Pairs of features (2i, 2i+1) are rotated by an angle proportional to
 * pos / 10000^(2i/head_size). This is how the model knows token order. */
static void rope(float *vec, int pos, int n_heads, int head_size) {
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_size;
        for (int i = 0; i < head_size; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
            float angle = (float)pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float v0 = v[i], v1 = v[i + 1];
            v[i]     = v0 * cs - v1 * sn;
            v[i + 1] = v0 * sn + v1 * cs;
        }
    }
}

/* ---------------------------------------------------------------------------
 * Matrix-vector multiply: out[d] = W(d,n) @ x[n].
 * This is >90% of total inference time, which is why it gets the parallel-for
 * and the optional AVX2 path. Rows are split across threads by threads.c.
 * ------------------------------------------------------------------------- */

typedef struct {
    const float   *x;    /* input vector, n floats                        */
    const float   *w;    /* fp32 matrix (d rows of n) — fp32 path         */
    const QMatrix *wq;   /* int8 matrix — quantized path                  */
    float         *out;  /* output vector, d floats                       */
    int            n;    /* row length (input dim)                        */
} MatVecCtx;

/* one row of W @ x, with the AVX2 8-wide FMA kernel when compiled for it */
static void dot_row(const float *x, const float *row, const QMatrix *wq,
                    int i, int n, float *out) {
#if defined(__AVX2__)
    if (wq == NULL) {
        const float *r = row + (size_t)i * n;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vr = _mm256_loadu_ps(r + j);
            acc = _mm256_fmadd_ps(vx, vr, acc);
        }
        float tail[8];
        _mm256_storeu_ps(tail, acc);
        float val = tail[0] + tail[1] + tail[2] + tail[3]
                  + tail[4] + tail[5] + tail[6] + tail[7];
        for (; j < n; j++) {
            val += r[j] * x[j];
        }
        out[i] = val;
        return;
    }
#endif
    if (wq == NULL) {
        const float *r = row + (size_t)i * n;
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += r[j] * x[j];
        }
        out[i] = val;
    } else {
        /* int8: dequantize one QGS-group of the row at a time */
        const int8_t *rq = wq->q + (size_t)i * n;
        const float  *rs = wq->s + (size_t)i * wq->groups_per_row;
        float val = 0.0f;
        for (int j = 0; j < n; j += QGS) {
            float scale = rs[j / QGS];
            int end = j + QGS < n ? j + QGS : n;
            for (int k = j; k < end; k++) {
                val += ((float)rq[k]) * scale * x[k];
            }
        }
        out[i] = val;
    }
}

static void matvec_rows(int start, int end, int id, void *vctx) {
    (void)id;
    MatVecCtx *c = (MatVecCtx *)vctx;
    for (int i = start; i < end; i++) {
        dot_row(c->x, c->w, c->wq, i, c->n, c->out);
    }
}

static void matmul_any(float *out, const float *x, const float *w_fp32,
                       const QMatrix *w_q, int quantized, int n, int d) {
    MatVecCtx ctx = { x, w_fp32, quantized ? w_q : NULL, out, n };
    nl_parallel_for(d, matvec_rows, &ctx, nl_get_threads());
}

/* ---------------------------------------------------------------------------
 * Quantization: fp32 -> grouped int8 (4x smaller files, faster matmuls)
 * ------------------------------------------------------------------------- */

/* Quantize one matrix, group by group WITHIN each row (llama.cpp q8_0 style).
 * Per-row grouping keeps the scale lookup O(1) in the hot matmul loop and
 * handles row lengths that are not a multiple of QGS. */
static void quantize_matrix(QMatrix *qm, float *src, size_t rows, size_t cols) {
    int gpr = (int)((cols + QGS - 1) / QGS);
    qm->groups_per_row = gpr;
    qm->s = (float *)malloc(rows * (size_t)gpr * sizeof(float));
    qm->q = (int8_t *)malloc(rows * cols * sizeof(int8_t));
    if (!qm->s || !qm->q) NL_ERROR("out of memory during quantization");

    for (size_t r = 0; r < rows; r++) {
        const float *row = src + r * cols;
        for (int g = 0; g < gpr; g++) {
            size_t start = (size_t)g * QGS;
            size_t gsize = cols - start < QGS ? cols - start : QGS;
            float max_abs = 0.0f;
            for (size_t j = 0; j < gsize; j++) {
                float a = row[start + j];
                if (a < 0) a = -a;
                if (a > max_abs) max_abs = a;
            }
            float scale = max_abs != 0.0f ? max_abs / 127.0f : 1.0f;
            qm->s[r * gpr + g] = scale;
            for (size_t j = 0; j < gsize; j++) {
                qm->q[r * cols + start + j] = (int8_t)roundf(row[start + j] / scale);
            }
        }
    }
}

/* Quantize the six big matmul weights (and the classifier, if standalone).
 * Embedding tables and RMSNorm gains stay fp32: they are tiny and accuracy-
 * sensitive, so we get ~4x savings where it actually matters. */
static void quantize_weights(Transformer *t) {
    Config *p = &t->cfg;
    int dim = p->dim, hidden = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = head_size * p->n_kv_heads;
    size_t L = (size_t)p->n_layers;

    t->qw.wq = (QMatrix *)malloc(L * sizeof(QMatrix));
    t->qw.wk = (QMatrix *)malloc(L * sizeof(QMatrix));
    t->qw.wv = (QMatrix *)malloc(L * sizeof(QMatrix));
    t->qw.wo = (QMatrix *)malloc(L * sizeof(QMatrix));
    t->qw.w1 = (QMatrix *)malloc(L * sizeof(QMatrix));
    t->qw.w2 = (QMatrix *)malloc(L * sizeof(QMatrix));
    t->qw.w3 = (QMatrix *)malloc(L * sizeof(QMatrix));
    if (!t->qw.wq || !t->qw.wk || !t->qw.wv || !t->qw.wo ||
        !t->qw.w1 || !t->qw.w2 || !t->qw.w3) {
        NL_ERROR("out of memory during quantization");
    }

    for (size_t l = 0; l < L; l++) {
        quantize_matrix(&t->qw.wq[l], t->w.wq + l * dim * dim,    (size_t)dim,    (size_t)dim);
        quantize_matrix(&t->qw.wk[l], t->w.wk + l * dim * kv_dim, (size_t)dim,    (size_t)kv_dim);
        quantize_matrix(&t->qw.wv[l], t->w.wv + l * dim * kv_dim, (size_t)dim,    (size_t)kv_dim);
        quantize_matrix(&t->qw.wo[l], t->w.wo + l * dim * dim,    (size_t)dim,    (size_t)dim);
        quantize_matrix(&t->qw.w1[l], t->w.w1 + l * hidden * dim, (size_t)hidden, (size_t)dim);
        quantize_matrix(&t->qw.w2[l], t->w.w2 + l * dim * hidden, (size_t)dim,    (size_t)hidden);
        quantize_matrix(&t->qw.w3[l], t->w.w3 + l * hidden * dim, (size_t)hidden, (size_t)dim);
    }
    if (!p->shared_classifier) {
        t->qw.wcls = (QMatrix *)malloc(sizeof(QMatrix));
        if (!t->qw.wcls) NL_ERROR("out of memory during quantization");
        quantize_matrix(t->qw.wcls, t->w.wcls, (size_t)p->vocab_size, (size_t)dim);
    }
    t->quantized = 1;
}

/* ---------------------------------------------------------------------------
 * Loading
 * ------------------------------------------------------------------------- */

void build_transformer(Transformer *t, const char *path, int quantize_flag) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "[nanollama] couldn't open model file %s\n", path);
        exit(EXIT_FAILURE);
    }

    int first_int;
    if (fread(&first_int, sizeof(int), 1, file) != 1) {
        NL_ERROR("model file is too small — is this a valid .bin model?");
    }

    int *cfg_ints = (int *)&t->cfg;
    int version = 0;                 /* 0 = legacy header (no magic number) */
    if (first_int == 0x616b3432) {   /* 'ak42' magic from newer exporters   */
        if (fread(&version, sizeof(int), 1, file) != 1) {
            NL_ERROR("failed to read model version");
        }
        if (fread(cfg_ints, sizeof(int), 7, file) != 7) {
            NL_ERROR("failed to read model header");
        }
    } else {
        /* legacy: no magic — the first int is already the config's dim */
        cfg_ints[0] = first_int;
        if (fread(&cfg_ints[1], sizeof(int), 6, file) != 6) {
            NL_ERROR("failed to read model header");
        }
    }

    Config *p = &t->cfg;
    int dim = p->dim, hidden = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = head_size * p->n_kv_heads;
    size_t L = (size_t)p->n_layers;
    size_t V = (size_t)p->vocab_size;
    size_t S = (size_t)p->seq_len;

    /* ---- layout detection -------------------------------------------------
     * Compute the expected blob size for every legal layout and pick the one
     * that matches the actual file size. base = the 11 tensors every file has;
     * legacy-era files additionally embed 2 RoPE lookup tables; the classifier
     * wcls is either its own tensor or shares token_embedding.                */
    size_t base = V * dim                         /* token embeddings       */
        + L * dim                                 /* rms_att                */
        + 2 * L * (size_t)dim * dim               /* wq + wo                */
        + 2 * L * (size_t)dim * kv_dim            /* wk + wv                */
        + L * dim                                 /* rms_ffn                */
        + 3 * L * (size_t)hidden * dim            /* w1 + w2 + w3           */
        + dim;                                    /* rms_final              */
    size_t freq = 2 * S * (size_t)(head_size / 2);/* legacy RoPE tables     */
    size_t vd = V * dim;                          /* standalone classifier  */

    /* v1+ files state the layout explicitly: read the shared flag BEFORE
     * measuring the blob, so the header-end position is exact */
    int has_freq, has_wcls;
    if (version >= 1) {
        has_freq = 0;
        if (fread(&p->shared_classifier, sizeof(int), 1, file) != 1) {
            NL_ERROR("failed to read shared_classifier flag");
        }
        has_wcls = !p->shared_classifier;
    }

    /* actual number of floats after the (complete) header */
    long file_pos = ftell(file);
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, file_pos, SEEK_SET);
    if ((file_size - file_pos) % sizeof(float) != 0) {
        NL_ERROR("model file size doesn't line up with the header");
    }
    size_t n_floats = (size_t)(file_size - file_pos) / sizeof(float);

    if (version >= 1) {
        size_t expect = base + (has_wcls ? vd : 0);
        if (n_floats != expect) {
            fprintf(stderr, "[nanollama] warning: header promises %zu floats, file has %zu\n",
                    expect, n_floats);
        }
    } else if (n_floats == base + freq + vd) {
        has_freq = 1; has_wcls = 1; p->shared_classifier = 0;
    } else if (n_floats == base + freq) {
        has_freq = 1; has_wcls = 0; p->shared_classifier = 1;
    } else if (n_floats == base + vd) {
        has_freq = 0; has_wcls = 1; p->shared_classifier = 0;
    } else if (n_floats == base) {
        has_freq = 0; has_wcls = 0; p->shared_classifier = 1;
    } else {
        fprintf(stderr, "[nanollama] file size doesn't match any known weight layout "
                        "(got %zu floats; base=%zu freq=%zu wcls=%zu)\n",
                n_floats, base, freq, vd);
        NL_ERROR("not a llama2.c model, or it's corrupted — re-download it with "
                 "python scripts/download_models.py");
    }

    /* every tensor is contiguous — read the whole blob in one fread */
    t->data_size = n_floats * sizeof(float);
    t->data = (float *)malloc(t->data_size);
    if (!t->data) NL_ERROR("out of memory for model weights");
    if (fread(t->data, 1, t->data_size, file) != t->data_size) {
        NL_ERROR("model file truncated while reading weights");
    }
    fclose(file);

    /* map the pointers, in exactly the order the exporter wrote them */
    float *ptr = t->data;
    TransformerWeights *w = &t->w;
    w->token_embedding = ptr; ptr += V * dim;
    w->rms_att_weight  = ptr; ptr += L * dim;
    w->wq              = ptr; ptr += L * (size_t)dim * dim;
    w->wk              = ptr; ptr += L * (size_t)dim * kv_dim;
    w->wv              = ptr; ptr += L * (size_t)dim * kv_dim;
    w->wo              = ptr; ptr += L * (size_t)dim * dim;
    w->rms_ffn_weight  = ptr; ptr += L * dim;
    w->w1              = ptr; ptr += L * (size_t)hidden * dim;
    w->w2              = ptr; ptr += L * (size_t)dim * hidden;
    w->w3              = ptr; ptr += L * (size_t)hidden * dim;
    w->rms_final_weight= ptr; ptr += dim;
    if (has_freq) {
        /* skipped: we compute RoPE on the fly, no lookup tables needed */
        w->freq_cis_real = ptr; ptr += S * (head_size / 2);
        w->freq_cis_imaj = ptr; ptr += S * (head_size / 2);
    } else {
        w->freq_cis_real = w->freq_cis_imaj = NULL;
    }
    w->wcls = p->shared_classifier ? w->token_embedding : ptr;
    t->quantized = 0;

    if (quantize_flag) {
        quantize_weights(t);
    }

    /* allocate the run state buffers */
    RunState *s = &t->s;
    s->x           = (float *)calloc(dim,               sizeof(float));
    s->xb          = (float *)calloc(dim,               sizeof(float));
    s->xb2         = (float *)calloc(dim,               sizeof(float));
    s->hb          = (float *)calloc(hidden,            sizeof(float));
    s->hb2         = (float *)calloc(hidden,            sizeof(float));
    s->q           = (float *)calloc(dim,               sizeof(float));
    s->k           = (float *)calloc(kv_dim,            sizeof(float));
    s->v           = (float *)calloc(kv_dim,            sizeof(float));
    s->att         = (float *)calloc((size_t)p->n_heads * S, sizeof(float));
    s->logits      = (float *)calloc(V,                 sizeof(float));
    s->key_cache   = (float *)calloc(L * S * (size_t)kv_dim, sizeof(float));
    s->value_cache = (float *)calloc(L * S * (size_t)kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k ||
        !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        NL_ERROR("out of memory for run-state buffers");
    }
}

void free_transformer(Transformer *t) {
    RunState *s = &t->s;
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    if (t->quantized) {
        Config *p = &t->cfg;
        for (int l = 0; l < p->n_layers; l++) {
            free(t->qw.wq[l].q); free(t->qw.wq[l].s);
            free(t->qw.wk[l].q); free(t->qw.wk[l].s);
            free(t->qw.wv[l].q); free(t->qw.wv[l].s);
            free(t->qw.wo[l].q); free(t->qw.wo[l].s);
            free(t->qw.w1[l].q); free(t->qw.w1[l].s);
            free(t->qw.w2[l].q); free(t->qw.w2[l].s);
            free(t->qw.w3[l].q); free(t->qw.w3[l].s);
        }
        free(t->qw.wq); free(t->qw.wk); free(t->qw.wv); free(t->qw.wo);
        free(t->qw.w1); free(t->qw.w2); free(t->qw.w3);
        if (!p->shared_classifier) {
            free(t->qw.wcls->q); free(t->qw.wcls->s); free(t->qw.wcls);
        }
    }
    free(t->data);
}

/* ---------------------------------------------------------------------------
 * The forward pass: one token in, next-token logits out.
 * token: the current token id; pos: its position in the sequence (0-based).
 * The KV cache makes this O(1) per token regardless of how far we are into
 * generation — the reason LLM text streaming is possible at all.
 * ------------------------------------------------------------------------- */
float *forward(Transformer *t, int token, int pos) {
    Config *p = &t->cfg;
    RunState *s = &t->s;
    TransformerWeights *w = &t->w;

    int dim = p->dim;
    int head_size = dim / p->n_heads;
    int kv_dim = head_size * p->n_kv_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; /* query heads per kv head (GQA) */
    int hidden = p->hidden_dim;
    int L = p->n_layers;

    /* copy the token's embedding vector as the initial activation */
    memcpy(s->x, w->token_embedding + (size_t)token * dim, dim * sizeof(float));

    for (int l = 0; l < L; l++) {
        /* --- attention block ------------------------------------------------ */
        rmsnorm(s->xb, s->x, w->rms_att_weight + (size_t)l * dim, dim);

        /* qkv projections — the only matmuls that happen per token */
        float *krow = s->key_cache   + (size_t)l * p->seq_len * kv_dim + (size_t)pos * kv_dim;
        float *vrow = s->value_cache + (size_t)l * p->seq_len * kv_dim + (size_t)pos * kv_dim;
        matmul_any(s->q,    s->xb, w->wq + (size_t)l * dim * dim,      &t->qw.wq[l], t->quantized, dim, dim);
        matmul_any(krow,    s->xb, w->wk + (size_t)l * dim * kv_dim,   &t->qw.wk[l], t->quantized, dim, kv_dim);
        matmul_any(vrow,    s->xb, w->wv + (size_t)l * dim * kv_dim,   &t->qw.wv[l], t->quantized, dim, kv_dim);

        /* give q and k their positional information */
        rope(s->q, pos, p->n_heads, head_size);
        rope(krow, pos, p->n_kv_heads, head_size);

        /* multi-head attention: each query head looks back at every token */
        for (int h = 0; h < p->n_heads; h++) {
            float *qh = s->q + h * head_size;
            float *att = s->att + (size_t)h * p->seq_len;

            /* scores against every cached key, scaled by 1/sqrt(head_size) */
            for (int step = 0; step <= pos; step++) {
                float *kh = s->key_cache + (size_t)l * p->seq_len * kv_dim
                          + (size_t)step * kv_dim + (size_t)(h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += qh[i] * kh[i];
                }
                att[step] = score / sqrtf((float)head_size);
            }
            softmax_inplace(att, pos + 1);

            /* blend the values weighted by attention probabilities */
            float *xbh = s->xb + h * head_size;
            memset(xbh, 0, head_size * sizeof(float));
            for (int step = 0; step <= pos; step++) {
                float *vh = s->value_cache + (size_t)l * p->seq_len * kv_dim
                          + (size_t)step * kv_dim + (size_t)(h / kv_mul) * head_size;
                float a = att[step];
                for (int i = 0; i < head_size; i++) {
                    xbh[i] += a * vh[i];
                }
            }
        }

        /* project heads back to model dimension, add the residual */
        matmul_any(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, &t->qw.wo[l], t->quantized, dim, dim);
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }

        /* --- feed-forward block (SwiGLU) ------------------------------------ */
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + (size_t)l * dim, dim);
        matmul_any(s->hb,  s->xb, w->w1 + (size_t)l * hidden * dim, &t->qw.w1[l], t->quantized, dim, hidden);
        matmul_any(s->hb2, s->xb, w->w3 + (size_t)l * hidden * dim, &t->qw.w3[l], t->quantized, dim, hidden);
        for (int i = 0; i < hidden; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val))); /* SiLU(x) = x * sigmoid(x) */
            val *= s->hb2[i];                    /* elementwise gate         */
            s->hb[i] = val;
        }
        matmul_any(s->xb, s->hb, w->w2 + (size_t)l * dim * hidden, &t->qw.w2[l], t->quantized, hidden, dim);
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i]; /* residual */
        }
    }

    /* final norm, then project to vocabulary logits */
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul_any(s->logits, s->x, w->wcls,
               t->qw.wcls, t->quantized && !p->shared_classifier,
               dim, p->vocab_size);
    return s->logits;
}
