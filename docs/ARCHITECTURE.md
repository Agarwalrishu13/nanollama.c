# Architecture: how a Llama-2 transformer runs in this engine

This is the model that `forward()` in `src/transformer.c` implements. Read
this once and the code becomes a checklist.

## The cast of tensors

Every weight in the model file is one of these:

| tensor | shape | what it does |
|---|---|---|
| `token_embedding` | `(vocab, dim)` | lookup table: token id → meaning vector |
| `rms_att` | `(layers, dim)` | normalization gains before attention |
| `wq` `wk` `wv` | `(layers, dim, ...)` | project the token into queries, keys, values |
| `wo` | `(layers, dim, dim)` | merge the heads back together |
| `rms_ffn` | `(layers, dim)` | normalization gains before the MLP |
| `w1` `w3` | `(layers, hidden, dim)` | the SwiGLU MLP's two "branches" |
| `w2` | `(layers, dim, hidden)` | project the MLP back down |
| `rms_final` | `(dim,)` | last normalization before classification |
| `wcls` | `(vocab, dim)` | map the final vector to a score per vocabulary word |

For the 15M model: `dim=288`, `hidden=768`, `layers=6`, `heads=6`,
`head_size=48`, `vocab=32000`, `ctx=256`.

## One forward pass, step by step

```
token id ──► embedding row ──► x                      (dim,)

repeat × n_layers:

  1. x1 = rmsnorm(x, rms_att[l])                     normalize (keeps scale stable)
  2. q = wq @ x1     (dim)                           "what am I looking for?"
     k = wk @ x1     (kv_dim)                        "what do I contain?"
     v = wv @ x1     (kv_dim)                        "what do I give back if matched?"
  3. rotate q,k by position (RoPE)                   inject word ORDER into q,k
  4. write k,v into the KV cache at slot `pos`       memory of this conversation
  5. for each head h:
        scores = (q_h · k_past) / sqrt(head_size)    match strength vs every past token
        probs  = softmax(scores)                     attention weights (sum to 1)
        out_h  = Σ probs · v_past                    weighted blend of memories
  6. x += wo @ concat(out_h)                         heads merged + residual add
  7. x1 = rmsnorm(x, rms_ffn[l])
  8. x += w2 @ ( silu(w1 @ x1) ⊙ (w3 @ x1) )         SwiGLU MLP + residual

final:
  x = rmsnorm(x, rms_final)
  logits = wcls @ x                                  (vocab,) score per token
```

## The ideas that make it work

**Residuals (the `+=` lines).** Every block *adjusts* the vector instead of
replacing it. Gradients flow through the skip connections, which is what makes
deep stacks trainable.

**RMSNorm.** Divides by the root-mean-square of the vector, then rescales with
a learned gain. Same job as LayerNorm but without mean-centering — one less
pass over the data, same stability.

**RoPE (Rotary Position Embeddings).** Attention itself is order-blind — a
set, not a sequence. RoPE rotates each *pair* of features in q and k by an
angle proportional to the token's position. When you later dot a query with a
key, the rotated components interact so the score depends on the *distance*
between tokens. Two lines of trig in `rope()`, and the model knows word order.

**Multi-head attention.** One attention head can only track one kind of
relationship. Splitting the `dim` vector into `n_heads` chunks lets different
heads specialize (syntax? coreference? the rhyme scheme?). GQA (`n_kv_heads <
n_heads`) lets several query heads *share* a key/value head — same model
quality, much smaller KV cache.

**The KV cache.** Generation is one token at a time, but each new token must
attend to *all* previous ones. Instead of recomputing k,v for the whole prefix
every step, we append each token's k,v to `key_cache`/`value_cache` once and
reuse them forever. This turns generation from O(n²) recomputation into O(n)
incremental work — it's the single most important systems idea in LLM
inference, and it's why `forward()` only costs one forward pass per token
regardless of story length (up to `seq_len`).

**SwiGLU.** The MLP runs the token through two parallel projections and
multiplies them: `silu(w1@x) ⊙ (w3@x)`. The `w3` branch acts as a learned
gate, deciding which dimensions of the `silu` branch matter for this token.
LLaMA chose it over the classic GELU MLP for a better quality/FLOP trade.

**Sampling.** The logits are just scores. Temperature reshapes them
(sharper/flatter), top-k and top-p cut off the implausible tail, then we roll
a weighted die. Temperature 0 = always the argmax = deterministic. See
`src/sampler.c`.

## Quantization (the `-q` flag)

Each big matmul matrix is split row-wise into groups of 64 floats. Each group
keeps one fp32 scale `s = max|values| / 127`; every value is stored as an int8
`round(v / s)`. Dequantization happens *inside the matmul loop* — `q * s` — so
nothing is ever fully expanded. Result: 4x smaller weights, tiny accuracy
change (compare `-q` output to fp32 with the same seed and see for yourself).

## File format

A `.bin` model is: a 7-int header (`dim, hidden_dim, n_layers, n_heads,
n_kv_heads, vocab_size, seq_len`), then every tensor above back-to-back as raw
little-endian fp32 — older exports also embed two RoPE lookup tables we skip,
and the classifier is either its own tensor or shared with the embedding
table. `build_transformer()` detects which layout it got by matching the file
size against all legal layouts, so both era formats just work.
