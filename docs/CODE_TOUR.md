# Code tour: reading nanollama.c in one sitting

~1,400 lines total. This is the reading order that makes everything click.

## 1. `src/nanollama.h` — the whole design on one page

Every struct you need: `Config` (7 integers describing the model),
`TransformerWeights` (pointers into the weight blob), `RunState` (scratch
buffers, including the KV cache), `Tokenizer`, `Sampler`. If you understand
these five structs, the implementation is just filling them in.

## 2. `src/transformer.c` — the model

Start at the bottom: **`forward()`** is ~80 lines and *is* the transformer.
Follow one token through: embedding lookup → per-layer (norm → qkv matmuls →
RoPE → attention over the cache → residual → norm → SwiGLU → residual) →
final norm → classifier.

Then read the helpers in the order `forward()` uses them:

- `rmsnorm()` — 8 lines. Notice it computes `1/sqrt(mean(x²)+ε)` and reuses
  the normalized vector for both the multiply and the gain.
- `rope()` — the two rotation lines that give the model word order.
- `matvec_rows()` / `dot_row()` — the hot loop. This is where 90% of wall-clock
  time goes: `out[i] = Σ row[j] * x[j]`. The `#if defined(__AVX2__)` block is
  the same math, 8 floats at a time with FMA instructions.
- `nl_parallel_for()` call in `matmul_any()` — how rows are split across
  threads (see `src/threads.c`).

Then go back to the top: **`build_transformer()`**. Watch it read the header,
detect which of the four legal weight layouts the file uses *by file size*
(`base + freq + wcls` vs `base` vs …), slurp the whole blob with one `fread`,
and carve it into tensor pointers with pointer arithmetic. This is what
"loading a model" actually is: no magic, just offsets.

Finally `quantize_weights()` / `quantize_matrix()` — grouping values into
chunks of 64, one scale per group, `int8 = round(v / scale)`.

## 3. `src/tokenizer.c` — text ⇄ ids

- `build_tokenizer()` walks the entry stream: `(score, len, utf-8 bytes)`
  repeated to EOF. The leading int is *not* the count (historical quirk — the
  comment explains), so we count as we walk.
- `decode()` — mostly about the two gotchas: the space after `<s>`, and
  `<0xNN>` byte-fallback tokens that decode to raw bytes.
- `encode()` is the heart. Pass 1: split the prompt into utf-8 codepoints,
  each a token if it exists in the vocab, else byte-fallback (`byte + 3`).
  Pass 2: repeatedly find the adjacent pair whose *concatenation* has the
  highest merge score and merge it. That loop — "merge the best pair until
  nothing merges" — is all BPE is.

## 4. `src/sampler.c` — picking the next token

`sample()` reads top to bottom exactly like the pipeline: greedy argmax if
`temperature <= 0`; else scale by temperature → top-k cutoff via a min-heap
(no sorting 32,000 floats) → softmax → top-p nucleus filter → one weighted
die roll from xorshift64*. Note `sample_topp()`'s quickfilter: any probability
below `(1-topp)/(n-1)` mathematically cannot be inside the nucleus, so the
insertion sort afterwards only ever sorts a handful of candidates.

## 5. `src/threads.c` — the pool, and the lesson

Read the header comment first: v0 of this file spawned threads per matmul and
was 40% slower than single-threaded. The pool version — workers parked on a
semaphore, chunk 0 running on the caller's stack, a done-semaphore as the
barrier — is what turned 34 tok/s into 172 tok/s.

Then the mechanics: `nl_worker()`'s park/wake/report loop, `nl_parallel_for()`'s
chunk splitting, and why every worker joins the barrier every time (empty
chunks keep the wake/done token counts balanced — see the comment).

## 6. `src/main.c` — putting it together

`run_generation()` is the loop every LLM app is secretly made of: prefill the
prompt tokens without printing, then sample one token at a time, decode, print.
`cmd_bench()` shows how to measure honestly (warmup, fixed token ids, forward
passes only). The arg parsing is deliberately boring.

---

## Things to try (great first contributions)

1. Run the same prompt fp32 vs `-q` with `-s 42` and diff the outputs —
   quantify what quantization costs.
2. Change `rope()`'s 10000.0 base and watch quality shift.
3. Break the KV cache (recompute k,v every step) and measure the slowdown —
   this is *the* proof it matters.
4. Add an ARM NEON kernel next to the AVX2 one in `dot_row()`.
5. Print the top-5 candidate tokens per step with their probabilities.
