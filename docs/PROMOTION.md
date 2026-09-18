# Promotion kit — nanollama.c × nanobrain

Everything below is ready to paste. Launch order matters: one channel at a
time, 24h apart, reply to every comment for 48h.

## Already done

- [x] Topics/tags on both repos (llm, inference, from-scratch, education…)
- [x] Cross-links between the repos; both pinned on the profile
- [x] Profile README with the nano-universe story
- [x] PR to codecrafters-io/build-your-own-x (#2011) — "Build your own AI Model"
- [x] PR to karpathy/llama2.c (#583) — the ecosystem list
- [ ] (you) upload social preview images: Repo → Settings → Social preview
      (use docs/demo.gif frames or benchmark charts as the image)
- [ ] (you) enable Discussions on nanollama.c (Settings → Features)

## 1. Reddit r/LocalLLaMA (post Tuesday–Thursday, morning US time)

**Title:**
> I trained my own LLM from scratch on my laptop's CPU — then ran it on the inference engine I also wrote from scratch in C

**Body:**
> After realizing I had called LLM APIs for years without knowing what
> happens under the hood, I built the whole stack myself.
>
> **nanollama.c** — the engine: a complete LLM inference engine in ~1,400
> lines of dependency-free C. No BLAS, no OpenMP (I wrote the thread pool —
> my first version spawning threads per matmul was 40% SLOWER, the pool is
> what took 15M-param inference to 172 tok/s), int8 quantization, BPE
> tokenizer, top-p sampling. CI-tested on Linux/macOS/Windows.
>
> **nanobrain** — the brain factory: a hand-written Llama-2 transformer +
> self-trained BPE tokenizer that exports models into the engine's format.
> My flagship (16.9M params) trained on CPU for 35 hours (my GPU driver NaN'd
> twice — log included) and now writes stories through my own engine.
>
> Everything is documented honestly, including every bug: a loss of 124 at
> step 1 (forgot init), an off-by-one-int my export caught in my own loader,
> and the thread-pool regression. Feedback welcome — especially on the
> quantization and the thread pool.
>
> Repos: links in profile / comments.

## 2. Hacker News — Show HN

**Title:** `Show HN: I built an LLM engine in C and trained the model that runs on it`

**First comment (yours):** the two-repo story, the "40% slower" thread-pool
lesson, the "engine caught the exporter's bug" bit, then the links. Expect
the "why not OpenMP/BLAS" question — the answer is the project.

## 3. X/Twitter thread (5 tweets)

1. "I spent 2 months building an LLM from the bottom up. No APIs. Here's the thread 🧵"
2. "The engine: 1,400 lines of C. Own thread pool (my first one was 40% SLOWER — screenshot of the benchmark). Own int8 quantization. Own tokenizer."
3. "The brain: I trained a 16.9M-param model on my CPU for 35 hours because my GPU driver was broken. It writes stories now."
4. "The full circle: model I trained → exported → running on the engine I wrote. The engine even caught a bug in my exporter."
5. "Both repos, MIT, with the full bug diary. Links below. ⭐ s appreciated but comments/feedback more so."

## 4. dev.to article (evergreen SEO)

**Title:** *What I learned building an LLM inference engine AND training
pipeline from scratch*
Outline: (1) matmul is 90% of the time, (2) the KV cache is the most
important systems idea in inference, (3) thread-pool vs spawn-per-call —
with the 40%-slower chart, (4) tokenizers are just "merge the best pair
until it stops", (5) what quantization actually costs — same seed, two
outputs, (6) the two-bug story: exporter finds loader bug; driver NaNs
move training to CPU. End: both repos.

## 5. Where NOT to spam

Don't post in r/MachineLearning (too research), don't open issues on
llama.cpp/ollama advertising, don't mass-DM. One good HN post outperforms
ten mediocre crossposts.

## Measuring

Traffic tab → clones + referrers. Star velocity after each channel. The
build-your-own-x and llama2.c PRs may take days/weeks to merge — each merge
is a slow-drip source of visitors forever.
