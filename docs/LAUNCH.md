# Launch kit — getting nanollama.c its first 100 stars

The code is done; this file is the marketing plan. Stars come from
*distribution*, not from code quality alone. Pick one channel, launch, answer
every comment for 48 hours.

## Before launching (10 minutes)

- [ ] Create the repo on GitHub and push (commands below)
- [ ] Edit README: replace every `Agarwalrishu13` with your GitHub username
- [ ] Check the Actions badge goes green (CI builds on Ubuntu + macOS + Windows)
- [ ] Star your own repo, pin it on your profile, pin the repo on your profile page
- [ ] Fill in your profile README with this project + the demo GIF

## Push commands

```bash
cd nanollama.c
git remote add origin https://github.com/Agarwalrishu13/nanollama.c.git
git push -u origin main
```

or with the `gh` CLI: `gh repo create nanollama.c --public --source=. --push`

## Where to post, in order

**1. r/LocalLLaMA** (biggest LLM-inference community — best fit)

Title ideas:
- "I wrote a complete LLM inference engine from scratch in ~1400 lines of dependency-free C"
- "nanollama.c: a from-scratch LLM engine — no BLAS, no OpenMP, not even a thread library (I wrote the pool myself)"

Body: 3-4 sentences. What it is, the "why" (understanding > calling APIs),
one number (5.1x thread scaling, 172 tok/s on 15M), the GIF, ask for feedback
on the thread pool. **Answer every comment.** Post Tuesday–Thursday, morning
US time.

**2. r/C_Programming** — angle: the systems engineering (file format
auto-detection, the Win32 handle-array bug story, q8 quantization from
scratch). This community loves the "I hit a segfault and here's the bug"
story in the threads.c header.

**3. Hacker News (Show HN)** — title: "Show HN: Nanollama.c – LLM inference
from scratch in 1.4k lines of C". Submit via the Show HN link. Prepare for
"why not use OpenMP?" — answer: that's the point, the pool IS the lesson, and
the measured 40% spawn-overhead regression is documented in the code.

**4. dev.to / your blog** — the long-form writeup: "What I learned building
an LLM inference engine from scratch". Five lessons: (1) matmul is 90% of the
time, (2) the KV cache is the single most important systems idea, (3) thread
spawn overhead vs a real pool — with the benchmark chart, (4) BPE is just
"merge the best pair until it stops", (5) quantization is one scale per
group of 64. End with the repo link. This is what ranks on Google for months.

**5. Twitter/X + LinkedIn** — clip the GIF, one-sentence pitch, link.

## The story hooks (use these, they're true)

- "The first threading version made it SLOWER. Here's the benchmark that
  proved it, and the 150-line pool that fixed it."
- "The loader auto-detects the weight layout by matching the file size
  against every legal format — because the model files in the wild come in
  three generations of the format."
- "No dependencies. Not even OpenMP. The thread pool is part of the lesson."
- "You can read the entire engine in one sitting and understand what happens
  between an API call and the words coming out."

## 48-hour launch rhythm

- Hour 0: post to one community. Reply to every comment within an hour.
- Hour 24: post to the next community. Cross-link where allowed.
- Hour 48: publish the dev.to writeup.
- Week 2: add one roadmap item (GGUF loader) and post the update — updates
  are second launches.

## Measuring

Watch star history (star-history.com), but optimize for *clone* counts in the
traffic tab and the quality of issues — that's who will contribute.
