/*
 * main.c — the nanollama.c command line interface.
 *
 *   nanollama generate -m model.bin -p "Once upon a time"   write a story
 *   nanollama chat     -m model.bin                          turn-by-turn REPL
 *   nanollama bench    -m model.bin                          tokens/sec sweep
 *
 * Full flag reference: ./nanollama --help
 */

#include "nanollama.h"

/* ---------------------------------------------------------------------------
 * tiny arg-parsing helpers
 * ------------------------------------------------------------------------- */
static int g_i = 0; /* current argv index, shared with the parse loop */

static char *next_arg(int argc, char **argv, const char *flag) {
    if (g_i + 1 >= argc) {
        fprintf(stderr, "[nanollama] missing value for %s\n", flag);
        exit(EXIT_FAILURE);
    }
    return argv[++g_i];
}

static void usage(void) {
    printf(
        "nanollama.c — a from-scratch LLM inference engine in pure C\n"
        "\n"
        "usage:\n"
        "  nanollama generate -m models/stories15M.bin [flags]\n"
        "  nanollama chat     -m models/stories15M.bin [flags]\n"
        "  nanollama bench    -m models/stories15M.bin [flags]\n"
        "\n"
        "flags:\n"
        "  -m <path>      model file (llama2.c .bin format)          [required]\n"
        "  -z <path>      tokenizer file          (models/tokenizer.bin)\n"
        "  -p <text>      prompt for generate     (empty = start from <s>)\n"
        "  -n <int>       tokens to generate      (256)\n"
        "  -t <float>     temperature             (0.8, 0 = greedy)\n"
        "  --topp <float> nucleus sampling        (0.9, 0 = off)\n"
        "  -k <int>       top-k                   (0 = off)\n"
        "  -s <int>       rng seed                (0 = time-based)\n"
        "  -i             interactive mode for generate\n"
        "  -th <int>      worker threads          (all cores)\n"
        "  -q             quantize weights to int8 at load time\n"
        "  --csv <path>   (bench) append results to a CSV file\n"
        "  -h, --help     this help\n"
        "\n"
        "models: python scripts/download_models.py   (TinyStories, 260K..110M params)\n"
    );
}

/* ---------------------------------------------------------------------------
 * generation loop, shared by `generate` and `chat`
 * ------------------------------------------------------------------------- */
/* returns tokens generated */
static int run_generation(Transformer *t, Tokenizer *tok, Sampler *smp,
                          const char *prompt, int n_gen, int echo) {
    Config *p = &t->cfg;
    int n_prompt = 0;
    int *prompt_tokens = NULL;

    if (prompt != NULL && prompt[0] != '\0') {
        prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
        if (!prompt_tokens) NL_ERROR("out of memory for prompt");
        encode(tok, prompt, 1, 0, prompt_tokens, &n_prompt);
        if (n_prompt < 1) NL_ERROR("prompt encoded to nothing");
        if (n_prompt > p->seq_len - 1) {
            fprintf(stderr, "[nanollama] prompt is longer than the model's context "
                            "(%d > %d) — truncating\n", n_prompt, p->seq_len - 1);
            n_prompt = p->seq_len - 1;
        }
    }

    int steps = p->seq_len;                    /* KV cache hard limit */
    int limit = (prompt_tokens ? n_prompt : 1) + n_gen;
    if (limit < steps) steps = limit;

    int token = prompt_tokens ? prompt_tokens[0] : 1; /* 1 = <s> */
    int pos = 0, generated = 0, prev_token = -1;

    while (pos < steps) {
        float *logits = forward(t, token, pos);

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];     /* prefill: no sampling */
        } else {
            next = sample(smp, logits);
        }
        pos++;

        if (next == 1 || next == 2) break;     /* <s> or </s> end generation */

        token = next;
        if (echo && pos > n_prompt - 1) {      /* only print newly generated */
            char *piece = decode(tok, prev_token, token);
            safe_print(piece);
            fflush(stdout);
        }
        prev_token = token;
        generated++;
    }
    if (echo) printf("\n");

    free(prompt_tokens);
    return generated;
}

static void cmd_generate(const char *model, const char *tokenizer, const char *prompt,
                         int n_gen, float temp, float topp, int topk, unsigned long long seed,
                         int threads, int quantize, int interactive) {
    Transformer t;
    Tokenizer tok;
    Sampler smp;

    double t0 = now_ms();
    build_transformer(&t, model, quantize);
    double t1 = now_ms();
    build_tokenizer(&tok, tokenizer, t.cfg.vocab_size);
    build_sampler(&smp, t.cfg.vocab_size, temp, topp, topk, seed);

    if (threads > 0) {
        nl_set_threads(threads);
    }

    printf("> model: %s  |  dim=%d layers=%d heads=%d vocab=%d ctx=%d%s\n",
           model, t.cfg.dim, t.cfg.n_layers, t.cfg.n_heads, t.cfg.vocab_size,
           t.cfg.seq_len, quantize ? "  [int8 quantized]" : "");
    fprintf(stderr, "[nanollama] loaded in %.1f ms\n", t1 - t0);

    if (interactive) {
        char buf[2048];
        printf("> interactive mode — empty line to quit\n");
        while (1) {
            printf(">> ");
            fflush(stdout);
            if (!fgets(buf, sizeof(buf), stdin)) break;
            if (buf[0] == '\n' || buf[0] == '\0') break;
            buf[strcspn(buf, "\r\n")] = '\0';
            run_generation(&t, &tok, &smp, buf, n_gen, 1);
        }
    } else {
        double g0 = now_ms();
        int n = run_generation(&t, &tok, &smp, prompt, n_gen, 1);
        double g1 = now_ms();
        if (n > 0) {
            fprintf(stderr, "[nanollama] %.1f tok/s (%d tokens in %.0f ms)\n",
                    (double)n / ((g1 - g0) / 1000.0), n, g1 - g0);
        }
    }

    free_sampler(&smp);
    free_tokenizer(&tok);
    free_transformer(&t);
}

/* ---------------------------------------------------------------------------
 * chat mode: turn-by-turn REPL with an optional turn template.
 * TinyStories models are not chat-tuned — they just continue the text —
 * which is exactly the kind of thing you discover by building this yourself.
 * ------------------------------------------------------------------------- */
static void cmd_chat(const char *model, const char *tokenizer, int n_gen,
                     float temp, float topp, int topk, unsigned long long seed,
                     int threads, int quantize) {
    Transformer t;
    Tokenizer tok;
    Sampler smp;

    build_transformer(&t, model, quantize);
    build_tokenizer(&tok, tokenizer, t.cfg.vocab_size);
    build_sampler(&smp, t.cfg.vocab_size, temp, topp, topk, seed);

    if (threads > 0) {
        nl_set_threads(threads);
    }

    printf("=====================================================\n"
           " nanollama chat\n"
           " model: %s\n"
           " note:  TinyStories models are not chat-tuned; they\n"
           "        continue your text. Use a Llama-2-chat model\n"
           "        (or wait for GGUF support) for real chatting.\n"
           " empty line to quit\n"
           "=====================================================\n", model);

    char buf[2048];
    char turn[2300];
    while (1) {
        printf("\n\x1b[36m you \x1b[0m> ");
        fflush(stdout);
        if (!fgets(buf, sizeof(buf), stdin)) break;
        if (buf[0] == '\n' || buf[0] == '\0') break;
        buf[strcspn(buf, "\r\n")] = '\0';

        snprintf(turn, sizeof(turn), "\n[INST] %s [/INST]\n", buf);
        printf("\x1b[35m nano \x1b[0m> ");
        fflush(stdout);
        run_generation(&t, &tok, &smp, turn, n_gen, 1);
    }

    free_sampler(&smp);
    free_tokenizer(&tok);
    free_transformer(&t);
}

/* ---------------------------------------------------------------------------
 * bench mode: pure forward-pass throughput, sweeping threads and dtype.
 * No tokenizer needed — determinism comes from feeding fixed token ids.
 * ------------------------------------------------------------------------- */
static double bench_one(const char *model, int quantize, int threads, int steps) {
    nl_set_threads(threads);
    Transformer t;
    build_transformer(&t, model, quantize);

    /* warmup: page in weights, touch the caches */
    for (int pos = 0; pos < 8; pos++) {
        forward(&t, 7, pos);
    }

    double t0 = now_ms();
    for (int pos = 0; pos < steps; pos++) {
        forward(&t, 7, pos);
    }
    double dt = now_ms() - t0;

    free_transformer(&t);
    return (double)steps / (dt / 1000.0);      /* tokens per second */
}

static void cmd_bench(const char *model, int steps, int max_threads,
                      int quantize_too, const char *csv_path) {
    int hw_threads = nl_thread_count();
    if (max_threads > 0 && max_threads < hw_threads) {
        hw_threads = max_threads;
    }

    int thread_counts[8];
    int n_th = 0;
    for (int th = 1; th <= hw_threads && n_th < 8; th *= 2) {
        thread_counts[n_th++] = th;
    }

    printf("benchmarking %s (%d forward passes per point, warmup 8)\n\n", model, steps);
    printf("| threads | dtype |      tok/s |\n");
    printf("|--------:|:------|-----------:|\n");

    FILE *csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "a");
        if (!csv) fprintf(stderr, "[nanollama] could not open %s for appending\n", csv_path);
    }

    for (int dtype = 0; dtype < (quantize_too ? 2 : 1); dtype++) {
        const char *name = dtype ? "int8" : "fp32";
        for (int i = 0; i < n_th; i++) {
            double tps = bench_one(model, dtype, thread_counts[i], steps);
            printf("| %7d | %-5s | %10.1f |\n", thread_counts[i], name, tps);
            if (csv) {
                fprintf(csv, "%d,%s,%.1f\n", thread_counts[i], name, tps);
            }
        }
    }
    if (csv) fclose(csv);
    printf("\n(forward passes only — end-to-end generation is a bit slower)\n");
}

/* ---------------------------------------------------------------------------
 * entry point
 * ------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);          /* stream tokens as they come */

    const char *command = "generate";
    const char *model = NULL;
    const char *tokenizer = "models/tokenizer.bin";
    const char *prompt = NULL;
    const char *csv_path = NULL;
    int n_gen = 256;
    float temp = 0.8f;
    float topp = 0.9f;
    int topk = 0;
    int threads = 0;                           /* 0 = all cores */
    int quantize = 0;
    int interactive = 0;
    unsigned long long seed = 0;

    if (argc > 1 && argv[1][0] != '-') {
        command = argv[1];
        g_i = 1;
    } else {
        g_i = 0;
    }

    while (++g_i < argc) {
        char *arg = argv[g_i];
        if (arg[0] != '-') {
            /* allow `nanollama "prompt"` as a shorthand for generate */
            if (strcmp(command, "generate") == 0 && prompt == NULL) {
                prompt = arg;
                continue;
            }
            fprintf(stderr, "[nanollama] unexpected argument: %s (see --help)\n", arg);
            return EXIT_FAILURE;
        }
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage();
            return 0;
        }
        if (strcmp(arg, "-m") == 0) model = next_arg(argc, argv, arg);
        else if (strcmp(arg, "-z") == 0) tokenizer = next_arg(argc, argv, arg);
        else if (strcmp(arg, "-p") == 0) prompt = next_arg(argc, argv, arg);
        else if (strcmp(arg, "-n") == 0) n_gen = atoi(next_arg(argc, argv, arg));
        else if (strcmp(arg, "-t") == 0) temp = (float)atof(next_arg(argc, argv, arg));
        else if (strcmp(arg, "--topp") == 0) topp = (float)atof(next_arg(argc, argv, arg));
        else if (strcmp(arg, "-k") == 0) topk = atoi(next_arg(argc, argv, arg));
        else if (strcmp(arg, "-s") == 0) seed = strtoull(next_arg(argc, argv, arg), NULL, 10);
        else if (strcmp(arg, "-th") == 0) threads = atoi(next_arg(argc, argv, arg));
        else if (strcmp(arg, "-q") == 0) quantize = 1;
        else if (strcmp(arg, "-i") == 0) interactive = 1;
        else if (strcmp(arg, "--csv") == 0) csv_path = next_arg(argc, argv, arg);
        else {
            fprintf(stderr, "[nanollama] unknown flag: %s (see --help)\n", arg);
            return EXIT_FAILURE;
        }
    }

    if (model == NULL) {
        fprintf(stderr, "[nanollama] no model given — use -m <path>\n\n");
        usage();
        return EXIT_FAILURE;
    }

    if (strcmp(command, "generate") == 0) {
        cmd_generate(model, tokenizer, prompt, n_gen, temp, topp, topk, seed,
                     threads, quantize, interactive);
    } else if (strcmp(command, "chat") == 0) {
        cmd_chat(model, tokenizer, n_gen, temp, topp, topk, seed, threads, quantize);
    } else if (strcmp(command, "bench") == 0) {
        int bench_steps = n_gen >= 32 ? n_gen : 32;  /* reuse -n as passes/point */
        cmd_bench(model, bench_steps, threads, 1, csv_path);
    } else {
        fprintf(stderr, "[nanollama] unknown command '%s' (use generate | chat | bench)\n",
                command);
        return EXIT_FAILURE;
    }

    return 0;
}
