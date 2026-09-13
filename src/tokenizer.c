/*
 * tokenizer.c — byte-pair encoding (BPE), written from scratch.
 *
 * Loads the tokenizer.bin file exported from a Llama-2 sentencepiece model:
 *
 *   int   vocab_count
 *   then per token:  float score, int len, utf-8 bytes[len]
 *
 * `score` is the merge priority: when two adjacent tokens could be merged,
 * the pair whose merged form has the HIGHEST score merges first.
 *
 * Token ids have a fixed layout in Llama-2:
 *   0 = <unk>, 1 = <s> (BOS), 2 = </s> (EOS)
 *   3..258     = the 256 raw bytes (byte-fallback for anything unmergeable)
 *   259+       = learned pieces, usually with a leading space
 */

#include "nanollama.h"

/* Raw bytes (<0x00>-<0xFF>) can't be printed; skip them instead of
 * corrupting the terminal. Called for every generated piece. */
void safe_print(const char *piece) {
    if (piece == NULL) return;
    if (piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = (unsigned char)piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; /* raw control byte — don't print */
        }
    }
    printf("%s", piece);
}

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, const char *path, int vocab_size) {
    /* the 256 byte-fallback tokens are materialized as 2-byte strings */
    t->byte_pieces[0] = '\0';
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "[nanollama] couldn't open tokenizer file %s\n"
                        "           (get it with: python scripts/download_models.py)\n", path);
        exit(EXIT_FAILURE);
    }

    t->max_token_length = 0;

    /* Historical quirk: the first int of tokenizer.bin files is NOT the token
     * count (it's a leftover from an older exporter). The entry stream after
     * it is self-describing — (float score, int len, bytes) repeated to EOF —
     * so we just walk it and count. */
    int first_int;
    (void)fread(&first_int, sizeof(int), 1, file); /* skipped */
    (void)first_int;

    int cap = 1024;
    t->vocab = (char **)malloc((size_t)cap * sizeof(char *));
    t->vocab_scores = (float *)malloc((size_t)cap * sizeof(float));
    if (!t->vocab || !t->vocab_scores) NL_ERROR("out of memory for tokenizer");
    t->vocab_size = 0;

    while (1) {
        float score;
        int len;
        if (fread(&score, sizeof(float), 1, file) != 1) break; /* EOF: done */
        if (fread(&len, sizeof(int), 1, file) != 1 || len < 0 || len > 1 << 20) {
            NL_ERROR("corrupted tokenizer entry stream");
        }
        if (t->vocab_size == cap) {
            cap *= 2;
            t->vocab = (char **)realloc(t->vocab, (size_t)cap * sizeof(char *));
            t->vocab_scores = (float *)realloc(t->vocab_scores, (size_t)cap * sizeof(float));
            if (!t->vocab || !t->vocab_scores) NL_ERROR("out of memory for tokenizer");
        }
        int id = t->vocab_size++;
        t->vocab_scores[id] = score;
        t->vocab[id] = (char *)malloc((size_t)len + 1);
        if (!t->vocab[id]) NL_ERROR("out of memory for tokenizer");
        if (fread(t->vocab[id], 1, (size_t)len, file) != (size_t)len) {
            NL_ERROR("failed to read tokenizer entry");
        }
        t->vocab[id][len] = '\0';
        if ((unsigned int)len > t->max_token_length) {
            t->max_token_length = (unsigned int)len;
        }
    }
    fclose(file);

    if (vocab_size > 0 && t->vocab_size != vocab_size) {
        fprintf(stderr, "[nanollama] warning: tokenizer has %d tokens, model expects %d — "
                        "make sure the tokenizer matches the model "
                        "(260K needs tok512.bin, the others need tokenizer.bin)\n",
                t->vocab_size, vocab_size);
    }

    t->sorted_vocab = NULL; /* built lazily on the first encode() */
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

/* Convert a token id to its printable string. `prev_token` is needed because
 * sentencepiece strips the space after <s>: the piece following BOS must not
 * print its leading space. "<0xNN>" byte tokens become the raw byte itself. */
char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    unsigned char byte_val;
    /* "<0xNN>"-style byte tokens: parse by hand — MinGW's msvcrt doesn't
     * support the %hhX scanf conversion */
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
        piece[5] == '>' && piece[6] == '\0') {
        char hex[3] = { piece[3], piece[4], '\0' };
        byte_val = (unsigned char)strtol(hex, NULL, 16);
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

/* binary search in the sorted vocab */
static int str_lookup(const char *str, const TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { (char *)str, 0 };
    const TokenIndex *res =
        (const TokenIndex *)bsearch(&tok, sorted_vocab, (size_t)vocab_size,
                                    sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

/* text -> tokens. If bos/eos are set, prepend/append id 1 / id 2. */
void encode(Tokenizer *t, const char *text, int bos, int eos, int *tokens, int *n_tokens) {
    if (text == NULL) NL_ERROR("cannot encode NULL text");

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex *)malloc((size_t)t->vocab_size * sizeof(TokenIndex));
        if (!t->sorted_vocab) NL_ERROR("out of memory for tokenizer sort");
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, (size_t)t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    /* scratch buffer big enough for any utf-8 codepoint + merged pieces */
    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 3));
    if (!str_buffer) NL_ERROR("out of memory for encoding");
    size_t str_len = 0;

    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1; /* <s> */

    /* sentencepiece's dummy prefix: text implicitly starts with a space */
    if (text[0] != '\0') {
        int dummy = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy;
    }

    /* pass 1: every utf-8 codepoint becomes a token (byte fallback if needed) */
    for (const char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0; /* not a continuation byte: new codepoint starts here */
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        /* keep consuming while we're mid-codepoint (max 4 bytes) */
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id; /* whole codepoint is a known token */
        } else {
            /* byte fallback: each raw byte maps to token id byte + 3 */
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    /* pass 2: greedily merge the best pair anywhere in the sequence.
     * "best" = highest merge score. Repeat until nothing merges. */
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < *n_tokens - 1; i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break; /* no more merges possible */

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < *n_tokens - 1; i++) {
            tokens[i] = tokens[i + 1]; /* shift left, delete the merged pair */
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2; /* </s> */

    free(str_buffer);
}
