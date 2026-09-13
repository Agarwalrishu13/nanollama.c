#!/usr/bin/env python3
"""Download TinyStories models + tokenizers for nanollama.c.

Models come from Andrej Karpathy's public repos in the llama2.c .bin format.
No API keys, no accounts — plain HTTPS downloads.

    python scripts/download_models.py              # 260K (smallest, ~1 MB)
    python scripts/download_models.py 260k 15m     # several at once

Note: the 260K model was trained with a 512-token tokenizer (tok512.bin);
the 15M/42M/110M models use the full 32,000-token Llama-2 tokenizer.
Each model needs its matching tokenizer — pass it with -z.
"""

import argparse
import os
import sys
import urllib.request

HF = "https://huggingface.co/karpathy/tinyllamas/resolve/main/"
TOKENIZER_URL = "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin"

# size -> (model_url, model_file, tokenizer_url, tokenizer_file)
MODELS = {
    "260k": (HF + "stories260K/stories260K.bin", "stories260K.bin",
             HF + "stories260K/tok512.bin", "tok512.bin"),
    "15m":  (HF + "stories15M.bin", "stories15M.bin", TOKENIZER_URL, "tokenizer.bin"),
    "42m":  (HF + "stories42M.bin", "stories42M.bin", TOKENIZER_URL, "tokenizer.bin"),
    "110m": (HF + "stories110M.bin", "stories110M.bin", TOKENIZER_URL, "tokenizer.bin"),
}


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  already have {dest}, skipping")
        return
    print(f"  downloading {url}")

    def report(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            pct = 100 * done / total
            sys.stdout.write(f"\r    {done/1e6:7.1f} / {total/1e6:.1f} MB ({pct:4.1f}%)")
            sys.stdout.flush()

    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=report)
    os.replace(tmp, dest)
    print("\r    done" + " " * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sizes", nargs="*", default=["260k"],
                        choices=sorted(MODELS.keys()),
                        help="model sizes to download (default: 260k)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    print("nanollama.c model downloader")
    for size in args.sizes:
        model_url, model_file, tok_url, tok_file = MODELS[size]
        download(model_url, os.path.join(models_dir, model_file))
        download(tok_url, os.path.join(models_dir, tok_file))
        print(f"  run it:  ./nanollama generate -m models/{model_file} "
              f"-z models/{tok_file} -p \"Once upon a time\"\n")


if __name__ == "__main__":
    main()
