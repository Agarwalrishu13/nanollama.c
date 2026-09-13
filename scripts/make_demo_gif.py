#!/usr/bin/env python3
"""Render a terminal-style demo GIF from REAL nanollama.c output.

Runs the actual engine, then draws the tokens appearing character-by-character
in a fake terminal window — the GIF that sits at the top of the README.

    python scripts/make_demo_gif.py            # 260K model (fast, tiny)
    python scripts/make_demo_gif.py 15m        # bigger model, nicer story
"""

import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT_PATH = os.path.join(ROOT, "docs", "demo.gif")

W, H = 860, 480
BG, TAB_BG, FG, PROMPT = "#0d1117", "#161b22", "#e6edf3", "#7ee787"
ACCENT, DIM = "#79c0ff", "#8b949e"

MODELS = {
    "260k": ("models/stories260K.bin", "models/tok512.bin"),
    "15m":  ("models/stories15M.bin", "models/tokenizer.bin"),
}


def find_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "couri.ttf", "lucon.ttf"):
        path = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def run_engine(model: str, tok: str) -> str:
    exe = os.path.join(ROOT, "nanollama.exe" if os.name == "nt" else "nanollama")
    if not os.path.exists(exe):
        sys.exit("build the engine first: make")
    out = subprocess.run(
        [exe, "generate", "-m", os.path.join(ROOT, model), "-z", os.path.join(ROOT, tok),
         "-t", "0.8", "-s", "42", "-n", "110", "-p", "Once upon a time"],
        capture_output=True, text=True, cwd=ROOT)
    return out.stdout


def frame(draw: ImageDraw.ImageDraw, lines: list[str], chars_visible: int,
          font, done: bool) -> None:
    draw.rectangle([0, 0, W, H], fill=BG)
    draw.rectangle([0, 0, W, 34], fill=TAB_BG)
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        draw.ellipse([16 + i * 24, 11, 26 + i * 24, 21], fill=c)
    draw.text((W // 2 - 92, 8), "nanollama.c — rishu@dev", fill=DIM, font=font)

    y = 52
    draw.text((24, y), "$ ./nanollama generate -m stories15M.bin", fill=PROMPT, font=font)
    draw.text((24, y + 26), '   -p "Once upon a time" -t 0.8', fill=PROMPT, font=font)
    y += 66

    # reveal `chars_visible` characters across the story lines
    remaining = chars_visible
    last_end_x = 24
    for line in lines:
        if remaining <= 0:
            break
        shown = line[:remaining]
        remaining -= len(line)
        draw.text((24, y), shown, fill=FG, font=font)
        if remaining > 0:
            last_end_x = 24 + draw.textlength(shown, font=font)
        y += 27

    if done:  # finished: fresh prompt line with a block cursor
        draw.text((24, y), "$ ", fill=PROMPT, font=font)
        draw.rectangle([24 + 18, y + 2, 24 + 30, y + 20], fill=FG)


def main() -> None:
    size = sys.argv[1] if len(sys.argv) > 1 else "15m"
    model, tok = MODELS[size]
    if shutil.which("python") is None:
        sys.exit("python required (for Pillow)")
    text = run_engine(model, tok)
    text = text.split("\n", 1)[1].strip() if "\n" in text else text.strip()
    text = "Once upon a time" + (" " + text if not text.startswith(",") else text)

    font = find_font(16)
    # wrap the story to ~62 chars per line
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 62:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    total_chars = sum(len(l) + 1 for l in lines)

    frames = []
    visible = 0
    while visible < total_chars:
        img = Image.new("RGB", (W, H), BG)
        frame(ImageDraw.Draw(img), lines, visible, font, done=False)
        frames.append(img)
        visible += 3                      # ~3 chars per frame: typing speed
        if visible > total_chars - 40:
            visible = min(visible + 2, total_chars)

    # hold the final frame a beat
    frames += [Image.new("RGB", (W, H), BG)] * 1
    img = Image.new("RGB", (W, H), BG)
    frame(ImageDraw.Draw(img), lines, total_chars, font, done=True)
    frames += [img] * 8

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    frames[0].save(OUT_PATH, save_all=True, append_images=frames[1:],
                   duration=45, loop=0, optimize=True)
    kb = os.path.getsize(OUT_PATH) / 1024
    print(f"wrote {os.path.normpath(OUT_PATH)} ({kb:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
