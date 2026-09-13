#!/usr/bin/env python3
"""Turn benchmark/results.csv into benchmark/bench.png.

Expects rows written by `nanollama bench --csv`:
    threads,dtype,tokens_per_sec

    python scripts/benchmark.py          # needs results.csv (run bench first)
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CSV_PATH = os.path.join(ROOT, "benchmark", "results.csv")
OUT_PATH = os.path.join(ROOT, "benchmark", "bench.png")

# GitHub-dark inspired palette
BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
COLORS = {"fp32": "#58a6ff", "int8": "#f0883e"}


def main() -> None:
    series = {}
    with open(CSV_PATH) as f:
        for row in csv.reader(f):
            if len(row) != 3 or row[0] == "threads":
                continue
            th, dtype, tps = int(row[0]), row[1].strip(), float(row[2])
            series.setdefault(dtype, {})[th] = tps

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for dtype, points in series.items():
        xs = sorted(points)
        ys = [points[x] for x in xs]
        color = COLORS.get(dtype, "#8b949e")
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=6, color=color, label=dtype)
        ax.annotate(f"{ys[-1]:.0f} tok/s", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    color=color, fontsize=10, fontweight="bold")

    best = max(p for s in series.values() for p in s.values())
    single = min(points.get(min(points), best) for points in series.values())
    speedup = best / single if single else 0

    ax.set_title(f"nanollama.c — stories15M forward-pass throughput "
                 f"({speedup:.1f}x at max threads)", color=FG, fontsize=13, pad=14)
    ax.set_xlabel("worker threads", color=FG, fontsize=11)
    ax.set_ylabel("tokens / second", color=FG, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    legend = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, title="weights",
                       title_fontproperties={"size": 10})
    legend.get_title().set_color(FG)

    fig.tight_layout()
    fig.savefig(OUT_PATH, facecolor=BG)
    print(f"wrote {os.path.normpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
