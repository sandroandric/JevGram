"""Draw docs/images/benchmark.png from the DetectionAI benchmark metrics.

Run after the benchmark:
    uv run --with matplotlib python -m benchmarks.plot_detectionai
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "benchmarks" / "results" / "detectionai" / "metrics.csv"
OUTPUT = ROOT / "docs" / "images" / "benchmark.png"

# Rows shown, top to bottom, with their display names. Jev judges whole passages,
# as the commercial detectors do.
ORDER = [
    ("Pangram", "Pangram"),
    ("Originality", "Originality"),
    ("GPTZero", "GPTZero"),
    ("Jev (whole passage)", "Jev"),
    ("RoBERTa", "RoBERTa"),
]
HIGHLIGHT = {"Jev (whole passage)"}
INK, PEN, GREY, RULE, SOFT = "#1C2533", "#2B3BC4", "#8A94A3", "#D5DAE1", "#5B6574"


def main() -> None:
    with METRICS.open() as handle:
        rows = {r["detector"]: r for r in csv.DictReader(handle) if r["subset"] == "all models"}

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial"], "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    y = list(range(len(ORDER)))[::-1]

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SOFT, length=0)
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels([name for _, name in ORDER])
    for label, (key, _) in zip(ax.get_yticklabels(), ORDER):
        label.set_color(INK)
        label.set_fontweight("bold" if key in HIGHLIGHT else "normal")

    for position, (key, _) in zip(y, ORDER):
        row = rows[key]
        color = PEN if key in HIGHLIGHT else GREY
        auc, low, high = float(row["AUROC"]), float(row["AUROC CI low"]), float(row["AUROC CI high"])
        ax.plot([low, high], [position, position], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(auc, position, "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
        if low - 0.08 > 0.5:
            ax.text(low - 0.012, position, f"{auc:.3f}", color=INK, fontsize=10,
                    va="center", ha="right")
        else:
            ax.text(high + 0.012, position, f"{auc:.3f}", color=INK, fontsize=10,
                    va="center", ha="left")

    ax.set_xlim(0.5, 1.005)
    ax.set_ylim(-0.6, len(ORDER) - 0.3)
    ax.set_title("AUROC (95% CI)", loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
    fig.text(0.01, -0.04,
             "DetectionAI benchmark (Imas & Jabarian): 200 human passages vs their GPT-4.1,\n"
             "Claude Opus 4 and Claude Sonnet 4 versions. Detector scores from the study's data.",
             color=SOFT, fontsize=9)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white", pad_inches=0.25)


if __name__ == "__main__":
    main()
