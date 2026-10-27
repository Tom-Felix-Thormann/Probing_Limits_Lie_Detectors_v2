"""
03_dataset_validation.py

Validates the inferred deception types and implied statements against the
manually annotated dataset.

Plot 1: share of correctly inferred deception types per actual type.
        - For the 10 deception types, an inference is correct when
          deception_type_misleading == deception_type.
        - For unrelated_or_nonsense, an inference is correct when the model
          answered that none of the deception types fits.
Plot 2: share of implied statements (misleading answers) that match the
        false_answer, per deception type.

Usage:
    python 03_dataset_validation.py [path/to/manual_entries_completed.xlsx] [output_dir]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from llm_utils import DATASETS_DIR, SCRIPT_DIR

INPUT = DATASETS_DIR / "manual_entries_completed_and_validated.xlsx"
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NONE_FITS = "None of the deception types fits"
UNRELATED = "unrelated_or_nonsense"

# Styling
BAR = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def bar_plot(summary: pd.DataFrame, title: str, ylabel: str, outfile: Path, separator_before=None):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = range(len(summary))
    ax.bar(x, summary["share"], width=0.6, color=BAR, zorder=3)
    for xi, (_, r) in zip(x, summary.iterrows()):
        ax.text(xi, r["share"] + 0.015, f"{r['share']:.0%}", ha="center", va="bottom",
                fontsize=9, color=TEXT)
    if separator_before is not None:
        ax.axvline(separator_before - 0.5, color=TEXT_2, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{t} (n={int(n)})" for t, n in summary["n"].items()],
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=6, length=0)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"saved {outfile}")


df = pd.read_excel(INPUT)

# ---------------------------------------------------------------------------
# Plot 1: correctly inferred deception types (10 types + unrelated_or_nonsense)
# ---------------------------------------------------------------------------
mis_correct = norm(df["deception_type_misleading"]) == norm(df["deception_type"])
per_type = (
    pd.DataFrame({"type": df["deception_type"], "correct": mis_correct})
    .groupby("type")["correct"].agg(share="mean", n="size")
    .sort_values("share", ascending=False)
)
unrel_correct = norm(df["deception_type_unrelated"]) == NONE_FITS.lower()
per_type.loc[UNRELATED] = [unrel_correct.mean(), unrel_correct.size]

bar_plot(
    per_type,
    "Share of correctly inferred deception types",
    "Share correctly inferred",
    OUT_DIR / "deception_type_accuracy.png",
    separator_before=len(per_type) - 1,
)

# ---------------------------------------------------------------------------
# Plot 2: implied statement (misleading) matches the false answer
# ---------------------------------------------------------------------------
match = norm(df["implied_misleading_matches_false_answer"]) == "yes"
per_type_match = (
    pd.DataFrame({"type": df["deception_type"], "match": match})
    .groupby("type")["match"].agg(share="mean", n="size")
    .sort_values("share", ascending=False)
)
bar_plot(
    per_type_match,
    "Share of implied statements matching the false answer",
    "Share matching",
    OUT_DIR / "implied_statement_match.png",
)

# Console summary
print("\nDeception type accuracy:\n", per_type.round(3).to_string())
print(f"\nOverall (10 types): {mis_correct.mean():.1%}")
print("\nImplied statement matches false answer:\n", per_type_match.round(3).to_string())
print(f"\nOverall: {match.mean():.1%}")
