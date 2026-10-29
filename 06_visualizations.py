"""Step 6: figures for the paper.

1. fig_dataset_validation          validation of the manual dataset: share of correctly inferred deception types
                                   (and of unrelated answers recognized as fitting none) and share of implied
                                   statements that match the false answer, per deception type
2. fig_forced_choice_responses     forced choice task (03): chosen options per model and condition (counts)
3. fig_forced_choice_rates         forced choice task: rate of the target option per condition, with Wilson CIs
                                   and the Holm-corrected McNemar tests (DWL vs LIE, DWL vs DWL-U)
4. fig_probe_generalization        probes (05): accuracy on held-out topics and held-out statement types, as
                                   heatmaps in the style of Truth_is_Universal (Buerger et al. 2024), Fig. 6
5. fig_probe_misleading            probes: share of true, false and misleading answers classified as false, with
                                   Wilson CIs and the Holm-corrected McNemar tests (false vs misleading)

Style as in the previous version of the paper (Fig. 4): stacked / grouped bars per model, yellow = deception
without lie, red = lie, green = honest, dark grey = invalid; legend with the condition abbreviations.
Every figure is saved as PNG in figures/. A figure whose input file is missing is skipped.

Usage:
    python 06_visualization.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from scipy.stats import binomtest

from llm_utils import DATASETS_DIR, RESULTS_DIR, SCRIPT_DIR, holm

FIG_DIR = SCRIPT_DIR / "figures"
VALIDATION_PATH = DATASETS_DIR / "manual_entries_completed_and_validated.xlsx"
FORCED_CHOICE_DIR = RESULTS_DIR / "forced_choice"
PROBES_DIR = RESULTS_DIR / "probes"

ALPHA = 0.01  # significance level of the (Holm-corrected) tests, as in the paper
TEXT_WIDTH = 7.0  # inches, two-column AAAI layout
NONE_FITS_TYPE = "None of the deception types fits"  # answer of the validation model for unrelated answers

# ---------------------------------------------------------------------------
# Labels (edit here to rename things in all figures)
# ---------------------------------------------------------------------------
MODEL_LABELS = {  # in panel order: by parameter count
    "mistral-7B-instruct": "mistral-7b",
    "qwen-2.5-7B-instruct": "qwen-2.5-7b",
    "llama-3.1-8B-instruct": "llama-3.1-8b",
    "gemma-3-27B-instruct": "gemma-3-27b",
    "llama-3.1-70B-instruct": "llama-3.1-70b",
    "qwen-2.5-72B-instruct": "qwen-2.5-72b",
}
FC_CONDITIONS = {  # forced choice condition -> (abbreviation, explanation)
    "lie": ("LIE", "Lie condition"),
    "mislead": ("DWL", "Deceive-without-lying condition"),
    "mislead_unfitting": ("DWL-U", "DWL condition, unrelated instead of misleading option"),
}
PROBE_CONDITIONS = {  # probe training condition -> (abbreviation, explanation)
    "raw": ("RAW", "Trained on raw statements"),
    "with_context": ("DIA", "Trained on statements in dialogues"),
    "with_context_and_misleading": ("DIA+DWL", "DIA plus misleading\nanswers (label false)"),
}
DECEPTION_TYPE_LABELS = {  # in the order of 01_dataset_sampling.py
    "abstraction": "abstraction",
    "contr_distraction": "contr. distraction",
    "equivocation": "equivocation",
    "FI_quantity": "FI (quantity)",
    "FI_relevance": "FI (relevance)",
    "understatement": "understatement",
    "overstatement": "overstatement",
    "obfuscation": "obfuscation",
    "half_truth": "half-truth",
    "augmentation": "augmentation",
}
TOPIC_LABELS = {
    "cities": "cities",
    "animal_class": "animals",
    "sp_en_trans": "translations",
    "element_symb": "elements",
    "inventors": "inventors",
    "facts": "facts",
}
STATEMENT_TYPE_LABELS = {  # as in Truth_is_Universal, Fig. 6b
    "conj": "Conjunctions",
    "disj": "Disjunctions",
    "de": "Affirmative German",
    "neg_de": "Negated German",
}

# ---------------------------------------------------------------------------
# Colors: answer types as in the paper; conditions of the probes as one sequential (blue) ramp
# ---------------------------------------------------------------------------
ANSWERS = {  # choice / kind -> (label, color), bottom to top in the stacked bars
    "misleading": ("deception without lie", "#ffe600"),
    "false": ("lie", "#d14444"),
    "true": ("honest", "#6aa06d"),
    "unrelated": ("unrelated", "#7b6fb5"),
    "none": ("none of these fits", "#bcbcbc"),
    "invalid": ("invalid", "#525252"),
}
HEATMAP_CMAP = "plasma"  # as in Truth_is_Universal, Fig. 6
HEATMAP_RANGE = (0.0, 1.0)  # color scale of the accuracies; (0.5, 1.0) would spread the colors over the above-chance range
TEXT = "#000000"
GRID = "#e7e7e7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.axisbelow": True,
    "hatch.linewidth": 0.6,
    "savefig.dpi": 300,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class Abbreviation:
    """Legend entry whose 'handle' is a bold abbreviation instead of a colored patch (as in the paper)."""

    def __init__(self, text):
        self.text = text


class AbbreviationHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return [Text(width / 2 - xdescent, height / 2 - ydescent, orig_handle.text, ha="center", va="center",
                     fontweight="bold", fontsize=fontsize, transform=trans)]


def figure_legend(fig, color_entries, abbreviations, ncol, extra=(), y=0.0):
    """Legend below the panels: colored patches, extra (handle, label) pairs, then the abbreviations."""
    handles = [Patch(facecolor=color, edgecolor="#7a6a00" if hatch else "none", linewidth=0, hatch=hatch)
               for _, color, hatch in color_entries]
    labels = [label for label, _, _ in color_entries]
    handles += [handle for handle, _ in extra]
    labels += [label for _, label in extra]
    handles += [Abbreviation(abbr) for abbr, _ in abbreviations]
    labels += [explanation for _, explanation in abbreviations]
    # the handle box must be wide enough for the longest bold abbreviation (~0.8 font sizes per character)
    handlelength = max(2.2, 0.8 * max(len(abbr) for abbr, _ in abbreviations))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               handler_map={Abbreviation: AbbreviationHandler()}, handlelength=handlelength, columnspacing=1.5)


def chance_handle():
    return plt.Line2D([], [], color=TEXT, linewidth=0.7, linestyle=(0, (3, 3)))


def style_axis(ax, percent=False):
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7)
    ax.tick_params(axis="x", length=0)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def wilson(k, n):
    """95% Wilson confidence interval of k successes in n trials."""
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low, ci.high


def significance_label(p):
    return "*" if p < ALPHA else "n.s."


def bracket(ax, x1, x2, y, label, height=0.025):
    """Significance bracket between two bars at height y (data coordinates)."""
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color=TEXT, linewidth=0.7)
    ax.text((x1 + x2) / 2, y + height, label, ha="center", va="bottom", fontsize=7)


CHANCE = 0.25  # forced choice: 1 of 4 options


def clear_of_chance(y, margin=0.04):
    """Move a bracket that would sit on the chance line just above it."""
    return CHANCE + margin if CHANCE - 0.03 < y < CHANCE + margin else y


def error_bars(ax, x, rates, lows, highs):
    ax.errorbar(x, rates, yerr=[np.array(rates) - lows, np.array(highs) - rates], fmt="none",
                ecolor=TEXT, elinewidth=0.7, capsize=1.8, capthick=0.7)


def ordered_models(models):
    present = set(models)
    return [m for m in MODEL_LABELS if m in present] + sorted(present - set(MODEL_LABELS))


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / name}.png")


def norm(series):
    return series.astype(str).str.strip().str.lower()


# ---------------------------------------------------------------------------
# 1. Dataset validation
# ---------------------------------------------------------------------------
def plot_dataset_validation():
    df = pd.read_excel(VALIDATION_PATH)
    types = [t for t in DECEPTION_TYPE_LABELS if t in set(df["deception_type"])]
    type_correct = norm(df["deception_type_misleading"]) == norm(df["deception_type"])
    unrelated_correct = norm(df["deception_type_unrelated"]) == NONE_FITS_TYPE.lower()
    implied_match = norm(df["implied_misleading_matches_false_answer"]) == "yes"

    def summary(values):
        grouped = pd.DataFrame({"type": df["deception_type"], "v": values}).groupby("type")["v"]
        return grouped.sum().reindex(types), grouped.size().reindex(types)

    panels = [
        ("Deception type inferred correctly", *summary(type_correct), True),
        ("Implied statement matches false answer", *summary(implied_match), False),
    ]
    rows = types + ["unrelated"]
    y = np.arange(len(rows))[::-1]  # first type at the top

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.2), sharey=True)
    for ax, (title, k, n, with_unrelated) in zip(axes, panels):
        rate = (k / n).to_numpy()
        lows, highs = zip(*[wilson(ki, ni) for ki, ni in zip(k, n)])
        colors = [ANSWERS["misleading"][1]] * len(types)
        if with_unrelated:  # unrelated answers: correct if the model said that no deception type fits
            rate = np.append(rate, unrelated_correct.mean())
            low, high = wilson(unrelated_correct.sum(), unrelated_correct.size)
            lows, highs = (*lows, low), (*highs, high)
            colors.append(ANSWERS["unrelated"][1])
        yy = y[: len(rate)]
        ax.barh(yy, rate, height=0.65, color=colors, edgecolor="white", linewidth=0.6)
        ax.errorbar(rate, yy, xerr=[rate - np.array(lows), np.array(highs) - rate], fmt="none",
                    ecolor=TEXT, elinewidth=0.7, capsize=1.8, capthick=0.7)
        for yi, r, h in zip(yy, rate, highs):
            ax.text(h + 0.02, yi, f"{r:.0%}", va="center", fontsize=7)
        if with_unrelated:
            ax.axhline(y[len(types)] + 0.5, color=TEXT, linewidth=0.6, linestyle=(0, (3, 3)))
        ax.set_xlim(0, 1.12)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.7)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left")
    n_per_type = df.groupby("deception_type").size()
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [f"{DECEPTION_TYPE_LABELS[t]} (n={n_per_type[t]})" for t in types]
        + [f"unrelated answers (n={unrelated_correct.size})"]
    )
    fig.tight_layout()
    save(fig, "fig_dataset_validation")

    print(f"\nDeception type inferred correctly: {type_correct.mean():.1%} overall; "
          f"unrelated answers recognized: {unrelated_correct.mean():.1%}")
    print(f"Implied statement matches false answer: {implied_match.mean():.1%} overall")


# ---------------------------------------------------------------------------
# 2. + 3. Forced choice task
# ---------------------------------------------------------------------------
def load_forced_choice():
    counts = pd.read_csv(FORCED_CHOICE_DIR / "counts.csv", index_col=[0, 1])
    counts = counts.reindex(columns=list(ANSWERS), fill_value=0)
    tests = pd.read_csv(FORCED_CHOICE_DIR / "tests.csv")
    if "p_holm" not in tests:  # tests.csv from before the Holm correction was added
        tests["p_holm"] = holm(tests["p"])
    return counts, tests


def plot_forced_choice_responses(counts):
    models = ordered_models(counts.index.get_level_values("model").unique())
    conditions = list(FC_CONDITIONS)
    ncols = 3
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXT_WIDTH, 2.3 * nrows), sharey=True, squeeze=False)
    n_items = counts.sum(axis=1).max()
    x = np.arange(len(conditions))
    for ax, model in zip(axes.flat, models):
        bottom = np.zeros(len(conditions))
        for choice, (_, color) in ANSWERS.items():
            values = counts.loc[model].reindex(conditions)[choice].to_numpy()
            ax.bar(x, values, width=0.6, bottom=bottom, color=color, edgecolor="white", linewidth=0.6)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels([FC_CONDITIONS[c][0] for c in conditions], fontweight="bold")
        ax.set_ylim(0, n_items * 1.02)
        ax.set_title(MODEL_LABELS.get(model, model))
        style_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Count")
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.5)
    figure_legend(
        fig,
        [(label, color, None) for label, color in ANSWERS.values()],
        list(FC_CONDITIONS.values()),
        ncol=3,
    )
    save(fig, "fig_forced_choice_responses")


def plot_forced_choice_rates(counts, tests):
    """Per model: misleading in DWL, misleading in LIE, unrelated in DWL-U, with Wilson CIs and test brackets."""
    models = ordered_models(counts.index.get_level_values("model").unique())
    bars = [  # (condition, choice, label, color, hatch)
        ("mislead", "misleading", "misleading answer chosen in DWL", ANSWERS["misleading"][1], None),
        ("lie", "misleading", "misleading answer chosen in LIE", ANSWERS["misleading"][1], "////"),
        ("mislead_unfitting", "unrelated", "unrelated answer chosen in DWL-U", ANSWERS["unrelated"][1], None),
    ]
    tests = tests.set_index(["model", "test"])
    width = 0.26
    offsets = (np.arange(len(bars)) - 1) * width

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.6))
    rows = []
    for i, model in enumerate(models):
        model_counts = counts.loc[model]
        n = model_counts.sum(axis=1)
        highs_per_bar = []
        for offset, (condition, choice, _, color, hatch) in zip(offsets, bars):
            k, total = model_counts.loc[condition, choice], n[condition]
            rate = k / total
            low, high = wilson(k, total)
            ax.bar(i + offset, rate, width=width, color=color, hatch=hatch,
                   edgecolor="#7a6a00" if hatch else "white", linewidth=0 if hatch else 0.6)
            error_bars(ax, [i + offset], [rate], [low], [high])
            highs_per_bar.append(high)
            rows.append({"model": model, "condition": condition, "choice": choice, "rate": rate, "low": low, "high": high})
        # brackets: DWL vs LIE (lower), DWL vs DWL-U (above it); both kept clear of the chance line
        y = clear_of_chance(max(highs_per_bar[:2]) + 0.04)
        bracket(ax, i + offsets[0], i + offsets[1], y, significance_label(tests.loc[(model, "mislead_vs_lie"), "p_holm"]))
        y = clear_of_chance(max(max(highs_per_bar) + 0.04, y + 0.13))
        bracket(ax, i + offsets[0], i + offsets[2], y, significance_label(tests.loc[(model, "mislead_vs_unfitting"), "p_holm"]))
    ax.axhline(CHANCE, color=TEXT, linewidth=0.7, linestyle=(0, (3, 3)))
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontweight="bold")
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_ylabel("Share of items")
    style_axis(ax, percent=True)
    fig.tight_layout()
    figure_legend(
        fig,
        [(label, color, hatch) for _, _, label, color, hatch in bars],
        list(FC_CONDITIONS.values()) + [("*", f"p < {ALPHA} (one-sided McNemar test, Holm-corrected)")],
        ncol=2,
        extra=[(chance_handle(), "chance (1 of 4 options)")],
    )
    save(fig, "fig_forced_choice_rates")

    print("\nForced choice: target option rates with 95% Wilson CIs")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 4. + 5. Probes
# ---------------------------------------------------------------------------
def plot_probe_generalization():
    """Accuracy heatmaps as in Truth_is_Universal, Fig. 6: (a) held-out topics, (b) held-out statement types.

    Rows are the held-out test sets plus their mean, columns the training conditions grouped by model.
    The cells show the accuracy in percent (one training run each, so without standard deviation).
    """
    panels = [  # (file, column with the held-out category, row labels, panel title)
        ("unseen_topic.csv", "test_topic", TOPIC_LABELS, "(a) Unseen topics"),
        ("unseen_statement_type.csv", "test_statement_type", STATEMENT_TYPE_LABELS, "(b) Unseen statement types"),
    ]
    data = [pd.read_csv(PROBES_DIR / file) for file, *_ in panels]
    models = ordered_models(data[0]["model"].unique())
    conditions = list(PROBE_CONDITIONS)
    columns = [(m, c) for m in models for c in conditions]
    cmap = plt.get_cmap(HEATMAP_CMAP)
    norm_ = matplotlib.colors.Normalize(*HEATMAP_RANGE)

    matrices, row_labels = [], []
    for (_, column, labels, _), df in zip(panels, data):
        categories = [c for c in labels if c in set(df[column])] + sorted(set(df[column]) - set(labels))
        table = df.pivot_table(index=column, columns=["model", "condition"], values="accuracy")
        matrix = table.reindex(index=categories, columns=pd.MultiIndex.from_tuples(columns)).to_numpy()
        matrices.append(np.vstack([matrix, np.nanmean(matrix, axis=0)]))  # last row: mean over the test sets
        row_labels.append([labels.get(c, c) for c in categories] + ["Mean"])

    heights = [len(r) for r in row_labels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(TEXT_WIDTH, 0.27 * sum(heights) + 2.2),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.45})
    for ax, matrix, labels, (*_, title) in zip(axes, matrices, row_labels, panels):
        image = ax.imshow(matrix, cmap=cmap, norm=norm_, aspect="auto")
        for (i, j), value in np.ndenumerate(matrix):
            if np.isnan(value):
                continue
            r, g, b, _ = cmap(norm_(value))
            color = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"
            ax.text(j, i, f"{value * 100:.0f}", ha="center", va="center", fontsize=7, color=color,
                    fontweight="bold" if i == len(labels) - 1 else "normal")
        # white gaps between the model groups and above the mean row
        for k in range(1, len(models)):
            ax.axvline(k * len(conditions) - 0.5, color="white", linewidth=3)
        ax.axhline(len(labels) - 1.5, color="white", linewidth=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([PROBE_CONDITIONS[c][0] for _, c in columns], fontsize=6, fontweight="bold")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for k, model in enumerate(models):  # model names above their condition columns
            ax.text((k + 0.5) * len(conditions) - 0.5, -1.25, MODEL_LABELS.get(model, model),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title(title, loc="left", pad=26)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, ticks=np.linspace(*HEATMAP_RANGE, 6))
    colorbar.outline.set_visible(False)
    colorbar.set_label("Accuracy")
    figure_legend(fig, [], list(PROBE_CONDITIONS.values()), ncol=3, y=0.07)
    save(fig, "fig_probe_generalization")

    for (_, column, *_), df in zip(panels, data):
        summary = df.groupby(["model", "condition"])["accuracy"].mean().unstack()[conditions]
        print(f"\nProbes: mean accuracy over the held-out {column[5:]}s\n", summary.round(3).to_string())


def plot_probe_misleading():
    df = pd.read_csv(PROBES_DIR / "misleading.csv")
    df["p_holm"] = holm(df["p"])
    models = ordered_models(df["model"].unique())
    conditions = list(PROBE_CONDITIONS)
    kinds = ["true", "false", "misleading"]  # bar order within a condition: honest, lie, deception without lie
    width = 0.26
    offsets = (np.arange(len(kinds)) - 1) * width
    x = np.arange(len(conditions))

    fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH, 2.6), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model].set_index("condition")
        for offset, kind in zip(offsets, kinds):
            rates = sub.reindex(conditions)[f"rate_{kind}"].to_numpy()
            n = sub.reindex(conditions)["n_test"].to_numpy()
            lows, highs = zip(*[wilson(round(r * ni), ni) for r, ni in zip(rates, n)])
            ax.bar(x + offset, rates, width=width, color=ANSWERS[kind][1], edgecolor="white", linewidth=0.6)
            error_bars(ax, x + offset, rates, np.array(lows), np.array(highs))
            if kind == "false":
                highs_false = np.array(highs)
            if kind == "misleading":
                highs_misleading = np.array(highs)
        for xi, condition, h1, h2 in zip(x, conditions, highs_false, highs_misleading):
            bracket(ax, xi + offsets[1], xi + offsets[2], max(h1, h2) + 0.04,
                    significance_label(sub.loc[condition, "p_holm"]))
        ax.set_xticks(x)
        ax.set_xticklabels([PROBE_CONDITIONS[c][0] for c in conditions], fontweight="bold", fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_title(MODEL_LABELS.get(model, model))
        style_axis(ax, percent=True)
    axes[0, 0].set_ylabel("Classified as false")
    fig.tight_layout()
    figure_legend(
        fig,
        [(ANSWERS[kind][0], ANSWERS[kind][1], None) for kind in kinds],
        list(PROBE_CONDITIONS.values()) + [("*", f"p < {ALPHA} lie vs. deception without\nlie (McNemar, Holm)")],
        ncol=3,
    )
    save(fig, "fig_probe_misleading")

    print("\nProbes: share classified as false (manual test rows)")
    print(df[["model", "condition", "n_test", "rate_true", "rate_false", "rate_misleading", "p", "p_holm"]]
          .round(4).to_string(index=False))


# ---------------------------------------------------------------------------
def run(name, function, *args):
    try:
        function(*args)
    except FileNotFoundError as error:
        print(f"Skipping {name}: {error}")


def main():
    run("dataset validation", plot_dataset_validation)
    try:
        counts, tests = load_forced_choice()
    except FileNotFoundError as error:
        print(f"Skipping the forced choice figures: {error}")
    else:
        plot_forced_choice_responses(counts)
        plot_forced_choice_rates(counts, tests)
    run("probe generalization", plot_probe_generalization)
    run("probe misleading", plot_probe_misleading)


if __name__ == "__main__":
    main()