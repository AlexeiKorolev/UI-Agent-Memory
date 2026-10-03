"""Figures for the report (static PNG; palette = dataviz reference instance, slots 1-2 + neutral)."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import PROJ

BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b4b2a9"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
LABELS = {"C0": "C0 no memory", "C1": "C1 clean memory", "C2": "C2 all past screens", "C3": "C3 text only",
          "C4": "C4 crops only", "C5": "C5 blank crops", "C6": "C6 shuffled crops", "C7": "C7 other episode",
          "C8": "C8 counterfactual", "C8t": "C8 text-edit only", "C8c": "C8 crop-edit only"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def hbars(ax, d, val, lo, hi, title, ref=None):
    d = d.iloc[::-1]
    y = range(len(d))
    ax.barh(y, d[val], height=0.55, color=BLUE, edgecolor=SURF, linewidth=2)
    ax.errorbar(d[val], y, xerr=[d[val] - d[lo], d[hi] - d[val]], fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
    for yi, v, h in zip(y, d[val], d[hi]):
        ax.text(h + 1, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels([LABELS.get(c, c) for c in d.condition], fontsize=9, color=INK)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    style(ax)


def main(prefix, md_subset):
    tab = pd.read_csv(f"{prefix}_summary_table.csv")
    order = [c for c in LABELS if c in set(tab.condition)]
    main_conds = [c for c in order if not c.startswith("C8")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    a = tab[(tab.subset == "all") & tab.condition.isin(main_conds)].set_index("condition").loc[main_conds].reset_index()
    hbars(axes[0], a, "AMS", "AMS_lo", "AMS_hi", f"AMS, all steps (n={int(a.n_steps.iloc[0])} per condition)")
    m = tab[(tab.subset == md_subset)].set_index("condition")
    m = m.loc[[c for c in order if c in m.index]].reset_index()
    c1 = float(m.loc[m.condition == "C1", "text_acc"].iloc[0]) if (m.condition == "C1").any() else None
    hbars(axes[1], m, "text_acc", "text_acc_lo", "text_acc_hi",
          f"Typed-text accuracy, {md_subset} steps (n={int(m.n_steps.max())})", ref=c1)
    axes[0].set_xlabel("% steps correct (official matcher)", color=INK2, fontsize=8)
    axes[1].set_xlabel("% steps with typed text ≈ needed string (sim ≥ 0.8)", color=INK2, fontsize=8)
    fig.text(0.01, 0.01, "Bars: point estimate; whiskers: 95% bootstrap CI over episodes (1,000 resamples). Dashed line: C1.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(PROJ / "results" / "figures" / "conditions_bar.png", dpi=180)

    ft = pd.read_csv(f"{prefix}_follow_table.csv")
    ft = ft[ft.subset == "MD_present_with_cf"] if (ft.subset == "MD_present_with_cf").any() else ft[ft.subset == "MD_with_cf"]
    ft = ft.set_index("condition").loc[[c for c in ["C1", "C8", "C8t", "C8c"] if c in set(ft.condition)]].reset_index()
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=SURF)
    y = list(range(len(ft)))[::-1]
    left = [0] * len(ft)
    for col, color, name in (("follow_cf", BLUE, "typed the counterfactual string"),
                             ("original", ORANGE, "typed the original string"),
                             ("neither", NEUTRAL, "neither / other action")):
        ax.barh(y, ft[col], left=left, height=0.55, color=color, edgecolor=SURF, linewidth=2, label=name)
        for yi, l, v in zip(y, left, ft[col]):
            if v >= 6:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white" if color != NEUTRAL else INK)
        left = [l + v for l, v in zip(left, ft[col])]
    ax.set_yticks(y); ax.set_yticklabels([LABELS[c] for c in ft.condition], fontsize=9, color=INK)
    ax.set_xlim(0, 100); ax.set_xlabel("% of MD steps (needed string present in clean memory)", color=INK2, fontsize=8)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    style(ax)
    fig.tight_layout()
    fig.savefig(PROJ / "results" / "figures" / "follow_rate.png", dpi=180)
    print("figures saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=str(PROJ / "results" / "n600"))
    ap.add_argument("--md_subset", default="MD_present")
    a = ap.parse_args()
    (PROJ / "results" / "figures").mkdir(parents=True, exist_ok=True)
    main(a.prefix, a.md_subset)
