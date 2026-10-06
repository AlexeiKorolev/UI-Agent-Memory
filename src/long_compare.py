"""Long-horizon follow-up: pre-registered comparisons L1-L4 (LOG.md, 2026-10-06), on the per-step tables of
runs/odylong and runs/memgui, with the n=600 main study as reference.

For each run and pair (a, b): paired rates on the same steps, discordant counts, exact McNemar p, and a 95% CI of the
difference a-b by bootstrapping episodes (1,000x, as in analyze.py).
  metric `correct` = official AMS step correctness (all steps); `text_ok` = typed/answered string matches the needed
  string (sim >= 0.8) on MD subsets; `follow` = typed the counterfactual alternative.
Also: difference-in-differences for L2 ((C1-C3) - (A1-A3), same steps) and for L4 ((C2-C1) on long vs n=600 episodes),
and AMS by step index for C0/C1/C2/C2b/C2w/A0/A1/A2/A2b.

Usage: python -m src.long_compare   (writes runs/long_compare_paired.csv, runs/long_compare_steps.csv)
"""
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from src.analyze import TEXT_T
from src.data import PROJ

RUNS = {"n600": [PROJ / "results" / "n600_at_per_step.parquet"],
        "odylong": [PROJ / "runs" / "odylong" / "results" / "n155_per_step.parquet"],
        "memgui": [PROJ / "runs" / "memgui" / "results" / "n295_per_step.parquet"]}
PAIRS = [  # (a, b, prediction)
    ("C1", "C2b", "L1"), ("A1", "A2b", "L1"), ("C1", "C2w", "L1"), ("C2", "C1", "L1/L4"), ("A2", "A1", "L1"),
    ("C1", "C3", "L2"), ("C1", "C5", "L2"), ("A1", "A3", "L2"), ("A1", "A5", "L2"),
    ("C1", "C0", "L3"), ("A1", "A0", "L3"), ("A1", "A7", "L3"), ("C1", "C7", "L3"), ("A0", "C1", "context"),
]
SUBSETS = {"all": lambda x: x, "MD_present": lambda x: x[x.is_md & x.present_any],
           "MD_strict_present": lambda x: x[x.is_md & x.md_strict & x.present_any]}
N_BOOT = 1000


def load(paths):
    d = pd.concat([pd.read_parquet(p) for p in paths])
    d = d.drop_duplicates(["episode_id", "step", "cond"])
    d["text_ok"] = d.text_sim_needed >= TEXT_T
    d["follow"] = d.text_sim_alt >= TEXT_T
    for c in ("is_md", "md_strict", "present_any"):
        d[c] = d[c].fillna(False).astype(bool)
    return d


def boot_diff(eps, va, vb, seed=0):
    """CI of mean(va) - mean(vb) over steps, resampling episodes."""
    g = pd.DataFrame({"e": eps, "a": va, "b": vb}).groupby("e").agg(["sum", "count"])
    sa, sb, n = g[("a", "sum")].to_numpy(float), g[("b", "sum")].to_numpy(float), g[("a", "count")].to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(n), size=(N_BOOT, len(n)))
    diff = 100 * (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def paired(d, a, b, sname, col):
    x = SUBSETS[sname](d[d.cond == a]).merge(d[d.cond == b], on=["episode_id", "step"], suffixes=("_a", "_b"))
    A, B = x[col + "_a"].astype(bool), x[col + "_b"].astype(bool)
    n10, n01 = int((A & ~B).sum()), int((~A & B).sum())
    lo, hi = boot_diff(x.episode_id.to_numpy(), A.to_numpy(float), B.to_numpy(float)) if len(x) else (np.nan, np.nan)
    return dict(n=len(x), a_pct=100 * A.mean(), b_pct=100 * B.mean(), diff=100 * (A.mean() - B.mean()), diff_lo=lo,
                diff_hi=hi, only_a=n10, only_b=n01,
                mcnemar_p=mcnemar([[0, n10], [n01, 0]], exact=True).pvalue if n10 + n01 else 1.0), x


def did(d, a1, b1, a2, b2, sname, col):
    """(a1-b1) - (a2-b2) on the steps all four conditions share, episode bootstrap."""
    keys = None
    vals = {}
    for c in (a1, b1, a2, b2):
        y = SUBSETS[sname](d[d.cond == c]).set_index(["episode_id", "step"])[col].astype(float)
        vals[c] = y
        keys = y.index if keys is None else keys.intersection(y.index)
    v = {c: vals[c].loc[keys].to_numpy() for c in vals}
    eps = keys.get_level_values(0).to_numpy()
    est = 100 * ((v[a1] - v[b1]) - (v[a2] - v[b2])).mean()
    lo, hi = boot_diff(eps, v[a1] - v[b1], v[a2] - v[b2])
    return dict(n=len(keys), did=est, did_lo=lo, did_hi=hi)


def main():
    rows, drows = [], []
    data = {r: load(p) for r, p in RUNS.items() if all(q.exists() for q in p)}
    for run, d in data.items():
        conds = set(d.cond)
        for a, b, pred in PAIRS:
            if a not in conds or b not in conds:
                continue
            for sname in SUBSETS:
                col = "correct" if sname == "all" else "text_ok"
                r, _ = paired(d, a, b, sname, col)
                rows.append(dict(run=run, prediction=pred, a=a, b=b, subset=sname, metric=col, **r))
        if {"A8", "C8"} <= conds:
            for sname in ("MD_present", "MD_strict_present"):
                for c in ("C8", "A8"):
                    x = SUBSETS[sname](d[(d.cond == c) & d.alternative.notna()])
                    rows.append(dict(run=run, prediction="L3", a=c, b="-", subset=sname, metric="follow", n=len(x),
                                     a_pct=100 * x.follow.mean()))
        if {"C1", "C3", "A1", "A3"} <= conds:
            for sname in SUBSETS:
                col = "correct" if sname == "all" else "text_ok"
                drows.append(dict(run=run, test="L2 (C1-C3)-(A1-A3)", subset=sname, metric=col,
                                  **did(d, "C1", "C3", "A1", "A3", sname, col)))
                drows.append(dict(run=run, test="L2 (C1-C5)-(A1-A5)", subset=sname, metric=col,
                                  **did(d, "C1", "C5", "A1", "A5", sname, col)))
    tab = pd.DataFrame(rows)
    # L4: C2-C1 all-step AMS gap, long vs n=600 episodes (unpaired: different episodes; bootstrap each run)
    for run in [r for r in ("odylong", "memgui") if r in data]:
        gaps = {}
        for r2 in ("n600", run):
            d = data[r2]
            x = d[d.cond == "C2"].merge(d[d.cond == "C1"], on=["episode_id", "step"], suffixes=("_a", "_b"))
            g = x.groupby("episode_id").agg(a=("correct_a", "sum"), b=("correct_b", "sum"), n=("correct_a", "size"))
            idx = np.random.default_rng(0).integers(0, len(g), size=(N_BOOT, len(g)))
            gaps[r2] = 100 * (g.a.to_numpy()[idx].sum(1) - g.b.to_numpy()[idx].sum(1)) / g.n.to_numpy()[idx].sum(1)
        diff = gaps[run] - gaps["n600"]
        drows.append(dict(run=run, test="L4 (C2-C1)[run] - (C2-C1)[n600]", subset="all", metric="correct",
                          did=float(np.mean(gaps[run]) - np.mean(gaps["n600"])),
                          did_lo=float(np.percentile(diff, 2.5)), did_hi=float(np.percentile(diff, 97.5))))
    dtab = pd.DataFrame(drows)
    # AMS by step index
    srows = []
    bins = [-1, 9, 19, 29, 39, 1000]
    labels = ["0-9", "10-19", "20-29", "30-39", "40+"]
    for run, d in data.items():
        x = d[d.cond.isin(["C0", "C1", "C2", "C2b", "C2w", "A0", "A1", "A2", "A2b"])].copy()
        x["bin"] = pd.cut(x.step, bins, labels=labels)
        t = x.groupby(["bin", "cond"], observed=True).correct.agg(["mean", "size"]).reset_index()
        t["run"] = run
        srows.append(t)
    steps = pd.concat(srows)
    steps["mean"] *= 100
    return tab, dtab, steps


if __name__ == "__main__":
    tab, dtab, steps = main()
    out = PROJ / "runs"
    tab.to_csv(out / "long_compare_paired.csv", index=False)
    dtab.to_csv(out / "long_compare_did.csv", index=False)
    steps.to_csv(out / "long_compare_steps.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(tab.round(3).to_string(index=False))
    print(dtab.round(3).to_string(index=False))
    print(steps.pivot_table(index=["run", "bin"], columns="cond", values="mean", observed=True).round(1).to_string())
