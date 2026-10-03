"""Metrics, statistics and tables.

AMS uses the *official* GUI-Odyssey matcher (third_party/GUI-Odyssey/src/eval_mm/GUIOdyssey_action_matching.py,
imported unmodified) together with the official simple_decode, exactly as evaluate_GUIOdyssey.py does
(any exception -> incorrect). 'macro' = step accuracy, 'micro' = mean over the 6 categories (their naming).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from statsmodels.stats.contingency_tables import mcnemar

from src.actions import odyssey_decode, text_of
from src.data import PROJ, decode_action, load_episode, sample
from src.ocr import norm

sys.path.insert(0, str(PROJ / "third_party" / "GUI-Odyssey" / "src" / "eval_mm"))
from GUIOdyssey_action_matching import action_matching  # noqa: E402  (official, unmodified)

RAW = PROJ / "results" / "raw"
RES = PROJ / "results"
CONDS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C8t", "C8c"]
TEXT_T = 0.8
N_BOOT = 1000


def official_correct(pred, gt, sam2_bbox):
    """Mirror of evaluate_GUIOdyssey.action_matching_evaluation for one sample."""
    g = odyssey_decode(gt)
    try:
        p = odyssey_decode(pred)
    except Exception:
        return False, "invalid"
    try:
        r = action_matching(p["action"], p["info"], g["action"], g["info"], sam2_bbox)
    except Exception:
        return False, "invalid"
    return r["is_correct"] == "yes", r["info"]


def tsim(a, b):
    if not isinstance(a, str) or not isinstance(b, str):  # None / NaN (no TYPE prediction) -> no match
        return 0.0
    return Levenshtein.normalized_similarity(norm(a), norm(b))


def same_action(a, b):
    """Change-rate equivalence: same type and same argument/target region."""
    da, db = odyssey_decode_safe(a), odyssey_decode_safe(b)
    if da is None or db is None:
        return a == b
    if da["action"] != db["action"]:
        return False
    if da["action"] in ("CLICK", "LONG_PRESS"):
        pa, pb = np.array(da["info"]) / 1000, np.array(db["info"]) / 1000
        return float(np.linalg.norm(pa - pb)) <= 0.14  # official click radius
    if da["action"] == "TYPE":
        return tsim(text_of(a), text_of(b)) >= TEXT_T
    return da["info"] == db["info"]


def odyssey_decode_safe(s):
    try:
        return odyssey_decode(s)
    except Exception:
        return None


def load_results(n):
    rows = []
    for e in sample(n):
        f = RAW / f"{e}.jsonl"
        if f.exists():
            rows += [json.loads(l) for l in open(f)]
    df = pd.DataFrame(rows)
    gt = []
    for e in df.episode_id.unique():
        ep = load_episode(e)
        for s in ep["steps"]:
            gt.append(dict(episode_id=e, step=s["step"], gt_cmd=decode_action(s["action"], s["info"]),
                           gt_action=s["action"], sam2_bbox=s.get("sam2_bbox") or None,
                           category=ep["task_info"]["category"]))
    df = df.merge(pd.DataFrame(gt), on=["episode_id", "step"], how="left")
    cr = [official_correct(p, g, b) for p, g, b in zip(df.action, df.gt_cmd, df.sam2_bbox)]
    df["correct"] = [c for c, _ in cr]
    df["match_info"] = [i for _, i in cr]
    df["pred_type"] = df.action.map(lambda a: a.split(":")[0].strip())
    df["pred_text"] = df.action.map(text_of)
    return df


def boot_ci(df, fn, n_boot=N_BOOT, seed=0):
    """Percentile CI from resampling episodes with replacement."""
    eps = df.episode_id.unique()
    groups = {e: g for e, g in df.groupby("episode_id")}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(eps, size=len(eps), replace=True)
        vals.append(fn(pd.concat([groups[e] for e in pick])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ams_macro(d):
    return 100 * d.correct.mean()


def ams_micro(d):
    return 100 * d.groupby("category").correct.mean().mean()


def text_acc(d):
    return 100 * (d.text_sim_needed >= TEXT_T).mean()


def summarize(df, md, pres, cf, out_prefix):
    md = md.copy()
    md["step"] = md.step.astype(int)
    key = ["episode_id", "step"]
    df = df.merge(md[key + ["needed_string", "md_strict"]].assign(is_md=True), on=key, how="left")
    df["is_md"] = df.is_md.fillna(False).astype(bool)
    df["md_strict"] = df.md_strict.fillna(False).astype(bool)
    if len(pres):
        p = pres.copy(); p["step"] = p.step.astype(int)
        df = df.merge(p[key + ["present_text", "present_crop"]], on=key, how="left")
    else:
        df["present_text"] = np.nan; df["present_crop"] = np.nan
    df["present_any"] = (df.present_text == True) | (df.present_crop == True)
    df["crop_only"] = (df.present_crop == True) & (df.present_text != True)
    if len(cf):
        c = cf.copy(); c["step"] = c.step.astype(int)
        df = df.merge(c[key + ["alternative"]], on=key, how="left")
    else:
        df["alternative"] = None
    df["text_sim_needed"] = [tsim(p, n) if isinstance(n, str) else np.nan for p, n in zip(df.pred_text, df.needed_string)]
    df["text_sim_alt"] = [tsim(p, a) if isinstance(a, str) else np.nan for p, a in zip(df.pred_text, df.alternative)]
    c1 = df[df.cond == "C1"].set_index(key).action
    df["changed_vs_C1"] = [None if (e, s) not in c1.index else (not same_action(a, c1.loc[(e, s)]))
                           for e, s, a in zip(df.episode_id, df.step, df.action)]
    df.to_parquet(f"{out_prefix}_per_step.parquet")

    subsets = {
        "all": lambda d: d,
        "MD": lambda d: d[d.is_md],
        "MD_strict": lambda d: d[d.is_md & d.md_strict],
        "MD_present": lambda d: d[d.is_md & d.present_any],
        "MD_strict_present": lambda d: d[d.is_md & d.md_strict & d.present_any],
        "MD_crop_only": lambda d: d[d.is_md & d.crop_only],
        "MD_absent": lambda d: d[d.is_md & ~d.present_any],
    }
    rows = []
    for cond in [c for c in CONDS if c in set(df.cond)]:
        dc = df[df.cond == cond]
        for sname, sf in subsets.items():
            d = sf(dc)
            if len(d) == 0:
                continue
            r = dict(condition=cond, subset=sname, n_steps=len(d), n_episodes=d.episode_id.nunique())
            r["AMS"] = ams_macro(d); r["AMS_lo"], r["AMS_hi"] = boot_ci(d, ams_macro)
            if sname == "all":
                r["AMS_micro"] = ams_micro(d)
            if sname != "all":
                r["text_acc"] = text_acc(d); r["text_acc_lo"], r["text_acc_hi"] = boot_ci(d, text_acc)
                r["follow_cf"] = 100 * (d.text_sim_alt >= TEXT_T).mean()
                r["follow_cf_lo"], r["follow_cf_hi"] = boot_ci(d, lambda x: 100 * (x.text_sim_alt >= TEXT_T).mean())
                r["pred_TYPE_rate"] = 100 * (d.pred_type == "TYPE").mean()
            ch = d.changed_vs_C1.dropna()
            if len(ch):
                r["change_vs_C1"] = 100 * ch.astype(float).mean()
                r["change_lo"], r["change_hi"] = boot_ci(d[d.changed_vs_C1.notna()], lambda x: 100 * x.changed_vs_C1.astype(float).mean())
            # paired McNemar vs C1 (text accuracy on MD subsets, AMS correctness on 'all')
            if cond != "C1":
                d1 = sf(df[df.cond == "C1"])
                m = d.merge(d1, on=key, suffixes=("", "_c1"))
                if sname == "all":
                    a, b = m.correct, m.correct_c1
                else:
                    a, b = m.text_sim_needed >= TEXT_T, m.text_sim_needed_c1 >= TEXT_T
                tab = [[int(((b) & (a)).sum()), int(((b) & (~a)).sum())], [int(((~b) & (a)).sum()), int(((~b) & (~a)).sum())]]
                r["mcnemar_n_pairs"] = len(m)
                r["c1_only_correct"], r["cond_only_correct"] = tab[0][1], tab[1][0]
                r["mcnemar_p"] = float(mcnemar(tab, exact=True).pvalue) if tab[0][1] + tab[1][0] > 0 else 1.0
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{out_prefix}_summary_table.csv", index=False)
    return df, tab


def follow_table(df):
    """C8 follow rates on MD steps: counterfactual vs original vs neither (incl. C1 baseline)."""
    rows = []
    for cond in ["C1", "C8", "C8t", "C8c"]:
        for sname, mask in [("MD_with_cf", df.is_md & df.alternative.notna()),
                            ("MD_present_with_cf", df.is_md & df.present_any & df.alternative.notna())]:
            d = df[(df.cond == cond) & mask]
            if not len(d):
                continue
            cfm = d.text_sim_alt >= TEXT_T
            orig = (d.text_sim_needed >= TEXT_T) & ~cfm
            rows.append(dict(condition=cond, subset=sname, n=len(d),
                             follow_cf=100 * cfm.mean(), original=100 * orig.mean(), neither=100 * (~cfm & ~orig).mean(),
                             follow_cf_lo=boot_ci(d, lambda x: 100 * (x.text_sim_alt >= TEXT_T).mean())[0],
                             follow_cf_hi=boot_ci(d, lambda x: 100 * (x.text_sim_alt >= TEXT_T).mean())[1]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--md", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    md = pd.read_parquet(a.md or RES / f"md_steps_n{a.n}.parquet")
    pres = pd.read_parquet(RES / f"presence{a.tag}.parquet") if (RES / f"presence{a.tag}.parquet").exists() else pd.DataFrame()
    cf = pd.read_parquet(RES / f"counterfactuals{a.tag}.parquet") if (RES / f"counterfactuals{a.tag}.parquet").exists() else pd.DataFrame()
    df = load_results(a.n)
    prefix = str(RES / f"n{a.n}{a.tag}")
    df, tab = summarize(df, md, pres, cf, prefix)
    ft = follow_table(df)
    ft.to_csv(f"{prefix}_follow_table.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(tab.round(2).to_string())
    print(ft.round(2).to_string())
