"""Paired comparisons within the action-trace follow-up (exact McNemar), beyond the vs-C1 tests in analyze.py.

Writes results/n600_at_paired.csv: for each pair, all-step AMS correctness and text accuracy on MD-present and
strict-MD-present steps.
"""
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from src.analyze import TEXT_T
from src.data import PROJ

RES = PROJ / "results"
PAIRS = [("A0", "C0"), ("A0", "C1"), ("A1", "C1"), ("A1", "A0"), ("A7", "A0"), ("A1", "A7"), ("A2", "C2"), ("A2", "A1"),
         ("A8", "C8")]
SUBSETS = {"all": lambda x: x, "MD_present": lambda x: x[x.is_md_a & x.present_any_a],
           "MD_strict_present": lambda x: x[x.is_md_a & x.md_strict_a & x.present_any_a]}


def main(per_step):
    d = pd.read_parquet(per_step)
    d["text_ok"] = d.text_sim_needed >= TEXT_T
    d["follow"] = d.text_sim_alt >= TEXT_T
    rows = []
    for a, b in PAIRS:
        x = d[d.cond == a].merge(d[d.cond == b], on=["episode_id", "step"], suffixes=("_a", "_b"))
        for sname, sf in SUBSETS.items():
            if a == "A8" and sname == "all":
                continue
            y = sf(x)
            for col in (["correct"] if sname == "all" else ["text_ok"] + (["follow"] if a == "A8" else [])):
                A, B = y[col + "_a"].astype(bool), y[col + "_b"].astype(bool)
                n10, n01 = int((A & ~B).sum()), int((~A & B).sum())
                p = mcnemar([[0, n10], [n01, 0]], exact=True).pvalue if n10 + n01 else 1.0
                rows.append(dict(a=a, b=b, subset=sname, metric=col, n=len(y), a_pct=100 * A.mean(), b_pct=100 * B.mean(),
                                 only_a=n10, only_b=n01, mcnemar_p=p))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tab = main(RES / "n600_at_per_step.parquet")
    tab.to_csv(RES / "n600_at_paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))
