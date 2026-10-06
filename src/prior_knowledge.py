"""How much typed text can the backbone produce WITHOUT seeing it? (follow-up to the action-trace run)

Every GT TYPE step (n=600, strings >= 3 chars) is assigned one source, in priority order:
  1 in instruction        needed string fuzzy-contained in the task instruction/task text
  2 on current screen     visible now (OCR or the dataset's screen description)
  3 earlier (strict MD)   only on an earlier screen / in earlier typed text  (= strict MD steps)
  4 seen nowhere          none of the above: the string never appears in any input up to step t
For source 4 a correct string can only come from the model's priors (world knowledge, the dataset's task templates,
or memorised training data) or from OCR misses. Reports fuzzy (sim >= 0.8, the study's text metric) and exact
(normalised equality) accuracy per condition, unconditional and given that the model typed.
"""
import argparse

import pandas as pd

from src.analyze import TEXT_T, tsim
from src.data import PROJ
from src.ocr import norm

RES = PROJ / "results"
CONDS = ["C0", "C1", "C2", "C7", "A0", "A1", "A2", "A7"]


def source(r):
    if r.in_instruction:
        return "1 in instruction"
    if r.on_current:
        return "2 on current screen"
    if r.md:
        return "3 earlier only (strict MD)"
    return "4 seen nowhere"


def main(n, per_step):
    t = pd.read_parquet(RES / f"type_steps_n{n}.parquet")
    t = t[~t.too_short.astype(bool)].copy()
    t["source"] = t.apply(source, axis=1)
    d = pd.read_parquet(per_step)
    d = d[d.cond.isin(CONDS)].merge(t[["episode_id", "step", "needed_string", "source"]].rename(columns={"needed_string": "need"}),
                                    on=["episode_id", "step"])
    d["typed"] = d.pred_type == "TYPE"
    d["fuzzy"] = [isinstance(p, str) and tsim(p, q) >= TEXT_T for p, q in zip(d.pred_text, d.need)]
    d["exact"] = [isinstance(p, str) and norm(p) == norm(q) for p, q in zip(d.pred_text, d.need)]
    rows = []
    for (s, c), g in d.groupby(["source", "cond"]):
        rows.append(dict(source=s, condition=c, n=len(g), typed=100 * g.typed.mean(), fuzzy=100 * g.fuzzy.mean(),
                         exact=100 * g.exact.mean(), fuzzy_given_typed=100 * g[g.typed].fuzzy.mean() if g.typed.any() else None))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--per_step", default=str(RES / "n600_at_per_step.parquet"))
    a = ap.parse_args()
    tab = main(a.n, a.per_step)
    tab.to_csv(RES / f"n{a.n}_at_prior_table.csv", index=False)
    pd.set_option("display.width", 200)
    for col in ["fuzzy", "exact", "typed"]:
        print(f"\n% {col}"); print(tab.pivot(index="source", columns="condition", values=col)[CONDS].round(1))
