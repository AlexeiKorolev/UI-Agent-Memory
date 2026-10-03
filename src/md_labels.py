"""Memory-dependent (MD) step labels.

A step is MD if its GT action is TYPE (GUI-Odyssey v2 'TEXT') and the typed text
  (a) does NOT appear on the current screen (OCR, or the dataset's per-step screen `description`), and
  (b) DOES appear on an earlier screen of the same episode (OCR) or in an earlier step's typed text.
Matching: normalized Levenshtein similarity >= 0.8 between the (normalized) needed string and the best
window of consecutive OCR words (src.ocr.best_window_sim).
Extra flag (deviation, see report): `in_instruction` -- needed string fuzzy-contained in the task
instruction/task text. Primary MD set = MD and not in_instruction ("md_strict"); both are saved.
"""
import argparse
import random

import pandas as pd

from src.data import PROJ, load_episode, sample
from src.ocr import SIM_THRESH, best_window_sim, norm, ocr_cached, text_sim_in

MIN_LEN = 3


def label_episode(eid):
    ep = load_episode(eid)
    steps = ep["steps"]
    instr = ep["task_info"]["instruction"] + " " + ep["task_info"]["task"]
    rows = []
    ocr = {}
    for s in steps:
        if s["action"] != "TEXT":
            continue
        t = s["step"]
        needed = str(s["info"]).strip()
        if len(norm(needed)) < MIN_LEN:
            rows.append(dict(episode_id=eid, step=t, needed_string=needed, too_short=True))
            continue
        if t not in ocr:
            ocr[t] = ocr_cached(s["screenshot"])
        cur_sim, _ = best_window_sim(needed, ocr[t]["words"])
        desc_sim = text_sim_in(needed, s.get("description", "") or "")
        srcs = []
        for k in range(t):
            if k not in ocr:
                ocr[k] = ocr_cached(steps[k]["screenshot"])
            sim, _ = best_window_sim(needed, ocr[k]["words"])
            if sim >= SIM_THRESH:
                srcs.append((k, "screen", sim))
            if steps[k]["action"] == "TEXT":
                tsim = text_sim_in(needed, str(steps[k]["info"]))
                if tsim >= SIM_THRESH:
                    srcs.append((k, "typed", tsim))
        on_current = cur_sim >= SIM_THRESH or desc_sim >= SIM_THRESH
        md = (not on_current) and len(srcs) > 0
        rows.append(dict(
            episode_id=eid, step=t, needed_string=needed, too_short=False,
            cur_ocr_sim=cur_sim, cur_desc_sim=desc_sim, on_current=on_current,
            in_instruction=text_sim_in(needed, instr) >= SIM_THRESH,
            source_step=max(k for k, _, _ in srcs) if srcs else None,       # most recent source
            first_source_step=min(k for k, _, _ in srcs) if srcs else None,
            source_kinds=",".join(sorted({kind for _, kind, _ in srcs})),
            src_sim=max((x for _, _, x in srcs), default=0.0),
            md=md, category=ep["task_info"]["category"]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    rows = []
    for e in sample(a.n):
        rows += label_episode(e)
    df = pd.DataFrame(rows)
    out = PROJ / "results"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / f"type_steps_n{a.n}.parquet")
    md = df[df.md == True].copy()
    md["md_strict"] = ~md.in_instruction.astype(bool)
    md[["episode_id", "step", "needed_string", "source_step", "first_source_step", "source_kinds",
        "in_instruction", "md_strict", "category", "src_sim", "cur_ocr_sim"]].to_parquet(out / f"md_steps_n{a.n}.parquet")
    # random non-MD comparison sample (any action type), seed 0
    md_keys = set(zip(md.episode_id, md.step))
    non = [(e, s["step"], s["action"]) for e in sample(a.n) for s in load_episode(e)["steps"] if (e, s["step"]) not in md_keys]
    rng = random.Random(0)
    pd.DataFrame(rng.sample(non, min(300, len(non))), columns=["episode_id", "step", "action"]).to_parquet(out / f"nonmd_sample_n{a.n}.parquet")
    print(f"TYPE steps: {len(df)}  too_short: {int(df.too_short.sum())}")
    print(f"MD (spec definition): {len(md)}  in {md.episode_id.nunique()} episodes")
    print(f"MD strict (not in instruction): {int(md.md_strict.sum())}")
    print(md.groupby('category').size())
    print(md.source_kinds.value_counts())
