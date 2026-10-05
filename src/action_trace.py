"""Action-trace follow-up: add the list of past (GT) actions to the "### Previous Actions" slot.

Built from the *existing* n=600 condition specs (results/specs/, never rebuilt), so memory contents, C7 donors and
C8 edited crops are byte-identical to the main study; only an action-trace block is added. Output specs go to
results/specs_at/ and backbone outputs to results/raw_at/, so the main study's files are never touched.

  A0  actions only (no memory): the backbone's native history format; the baseline C0 lacks
  A1  C1 memory + actions                  -> does knowing what was done fix state errors? (vs C1, vs A0)
  A2  C2 raw past screenshots, each followed by the action taken on it (the 20-screenshot cap of C2 kept;
      actions of older steps listed as text)
  A7  C7 (other episode's memory) + the recipient's own actions  -> is A1's gain over A0 memory *content*?
  A8  C8 (MD steps: needed string -> alternative in notes and crops) + actions, with the needed string also replaced
      in any earlier Type(...) action, so the original string appears nowhere in the history

Past actions are the ground-truth actions (teacher-forced, like the memory itself), formatted by gt_to_venus in the
backbone's own action syntax. "[step k] <action>" is the action taken on screen k, matching the memory labels.
"""
import argparse
import json

import pandas as pd

from src.actions import gt_to_venus
from src.conditions import C2_CAP, MEM_HEADER, SPECS
from src.counterfactual import edit_text
from src.data import PROJ, load_episode, sample, shot_path
from src.vlm import PX_HISTORY

SPECS_AT = PROJ / "results" / "specs_at"
TRACE_HEADER = "Actions taken so far:\n"
AT_CONDS = ["A0", "A1", "A2", "A7", "A8"]


def block_of(spec_parts):
    """The parts placed in the Previous Actions slot: spec = [prompt_pre, *block, prompt_post, current screenshot]."""
    return spec_parts[1:-2]


def with_block(spec_parts, block):
    return [spec_parts[0], *block, *spec_parts[-2:]]


def past_actions(ep, t):
    return [gt_to_venus(ep["steps"][k]["action"], ep["steps"][k]["info"]) for k in range(t)]


def trace_block(actions):
    return [{"text": TRACE_HEADER + "".join(f"[step {k}] {a}\n" for k, a in enumerate(actions))}]


def raw_history_with_actions(eid, actions):
    t = len(actions)
    parts = [{"text": MEM_HEADER}]
    for k in range(t):
        if k < t - C2_CAP:
            parts.append({"text": f"[step {k}] Action: {actions[k]}\n"})
        else:
            parts += [{"text": f"[step {k}]\n"}, {"image": str(shot_path(eid, k)), "px": PX_HISTORY},
                      {"text": f"\nAction: {actions[k]}\n"}]
    return parts


def build(n, cf_path):
    SPECS_AT.mkdir(parents=True, exist_ok=True)
    cf = pd.read_parquet(cf_path)
    alt_of = {(r.episode_id, int(r.step)): (r.needed_string, r.alternative) for r in cf.itertuples() if pd.notna(r.alternative)}
    stats = dict(steps=0, A8=0, A8_trace_edits=0, A7_none=0)
    for eid in sample(n):
        ep = load_episode(eid)
        spec = json.load(open(SPECS / f"{eid}.json"))
        out = {}
        for ts, cs in spec.items():
            t = int(ts)
            acts = past_actions(ep, t)
            c = {}
            if t == 0:  # no history: identical to C0 (deduplicated by the backbone); MD steps never occur at t=0
                c = {k: cs["C0"] for k in ["A0", "A1", "A2", "A7"]}
            else:
                tr = trace_block(acts)
                c["A0"] = with_block(cs["C0"], tr)
                c["A1"] = with_block(cs["C1"], block_of(cs["C1"]) + tr)
                c["A2"] = with_block(cs["C2"], raw_history_with_actions(eid, acts))
                if cs.get("C7") is None:
                    c["A7"] = None
                    stats["A7_none"] += 1
                else:
                    blk = [] if cs["C7"] == cs["C0"] else block_of(cs["C7"])
                    c["A7"] = with_block(cs["C7"], blk + tr)
                if cs.get("C8"):
                    needed, alt = alt_of[(eid, t)]
                    edited = []
                    for a in acts:
                        a2, k = edit_text(a, needed, alt) if a.startswith("Type(") else (a, 0)
                        edited.append(a2)
                        stats["A8_trace_edits"] += k
                    c["A8"] = with_block(cs["C8"], block_of(cs["C8"]) + trace_block(edited))
            stats["steps"] += 1
            stats["A8"] += bool(c.get("A8"))
            out[ts] = c
        json.dump(out, open(SPECS_AT / f"{eid}.json", "w"))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--cf", default=str(PROJ / "results" / "counterfactuals.parquet"))
    a = ap.parse_args()
    print(build(a.n, a.cf))
