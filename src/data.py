"""GUIOdyssey v2 loading, official GT decoding, stratified episode sampling.

Follow-up runs (long GUI-Odyssey episodes, MemGUI-3K) are selected with two environment variables:
  MG_DATASET  odyssey (default) | memgui   -- which converted dataset load_episode()/shot_path() read
  MG_RUN      run name (default: unset = main study). If set, every output (controller cache, specs, raw backbone
              outputs, tables) goes under runs/<MG_RUN>/ and sample(n) returns the first n episodes of
              runs/<MG_RUN>/episodes.txt, so the main study's files are never touched.
"""
import json
import math
import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

# project root: $MG_PROJ if set, else the repo root (parent of src/)
PROJ = Path(os.environ.get("MG_PROJ", Path(__file__).resolve().parents[1]))
DATASET = os.environ.get("MG_DATASET", "odyssey")
RUN = os.environ.get("MG_RUN", "")
if DATASET == "memgui":
    DATA = PROJ / "data" / "memgui3k"   # converted by src.memgui into the GUI-Odyssey annotation schema
    CATEGORIES = ["MemGUI"]
else:
    DATA = PROJ / "data" / "guiodyssey_v2"
    CATEGORIES = ["General_Tool", "Information_Management", "Media_Entertainment",
                  "Multi_Apps", "Social_Sharing", "Web_Shopping"]
ANNO = DATA / "annotations"
SHOTS = DATA / "screenshots"
SPLIT = "random_split"
OUT = PROJ / "runs" / RUN if RUN else PROJ   # root for cache/ and results/
CACHE = OUT / "cache"
RESULTS = OUT / "results"


@lru_cache(maxsize=None)
def load_episode(eid):
    return json.load(open(ANNO / f"{eid}.json"))


def scored(step):
    """Steps evaluated by the backbone. All GUI-Odyssey steps; MemGUI-3K steps the dataset's evaluator marked
    reasonable (unreasonable steps stay in the trajectory, so they appear in history and memory)."""
    return step.get("score", True)


def test_ids():
    return [x.replace(".json", "") for x in json.load(open(DATA / "splits" / f"{SPLIT}.json"))["test"]]


def decode_action(action, info):
    """Verbatim logic of GUI-Odyssey/data/format_converter.py::decode_action (official GT string)."""
    if action == 'CLICK' or action == "LONG_PRESS":
        if info == 'KEY_HOME':
            gt = 'PRESS_HOME'
        elif info == 'KEY_BACK':
            gt = 'PRESS_BACK'
        elif info == 'KEY_APPSELECT':
            gt = 'PRESS_RECENT'
        elif type(info) == list:
            gt = f"{action}: {tuple(info[0])}"
        else:
            raise ValueError(f'Unknown click action {info}')
    elif action == 'SCROLL':
        start = np.array(info[0])
        end = np.array(info[1])
        delta = end - start
        delta_abs = np.abs(delta)
        lr = 'LEFT' if delta[0] < 0 else 'RIGHT'
        ud = 'UP' if delta[1] < 0 else 'DOWN'
        if delta_abs[0] > delta_abs[1]:
            gt = f"SCROLL: {lr}"
        else:
            gt = f"SCROLL: {ud}"
    elif action == 'TEXT':
        gt = f'TYPE: {info}'
    elif action == 'COMPLETE':
        gt = action
    elif action == 'INCOMPLETE':
        gt = 'IMPOSSIBLE'
    # MemGUI-3K actions with no GUI-Odyssey counterpart (not in the official converter): scored by the official
    # matcher's generic rule (action type must match)
    elif action == 'ENTER':
        gt = 'PRESS_ENTER'
    elif action == 'WAIT':
        gt = 'WAIT'
    elif action == 'ANSWER':
        gt = f'ANSWER: {info}'
    else:
        raise ValueError(f'Unknown action {action}')
    return gt


def stratified_order(seed=0):
    """Deterministic stratified ordering of test episodes.

    Each category list is shuffled with `seed`; the i-th episode of category c gets key
    (i + 0.5) / n_c. Sorting by key gives an ordering where every prefix (N=10, 300, 600, ...)
    is category-proportional to within +-1 episode, so the 600 extension is a superset of 300.
    """
    rng = random.Random(seed)
    by_cat = {c: [] for c in CATEGORIES}
    for e in sorted(test_ids()):
        by_cat[load_episode(e)["task_info"]["category"]].append(e)
    keyed = []
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])
        n = len(by_cat[c])
        keyed += [((i + 0.5) / n, CATEGORIES.index(c), e) for i, e in enumerate(by_cat[c])]
    return [e for _, _, e in sorted(keyed)]


def sample(n, seed=0):
    if RUN:
        return [l.strip() for l in open(OUT / "episodes.txt") if l.strip()][:n]
    return stratified_order(seed)[:n]


def shot_path(eid, step):
    return SHOTS / f"{eid}_{step}.png"


if __name__ == "__main__":
    import collections
    o = stratified_order(0)
    assert len(set(o)) == len(o) == len(test_ids())
    for n in (10, 300, 600):
        print(n, collections.Counter(load_episode(e)["task_info"]["category"] for e in o[:n]))
