"""GUIOdyssey v2 loading, official GT decoding, stratified episode sampling."""
import json
import math
import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

# project root: $MG_PROJ if set, else the repo root (parent of src/)
PROJ = Path(os.environ.get("MG_PROJ", Path(__file__).resolve().parents[1]))
DATA = PROJ / "data" / "guiodyssey_v2"
ANNO = DATA / "annotations"
SHOTS = DATA / "screenshots"
SPLIT = "random_split"
CATEGORIES = ["General_Tool", "Information_Management", "Media_Entertainment",
              "Multi_Apps", "Social_Sharing", "Web_Shopping"]


@lru_cache(maxsize=None)
def load_episode(eid):
    return json.load(open(ANNO / f"{eid}.json"))


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
    return stratified_order(seed)[:n]


def shot_path(eid, step):
    return SHOTS / f"{eid}_{step}.png"


if __name__ == "__main__":
    import collections
    o = stratified_order(0)
    assert len(set(o)) == len(o) == len(test_ids())
    for n in (10, 300, 600):
        print(n, collections.Counter(load_episode(e)["task_info"]["category"] for e in o[:n]))
