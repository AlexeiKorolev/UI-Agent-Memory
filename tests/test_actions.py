import random

from src.actions import gt_to_venus, odyssey_decode, parse_venus, venus_to_odyssey
from src.data import decode_action, load_episode, sample


def _gt_steps(n=20, seed=0):
    """20 GT steps from the sample, covering every action type present (stratified by type)."""
    by_type = {}
    for e in sample(300):
        for s in load_episode(e)["steps"]:
            key = s["action"] if not isinstance(s["info"], str) or s["action"] == "TEXT" else f"{s['action']}:{s['info']}"
            by_type.setdefault(key, []).append(s)
    rng = random.Random(seed)
    out = []
    keys = sorted(by_type)
    while len(out) < n:
        for k in keys:
            if len(out) < n:
                out.append(rng.choice(by_type[k]))
    return out


def test_round_trip_20_gt_steps():
    steps = _gt_steps()
    assert len(steps) == 20
    for s in steps:
        gt = decode_action(s["action"], s["info"])
        venus = gt_to_venus(s["action"], s["info"])
        back = venus_to_odyssey(parse_venus(f"<think>x</think><action>{venus}</action><conclusion>y</conclusion>"))
        assert back == gt, (s, venus, back, gt)
        assert odyssey_decode(back) == odyssey_decode(gt)


def test_type_with_quotes_and_commas():
    raw = "<action>Type(content='Arthur Ashkin, Gerard Mourou (2018) it's \"great\"')</action>"
    assert venus_to_odyssey(parse_venus(raw)) == "TYPE: Arthur Ashkin, Gerard Mourou (2018) it's \"great\""


def test_click_box_centre_and_no_tags():
    assert venus_to_odyssey(parse_venus("Click(box=(100,200,300,400))")) == "CLICK: (200, 300)"
    assert venus_to_odyssey(parse_venus("I will tap. Click(box=(5, 7))")) == "CLICK: (5, 7)"


def test_scroll_semantics_match_official():
    # finger moves up (y decreases) -> official "UP"
    assert venus_to_odyssey(parse_venus("<action>Scroll(start=(500,800), end=(500,200))</action>")) == "SCROLL: UP"
    assert venus_to_odyssey(parse_venus("<action>Scroll(start=(800,500), end=(100,500))</action>")) == "SCROLL: LEFT"


def test_special_keys_and_terminal():
    assert venus_to_odyssey(parse_venus("<action>PressBack()</action>")) == "PRESS_BACK"
    assert venus_to_odyssey(parse_venus("<action>Finished(content='done')</action>")) == "COMPLETE"
    assert venus_to_odyssey(parse_venus("<action>CallUser(content='The task is impossible.')</action>")) == "IMPOSSIBLE"
    assert venus_to_odyssey(parse_venus("garbage")) == "INVALID"
