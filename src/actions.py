"""Converters between UI-Venus-1.5 actions and GUI-Odyssey (official eval) action strings.

UI-Venus-1.5 action space (tech report arXiv 2602.09082, Table 8 / App. A.3), coordinates in [0,1000]:
  Click(box=(x,y)) LongPress(box=(x,y)) Scroll(start=(x1,y1), end=(x2,y2)) Drag(start=..., end=...)
  Type(content='') Launch(app='') Wait() Finished(content='') CallUser(content='')
  PressBack() PressHome() PressEnter() PressRecent()

GUI-Odyssey official command strings (format_converter.decode_action + evaluate_GUIOdyssey.simple_decode):
  "CLICK: (x, y)"  "LONG_PRESS: (x, y)"  "SCROLL: UP|DOWN|LEFT|RIGHT"  "TYPE: text"
  "PRESS_HOME" "PRESS_BACK" "PRESS_RECENT" "COMPLETE" "IMPOSSIBLE"
Both use [0,1000] normalized coordinates and finger-trajectory scroll semantics, so no rescaling.

Mappings with no exact counterpart (documented in the report):
  Drag -> SCROLL (direction of start->end), CallUser -> COMPLETE, or IMPOSSIBLE if its content says the
  task is infeasible; PressEnter/Wait/Launch -> kept as PRESS_ENTER/WAIT/LAUNCH (never match GUI-Odyssey GT).
MemGUI-3K (MG_DATASET=memgui) adds GT PRESS_ENTER, WAIT and ANSWER (its answer action); there CallUser -> ANSWER.
"""
import os
import re

import numpy as np

# MemGUI-3K has an explicit answer action, so there CallUser(content) is scored as ANSWER (see src/memgui.py)
DATASET = os.environ.get("MG_DATASET", "odyssey")

_NUM = r"-?\d+(?:\.\d+)?"
IMPOSSIBLE_PAT = re.compile(r"impossible|infeasible|cannot be (completed|done)|not possible|unable to complete", re.I)


def extract_tag(tag, text):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else None


def _str_arg(argstr, key):
    """Extract a quoted string argument, tolerant to quotes/commas/parens inside the content."""
    m = re.search(rf"{key}\s*=\s*(['\"‘“])", argstr)
    if not m:
        m2 = re.search(rf"{key}\s*=\s*(.*)$", argstr, re.S)
        return m2.group(1).strip() if m2 else ""
    q = m.group(1)
    close = {"‘": "’", "“": "”"}.get(q, q)
    body = argstr[m.end():]
    j = body.rfind(close)
    if j < 0 and close != q:
        j = body.rfind(q)
    return body[:j] if j >= 0 else body


def _point(argstr, key):
    m = re.search(rf"{key}\s*=\s*[\(\[]\s*({_NUM})\s*,\s*({_NUM})(?:\s*,\s*({_NUM})\s*,\s*({_NUM}))?\s*[\)\]]", argstr)
    if not m:
        return None
    x1, y1 = float(m.group(1)), float(m.group(2))
    if m.group(3) is not None:  # a box was given: use its centre
        x1, y1 = (x1 + float(m.group(3))) / 2, (y1 + float(m.group(4))) / 2
    return (int(round(x1)), int(round(y1)))


def parse_venus(raw):
    """Parse raw UI-Venus output -> dict(name, args). Returns name='PARSE_FAIL' when unparseable."""
    act = extract_tag("action", raw)
    if act is None:
        act = raw.strip()
        m = re.search(r"(Click|LongPress|Scroll|Drag|Type|Launch|Wait|Finished|CallUser|PressBack|PressHome|PressEnter|PressRecent)\s*\(", act)
        if not m:
            return {"name": "PARSE_FAIL", "args": {}, "action_str": act}
        act = act[m.start():]
    act = act.strip().strip("`").strip()
    m = re.match(r"(\w+)\s*\((.*)\)\s*$", act, re.S)
    if not m:
        m = re.match(r"(\w+)\s*\((.*)$", act, re.S)
        if not m:
            return {"name": "PARSE_FAIL", "args": {}, "action_str": act}
    name, argstr = m.group(1), m.group(2)
    args = {}
    if name in ("Click", "LongPress"):
        args["box"] = _point(argstr, "box") or _point("box=" + argstr, "box")
    elif name in ("Scroll", "Drag"):
        args["start"] = _point(argstr, "start")
        args["end"] = _point(argstr, "end")
        d = re.search(r"direction\s*=\s*['\"](\w+)['\"]", argstr)
        if d:
            args["direction"] = d.group(1).lower()
    elif name in ("Type", "Finished", "CallUser"):
        args["content"] = _str_arg(argstr, "content")
    elif name == "Launch":
        args["app"] = _str_arg(argstr, "app")
    return {"name": name, "args": args, "action_str": act}


def scroll_dir(start, end):
    """Official GUI-Odyssey direction rule (format_converter.decode_action)."""
    delta = np.array(end) - np.array(start)
    delta_abs = np.abs(delta)
    lr = 'LEFT' if delta[0] < 0 else 'RIGHT'
    ud = 'UP' if delta[1] < 0 else 'DOWN'
    return lr if delta_abs[0] > delta_abs[1] else ud


def venus_to_odyssey(parsed):
    """Parsed UI-Venus action -> official GUI-Odyssey command string."""
    n, a = parsed["name"], parsed["args"]
    if n in ("Click", "LongPress"):
        if not a.get("box"):
            return "INVALID"
        tag = "CLICK" if n == "Click" else "LONG_PRESS"
        return f"{tag}: {tuple(a['box'])}"
    if n in ("Scroll", "Drag"):
        if a.get("start") and a.get("end") and tuple(a["start"]) != tuple(a["end"]):
            return f"SCROLL: {scroll_dir(a['start'], a['end'])}"
        if a.get("direction") in ("up", "down", "left", "right"):
            # Direction-only scroll: UI-Venus framework maps SWIPE_UP to a finger moving up, which is
            # GUI-Odyssey "UP" as well (finger semantics).
            return f"SCROLL: {a['direction'].upper()}"
        return "INVALID"
    if n == "Type":
        return f"TYPE: {a.get('content', '')}"
    if n == "PressHome":
        return "PRESS_HOME"
    if n == "PressBack":
        return "PRESS_BACK"
    if n == "PressRecent":
        return "PRESS_RECENT"
    if n == "Finished":
        return "COMPLETE"
    if n == "CallUser":
        if IMPOSSIBLE_PAT.search(a.get("content", "")):
            return "IMPOSSIBLE"
        return f"ANSWER: {a.get('content', '')}" if DATASET == "memgui" else "COMPLETE"
    if n == "PressEnter":
        return "PRESS_ENTER"
    if n == "Wait":
        return "WAIT"
    if n == "Launch":
        return "LAUNCH"
    return "INVALID"


def gt_to_venus(action, info):
    """GUI-Odyssey raw GT step (action, info) -> UI-Venus action string (used for round-trip tests and
    for describing the previous GT action to the controller)."""
    if action in ("CLICK", "LONG_PRESS"):
        if info == "KEY_HOME":
            return "PressHome()"
        if info == "KEY_BACK":
            return "PressBack()"
        if info == "KEY_APPSELECT":
            return "PressRecent()"
        x, y = info[0]
        return f"{'Click' if action == 'CLICK' else 'LongPress'}(box=({x},{y}))"
    if action == "SCROLL":
        (x1, y1), (x2, y2) = info[0], info[1]
        return f"Scroll(start=({x1},{y1}), end=({x2},{y2}))"
    if action == "TEXT":
        return f"Type(content='{info}')"
    if action == "COMPLETE":
        return "Finished(content='')"
    if action == "INCOMPLETE":
        return "CallUser(content='The task is impossible to complete.')"
    if action == "ENTER":
        return "PressEnter()"
    if action == "WAIT":
        return "Wait()"
    if action == "ANSWER":
        return f"CallUser(content='{info}')"
    raise ValueError(action)


def odyssey_decode(s):
    """Official simple_decode from evaluate_GUIOdyssey.py (verbatim semantics)."""
    gts = s.split(':')
    gt_action = gts[0].strip()
    if len(gts) > 1:
        action = gt_action
        info = gts[1].strip()
        if action in ['CLICK', "LONG_PRESS"]:
            info = eval(info)
    else:
        action = gt_action
        info = ""
    return {"action": action, "info": info}


def text_of(cmd):
    """Typed (or, for MemGUI-3K, answered) text of an official command string (None otherwise). Uses split(':',1) so
    text with ':' survives."""
    if cmd.startswith("TYPE:") or cmd.startswith("ANSWER:"):
        return cmd.split(":", 1)[1].strip()
    return None
