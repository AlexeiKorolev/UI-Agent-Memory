"""MemGUI-3K (lgy0404/MemGUI-3K, Apache-2.0) test split -> the GUI-Odyssey annotation schema used by this pipeline.

Each released step is one MemGUI-Agent model call (teacher rollout). Ground truth = the UI action in the step's
<tool_call> (coordinates normalized to 0-1000, like GUI-Odyssey). The step's `action` field is the executed log in
screen pixels and is not used. Conversion:
  click / long_press (x, y)      -> CLICK / LONG_PRESS [[x, y]]
  swipe (x, y) -> (x2, y2)       -> SCROLL [[x, y], [x2, y2]]  (finger trajectory, as in GUI-Odyssey)
  type text                      -> TEXT text
  system_button Home/Back/Enter  -> CLICK KEY_HOME / CLICK KEY_BACK / ENTER
  wait                           -> WAIT
  answer text                    -> ANSWER text      (UI-Venus answers with CallUser(content))
  terminate success / failure    -> COMPLETE / INCOMPLETE
  memory_add/update/delete       -> dropped (the agent's own context actions; the screen does not change)
  steps without a parsable tool call (4 in the test split) -> dropped
Steps the dataset's evaluator marked unreasonable stay in the trajectory (they happened, so they shape later screens,
memory and history) but get score=False and are not evaluated. The teacher's responses (reasoning, folded history,
memory contents) are never shown to any model. Its <ui_observation> is stored as `teacher_observation` but NOT used as
the step `description` (empty): the pilot showed it often describes the screen *after* the action (e.g. "the title
field now contains 'First Day of Summer'" while the field is empty), so MD labels rely on OCR alone.

Usage (login node; needs internet):  MG_DATASET=memgui python -m src.memgui convert | download
"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.data import PROJ
from src.remote_zip import SplitZip

REPO = "lgy0404/MemGUI-3K"
REV = "003822b5aa3d5450232b8e6305d3094f564a1cf1"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/image_archives/"
PARTS = ["images.z01", "images.z02", "images.z03", "images.z04", "images.zip"]
SIZES = [10737418240] * 4 + [5223032870]   # HF tree API at REV
D = PROJ / "data" / "memgui3k"
ANNO, SHOTS = D / "annotations", D / "screenshots"
BUTTONS = {"home": ("CLICK", "KEY_HOME"), "back": ("CLICK", "KEY_BACK"), "enter": ("ENTER", None)}


def tool_call(resp):
    m = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", resp, re.S)
    if len(m) != 1:
        return None
    try:
        return json.loads(m[0])["arguments"]
    except (json.JSONDecodeError, KeyError):
        return None


def to_gt(a):
    """MemGUI tool-call arguments -> (GUI-Odyssey action, info), or None for context actions."""
    k = a.get("action")
    if k in ("click", "long_press"):
        return k.upper(), [list(map(int, a["coordinate"]))]
    if k == "swipe":
        return "SCROLL", [list(map(int, a["coordinate"])), list(map(int, a["coordinate2"]))]
    if k == "type":
        return "TEXT", a["text"]
    if k == "system_button":
        return BUTTONS[a["button"].strip().lower()]
    if k == "wait":
        return "WAIT", None
    if k == "answer":
        return "ANSWER", a["text"]
    if k == "terminate":
        return ("COMPLETE" if a.get("status") == "success" else "INCOMPLETE"), None
    if k.startswith("memory_"):
        return None
    raise ValueError(f"unknown action {k}")


def query_of(user_prompt):
    m = re.search(r"### User Query\n(.*?)\n\n###", user_prompt, re.S)
    return m.group(1).strip()


def convert():
    ANNO.mkdir(parents=True, exist_ok=True)
    stats = dict(trajectories=0, steps_in=0, steps_out=0, scored=0, dropped_memory=0, dropped_no_call=0)
    members = {}
    for line in open(D / "test_trajectories.jsonl"):
        tr = json.loads(line)
        eid = tr["task_id"]
        src = sorted(tr["steps"], key=lambda s: int(s["step"]))
        instr = query_of(src[0]["user_prompt"])
        steps = []
        for s in src:
            stats["steps_in"] += 1
            a = tool_call(s["assistant_response"])
            if a is None:
                stats["dropped_no_call"] += 1
                continue
            gt = to_gt(a)
            if gt is None:
                stats["dropped_memory"] += 1
                continue
            k = len(steps)
            obs = re.search(r"<ui_observation>(.*?)</ui_observation>", s["assistant_response"], re.S)
            ok = s["is_reasonable"] in (True, "True", "true")
            steps.append(dict(step=k, action=gt[0], info=gt[1], screenshot=f"{eid}_{k}.png", score=ok, description="",
                              teacher_observation=obs.group(1).strip() if obs else "", src_step=int(s["step"]),
                              src_screenshot=s["screenshot"]))
            members[f"{eid}_{k}.png"] = Path(s["screenshot"]).name
            stats["scored"] += ok
        stats["trajectories"] += 1
        stats["steps_out"] += len(steps)
        ep = dict(episode_id=eid, task_info=dict(instruction=instr, task=instr, category="MemGUI", meta_task="",
                                                 trajectory_id=f"{tr['session']}/{eid}/{tr['agent']}/{tr['attempt']}"),
                  steps=steps)
        json.dump(ep, open(ANNO / f"{eid}.json", "w"), indent=1)
    json.dump(members, open(D / "members.json", "w"))
    eids = sorted(p.stem for p in ANNO.glob("*.json"))
    random.Random(0).shuffle(eids)   # pilot = first 10 of this fixed order
    run = PROJ / "runs" / "memgui"
    run.mkdir(parents=True, exist_ok=True)
    (run / "episodes.txt").write_text("\n".join(eids) + "\n")
    print(stats)


def download(workers=8):
    members = json.load(open(D / "members.json"))
    SHOTS.mkdir(parents=True, exist_ok=True)
    z = SplitZip(base=BASE, parts=PARTS, sizes=SIZES)
    idx_path = D / "zip_index.json"
    if not idx_path.exists():
        json.dump(z.central_directory(), open(idx_path, "w"))
    idx = json.load(open(idx_path))
    print("central directory entries:", len(idx), flush=True)
    todo = [(out, m) for out, m in members.items() if not (SHOTS / out).exists()]
    missing = [m for _, m in todo if m not in idx]
    print(f"members {len(members)}, todo {len(todo)}, missing from zip {len(missing)}", flush=True)

    def job(x):
        out, m = x
        if m not in idx:
            return False
        tmp = SHOTS / (out + ".tmp")
        tmp.write_bytes(z.read_member(idx[m]))
        tmp.rename(SHOTS / out)
        return True

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for ok in ex.map(job, todo):
            done += ok
            if done % 500 == 0:
                print("downloaded", done, flush=True)
    print("done", done, "missing", missing[:20])


if __name__ == "__main__":
    {"convert": convert, "download": download}[sys.argv[1]]()
