"""MementoGUI-style *working memory* built by a prompted controller (Qwen3-VL-8B-Instruct, no training).

For each episode we walk the ground-truth trajectory. At step t the controller sees: task instruction,
the previous GT action (the one that led to this screen), the current memory (text), and the current
screenshot. It returns JSON {salience, write, summary, roi_bbox}. If write, we append
{step, summary, salience, crop(roi_bbox)}. If memory > K=8 entries, the controller merges the two
oldest entries into one text summary. At most 4 crops are kept (most salient; ties -> most recent).

Cache: cache/{episode}/{t}.json is the memory state *before* step t (entries from steps < t only, so
the current screen never leaks into memory). Crops: cache/{episode}/crops/{k}.png.
Episodes run step-synchronously in one vLLM batch, so the controller is called once per (episode, step).
"""
import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image

from src.actions import gt_to_venus
from src.data import CACHE, RESULTS, load_episode, sample, shot_path
from src.vlm import PX_CONTROLLER, PX_CROP, greedy, make_llm, resize_to_budget, to_messages

K_MAX = 8
MAX_CROPS = 4
MAX_RETRIES = 2

WRITE_PROMPT = """You are the memory controller of a mobile GUI agent. The agent only sees the current screen, so you must decide what to remember from it for later steps of the task.

### Task
{task}

### Action just executed (it led to the current screen)
{prev_action}

### Current memory (from earlier steps)
{memory}

### Current screenshot
"""

WRITE_INSTR = """
### What to do
1. Rate how important the current screen is to remember for completing the task later (salience, 0 to 1). Screens that show information the task will need later (names, numbers, dates, addresses, prices, titles, links, search results, text to copy) are important; transitional screens (home screen, loading, menus) are not.
2. Decide whether to write a memory entry (write = true/false).
3. If writing, give a one-line summary of at most 30 words. Include the concrete values/names you see (copy text exactly as shown), not generic descriptions.
4. If a specific region of the screen holds the important information, give its bounding box roi_bbox = [x1, y1, x2, y2] in coordinates normalized to 0-1000 (x to the right, y downward). Otherwise roi_bbox = null.

Respond with ONLY a JSON object, no other text:
{{"salience": <number 0-1>, "write": <true/false>, "summary": "<=30 words>", "roi_bbox": [x1, y1, x2, y2] or null}}"""

MERGE_PROMPT = """You are the memory controller of a mobile GUI agent working on this task:
{task}

Merge the following older memory entries into ONE concise summary of at most 60 words. Keep every concrete value (names, numbers, dates, addresses, titles, links) exactly as written; drop navigation details.

{entries}

Respond with ONLY a JSON object: {{"summary": "<merged summary>"}}"""


def memory_text(entries):
    if not entries:
        return "(empty)"
    return "\n".join(f"[{label(e)}] {e['summary']}" for e in entries)


def label(e):
    return f"steps {e['step']}" if isinstance(e["step"], str) else f"step {e['step']}"


def extract_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no json")
    return json.loads(m.group(0))


def validate_write(d):
    sal = float(d["salience"])
    if not 0 <= sal <= 1:
        raise ValueError("salience range")
    w = d["write"]
    if isinstance(w, str):
        w = w.strip().lower() == "true"
    w = bool(w)
    summ = str(d.get("summary", "") or "").strip()
    words = summ.split()
    if len(words) > 30:  # enforce the 30-word cap deterministically rather than failing
        summ = " ".join(words[:30])
    bb = d.get("roi_bbox")
    if bb is not None:
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            bb = [float(v) for v in bb]
            x1, y1, x2, y2 = bb
            x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
            bb = [max(0, min(1000, v)) for v in (x1, y1, x2, y2)]
            if bb[2] - bb[0] < 5 or bb[3] - bb[1] < 5:
                bb = None
        else:
            raise ValueError("bad bbox")
    if w and not summ:
        raise ValueError("write without summary")
    return {"salience": sal, "write": w, "summary": summ, "roi_bbox": bb}


def crop_of(eid, step, bb):
    im = Image.open(shot_path(eid, step)).convert("RGB")
    W, H = im.size
    box = (int(bb[0] / 1000 * W), int(bb[1] / 1000 * H), int(bb[2] / 1000 * W), int(bb[3] / 1000 * H))
    c = im.crop(box)
    return resize_to_budget(c, PX_CROP), box


def prune_crops(entries):
    with_crop = [e for e in entries if e.get("crop")]
    if len(with_crop) <= MAX_CROPS:
        return
    keep = sorted(with_crop, key=lambda e: (e["salience"], e["step"] if isinstance(e["step"], int) else -1), reverse=True)[:MAX_CROPS]
    keep_ids = {id(e) for e in keep}
    for e in with_crop:
        if id(e) not in keep_ids:
            e["dropped_crop"] = e.pop("crop")


class EpisodeState:
    def __init__(self, eid):
        self.eid = eid
        self.ep = load_episode(eid)
        self.dir = CACHE / eid
        (self.dir / "crops").mkdir(parents=True, exist_ok=True)
        self.entries = []
        self.log = []
        self.n = len(self.ep["steps"])

    def save_state(self, t):
        json.dump({"episode_id": self.eid, "step": t, "entries": self.entries}, open(self.dir / f"{t}.json", "w"), indent=1)


def write_parts(st, t):
    ep = st.ep
    prev = "None (this is the first step)" if t == 0 else gt_to_venus(ep["steps"][t - 1]["action"], ep["steps"][t - 1]["info"])
    txt = WRITE_PROMPT.format(task=ep["task_info"]["instruction"], prev_action=prev, memory=memory_text(st.entries))
    img = resize_to_budget(Image.open(shot_path(st.eid, t)), PX_CONTROLLER)
    return [txt, img, WRITE_INSTR]


def run(eids, llm):
    sp_write, sp_merge = greedy(256), greedy(256)
    states = []
    for e in eids:
        if (CACHE / e / "done").exists():
            continue
        states.append(EpisodeState(e))
    print(f"{len(states)} episodes to process", flush=True)
    stats = {"calls": 0, "parse_fail_first": 0, "retries": 0, "final_fail": 0, "merge_calls": 0, "merge_fail": 0}
    T = max([s.n for s in states], default=0)
    for t in range(T):
        active = [s for s in states if t < s.n]
        for s in active:
            s.save_state(t)  # memory before step t
        # --- write decisions (with up to 2 retries on invalid JSON) ---
        pending = {id(s): s for s in active}
        results, raws = {}, {}
        for attempt in range(MAX_RETRIES + 1):
            if not pending:
                break
            batch = list(pending.values())
            outs = llm.chat([to_messages(write_parts(s, t)) for s in batch], sp_write, use_tqdm=False)
            stats["calls"] += len(batch)
            if attempt > 0:
                stats["retries"] += len(batch)
            for s, o in zip(batch, outs):
                raw = o.outputs[0].text
                raws.setdefault(id(s), []).append(raw)
                try:
                    results[id(s)] = validate_write(extract_json(raw))
                    pending.pop(id(s))
                except Exception:
                    if attempt == 0:
                        stats["parse_fail_first"] += 1
        for s in pending.values():
            stats["final_fail"] += 1
            results[id(s)] = {"salience": 0.0, "write": False, "summary": "", "roi_bbox": None, "failed": True}
        for s in active:
            r = results[id(s)]
            s.log.append({"step": t, "raw": raws.get(id(s)), "parsed": r})
            if r["write"]:
                e = {"step": t, "summary": r["summary"], "salience": r["salience"], "roi_bbox": r["roi_bbox"]}
                if r["roi_bbox"] is not None:
                    c, box = crop_of(s.eid, t, r["roi_bbox"])
                    fn = f"crops/{t}.png"
                    c.save(s.dir / fn)
                    e["crop"], e["crop_box_px"] = fn, box
                s.entries.append(e)
                prune_crops(s.entries)
        # --- compression: merge two oldest entries while > K_MAX ---
        need = [s for s in active if len(s.entries) > K_MAX]
        while need:
            prompts = []
            for s in need:
                ents = "\n".join(f"[{label(e)}] {e['summary']}" for e in s.entries[:2])
                prompts.append(to_messages([MERGE_PROMPT.format(task=s.ep["task_info"]["instruction"], entries=ents)]))
            outs = llm.chat(prompts, sp_merge, use_tqdm=False)
            stats["merge_calls"] += len(need)
            for s, o in zip(need, outs):
                a, b = s.entries[0], s.entries[1]
                raw = o.outputs[0].text
                try:
                    summ = str(extract_json(raw)["summary"]).strip()
                except Exception:
                    stats["merge_fail"] += 1
                    summ = f"{a['summary']} {b['summary']}"  # fallback: concatenate
                first = a["step"].split("-")[0] if isinstance(a["step"], str) else a["step"]
                last = b["step"].split("-")[-1] if isinstance(b["step"], str) else b["step"]
                merged = {"step": f"{first}-{last}", "summary": summ, "salience": max(a["salience"], b["salience"]),
                          "merged": True, "merged_from": [a, b]}
                for x in (a, b):  # merged entries lose their crops (text-only), keep a record
                    if x.get("crop"):
                        x["dropped_crop"] = x.pop("crop")
                s.entries = [merged] + s.entries[2:]
                s.log.append({"step": t, "merge_raw": raw})
            need = [s for s in need if len(s.entries) > K_MAX]
        for s in active:
            if t == s.n - 1:
                s.save_state(s.n)  # final state (after last step), unused by backbone
                with open(s.dir / "controller_log.jsonl", "w") as f:
                    for row in s.log:
                        f.write(json.dumps(row) + "\n")
                (s.dir / "done").touch()
        print(f"t={t} active={len(active)} stats={stats}", flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    eids = sample(a.n)[a.shard::a.nshards]
    t0 = time.time()
    llm = make_llm("controller", max_model_len=8192, max_images=1)
    stats = run(eids, llm)
    stats["seconds"] = time.time() - t0
    out = RESULTS / "controller_stats"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(out / f"n{a.n}_shard{a.shard}of{a.nshards}_{int(time.time())}.json", "w"), indent=1)
    print("DONE", stats)
