"""Build backbone prompt specs for every (episode, step, condition) from the controller cache.

All conditions share the official UI-Venus-1.5 mobile prompt (tech report App. A.3) and the current
screenshot; they differ ONLY in the block placed in the "### Previous Actions" slot:
  C0  None
  C1  "Memory of earlier steps:" + per entry "[step k] summary" + crop image
  C2  all past GT screenshots (most recent 20, downsampled to 0.35 MP), each "[step k]" + image
  C3  C1 without crops            C4  C1 with summaries replaced by "[step k]"
  C5  crops -> uniform gray (128) image of the same size
  C6  crops deranged among entries of the same memory (donor crops from earlier dropped entries when
      fewer than 2 crops; n/a if impossible)
  C7  memory of another sampled episode, same category, matched #entries (then #crops), relabelled
  C8  MD steps: needed string replaced by a same-type alternative in text AND crops
  C8t text-only edit            C8c crop-only edit
Specs are JSON lists of parts: {"text": str} | {"image": path, "px": budget, "gray": bool}.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

from src.counterfactual import choose_alternative, edit_crop, edit_text
from src.data import PROJ, load_episode, sample, shot_path
from src.ocr import SIM_THRESH, best_window_sim, ocr_cached, ocr_image, reading_order, text_sim_in
from src.vlm import PX_CROP, PX_CURRENT, PX_HISTORY

CACHE = PROJ / "cache"
SPECS = PROJ / "results" / "specs"
CROP_OCR = PROJ / "data" / "ocr_crops"
C2_CAP = 20

VENUS_PROMPT_PRE = """**You are a GUI Agent**.
Your task is to analyze a given user task, review current screenshot and previous actions, and determine the next action to complete the task.
### Available Actions
You may execute one of the following functions:
- Click(box=(x1,y1))
- Drag(start=(x1,y1), end=(x2,y2))
- Scroll(start=(x1,y1), end=(x2,y2))
- Type(content='')
- Launch(app='')
- Wait()
- Finished(content='')
- CallUser(content='')
- LongPress(box=(x1,y1))
- PressBack()
- PressHome()
- PressEnter()
- PressRecent()
### User Task
{problem}
### Previous Actions
"""
VENUS_PROMPT_POST = """
### Output Format
<think> your thinking process </think>
<action> the next action </action>
<conclusion> the conclusion about the next action </conclusion>
### Instruction
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- For requests that are questions (or chat messages), remember to use the 'CallUser' action to reply to user explicitly before finishing! Then, after you have replied, use the Finished action if the goal is achieved.
- Consider exploring the screen by using the 'scroll' action with different directions to reveal additional content.
- To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the 'copy' button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a 'paste' button in it.
### Current Screenshot
"""
MEM_HEADER = "Memory of earlier steps:\n"


def lbl(step):
    return f"[steps {step}]" if isinstance(step, str) else f"[step {step}]"


def load_mem(eid, t):
    return json.load(open(CACHE / eid / f"{t}.json"))["entries"]


def crop_path(eid, rel):
    return str(CACHE / eid / rel)


def wrap(problem, block_parts, eid, t):
    parts = [{"text": VENUS_PROMPT_PRE.format(problem=problem)}]
    parts += block_parts if block_parts else [{"text": "None\n"}]
    parts += [{"text": VENUS_PROMPT_POST}, {"image": str(shot_path(eid, t)), "px": PX_CURRENT}]
    return parts


def mem_block(entries, eid, text=True, crops=True, gray=False, crop_override=None, labels=None, text_override=None):
    if not entries:
        return []
    parts = [{"text": MEM_HEADER}]
    for i, e in enumerate(entries):
        lab = lbl(labels[i] if labels else e["step"])
        summ = text_override[i] if text_override else e["summary"]
        parts.append({"text": f"{lab} {summ}\n" if text else f"{lab}\n"})
        cp = (crop_override or {}).get(i, e.get("crop") and crop_path(e["_eid"] if "_eid" in e else eid, e["crop"]))
        if crops and cp:
            parts.append({"image": cp, "px": PX_CROP, "gray": gray})
            parts.append({"text": "\n"})
    return parts


def crop_ocr(path):
    p = Path(path)
    out = CROP_OCR / (p.parent.parent.name + "_" + p.name + ".json")
    if out.exists():
        res = json.load(open(out))
        res["words"] = reading_order(res["words"])
        return res
    res = ocr_image(Image.open(p))
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"))
    return res


def dropped_crops(entries):
    """Crops that existed in this episode's past but are no longer attached (pruned or merged)."""
    out = []

    def walk(e):
        if e.get("dropped_crop"):
            out.append(e["dropped_crop"])
        for c in e.get("merged_from", []):
            walk(c)
    for e in entries:
        walk(e)
    return out


def derange(n, rng):
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def seeded(*key):
    return random.Random(int(hashlib.md5("|".join(map(str, key)).encode()).hexdigest()[:8], 16))


def presence(entries, eid, needed):
    """Is the needed string in memory text, and/or visible in a memory crop (crop OCR)?"""
    in_text = any(text_sim_in(needed, e["summary"]) >= SIM_THRESH for e in entries)
    in_crop = False
    for e in entries:
        if e.get("crop"):
            if best_window_sim(needed, crop_ocr(crop_path(eid, e["crop"]))["words"])[0] >= SIM_THRESH:
                in_crop = True
    return in_text, in_crop


def build(n, md_path, only=None):
    eids = sample(n)
    if only:
        eids = [e for e in eids if e in set(only)]
    SPECS.mkdir(parents=True, exist_ok=True)
    md = pd.read_parquet(md_path)
    md_keys = {(r.episode_id, int(r.step)): r for r in md.itertuples()}
    # counterfactual pool: (string, meta_task, category, episode) so alternatives fill the same semantic slot
    pool = sorted({(r.needed_string, load_episode(r.episode_id)["task_info"]["meta_task"],
                    load_episode(r.episode_id)["task_info"]["category"], r.episode_id) for r in md.itertuples()})
    # index memory states for C7 donors: category -> n_entries -> list of (eid, t, n_crops)
    states = defaultdict(lambda: defaultdict(list))
    all_sample = sample(n)
    for e in all_sample:
        if not (CACHE / e / "done").exists():
            continue
        cat = load_episode(e)["task_info"]["category"]
        for t in range(len(load_episode(e)["steps"])):
            ents = load_mem(e, t)
            states[cat][len(ents)].append((e, t, sum(1 for x in ents if x.get("crop"))))
    pres_rows, cf_rows = [], []
    for eid in eids:
        if not (CACHE / eid / "done").exists():
            print("skip (no controller cache)", eid)
            continue
        ep = load_episode(eid)
        problem = ep["task_info"]["instruction"]
        cat = ep["task_info"]["category"]
        spec = {}
        for t in range(len(ep["steps"])):
            ents = load_mem(eid, t)
            c = {}
            c["C0"] = wrap(problem, [], eid, t)
            c["C1"] = wrap(problem, mem_block(ents, eid), eid, t)
            hist = []
            if t > 0:
                hist = [{"text": MEM_HEADER}]
                for k in range(max(0, t - C2_CAP), t):
                    hist += [{"text": f"[step {k}]\n"}, {"image": str(shot_path(eid, k)), "px": PX_HISTORY}, {"text": "\n"}]
            c["C2"] = wrap(problem, hist, eid, t)
            c["C3"] = wrap(problem, mem_block(ents, eid, crops=False), eid, t)
            c["C4"] = wrap(problem, mem_block(ents, eid, text=False), eid, t)
            c["C5"] = wrap(problem, mem_block(ents, eid, gray=True), eid, t)
            # C6
            idx = [i for i, e in enumerate(ents) if e.get("crop")]
            rng = seeded("C6", eid, t)
            if len(idx) >= 2:
                p = derange(len(idx), rng)
                over = {idx[i]: crop_path(eid, ents[idx[p[i]]]["crop"]) for i in range(len(idx))}
                c["C6"] = wrap(problem, mem_block(ents, eid, crop_override=over), eid, t)
            elif len(idx) == 1 and dropped_crops(ents):
                donor = rng.choice(dropped_crops(ents))
                c["C6"] = wrap(problem, mem_block(ents, eid, crop_override={idx[0]: crop_path(eid, donor)}), eid, t)
            else:
                c["C6"] = None if idx else c["C1"]  # no crops at all -> identical to C1 (kept for AMS)
            # C7
            n_ent = len(ents)
            if n_ent == 0:
                c["C7"] = c["C0"]
            else:
                rng = seeded("C7", eid, t)
                n_cr = len(idx)
                cands = [x for x in states[cat].get(n_ent, []) if x[0] != eid]
                if not cands:  # nearest entry count
                    ks = sorted((k for k in states[cat] if any(x[0] != eid for x in states[cat][k])), key=lambda k: (abs(k - n_ent), k))
                    cands = [x for x in states[cat][ks[0]] if x[0] != eid] if ks else []
                best = [x for x in cands if x[2] == n_cr] or cands
                if best:
                    de, dt, _ = rng.choice(sorted(best))
                    dents = load_mem(de, dt)
                    for x in dents:
                        x["_eid"] = de
                    labels = [e["step"] for e in ents][:len(dents)]
                    labels += [f"{t - 1}" for _ in range(len(dents) - len(labels))]
                    labels = [int(l) if isinstance(l, str) and l.isdigit() else l for l in labels]
                    c["C7"] = wrap(problem, mem_block(dents, eid, labels=labels), eid, t)
                    c["C7_donor"] = [de, dt]
                else:
                    c["C7"] = None
            # C8 on MD steps
            r = md_keys.get((eid, t))
            if r is not None:
                needed = r.needed_string
                in_text, in_crop = presence(ents, eid, needed)
                pres_rows.append(dict(episode_id=eid, step=t, needed_string=needed, n_entries=n_ent, n_crops=len(idx),
                                      present_text=in_text, present_crop=in_crop))
                # the alternative must not occur on ANY screen seen so far (0..t) nor in the task text, so
                # outputting it can only come from the edited memory
                seen_ocr = [" ".join(w["text"] for w in ocr_cached(ep["steps"][k]["screenshot"])["words"]) for k in range(t + 1)]
                alt, how = choose_alternative(needed, [(p, m, c) for p, m, c, pe in pool if pe != eid and p != needed],
                                              seeded("C8", eid, t), forbid_texts=(problem + " " + ep["task_info"]["task"], *seen_ocr),
                                              meta_task=ep["task_info"]["meta_task"], category=cat)
                n_txt, n_crop = 0, 0
                if alt is not None:
                    new_txt = []
                    for e in ents:
                        s2, k = edit_text(e["summary"], needed, alt)
                        new_txt.append(s2); n_txt += k
                    over = {}
                    cfdir = CACHE / eid / "cf"
                    cfdir.mkdir(exist_ok=True)
                    for i in idx:
                        cp = crop_path(eid, ents[i]["crop"])
                        im2, found = edit_crop(Image.open(cp), needed, alt, ocr=crop_ocr(cp))
                        if found:
                            fn = cfdir / f"{t}_{i}.png"
                            im2.save(fn)
                            over[i] = str(fn); n_crop += 1
                    c["C8"] = wrap(problem, mem_block(ents, eid, text_override=new_txt, crop_override=over), eid, t)
                    c["C8t"] = wrap(problem, mem_block(ents, eid, text_override=new_txt), eid, t)
                    c["C8c"] = wrap(problem, mem_block(ents, eid, crop_override=over), eid, t)
                cf_rows.append(dict(episode_id=eid, step=t, needed_string=needed, alternative=alt, alt_method=how,
                                    n_text_edits=n_txt, n_crop_edits=n_crop))
            spec[str(t)] = c
        json.dump(spec, open(SPECS / f"{eid}.json", "w"))
    return pd.DataFrame(pres_rows), pd.DataFrame(cf_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--md", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    md_path = a.md or (PROJ / "results" / f"md_steps_n{a.n}.parquet")
    pres, cf = build(a.n, md_path)
    pres.to_parquet(PROJ / "results" / f"presence{a.tag}.parquet")
    cf.to_parquet(PROJ / "results" / f"counterfactuals{a.tag}.parquet")
    if len(pres):
        print("MD steps:", len(pres), "present_text:", pres.present_text.mean().round(3),
              "present_crop:", pres.present_crop.mean().round(3),
              "present_any:", (pres.present_text | pres.present_crop).mean().round(3))
    if len(cf):
        print("C8 alternatives:", cf.alt_method.value_counts().to_dict(),
              "text edits>0:", (cf.n_text_edits > 0).mean().round(3), "crop edits>0:", (cf.n_crop_edits > 0).mean().round(3))
