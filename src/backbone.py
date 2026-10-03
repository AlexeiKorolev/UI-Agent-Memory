"""Run the frozen backbone (UI-Venus-1.5-8B, vLLM, greedy) on all condition specs.

Identical prompts (e.g. C1 at step 0 == C0) are generated once and shared. Output per episode:
results/raw/{episode}.jsonl with episode_id, step, cond, raw, action (official GUI-Odyssey string),
n_images, prompt_hash. Resumable: an episode whose output file exists is skipped.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from src.actions import parse_venus, venus_to_odyssey
from src.data import PROJ, sample
from src.vlm import greedy, make_llm, resize_to_budget, to_messages

SPECS = PROJ / "results" / "specs"
RAW = PROJ / "results" / "raw"
_img_cache = {}


def load_img(p):
    key = (p["image"], p.get("px"), p.get("gray", False))
    if key not in _img_cache:
        im = resize_to_budget(Image.open(p["image"]), p["px"])
        if p.get("gray"):
            im = Image.new("RGB", im.size, (128, 128, 128))
        _img_cache[key] = im
    return _img_cache[key]


def spec_hash(parts):
    return hashlib.md5(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def to_parts(spec):
    return [p["text"] if "text" in p else load_img(p) for p in spec]


def run(eids, llm, conds, max_tokens, batch_eps):
    sp = greedy(max_tokens)
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eids if not (RAW / f"{e}.jsonl").exists() and (SPECS / f"{e}.json").exists()]
    print(f"{len(todo)} episodes to run", flush=True)
    for b in range(0, len(todo), batch_eps):
        chunk = todo[b:b + batch_eps]
        t0 = time.time()
        jobs, uniq = [], {}
        for e in chunk:
            spec = json.load(open(SPECS / f"{e}.json"))
            for t, cs in spec.items():
                for c in conds:
                    if cs.get(c) is None:
                        continue
                    h = spec_hash(cs[c])
                    jobs.append((e, int(t), c, h, sum("image" in p for p in cs[c])))
                    uniq.setdefault(h, cs[c])
        hs = list(uniq)
        outs = llm.chat([to_messages(to_parts(uniq[h])) for h in hs], sp, use_tqdm=False)
        res = {h: o.outputs[0].text for h, o in zip(hs, outs)}
        ntok = sum(len(o.prompt_token_ids) for o in outs)
        by_ep = {}
        for e, t, c, h, nimg in jobs:
            raw = res[h]
            by_ep.setdefault(e, []).append(dict(episode_id=e, step=t, cond=c, raw=raw,
                                                action=venus_to_odyssey(parse_venus(raw)), n_images=nimg, prompt_hash=h))
        for e, rows in by_ep.items():
            tmp = RAW / f"{e}.jsonl.tmp"
            with open(tmp, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            tmp.rename(RAW / f"{e}.jsonl")
        _img_cache.clear()
        dt = time.time() - t0
        print(f"batch {b // batch_eps}: {len(chunk)} eps, {len(jobs)} jobs, {len(hs)} unique prompts, "
              f"{ntok} prompt tokens, {dt:.0f}s ({len(hs) / dt:.2f} prompts/s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--conds", default="C0,C1,C2,C3,C4,C5,C6,C7,C8,C8t,C8c")
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--batch_eps", type=int, default=10)
    a = ap.parse_args()
    eids = sample(a.n)[a.shard::a.nshards]
    llm = make_llm("backbone", max_model_len=24576, max_images=32)
    run(eids, llm, a.conds.split(","), a.max_tokens, a.batch_eps)
    print("DONE")
