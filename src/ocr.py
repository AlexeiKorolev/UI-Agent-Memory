"""Tesseract OCR with per-image JSON cache + fuzzy string search utilities.

OCR config: tesseract 4.1.1 (system /usr/bin/tesseract), lang=eng, --psm 11 (sparse text; best for UIs).
Dark screens (mean luminance < 110) are additionally OCR'd inverted; word lists are unioned.
"""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein

from src.data import PROJ, SHOTS

OCR_DIR = PROJ / "data" / "ocr"
SIM_THRESH = 0.8


def ocr_image(im, min_conf=30):
    import pytesseract
    im = im.convert("RGB")
    passes = [im]
    lum = np.asarray(im.convert("L"), dtype=np.float32).mean()
    if lum < 110:
        passes.append(ImageOps.invert(im))
    words = []
    for p in passes:
        d = pytesseract.image_to_data(p, output_type=pytesseract.Output.DICT, config="--psm 11")
        for i, t in enumerate(d["text"]):
            t = t.strip()
            if not t or float(d["conf"][i]) < min_conf:
                continue
            words.append({"text": t, "conf": float(d["conf"][i]), "x": d["left"][i], "y": d["top"][i],
                          "w": d["width"][i], "h": d["height"][i]})
    return {"size": im.size, "lum": float(lum), "words": reading_order(words)}


def reading_order(words):
    """Cluster words into lines by vertical-centre proximity, then order lines top-down, words left-right."""
    lines = []
    for w in sorted(words, key=lambda w: w["y"] + w["h"] / 2):
        yc = w["y"] + w["h"] / 2
        if lines and abs(yc - lines[-1]["yc"]) <= 0.5 * max(w["h"], lines[-1]["h"]):
            L = lines[-1]
            L["ws"].append(w)
            L["yc"] = sum(x["y"] + x["h"] / 2 for x in L["ws"]) / len(L["ws"])
            L["h"] = max(L["h"], w["h"])
        else:
            lines.append({"yc": yc, "h": w["h"], "ws": [w]})
    out = []
    for L in lines:
        out += sorted(L["ws"], key=lambda w: w["x"])
    return out


def ocr_cached(name):
    out = OCR_DIR / (name + ".json")
    if out.exists():
        res = json.load(open(out))
        res["words"] = reading_order(res["words"])  # caches written before the ordering fix
        return res
    res = ocr_image(Image.open(SHOTS / name))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    json.dump(res, open(tmp, "w"))
    tmp.rename(out)
    return res


def norm(s):
    s = s.lower()
    s = re.sub(r"[^0-9a-zÀ-￿]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def best_window_sim(needle, words):
    """Max normalized Levenshtein similarity between `needle` and any run of consecutive OCR words
    (normalized text). Returns (sim, (i, j)) with word index range [i, j)."""
    nd = norm(needle)
    if not nd:
        return 0.0, None
    toks = [norm(w["text"]) if isinstance(w, dict) else norm(w) for w in words]
    L = len(nd)
    best, arg = 0.0, None
    n = len(toks)
    for i in range(n):
        if not toks[i]:
            continue
        s = ""
        for j in range(i, n):
            if not toks[j]:
                continue
            s = (s + " " + toks[j]).strip()
            if len(s) >= 0.6 * L:
                sim = Levenshtein.normalized_similarity(nd, s)
                if sim > best:
                    best, arg = sim, (i, j + 1)
            if len(s) > 1.4 * L + 2:
                break
    return best, arg


def text_sim_in(needle, text):
    """Fuzzy containment of needle in a free text (memory summaries, typed text, descriptions)."""
    return best_window_sim(needle, text.split())[0]


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    names = [l.strip() for l in open(a.files) if l.strip()]
    todo = [n for n in names if not (OCR_DIR / (n + ".json")).exists() and (SHOTS / n).exists()]
    print(f"{len(names)} names, {len(todo)} to OCR", flush=True)
    with Pool(a.procs) as p:
        for k, _ in enumerate(p.imap_unordered(ocr_cached, todo, chunksize=4)):
            if k % 500 == 0:
                print("ocr", k, flush=True)
    print("done")
