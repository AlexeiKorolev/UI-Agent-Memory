"""Targeted counterfactual editing of memory (condition C8).

- choose_alternative(): a same-type replacement for the needed string
    * digit-bearing strings: every digit replaced by a different digit (same digit count/format,
      leading digit kept non-zero), letters/punctuation unchanged
    * other strings: another MD needed string from a *different* episode with the same word count
      (closest length) and same type class (url/email/alpha), dissimilar to the original
- edit_text(): replace fuzzy occurrences (sim>=0.8 windows) of the needed string in a summary
- edit_crop(): cover the string's OCR box with the local background colour and render the alternative
  in a similar font size (Droid Sans, the Android UI font family)
"""
import random
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.distance import Levenshtein

from src.ocr import SIM_THRESH, best_window_sim, norm, ocr_image

FONT = "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf"


def type_class(s):
    if re.search(r"https?://|www\.|\.com\b", s):
        return "url"
    if "@" in s:
        return "email"
    if re.search(r"\d", s):
        return "digits"
    return "alpha"


def perturb_digits(s, rng):
    out = []
    first = True
    for ch in s:
        if ch.isdigit():
            choices = [d for d in "0123456789" if d != ch and not (first and d == "0" and ch != "0")]
            out.append(rng.choice(choices))
            first = False
        else:
            out.append(ch)
    return "".join(out)


def choose_alternative(needed, pool, rng, forbid_texts=()):
    """pool: list of (episode_id, needed_string) from other episodes. forbid_texts: strings in which the
    alternative must not already occur (instruction, current-screen OCR text) so following is unambiguous."""
    cls = type_class(needed)

    def ok(alt):
        if norm(alt) == norm(needed):
            return False
        if cls != "digits" and Levenshtein.normalized_similarity(norm(alt), norm(needed)) >= 0.5:
            return False
        for f in forbid_texts:
            if best_window_sim(alt, f.split())[0] >= SIM_THRESH:
                return False
        return True

    if cls == "digits":
        for _ in range(50):
            alt = perturb_digits(needed, rng)
            if alt != needed and ok(alt):
                return alt, "digit_perturb"
    nw = len(needed.split())
    cands = [p for p in pool if type_class(p) == cls and len(p.split()) == nw and norm(p) != norm(needed)]
    if not cands:
        cands = [p for p in pool if type_class(p) == cls and norm(p) != norm(needed)]
    cands.sort(key=lambda p: (abs(len(p) - len(needed)), p))
    cands = cands[:15]
    rng.shuffle(cands)
    for alt in cands:
        if ok(alt):
            return alt, "pool_swap"
    # last resort: letter-level scramble preserving case/shape
    letters = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(50):
        alt = "".join((rng.choice(letters).upper() if c.isupper() else rng.choice(letters)) if c.isalpha() else
                      (rng.choice("0123456789") if c.isdigit() else c) for c in needed)
        if ok(alt):
            return alt, "scramble"
    return None, "none"


def edit_text(summary, needed, alt):
    """Replace every fuzzy occurrence of `needed` in `summary` with `alt`. Returns (new, n_replaced)."""
    n_rep = 0
    # exact (case-insensitive) occurrences first
    pat = re.compile(re.escape(needed.strip()), re.I)
    new, k = pat.subn(alt, summary)
    n_rep += k
    # fuzzy word windows
    for _ in range(5):
        words = new.split(" ")
        sim, rng_ = best_window_sim(needed, words)
        if sim < SIM_THRESH or rng_ is None:
            break
        i, j = rng_
        if norm(" ".join(words[i:j])) == norm(alt):
            break
        # keep surrounding punctuation of the window
        lead = re.match(r"^\W*", words[i]).group(0)
        trail = re.search(r"\W*$", words[j - 1]).group(0)
        words[i:j] = [lead + alt + trail]
        new = " ".join(words)
        n_rep += 1
    return new, n_rep


def _bg_color(im, box):
    x1, y1, x2, y2 = box
    a = np.asarray(im.convert("RGB"))
    H, W = a.shape[:2]
    pad = 3
    ring = np.concatenate([
        a[max(0, y1 - pad):y1, max(0, x1):min(W, x2)].reshape(-1, 3),
        a[y2:min(H, y2 + pad), max(0, x1):min(W, x2)].reshape(-1, 3),
        a[max(0, y1):min(H, y2), max(0, x1 - pad):x1].reshape(-1, 3),
        a[max(0, y1):min(H, y2), x2:min(W, x2 + pad)].reshape(-1, 3)])
    if len(ring) == 0:
        ring = a.reshape(-1, 3)
    return tuple(int(v) for v in np.median(ring, axis=0))


def _text_color(im, box, bg):
    a = np.asarray(im.convert("RGB").crop(box)).reshape(-1, 3).astype(int)
    d = np.abs(a - np.array(bg)).sum(1)
    if len(d) == 0 or d.max() < 60:
        return (0, 0, 0) if sum(bg) > 380 else (255, 255, 255)
    return tuple(int(v) for v in a[d >= np.percentile(d, 90)].mean(0))


def edit_crop(crop, needed, alt, ocr=None):
    """Find `needed` in the crop via OCR, paint over it and render `alt`. Returns (new_crop, found: bool)."""
    ocr = ocr or ocr_image(crop)
    words = ocr["words"]
    sim, rng_ = best_window_sim(needed, words)
    if sim < SIM_THRESH or rng_ is None:
        return crop, False
    ws = [w for w in words[rng_[0]:rng_[1]] if norm(w["text"])]
    im = crop.convert("RGB").copy()
    # group window words into lines
    lines = []
    for w in sorted(ws, key=lambda w: (w["y"], w["x"])):
        if lines and abs(w["y"] - lines[-1][0]["y"]) < max(8, 0.6 * w["h"]):
            lines[-1].append(w)
        else:
            lines.append([w])
    boxes = [(min(w["x"] for w in L) - 2, min(w["y"] for w in L) - 2,
              max(w["x"] + w["w"] for w in L) + 2, max(w["y"] + w["h"] for w in L) + 2) for L in lines]
    # split alt text across the same number of lines (by words, proportional)
    alt_words = alt.split()
    per = max(1, int(np.ceil(len(alt_words) / len(boxes))))
    chunks = [" ".join(alt_words[i * per:(i + 1) * per]) for i in range(len(boxes))]
    if len(boxes) == 1:
        chunks = [alt]
    d = ImageDraw.Draw(im)
    for box, chunk in zip(boxes, chunks):
        box = tuple(int(v) for v in box)
        bg = _bg_color(im, box)
        fg = _text_color(crop, box, bg)
        d.rectangle(box, fill=bg)
        if not chunk:
            continue
        h = box[3] - box[1]
        size = max(8, int(h * 0.85))
        font = ImageFont.truetype(FONT, size)
        # shrink until it fits horizontally (allow 30% overflow into background)
        maxw = (box[2] - box[0]) * 1.3 + 10
        while size > 8 and d.textlength(chunk, font=font) > maxw:
            size -= 1
            font = ImageFont.truetype(FONT, size)
        d.text((box[0] + 1, box[1] + (h - size) / 2 - 1), chunk, fill=fg, font=font)
    return im, True
