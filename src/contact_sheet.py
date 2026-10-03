"""Contact sheets for hand-checking MD labels: [source screen | current screen] + needed string.

The matched OCR window on the source screen is outlined in red; the current screen is shown so the
checker can verify the string is *not* visible there.
"""
import argparse
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.counterfactual import FONT
from src.data import PROJ, load_episode, shot_path
from src.ocr import best_window_sim, ocr_cached

H = 640


def panel(eid, step, needed, src):
    ep = load_episode(eid)
    out = []
    zoom = None
    for k, tag in ((src, "source"), (step, "current")):
        im = Image.open(shot_path(eid, k)).convert("RGB")
        if tag == "source":
            words = ocr_cached(ep["steps"][k]["screenshot"])["words"]
            sim, rng = best_window_sim(needed, words)
            if rng:
                ws = words[rng[0]:rng[1]]
                bx = (max(0, min(w["x"] for w in ws) - 60), max(0, min(w["y"] for w in ws) - 40),
                      min(im.width, max(w["x"] + w["w"] for w in ws) + 60), min(im.height, max(w["y"] + w["h"] for w in ws) + 40))
                zoom = im.crop(bx)
                d = ImageDraw.Draw(im)
                d.rectangle((min(w["x"] for w in ws) - 6, min(w["y"] for w in ws) - 6,
                             max(w["x"] + w["w"] for w in ws) + 6, max(w["y"] + w["h"] for w in ws) + 6),
                            outline=(255, 0, 0), width=8)
        s = H / im.height
        out.append(im.resize((int(im.width * s), H)))
    w = max(sum(i.width for i in out) + 30, 900)
    zh = 0
    if zoom is not None:
        zs = min(880 / zoom.width, 220 / zoom.height)
        zoom = zoom.resize((max(1, int(zoom.width * zs)), max(1, int(zoom.height * zs))))
        zh = zoom.height + 10
    canvas = Image.new("RGB", (w, H + 120 + zh), (255, 255, 255))
    if zoom is not None:
        canvas.paste(zoom, (10, 110))
    x = 10
    for i in out:
        canvas.paste(i, (x, 110 + zh)); x += i.width + 10
    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, 18)
    d.text((10, 5), f"{eid} step {step}  (source step {src} | current)", fill=(0, 0, 0), font=f)
    d.text((10, 30), "\n".join(textwrap.wrap("NEEDED: " + needed, 70)[:3]), fill=(200, 0, 0), font=f)
    return canvas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(PROJ / "results" / "md_steps_n600.parquet"))
    ap.add_argument("--k", type=int, default=30)
    a = ap.parse_args()
    md = pd.read_parquet(a.md).sample(n=a.k, random_state=0).reset_index(drop=True)
    md.to_csv(PROJ / "reports" / "label_check_sample.csv", index=False)
    figdir = PROJ / "reports" / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    panels = [panel(r.episode_id, int(r.step), r.needed_string, int(r.source_step)) for r in md.itertuples()]
    for s in range(0, len(panels), 3):
        group = panels[s:s + 3]
        cols = 3
        rows = (len(group) + cols - 1) // cols
        pw = max(p.width for p in group); ph = max(p.height for p in group)
        sheet = Image.new("RGB", (cols * pw, rows * ph), (230, 230, 230))
        for i, p in enumerate(group):
            sheet.paste(p, ((i % cols) * pw, (i // cols) * ph))
        sheet.save(figdir / f"md_label_check_{s // 3 + 1:02d}.png")
    print("saved", len(panels))
