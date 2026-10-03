"""Select and render annotated example cases for the final report.

Picks MD steps (needed string present in clean memory) across outcome types:
  - C1 wrong although the string is in memory          - C8 follows the counterfactual
  - C8 ignores the counterfactual (types the original)  - C7 (other episode) still right
  - C0 right without memory (string inferable)          - C5 (blank crops) breaks a crop-only case
Each panel: current screenshot, memory crops, memory text, and the action under each condition.
"""
import argparse
import json
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.counterfactual import FONT
from src.data import PROJ, load_episode, shot_path

CACHE = PROJ / "cache"
FIG = PROJ / "reports" / "figs"


def render(e, t, rows, needed, alt, title):
    ep = load_episode(e)
    mem = json.load(open(CACHE / e / f"{t}.json"))["entries"]
    cur = Image.open(shot_path(e, t)).convert("RGB")
    sc = min(700 / cur.height, 760 / cur.width)
    cur = cur.resize((int(cur.width * sc), int(cur.height * sc)))
    crops = [Image.open(CACHE / e / x["crop"]).convert("RGB") for x in mem if x.get("crop")]
    W = 1700
    canvas = Image.new("RGB", (W, 1000), "white")
    canvas.paste(cur, (10, 280))
    d = ImageDraw.Draw(canvas)
    f, fb, fs = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 15)
    d.text((10, 8), title, fill=(0, 0, 0), font=fb)
    d.text((10, 36), "\n".join(textwrap.wrap("TASK: " + ep["task_info"]["instruction"], 150)[:2]), fill=(60, 60, 60), font=fs)
    d.text((10, 80), f"GT: {rows['gt_cmd']}   needed: {needed!r}   C8 alternative: {alt!r}", fill=(170, 0, 0), font=f)
    y = 115
    for c in ["C0", "C1", "C2", "C3", "C5", "C7", "C8", "C8t", "C8c"]:
        if c in rows["preds"]:
            a, ok = rows["preds"][c]
            d.text((10 + (["C0", "C1", "C2", "C3", "C5", "C7", "C8", "C8t", "C8c"].index(c) % 3) * 560,
                    y + (["C0", "C1", "C2", "C3", "C5", "C7", "C8", "C8t", "C8c"].index(c) // 3) * 24),
                   f"{c}: {a[:48]} {'[correct]' if ok else '[wrong]'}", fill=(0, 110, 0) if ok else (0, 0, 0), font=f)
    x0 = cur.width + 30
    d.text((x0, 270), "Memory (C1):", fill=(0, 0, 0), font=fb)
    yy = 298
    for m in mem:
        lab = f"[steps {m['step']}]" if isinstance(m["step"], str) else f"[step {m['step']}]"
        for line in textwrap.wrap(f"{lab} {m['summary']}", max(40, int((W - x0 - 20) / 8.2)))[:4]:
            d.text((x0, yy), line, fill=(30, 30, 30), font=fs); yy += 19
        yy += 4
    cx, cy = x0, max(yy + 10, 600)
    for c in crops:
        s = min(1.0, 170 / c.height, 380 / c.width)
        c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))))
        if cx + c.width > W - 10:
            cx, cy = x0, cy + 180
        canvas.paste(c, (cx, cy)); d.rectangle((cx, cy, cx + c.width, cy + c.height), outline=(42, 120, 214), width=2)
        cx += c.width + 12
    return canvas


def main(prefix, k):
    df = pd.read_parquet(f"{prefix}_per_step.parquet")
    md = df[df.is_md & df.present_any]
    piv = md.pivot_table(index=["episode_id", "step"], columns="cond", values="action", aggfunc="first")
    cor = md.pivot_table(index=["episode_id", "step"], columns="cond", values="correct", aggfunc="first")
    sim_alt = md[md.cond == "C8"].set_index(["episode_id", "step"]).text_sim_alt
    sim_need = md.pivot_table(index=["episode_id", "step"], columns="cond", values="text_sim_needed", aggfunc="first")
    meta = md[md.cond == "C1"].set_index(["episode_id", "step"])
    buckets = {
        "C8 follows the counterfactual": [i for i in sim_alt.index if sim_alt[i] >= 0.8],
        "C8 ignores the edit (types original)": [i for i in sim_alt.index if sim_alt[i] < 0.8 and sim_need.loc[i].get("C8", 0) >= 0.8],
        "C1 wrong although string is in memory": [i for i in sim_need.index if sim_need.loc[i].get("C1", 1) < 0.8],
        "C7 (other episode) still right": [i for i in sim_need.index if sim_need.loc[i].get("C7", 0) >= 0.8],
        "C0 right without memory": [i for i in sim_need.index if sim_need.loc[i].get("C0", 0) >= 0.8 and not meta.loc[i].md_strict],
        "C5 (blank crops) breaks crop-only case": [i for i in sim_need.index if meta.loc[i].crop_only and sim_need.loc[i].get("C1", 0) >= 0.8 and sim_need.loc[i].get("C5", 1) < 0.8],
    }
    per = {"C8 follows the counterfactual": 2, "C8 ignores the edit (types original)": 2, "C1 wrong although string is in memory": 2,
           "C7 (other episode) still right": 2, "C0 right without memory": 1, "C5 (blank crops) breaks crop-only case": 1}
    chosen, used = [], set()
    for b, items in buckets.items():
        for i in sorted(items)[:per[b]]:
            if i not in used:
                chosen.append((b, i)); used.add(i)
    FIG.mkdir(parents=True, exist_ok=True)
    out = []
    for n, (b, (e, t)) in enumerate(chosen[:k], 1):
        r = meta.loc[(e, t)]
        rows = {"gt_cmd": r.gt_cmd, "preds": {c: (piv.loc[(e, t)][c], bool(cor.loc[(e, t)][c]) if c in cor.columns else False)
                                              for c in piv.columns if isinstance(piv.loc[(e, t)][c], str)}}
        img = render(e, int(t), rows, r.needed_string, r.alternative, f"Case {n}: {b}  —  episode {e}, step {t}")
        fn = FIG / f"case_{n:02d}.png"
        img.save(fn)
        out.append(dict(case=n, bucket=b, episode_id=e, step=int(t), needed=r.needed_string, alternative=r.alternative,
                        md_strict=bool(r.md_strict), crop_only=bool(r.crop_only), **{c: a for c, (a, _) in rows["preds"].items()}))
    pd.DataFrame(out).to_csv(PROJ / "reports" / "failure_cases.csv", index=False)
    print(pd.DataFrame(out)[["case", "bucket", "episode_id", "step", "needed", "alternative"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=str(PROJ / "results" / "n600"))
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    main(a.prefix, a.k)
