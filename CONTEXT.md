# CONTEXT — handoff document

Everything needed to continue this project on another machine or with a new assistant session. The original task
specification is in [`docs/task_spec.md`](docs/task_spec.md) (verbatim). The dated running log is
[`LOG.md`](LOG.md). This file summarises **what was decided, why, what is verified, and what remains**.

_Last updated: 2026-10-03, ~01:15 EDT. Written on Adroit (Princeton RC) before moving to Neuronic._

---

## 1. Research question (one paragraph)

MementoGUI (Zeng et al., arXiv 2605.18652) adds a plug-in memory (text summaries + cropped screen regions, "ROIs") to a
**frozen** GUI agent and reports UI-Venus-1.5-8B on GUI-Odyssey going from 54.58 AMS (no history) → 68.32 (memory).
**We ask whether the backbone uses the *content* of the memory, or whether gains come from presence/format effects**
(extra images, memory-shaped context). We answer with **interventions on memory content holding everything else fixed**.
MementoGUI released no code/weights, so we re-implement a **MementoGUI-style working memory with a prompted controller**
(no episodic memory). This is *not* an exact reproduction; the report must say so.

## 2. Design

* **Backbone (frozen, never modified):** `inclusionAI/UI-Venus-1.5-8B` @ `a06ff6c6f15a9eca210769dacc1603f73b4a500c`.
* **Controller (prompted, NOT trained):** `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
  Replays each episode along the **ground-truth** trajectory. At step t it sees: task instruction, the previous GT action
  (as a UI-Venus action string), current memory (text), current screenshot. Returns JSON
  `{"salience":0-1,"write":bool,"summary":"<=30 words, concrete values","roi_bbox":[x1,y1,x2,y2] (0-1000) | null}`.
  If write → append {step, summary, crop of roi_bbox}. >8 entries → controller merges the two oldest into one text entry
  (crops of merged entries dropped). ≤4 crops kept (most salient, ties → most recent). JSON retried ≤2×; failures logged.
  **Cache = memory state *before* step t** (`cache/{episode}/{t}.json` + `crops/`), so the current screen never leaks.
* **Memory-dependent (MD) steps:** GT action is TYPE (v2 calls it `TEXT`) and the typed string (a) is not on the current
  screen (OCR, or the dataset's per-step `description`) and (b) is on an earlier screen (OCR) or in earlier typed text.
  Fuzzy match = normalized Levenshtein ≥ 0.8 on the best window of consecutive OCR words.
  **Added flag `in_instruction`** (deviation): if the string is in the task instruction the agent can copy it without
  memory; `md_strict` = MD and not in instruction. Report both.
* **Conditions** — identical official prompt + current screenshot; only the "### Previous Actions" slot changes:

| ID | slot content | tests |
|---|---|---|
| C0 | `None` | lower baseline |
| C1 | `Memory of earlier steps:` + `[step k] summary` + crop image per entry | main |
| C2 | all past GT screenshots (most recent 20, 0.35 MP each), `[step k]` + image | paper's keep-everything baseline |
| C3 | C1 without crops | crop contribution |
| C4 | C1 with summaries replaced by `[step k]` | text contribution |
| C5 | crops → uniform gray (128) image of same size | pixels vs image presence |
| C6 | crops deranged among entries (donor = earlier dropped crops if only 1 crop; n/a if none) | text–image alignment |
| C7 | memory of another sampled episode, same category, matched #entries (then #crops), relabelled to recipient's step labels | format/presence control |
| C8 | MD steps: needed string → same-type alternative in text **and** crops | causal following |
| C8t / C8c | text-only / crop-only edit | which channel drives following |

  C8 alternative: digit strings → every digit changed (format kept); otherwise another MD string from a different episode
  with same word count/type class (url/email/alpha), Levenshtein sim < 0.5, and **absent from the instruction and from every
  screen 0..t** (so typing it can only come from edited memory). Crop edit: OCR box of the string painted with local
  background colour, alternative rendered in Droid Sans at matched height (`src/counterfactual.py`).
* **Metrics:** official AMS (macro = step accuracy; micro = mean over 6 categories, which is what their code uses for
  random_split) for all and MD steps; MD text-argument accuracy (sim ≥ 0.8 to needed string); change rate vs C1 (same type
  + same argument; clicks within 0.14); C8 follow rate (cf vs original vs neither), with C1 as the "spontaneous cf" baseline.
  95% CIs by bootstrapping **episodes** 1,000×; exact McNemar C1 vs each condition (paired). Subsets: all, MD, MD_strict,
  MD_present (needed string present in clean memory text or visible in a memory crop), MD_crop_only, MD_absent.
* **Pre-registered:** H1 C1 > C0 on MD. H2 C7 ≈ C0 ≪ C1 on MD. H3 C5 < C1 on MD steps whose string is only in a crop.
  H4 C8 follow rate ≫ rate of cf string under C1 (~0). Calibration only: C0/C2 vs paper's 54.58/66.31 — don't tune.

## 3. Verified facts & decisions (with sources)

| Topic | Finding / decision | Source |
|---|---|---|
| Dataset version | `OpenGVLab/GUI-Odyssey` README says use **v2 `hflqf88888/GUIOdyssey`** (8,334 eps, adds `sam2_bbox`, low-level instr.). We use v2 @ `61632d0f3f4d51d7e9561ce4f84347dd06b2019d`. | HF dataset cards |
| Split | `random_split` test: 1,666 episodes, 25,807 steps (CLICK 18,764; TEXT 2,666; SCROLL 2,622; COMPLETE 1,572; INCOMPLETE 94; LONG_PRESS 89). MementoGUI's split unknown. | `data/guiodyssey_v2/splits` |
| Leakage | v2 annotations have `context`/`intention` fields summarising history → **never shown to any model**. `description` used only for MD labelling. | — |
| Screenshots | v2 ships only a 92.6 GB split zip (`screenshots.z01–z08` + `.zip`). We extract members by HTTP range reads of the ZIP64 central directory (`src/remote_zip.py`). HF "resolvers" rate limit = 3000 req / 5 min **per IP** → cache CDN redirect, single range read per member. | — |
| UI-Venus-1.5 prompt | Not in model card; repo history rewritten at UI-Venus-2 (1.5 code gone; only a Chinese framework prompt remains). **Used the English "Mobile Prompt" verbatim from the UI-Venus-1.5 tech report, arXiv 2602.09082v2, App. A.3** (`src/conditions.py`). We add a `### Current Screenshot` header in every condition. | tech report |
| Coordinates | UI-Venus: normalized [0,1000] (report §2; framework `normalize_to_screen`). GUI-Odyssey: [0,1000]. No rescaling. | report + `third_party/UI-Venus/.../ui_venus_policy.py` |
| Scroll semantics | Both finger trajectory (start→end). Official direction rule = sign of larger delta component. **To verify empirically in pilot** (if C0 scroll accuracy ≈ 0, convention is inverted). | `format_converter.decode_action` |
| AMS | Official `GUIOdyssey_action_matching.py` imported **unmodified**: type must match; CLICK/LONG_PRESS correct if inside `sam2_bbox` or L2 ≤ 0.14; TYPE ANLS ≥ 0.5 or substring; SCROLL direction string. Exceptions → incorrect (mirrors official). Official `simple_decode` splits on ':' ⇒ TYPE text with ':' (URLs) truncated on both sides — kept; our MD text metric uses split(':',1). | `third_party/GUI-Odyssey/src/eval_mm/` @ `5cdf76d9` |
| Action mapping | Click/LongPress/Type/PressBack/PressHome/PressRecent/Finished map 1:1. Drag→SCROLL. CallUser→COMPLETE, or IMPOSSIBLE if content says infeasible (GT INCOMPLETE ↔ `CallUser('...impossible...')`). PressEnter/Wait/Launch have no GT counterpart → always wrong (report their frequency). 20-step GT round-trip test passes. | `src/actions.py`, `tests/test_actions.py` |
| Image budgets | We resize ourselves (multiples of 32): current screenshot ≤ 2.5 MP (backbone), ≤ 1.2 MP (controller), crops ≤ 0.3 MP, C2 history ≤ 0.35 MP each. Processor `longest_edge`=16.7 MP so no further resize. | `src/vlm.py` |
| Decoding | vLLM 0.30, bf16, greedy (T=0, seed 0), prefix caching; backbone max_tokens 768, controller 256. Identical prompts deduplicated (e.g. C1 at step 0 = C0). | `src/backbone.py` |
| Sampling | `stratified_order(seed=0)`: per-category shuffled lists merged by fractional rank ⇒ any prefix (10/300/600) is category-proportional (±1) and nested. Pilot = first 10. | `src/data.py` |
| Episode count | n=300 gave **134 MD** (<150) ⇒ **extended to 600** as pre-specified: **275 MD** (164 strict) in 245 episodes; 995 TYPE steps. | `results/md_steps_n600.parquet` |
| Label precision | Hand-checked 30 random MD labels from images: **27/30 = 0.90** (Wilson 95% CI 0.74–0.97). Errors: OCR missed white-on-colour text on current screen; stale browser history matched; loose fuzzy paraphrase. | `reports/label_check.md`, `reports/figs/md_label_check_*.png` |
| OCR | System tesseract 4.1.1, eng, `--psm 11`, + inverted pass if mean luminance < 110; conf ≥ 30. Reading order via line clustering (a bucket-sort bug was found by unit test and fixed before the OCR job). Cache shipped as `artifacts/ocr_cache_600.tar.gz`. | `src/ocr.py` |

## 4. Status (2026-10-03 01:15)

**Done:** env, model download, annotations, 9,388 screenshots (600 eps), OCR, MD labels (n=300 and n=600), label
hand-check, all pipeline code, 10/10 unit tests, CPU smoke test (all conditions build; max prompt ≈ 5.1k tokens on a test
episode), provenance, sbatch scripts.

**Not done (nothing has run on a GPU yet):**
1. **Pilot** (10 episodes): `slurm/pilot.sbatch` = controller → conditions → backbone. On Adroit it was job 3389614,
   pending ≥ 5 h behind ~17 jobs (4×A100 busy; both A40s used by the user's other jobs). Then write `reports/pilot.md`
   (setup checks, timing, sample prompts/outputs, surprises). **Check in pilot:** controller JSON failure rate; backbone
   parse failures / INVALID rate; C0 scroll accuracy (direction convention); C0 AMS roughly plausible vs 54.58;
   that memory crops are sensible; throughput (prompts/s) to plan the full run.
2. **Full run (n=600):** controller array (4 shards) → `build_conditions` (CPU) → backbone array (4 shards) → analyze →
   figures. ~9.4k steps × 8 conditions + 3×275 C8 prompts (~65–75k unique prompts after dedup).
3. **Reports:** `reports/final.md` (setup, deviations, label precision, presence rate, main table, H1–H4 verdicts,
   10 annotated failure cases with images, limitations, GPU-hours) and README polish.
4. Optional §7 activation patching (HF model, 50 MD steps) — only after a clean main result.

## 5. Running on a new cluster (e.g. Neuronic) — checklist

```bash
git clone https://github.com/AlexeiKorolev/UI-Agent-Memory.git mementogui_causal   # put it on large scratch storage
cd mementogui_causal
# edit if needed: MG_ANACONDA_MODULE (module name for conda), then:
bash scripts/setup_env.sh          # LOGIN node: conda env in ./env with vllm 0.30 (torch cu130 → needs driver ≥ 580 / CUDA 13)
bash scripts/download.sh           # LOGIN node: models (~35 GB), annotations, screenshots (~6.7 GB), unpacks OCR cache
source env.sh && python -m pytest -q tests      # expect 10 passed
```

Adapt the `#SBATCH` lines in `slurm/*.sbatch` to the cluster: `--partition`, `--gres`/`--constraint`
(needs ≥ 40 GB GPU for an 8B bf16 VLM; on Adroit we used `--constraint="gpu80|a40"`), time limits (Adroit gpu-short QOS
= ≤4 h, 4 GPUs/user — all jobs are resumable so short limits are fine), and add
`--mail-user=<you>` on the command line (removed from tracked files because this repo is public). **Submit from the
repo root** (log paths and `source "$SLURM_SUBMIT_DIR/env.sh"` are relative to it):

```bash
mkdir -p logs
sbatch --mail-user=<you> slurm/pilot.sbatch                           # pilot (10 eps)
sbatch --mail-user=<you> slurm/controller.sbatch 600                  # array 0-3
sbatch --mail-user=<you> --dependency=afterok:<ctrl> slurm/build_conditions.sbatch 600
sbatch --mail-user=<you> --dependency=afterok:<cond> slurm/backbone.sbatch 600
sbatch --mail-user=<you> --dependency=afterok:<bb>   slurm/analyze.sbatch 600
python -m src.figures --prefix results/n600 && python -m src.provenance
```

Other notes: compute nodes may have no internet → download on the login node; jobs set `HF_HUB_OFFLINE=1`. If the
driver is older than CUDA 13, pin an older vLLM/torch pair (UI-Venus card requires vllm ≥ 0.11, transformers ≥ 4.57).
If you get HF 429s, they are per-IP: wait out the 5-minute window (the extractor backs off automatically).
OCR needs `tesseract` (4.x, eng) on PATH only if you re-run OCR; the shipped cache covers the 600 sampled episodes.

## 6. File map

```
CONTEXT.md            this file          LOG.md     dated log          docs/task_spec.md   original task
env.sh                environment (paths/caches relative to repo)
scripts/              setup_env.sh, download.sh
slurm/                ocr, pilot, controller (array), build_conditions, backbone (array), analyze
src/data.py           loading, official GT decode, stratified sampling
src/remote_zip.py     extract screenshots from the HF split zip via range reads
src/actions.py        UI-Venus ⇄ GUI-Odyssey converters + parser
src/ocr.py            tesseract OCR cache + fuzzy window matching
src/md_labels.py      MD labels → results/md_steps_n{N}.parquet, type_steps, nonmd_sample
src/contact_sheet.py  hand-check sheets
src/controller.py     prompted Qwen3-VL working-memory builder (cache/)
src/counterfactual.py C8 alternative selection, text + crop editing
src/conditions.py     C0–C8 prompt specs (results/specs/), presence + counterfactual tables
src/vlm.py            vLLM helpers, model revisions, pixel budgets
src/backbone.py       UI-Venus runner (results/raw/)
src/analyze.py        official AMS, metrics, bootstrap, McNemar → summary_table.csv, follow_table.csv
src/figures.py        bar chart + follow-rate chart
src/provenance.py     results/provenance.json
tests/                converter + counterfactual tests
artifacts/            OCR cache (600 eps), zip central-directory index
results/              md_steps_*.parquet, type_steps_*.parquet, nonmd_sample_*.parquet, provenance.json
reports/              label_check.md, figs/ (contact sheets)
```

## 7. Ground rules carried over

No fine-tuning/modification of the backbone; controller prompted only; no API models; no non-public data; never invent
numbers; investigate numbers that look too good/bad (parsing, coordinate convention, leakage); if compute is short cut
episodes, not conditions C0/C1/C5/C7/C8. Keep `LOG.md` updated and commit often.
