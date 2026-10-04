# CONTEXT — handoff document

Everything needed to continue this project on another machine or with a new assistant session. The original task
specification is in [`docs/task_spec.md`](docs/task_spec.md) (verbatim). The dated running log is
[`LOG.md`](LOG.md). This file summarises **what was decided, why, what is verified, and what remains**.

_Last updated: 2026-10-03 20:15 EDT (Adroit). **A fresh session should start at §4 "PICK UP HERE".**_

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

## 4. Status — PICK UP HERE (updated 2026-10-03 23:59 EDT)

### 4.1 Where things stand: **the main study is COMPLETE**

All stages finished: controller (job 3390346), conditions (3390347), backbone (3390348; 4 shards: two on A40s
~3 h each, two on A100s ~1.6 h each), analysis (3390782, after replacing a too-slow bootstrap with an equivalent
vectorised one). **Final report: `reports/final.md`.** Results: `results/n600_summary_table.csv` (= `summary_table.csv`),
`results/n600_follow_table.csv`, `results/n600_per_step.parquet`, figures in `results/figures/`, 10 annotated cases in
`reports/figs/case_*.png` + `reports/failure_cases.csv`. Compute: 11.4 GPU-hours.

Headline: H1 supported (MD-present text acc C1 40.6 vs C0 10.2); H2 mostly supported (C7 17.2 ≪ C1, but C7 > C0, and on
*all steps* C7 recovers ~48% of the AMS gain = presence/format effect); H3 not testable as pre-registered (crop-only n=10)
but indirect evidence that crop content matters (C5 ≈ C3 < C1); H4 supported (C8 follow 20.7% vs 0% under C1; text
channel 14.1% ≫ crop channel 1.6%). C2 (raw history) beats C1.

**Possible next steps (none required):** (a) optional activation patching (task spec §7) on ≤ 50 MD steps where C1 is
right and C8 follows (`reports/failure_cases.csv`/per-step parquet identify them); (b) a closed-loop (predicted-history)
variant; (c) a larger crop-only subset to test H3 directly; (d) ask the user whether to scrub the netid from GitHub
history: paths in pilot specs (commit `e5c5dfb`) and tar owner names in the first `artifacts/ocr_cache_600.tar.gz`
(commits `ae6ec2b`–`aeeba79`; rebuilt with anonymous owner in the final commit). Needs a force-push.

### 4.2 How to check

```bash
cd <project dir> && source env.sh        # on Adroit: /scratch/network/<netid>/mementogui_causal
squeue -u $USER -o "%i %j %T %M %N %R"
sacct -j 3390348,3390349 -X -o JobID%14,State,Start,Elapsed,NodeList
ls results/raw | wc -l                                  # 600 when the backbone is complete
grep -c '^batch' logs/bb_3390348_*.out                  # 19 batches = shard complete
grep -l Traceback logs/bb_3390348_*.err                 # should print nothing
python3 scripts/gpu_hours.py                            # compute used (job names mg_*)
```

### 4.3 If something goes wrong

* **A backbone shard hits the 4 h limit (TIMEOUT) or fails:** everything is resumable per episode (an episode file in
  `results/raw/` is only written when complete; finished episodes are skipped). The analysis job (3390349) will then
  never start because its dependency failed → `scancel 3390349`, resubmit the missing shard(s), then analysis:
  `sbatch --mail-user=<you> --array=<k> slurm/backbone.sbatch 600` and
  `sbatch --mail-user=<you> --dependency=afterok:<new> slurm/analyze.sbatch 600`.
* **Never re-run the controller or the condition build** for n=600: the backbone outputs depend on these exact specs.
  (Re-running conditions would also overwrite the C8 edited crops in `cache/*/cf/`.)
* All jobs: submit **from the repo root**, pass `--mail-user=<user's addresses>` on the
  command line (addresses: see the user's `princeton-adroit` Claude skill) (removed from tracked files because the repo is public). Keep jobs ≤ 4 h (gpu-short QOS, 4 GPUs/user)
  so they don't compete with the user's own gpu-long jobs.

### 4.4 Remaining work, in order

1. **Wait for backbone + analysis.** Then sanity-check before believing anything (task spec §9): INVALID/parse rate per
   condition, Launch/Wait/PressEnter rates, truncation (`</action>` present), C6/C7 n/a counts, that C1 ≡ C0 at step 0, that
   C8 rows exist only on MD steps, that `changed_vs_C1` for C1 is 0.
2. **Figures:** `python -m src.figures --prefix results/n600 --md_subset MD_present` → `results/figures/conditions_bar.png`,
   `results/figures/follow_rate.png`. Look at them (render and inspect) before using.
3. **Failure/annotated cases:** `python -m src.failure_cases --prefix results/n600 --k 10` → `reports/figs/case_01..10.png`,
   `reports/failure_cases.csv`. Inspect each image and write a 1–2 line annotation per case in the report.
4. **Copy deliverable names:** `results/n600_summary_table.csv` → also save as `results/summary_table.csv`
   (task spec §8 name); per-step parquet = `results/n600_per_step.parquet`.
5. **`python -m src.provenance`** (refresh commit hash) and **`python3 scripts/gpu_hours.py`** for the compute section.
6. **Write `reports/final.md`** (task spec §8): setup; deviations from MementoGUI (prompted controller, no episodic
   memory, v2 data / random_split, official prompt from tech report, Launch/CallUser mapping, MD `in_instruction` flag,
   template-aware C8 alternatives); label precision (0.90, CI 0.74–0.97); presence rate (above); main table
   (conditions × {all, MD, MD_present, MD_strict(_present), MD_crop_only} with CIs); **H1–H4 verdicts** stated against
   the pre-registered predictions (H1: C1 > C0 on MD; H2: C7 ≈ C0 ≪ C1 on MD; H3: C5 < C1 on MD_crop_only; H4: C8 follow
   ≫ C1 cf-rate); C8 vs C8t vs C8c (which channel drives following); calibration C0/C2 vs paper 54.58/66.31 (don't tune;
   explain the Launch/no-history effect seen in the pilot); 10 annotated cases; limitations (label precision 0.90, small
   MD_crop_only subset, prompted controller, GT-trajectory memory, single seed/greedy, A40 vs A100 mixed hardware —
   greedy outputs can differ slightly across GPUs); compute (GPU-hours). Pilot-scale hints (n=7 MD steps, not
   evidence): C1 67.1 vs C0 43.8 AMS on all steps; C8 follow 2/7 via text edits, 0/7 via crop-only edits.
7. Update `README.md` status line, `LOG.md`, commit, and push to GitHub (check `git grep` for personal info first; specs
   are gitignored because they contain absolute paths).
8. Optional (task spec §7): activation patching on ≤ 50 MD steps with the HF model — only after 1–7.

### 4.5 Open items / caveats to carry forward

* Public GitHub history: commit `e5c5dfb` contains 10 pilot spec JSONs whose image paths include the netid; removed from
  the tree in `7ecd050`. Scrubbing history needs a force-push — **ask the user first**.
* Pilot outputs (`results/pilot/`) used the *old* C8 alternative picker; the full run regenerates those 10 episodes with
  the template-aware picker. Don't mix pilot and full-run numbers.
* `/home` was cleaned on 2026-10-03 (10 → 3.4 GiB). Keep all caches in the project (env.sh does this, incl. FlashInfer).
* Mixed hardware: shard 2 runs on an A100, shards 0/1 on A40s. Greedy decoding is deterministic per device but not
  guaranteed bit-identical across GPU types; mention in limitations.

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
scripts/              setup_env.sh, download.sh, engine_check.py (GPU smoke test), gpu_hours.py (compute tally)
slurm/                ocr, pilot, engine_check, controller (array), build_conditions, backbone (array), analyze
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
src/failure_cases.py  10 annotated example cases (reports/figs/case_*.png)
src/provenance.py     results/provenance.json
tests/                converter + counterfactual tests
artifacts/            OCR cache (600 eps), zip central-directory index
results/              md_steps_*.parquet, type_steps_*.parquet, nonmd_sample_*.parquet, provenance.json,
                      presence.parquet, counterfactuals.parquet, specs/ (gitignored), raw/ (backbone outputs, gitignored),
                      pilot/ (archived pilot outputs), controller_stats/
reports/              pilot.md, label_check.md, figs/ (contact sheets; case panels later)
cache/                controller memory states + crops + C8 edited crops (gitignored; Adroit only)
```

## 7. Ground rules carried over

No fine-tuning/modification of the backbone; controller prompted only; no API models; no non-public data; never invent
numbers; investigate numbers that look too good/bad (parsing, coordinate convention, leakage); if compute is short cut
episodes, not conditions C0/C1/C5/C7/C8. Keep `LOG.md` updated and commit often.
