# Running log — MementoGUI-style causal memory study

## 2026-10-02 (Fri)
- 19:40 Created <project dir> (/scratch/network/<netid>/mementogui_causal on Adroit). Login node adroit5.
- Cluster check (`sinfo`/`snodes`): partitions all/class/gpu. GPU nodes: adroit-h11g1 4×A100-80GB, adroit-h11g2 8×A100 MIG 3g.20gb, **adroit-h11g3 2×A40 (48GB)**. GPU QOS walltime cap 48 h.
  Modules: anaconda3 up to 2026.7 (using 2025.12), cudatoolkit 11.8–13.3. Tesseract is system-installed (/usr/bin/tesseract).
- **Problem: scratch quota 362.8/390.6 GiB (~28 GB free), /home over soft limit (grace 6 days).** Not enough for 2×17.5 GB models.
  With user approval, deleted unused, re-downloadable HF caches (last accessed May 2026; checked that no running job references them):
  Gemma/hf_cache/models--google--gemma-2-9b (35G), gemma-3-12b-pt (23G), gemma-2-2b (9.8G). Scratch now 295.8 GiB used.
  All caches (pip/conda/HF/XDG/vllm/triton) redirected into the project dir via env.sh so /home is untouched.
- Hugging Face findings:
  - `OpenGVLab/GUI-Odyssey` (sha 71e0e7e2) README points to a **newer version `hflqf88888/GUIOdyssey`** (sha 61632d0f): 8,334 episodes, adds sam2_bbox for clicks, low-level instructions, etc. Action set differs from v1 (COMPLETE/INCOMPLETE; KEY_APPSELECT).
  - v2 screenshots ship only as a 92.6 GB split zip (screenshots.z01–z08 + .zip). Plan: extract only the needed episodes with HTTP range reads of the zip central directory (no 92 GB download).
  - UI-Venus-1.5-8B sha a06ff6c6; Qwen3-VL-8B-Instruct sha 0c351dd0. Each ~17.5 GB.
- User chose: fresh conda env with vLLM.
- 20:05 Env created at env/ (python 3.11): torch 2.13.0+cu130, transformers 5.18.0, vllm 0.30.0 (+qwen-vl-utils, pytesseract, rapidfuzz, statsmodels...). 6.5 GB.
- Cloned refs into third_party/: inclusionAI/UI-Venus @1ffbaf65 (repo history was rewritten at UI-Venus-2 release; 1.5-era code not in git),
  OpenGVLab/GUI-Odyssey @5cdf76d9 (official eval: src/eval_mm/GUIOdyssey_action_matching.py — uses sam2_bbox ⇒ targets v2 data).
- UI-Venus-1.5 navigation prompt: NOT in the model card nor current repo. Taken verbatim from the UI-Venus-1.5 tech report (arXiv 2602.09082v2, App. A.3 "Mobile Prompt").
  Coordinates: normalized [0,1000] (report §2 + mobile framework normalize_to_screen) — same convention as GUI-Odyssey. Scroll(start,end) = finger trajectory, same as GUI-Odyssey SCROLL info.
- Official AMS details: action type must match; CLICK/LONG_PRESS correct if inside sam2_bbox OR L2 dist ≤ 0.14 (in [0,1] units); TYPE via ANLS≥0.5 or substring; SCROLL via direction string; others type-only.
  'macro' in their code = plain step accuracy; 'micro' = mean over 6 categories (used for random_split). We report both.
- Data: using v2 `hflqf88888/GUIOdyssey`, **random_split test (1666 episodes, 25,807 steps)**. Downloaded all 1666 test annotation JSONs.
  TEXT(=TYPE) steps: 2666; of those 1281 (in 935 episodes) have text not fuzzy-contained in the instruction.
  Annotations contain `context`/`intention` fields that summarise history → never shown to controller/backbone (leakage).
- MD label refinement (deviation, documented): also require needed string NOT in the task instruction (fuzzy ≥0.8), since the instruction is always in the prompt. Flag kept so both definitions can be reported.
- Slurm QOS (sacctmgr): gpu-short (≤4h) 4 GPUs/user; gpu-medium (≤24h) & gpu-long 2 GPUs/user. User's other (NASA) jobs occupy gpu-long.
  ⇒ All jobs here are ≤4h resumable array tasks so they use the gpu-short allowance and never block the user's gpu-long jobs.
  GPU constraint: "gpu80|a40" (8B bf16 + vision tokens is too tight for 20GB MIG slices).
- 20:16 Submitted OCR+MD job 3389613 (32 CPUs, all) and pilot job 3389614 (afterok).
- 20:15 Screenshots: v2 split zip parsed via HTTP range reads (src/remote_zip.py; 127,893 members, deflate, CRC-checked).
  First attempt hit HF **429 rate limit** ("resolvers" bucket: 3000 req/5 min per IP; the login-node NAT IP is shared by other Adroit users).
  Fix: cache the CDN redirect URL per part, hard-code part sizes, single range read per member, exponential backoff, 8 workers.
  All 9,388 screenshots for the first 600 stratified episodes downloaded (~6.7 GB).
- Sampling: src/data.py::stratified_order(seed=0) — per-category shuffled lists merged by fractional rank, so every prefix
  (10/300/600) is category-proportional (±1) and 600 ⊃ 300 ⊃ 10. n=300: GT 64/IM 54/ME 41/MA 63/SS 44/WS 34.
- Converter tests: 20 GT steps (stratified over action types) GT→UI-Venus→GUI-Odyssey round-trip exact; + edge cases. All pass.
  Note: official simple_decode splits on ':' so TYPE text containing ':' (URLs) is truncated for BOTH gt and pred in AMS (official behaviour, kept);
  our MD text accuracy uses split(':',1).
- OCR bug found by counterfactual unit test: reading order used fixed 20px y-buckets → words of one line could be mis-ordered.
  Replaced by line clustering (src/ocr.py::reading_order), applied on load, so caches written earlier are fixed too. Fixed before OCR job ran.
- 20:25 OCR+MD job 3389613 (class partition after `all` was full; est. wait there was 3h): 9,388 shots OCR'd in ~25 min.
  MD labels n=300: 498 TYPE steps, **134 MD** (76 strict, i.e. not in instruction) in 121 episodes → <150 ⇒ **extend to 600 episodes** (as pre-specified).
  n=600: 995 TYPE steps, **275 MD** (164 strict) in 245 episodes. Sources: 180 screen-only, 95 screen+typed.
- Hand-check of 30 MD labels (reports/label_check.md, contact sheets reports/figs/md_label_check_*.png): **27/30 correct (precision 0.90, Wilson 95% CI 0.74–0.97)**.
  Errors: OCR missed white-on-purple text on current screen; stale browser history matched; loose fuzzy paraphrase match.
- CPU smoke test of controller-prompt/conditions/backbone-prompt build with fake memory: all conditions build; max prompt 5.1k tokens on the test episode.
  Tightened C8: alternative must not appear on any screen 0..t of the episode (not just current).
- 20:50 GPU queue is congested: pilot 3389614 est. start Oct 4 (≈20 jobs ahead at similar priority; 4×A100 busy, both A40s used by user's own NASA jobs).
  Decided NOT to quantize to fit 20GB MIG slices (would alter the backbone). Waiting.

## 2026-10-03 (Sat)
- 01:00 Made paths portable (MG_PROJ / repo root), removed personal info from tracked files, wrote CONTEXT.md handoff, pushed
  a single fresh commit to github.com/AlexeiKorolev/UI-Agent-Memory (local full history kept on branch `adroit-history`).
- 01:30 /home was at its 10 GiB hard limit (not caused by this project). With user approval cleaned re-downloadable caches
  (4 old VS Code CLI servers, 6 legacy VS Code server builds, cached VSIXs, ~/.cache/pip): /home 10 → 3.4 GiB.
- 11:22 Pilot 3389614 started (after ~14 h queued) on adroit-h11g1 (A100-80GB). Controller model loaded fine
  (16.8 GiB weights, 51.5 GiB KV cache) but **engine warm-up crashed**: vLLM's default FlashInfer top-k/top-p sampler
  JIT-compiles a CUDA kernel that needs `curand.h` (not available without a CUDA toolkit module), and it wrote its JIT cache
  to ~/.cache/flashinfer (bypassing our cache redirects). Job reported COMPLETED only because the script's final echo ran.
  Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` (greedy = argmax, outputs unaffected), `FLASHINFER_WORKSPACE_BASE=$PROJ`,
  load `cudatoolkit/13.0` in jobs (headers for any other JIT), `set -eo pipefail` in all sbatch scripts. Removed the 696K
  ~/.cache/flashinfer it created. Resubmitted pilot (3390145) + a 20-min engine check (3390146, any GPU incl. MIG).
- 14:34 Pilot OK (reports/pilot.md). C8 alternatives made template-aware. Pilot outputs archived to results/pilot/. Full run submitted: controller=3390346 (array 0-3) -> conditions=3390347 -> backbone=3390348 (array 0-3) -> analyze=3390349.
- 16:20 Controller array 3390346 done (4×~25 min, adroit-h11g3 A40s): 600/600 episodes cached. 9,242 write calls (+146 pilot cached), first-try JSON failures 2 (0.02%), resolved by retry, final failures 0; merge calls 2,517, merge JSON failures 2 (fallback: concatenation). Conditions build 3390347 started on adroit-h11n2.
- 16:25 Conditions 3390347 done (4.4 min): specs for 600 episodes. Presence on 275 MD steps: text 89.5%, crop 65.5%, any 93.1%. C8 alternatives: same-template 203, same-category 60, digit 11, any 1; text edited on 89.8%, crop on 65.5% of MD steps. Backbone array 3390348 queued.
- 16:30–20:15 Backbone array 3390348: shard 0 COMPLETED 16:30→19:40 (3 h 09 m, A40, ~1.5 prompts/s); shard 1 running
  since 19:40 (A40); shard 2 running since 20:06 (A100 — mixed hardware, note in limitations); shard 3 pending.
  182/600 episodes done at 20:13. GPU-hours so far ≈ 5.3. Added src/failure_cases.py (tested on pilot data) and
  scripts/gpu_hours.py. Rewrote CONTEXT.md §4 as a "PICK UP HERE" handoff; added CLAUDE.md pointing to it.
- 22:46 Backbone array 3390348 all 4 shards COMPLETED (A40: 3h09, 3h06; A100: 1h39, 1h34); 600/600 episodes. Analysis job 3390349 cancelled: per-resample pd.concat bootstrap would take many hours at n=600 (~270 CIs). Replaced with an equivalent vectorised episode bootstrap (per-episode sums/counts, numpy resampling; verified against the old implementation on pilot data, CIs agree within MC noise). Resubmitted analysis as 3390782.
- 23:00 Analysis 3390782 COMPLETED (16 s with the vectorised bootstrap). Sanity checks all pass: INVALID ≤ 0.03%,
  truncation 4/75,072, step-0 prompts identical across C0/C1/C3/C7, C8 rows only on MD steps, C1 self-change 0, no A40/A100
  difference (< 1 AMS point). Diagnostics: C1 predicts TYPE on 57.8% of MD-present steps (rest ≈ all CLICK), 67.9% correct
  when typing; C8 edits clean (original survives in 1/275 texts, 0 crops); C7 vs C0 McNemar p = 0.002 (strict p = 0.03).
- Figures (`results/figures/`), 10 annotated cases (`reports/figs/case_*.png`; fixed a selection bug where de-duplication
  dropped cases), provenance refreshed, GPU-hours 11.43. Wrote `reports/final.md` (numbers generated from the CSVs;
  claims checked against computed values; softened one unverified explanation in §5.2). Study complete.

## 2026-10-04 (Sun)
- Failure analysis of C1 (all steps, 3,167/9,388 wrong): 1,917 wrong action type, 1,123 wrong click location, 82 wrong
  text, 45 wrong scroll direction. GT TEXT steps: C1 clicks on 393/995 (reasoning often "click the search bar to start
  typing <right string>"); GT COMPLETE: 183 clicks. Accuracy decays with step index (C1 70.0 at steps 1-2 -> 57.5 at
  20+; C2 66.1). Controller salience is saturated (~89% of entries >= 0.8), writes on 98% of steps. C1 replaces the
  prompt's "Previous Actions" slot with screen summaries, so the backbone never sees which actions were taken.
- Follow-up 1, action trace (pre-specified, no tuning; evaluated once on the same n=600 episodes, reusing the frozen
  specs/cache): `src/action_trace.py` builds results/specs_at/ from results/specs/ (never rebuilt) adding
  "Actions taken so far: [step k] <GT action in UI-Venus syntax>". Conditions A0 actions only, A1 C1+actions,
  A2 C2 screenshots interleaved with actions, A7 C7+recipient's actions, A8 C8+actions (needed string also replaced in
  earlier Type actions: 89 edits; original survives in 1/275 A8 histories, as in C8). 26 MD steps have the needed
  string in the clean action trace (typed earlier). backbone.py/analyze.py take --specs/--raw/--out_tag (defaults =
  main study, whose files are untouched). Submitted backbone 3392521 (array 0-3) -> analysis 3392522
  (outputs results/n600_at_*).
- 2026-10-05 01:51 Action-trace run done: backbone 3392521 (3 shards A100 ~1.0 h, 1 shard h11g3 1.8 h), analysis
  3392522 -> results/n600_at_{summary_table,follow_table}.csv, n600_at_per_step.parquet (C* rows identical to the
  main study). Sanity: INVALID <= 0.011%, truncation 2/37,827, step-0 prompts identical to C0, A8 only on MD steps.
  AMS all steps: A0 68.2, A1 71.8, A2 76.6, A7 64.1 (vs C0 38.3, C1 66.3, C2 71.0, C7 51.6).
  MD-present text acc: A0 40.2, A1 53.5, A2 64.1, A7 36.7; strict-present: A0 19.7, A1 45.4, A2 57.9, A7 16.4.
  A8 follow 25.4% (C8 20.7%); strict 32.9%. TEXT steps predicted as CLICK: C1 39.5% -> A1 19.8%.
  Leakage checks: copying the previous GT action ~ base rate (8-9% vs GT repeat rate 7.9%); gains hold excluding repeats.
  Needed string in trace on only 25/256 MD-present steps; gains hold on the other 231.
  Finding: with the trace the backbone types the needed string on 30/152 strict-present steps with NO memory (A0), e.g.
  'BTS' (world knowledge), 'Los Angeles' (guess), and a ~200-char recipe verbatim (possible training-data
  memorisation; GUI-Odyssey is not named in the UI-Venus-1.5 report's data list, which says "30+ sources ... and so
  on"). C0 had the same knowledge in its reasoning but clicked instead of typing. => "strict MD" is not memory-only;
  content effects should be read as A1 vs A0/A7 (45.4 vs 19.7/16.4 strict), not vs C0.

## 2026-10-06 (Tue)
- Clarified for the write-up: the controller receives the previous GT action, the backbone never did; C1 replaced the
  backbone's native "Previous Actions" history (only 13% of memory entries mention an action verb).
- Prior-knowledge analysis (`src/prior_knowledge.py` → results/n600_at_prior_table.csv): of 982 GT TYPE steps, 248 have
  a string that appears in no input; A0 types 31.5% of them exactly (C0 13.7%). Dataset typos are not reproduced
  (Italin→Italian, prosche→Porsche) → world knowledge + template conventions, no evidence of verbatim episode recall.
  A1 < A0 on strings visible on the current screen (47.7 vs 64.0 exact, n=86).
- Paired tests (`src/at_compare.py` → results/n600_at_paired.csv): A1 vs A0 strict MD 45.4 vs 19.7 (49/10, p=3e-7);
  A7 vs A0 all-step AMS 64.1 vs 68.2 (362/752); A2 vs A1 strict 57.9 vs 45.4 (p=0.004); A8 vs C8 follow 25.4 vs 20.7.
- Related-work snapshot (docs/related_work.md) and task-graph memory design (docs/graph_memory_design.md, not run).
- Report §9 + TL;DR bullet, CONTEXT §4.0, README status updated. GPU-hours total 16.3 (follow-up 4.9).
- Follow-up 2, long-horizon (design fixed before any result; no tuning). Question: does a MementoGUI-style memory pay
  off once episodes are long, and against raw screenshots under the *same* visual budget?
  * Runs (outputs under runs/<name>/, selected by MG_DATASET/MG_RUN; main-study files untouched):
    `odylong` = all 155 GUI-Odyssey random_split test episodes with >= 25 steps (4,668 steps; 131 Multi_Apps); 55
    overlap the n=600 sample and reuse its controller caches (copied, without the C8 edits in cf/).
    `memgui` = MemGUI-3K test split (lgy0404/MemGUI-3K @ 003822b5, Apache-2.0; teacher rollouts of MemGUI-Agent,
    released after UI-Venus-1.5): 295 episodes, 7,958 steps after dropping 334 memory_* context actions and 4 steps
    without a tool call (median 22, p90 50, max 115); 6,146 steps marked reasonable are scored, the rest stay in the
    history. GT = the step's <tool_call> (0-1000 coords); `src/memgui.py` converts to the GUI-Odyssey schema; new GT
    types PRESS_ENTER/WAIT/ANSWER are scored by the official matcher's generic type-match rule; clicks by L2 <= 0.14
    (no SAM2 boxes); CallUser -> ANSWER for MemGUI. MD labels also cover ANSWER steps; the teacher's <ui_observation>
    is used like GUI-Odyssey's `description` (current-screen check only, never shown to a model).
  * New conditions: C2b = the n most recent past screenshots, n = #crops in C1 at that step, each downsampled so the
    total pixels equal C1's crops; C2w = last 4 screenshots at 0.35 MP; A3/A5 = C3/C5 + action trace; A2b = C2b +
    action trace. Backbone runs C0,C1,C2,C2b,C2w,C3,C4,C5,C7,C8,C8t,C8c and A0,A1,A2,A2b,A3,A5,A7,A8 (C6 dropped).
  * Predictions: L1 at matched pixels memory beats screenshots on MD steps (C1 > C2b, A1 > A2b); L2 the crop effect
    shrinks once actions are given (A1-A3 < C1-C3); L3 content effect replicates (A1 > A7, C8 follow > 0);
    L4 (exploratory) C2-C1 gap smaller on long episodes than in n=600.
  * Pilots: 10 episodes per run in runs/<name>_pilot (job 3394140 = odylong_pilot).
- Pilot fix (memgui, before any full-run labels): the teacher's <ui_observation> often describes the screen *after*
  the action ("the title field now contains 'First Day of Summer'" with an empty field on the screenshot; 6/14 pilot
  TYPE steps had desc-sim 1.0 vs OCR <= 0.52), which wrongly marked needed strings as visible. It is no longer used:
  MemGUI MD labels rely on OCR alone (teacher text kept as `teacher_observation`, unused). Pilot MD: 2 -> 5 of 14
  scored TYPE/ANSWER steps (1 strict). odylong pilot (job 3394140, 296 steps) ran end to end: 0 INVALID, 0 truncated,
  C2b image count = C1's. Full odylong chain: build 3394265 -> backbone 3394266 (8 shards) -> analysis 3394267.
- memgui pilot (job 3394142, 157 scored steps) ran end to end: 0 truncated, 1 INVALID (C0); WAIT/COMPLETE scored by
  type; scroll acc 31-46% (convention OK). Full memgui chain: build 3394295 -> backbone 3394296 (8 shards) -> analysis 3394297.
- 08:06 odylong done: controller 3394145 (4 x ~11 min), OCR/MD 3394143 (148 MD, 67 strict, 115 eps), build 3394265
  (presence 94.6%), backbone 3394266 (8 shards: 7 on A100 ~1 h, 1 on A40 2.3 h), analysis 3394267 ->
  runs/odylong/results/n155_*; paired tests `src/long_compare.py` -> runs/long_compare_{paired,did,steps}.csv.
  Sanity: INVALID <= 0.02%, 0 truncated, step-0 prompts = C0 for all conditions, C8/A8 only on MD, C0 Launch 16%.
  All-step AMS: C0 34.4, C1 62.0, C2 68.7, C2b 61.1, C2w 63.7, C3 59.2, C5 59.7, C7 48.8; A0 67.5, A1 68.3, A2 73.5,
  A2b 73.1, A3 66.9, A5 67.8, A7 63.2. MD-present text acc (n=140): C1 35.7, C2b 47.9, C2w 48.6, C2 48.6, A0 52.1,
  A1 46.4, A2b 67.1, A2 68.6, A7 42.9; strict-present (n=63): C1 20.6, C2b 20.6, C2 34.9, A0 25.4, A1 34.9, A2b 49.2,
  A2 60.3, A7 20.6. C8 follow 17.1% (strict 12.7), A8 20.7%.
  L1 not supported: at matched pixels screenshots >= memory (MD-present C2b-C1 +12.1 [4.9, 20.7], p=0.009; A2b-A1
  +20.7, p<1e-4; strict: C tie, A +14.3 p=0.09; all steps C1 ~ C2b, A2b > A1 by 4.9). Checked: not because the source
  screen is in the window (C2b 50.0% when it is outside, 44.4% inside). Mechanism: screenshots raise the TYPE rate
  (C1 57.9 -> C2b 66.4) and memory text *distracts* when the string is in the instruction (acc|typed C1 74.0, A1 68.3
  vs C2b 91.5, A0 91.9).
  L2 supported (weakly): crops matter without actions (C1-C3 +7.1 on MD-present, p=0.04) but not with them (A1-A3
  -0.7); DiD 7.9 [0.0, 15.8]; all steps DiD 1.4 [0.5, 2.5].
  L3 partly: content effect only on strict steps (A1 vs A7 34.9 vs 20.6, p=0.049; A1 vs A0 +9.5 ns); C8 follow
  17.1% > 0 (C1: 0). L4 opposite: C2-C1 gap is larger on long episodes (+6.7 vs +4.8; diff 1.9 [0.5, 3.4]).
- 13:5x memgui done: controller 3394150 (295/295; 1 write fail, 8 merge fallbacks), OCR/MD 3394149 (238 MD, 144
  strict), build 3394295 (presence 67.6%), backbone 3394296 (8 shards, 1.1-2.9 h), analysis 3394297 ->
  runs/memgui/results/n295_*. Sanity as for odylong; C0 Launch 37.5% (home-screen starts).
  All-step AMS: C0 23.5, C1 51.4, C2 54.3, C2b 48.6, C2w 51.3, A0 54.6, A1 59.2, A2 62.4, A2b 60.0, A7 53.5.
  Strict MD-present (n=81): C1 27.2, C2 18.5, C2b 2.5, C2w 4.9, A0 9.9, A1 50.6, A2 44.4, A2b 18.5, A3 38.3, A7 7.4;
  A8 follow 39.5% strict (21.7% MD-present).
  L1 supported on MemGUI (A1 vs A2b strict 30 vs 4; C1 vs C2b 22 vs 2) - opposite of odylong. Checked: strict strings
  are exact values (prices, long numbers) a median 5 steps back; source outside the C2b window on 65%, and inside it
  the downsampled screens are mostly unreadable (C2b 2/28; C2 at 0.35 MP 32%). L2 not supported on MemGUI (A1-A3
  +12.3 strict, p=0.006). L3 strongly supported (A1 vs A7 35 vs 0 strict). L4 supported on MemGUI (C2-C1 3.0 vs 4.8).
  Long-horizon follow-up total 26.1 GPU-hours (project ~42.4). Report §10 + TL;DR, README, CONTEXT §4.0a updated.
  Naming: this is the "long-horizon follow-up"; the task-graph design (docs/graph_memory_design.md) is separate.
