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
