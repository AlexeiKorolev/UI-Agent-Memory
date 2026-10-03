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
