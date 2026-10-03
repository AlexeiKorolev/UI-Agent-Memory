# Task: Do GUI agents causally use their memory? A MementoGUI-style intervention study on GUI-Odyssey

You are running on Princeton's Adroit cluster (Slurm). I'm a Princeton senior; this is part of my thesis on whether VLMs actually use explicit state. Work carefully, keep a running log, and never invent numbers. If something fails or differs from what this prompt assumes, write it down and adapt. Don't paper over it.

## 1. Background (read fully before starting)

MementoGUI (Zeng et al., arXiv 2605.18652, May 2026) adds a plug-in memory to a *frozen* GUI action model (the "backbone"). Its controller ("MementoCore") does four things:
1. At each step, scores how important the new screen is to remember, writes a one-line event summary, outputs a bounding box around the relevant region of the screen (the ROI), and decides whether to retrieve past episodes.
2. Compresses older working-memory entries.
3. Writes finished trajectories to episodic memory.
4. Selects relevant episodic entries.

The memory reaches the backbone as ordinary prompt content: text summaries plus cropped ROI images. Reported result: UI-Venus-1.5-8B on GUI-Odyssey goes from 54.58 AMS (no history) to 68.32 AMS (working + episodic memory). Keeping all predicted history gives 66.31. Removing the ROI crops hurts.

**Research question:** does the backbone use the *content* of the memory, or do the gains come from presence or format effects (extra images in the prompt, memory-shaped context)? We answer it with interventions on the memory content, holding everything else fixed.

**Constraint:** MementoGUI has released no code, weights or benchmark. Their GitHub repo is only a project page. So we re-implement a **MementoGUI-style working memory** with a *prompted* controller. This is not an exact reproduction, and the report must say so. **Skip episodic memory.** Working memory is the clean object for causal tests.

## 2. Adroit environment rules

- Verify everything below yourself; don't trust my assumptions. Run `sinfo`, `snodes` (if available), `checkquota`, and `module avail` to find GPU partitions, GPU types and storage limits.
- Compute nodes likely have **no internet**. Do every download (pip/conda packages, Hugging Face models and datasets) on the **login node** first.
- Put all large files under `/scratch/network/$USER/mementogui_causal/`. Set `HF_HOME` there and `HF_HUB_OFFLINE=1` inside jobs. Keep code in a git repo in that directory and commit often.
- Environment: `module load anaconda3/<version>`, then create a conda env with Python 3.11, PyTorch matching the cluster's CUDA, `transformers` (a version that supports Qwen3-VL), `vllm` (if it installs cleanly; otherwise fall back to HF `generate` with batching), `datasets`, `pillow`, `pandas`, `scipy`, `statsmodels`.
- One 8B VLM in bf16 needs about 17–20 GB. Prefer A100s. Run the controller and the backbone as separate passes so only one model is loaded at a time.
- Slurm jobs: make them resumable. Write a cache per episode, skip finished episodes, and request time limits sensibly. Use `sbatch` and job arrays for sharding.

## 3. Models and data (download on the login node)

- **Backbone:** `inclusionAI/UI-Venus-1.5-8B` (Qwen3-VL architecture, Apache-2.0). Read its model card and the GitHub repo `inclusionAI/UI-Venus` for the **exact navigation prompt, the action format, and the coordinate convention** (absolute vs 0–1000 normalized). Write a converter between UI-Venus actions and GUI-Odyssey actions. Unit-test it on 20 ground-truth steps (GT → UI-Venus format → back must round-trip).
- **Controller:** `Qwen/Qwen3-VL-8B-Instruct`, prompted, no training.
- **Data:** `OpenGVLab/GUI-Odyssey` on Hugging Face (check the dataset card for the newer version and its splits). Use the official evaluation code from the GUI-Odyssey GitHub repo for **AMS**. Do not re-implement the matching thresholds yourself unless the official code is unavailable. If you have to, document the exact thresholds and their source.
- Record exact model and dataset revisions (commit hashes) in `results/provenance.json`.

## 4. Pipeline

### 4a. Episode sampling
- From the test split, sample **300 episodes**, stratified by GUI-Odyssey task category, with a fixed seed (0).
- Pilot first on **10 episodes**. Stop after the pilot and write `reports/pilot.md` (setup checks, timing, sample prompts and outputs, any surprises). Then continue to the full run unless something is clearly broken. If it is, fix it and document what you changed.

### 4b. Memory-dependent step labels
Mark a step **memory-dependent (MD)** if its ground-truth action is TYPE (or another action with a text argument) and the text:
- does not appear on the current screen, and
- does appear on an earlier screen in the same episode (or in an earlier step's typed text).

Use OCR (e.g. PaddleOCR or Tesseract; install on the login node) plus any accessibility or UI text in the dataset, with fuzzy matching (normalized edit similarity ≥ 0.8).
- Save `md_steps.parquet` with columns episode_id, step, needed_string, source_step.
- Report how many MD steps you found. If there are fewer than 150, extend to 600 episodes. Also keep a random sample of non-MD steps for comparison.
- Hand-check 30 MD labels (inspect the images and save a contact sheet). Report label precision.

### 4c. Build the clean working memory (controller pass, ground-truth history)
For each episode, step through the ground-truth trajectory. At step t, give the controller the current screenshot, the previous ground-truth action, the task instruction, and the current memory. It must return JSON:

```json
{"salience": 0-1, "write": true/false, "summary": "<=30 words, include concrete values/names seen>", "roi_bbox": [x1,y1,x2,y2] or null}
```

- If `write`, append {step, summary, crop of roi_bbox from that screenshot} to memory.
- Compression: if memory has more than K=8 entries, have the controller merge the oldest entries into one text summary. Keep at most 4 crops, the most recent or most salient ones.
- Validate the JSON and retry up to 2× on parse failure. Log the failure rate.
- Cache the memory state *before* each step t as `cache/{episode}/{t}.json` plus PNG crops. Every condition below is built from this cache, so the controller runs only once.
- Sanity check: report how often an MD step's needed string appears in the memory text, or is visible in a memory crop (OCR the crops). This is the "presence" rate. Analyze causal use only on MD steps where the information is present in clean memory, and report results for both subsets.

### 4d. Backbone conditions (all use the same base prompt and the current screenshot)
Insert memory as one block: "Memory of earlier steps:", then for each entry `[step k] summary` followed by its crop image.

| ID | Condition | What it tests |
|---|---|---|
| C0 | No memory | lower baseline |
| C1 | Clean memory (text + crops) | main condition |
| C2 | All past ground-truth screenshots (cap at the context budget; downsample if needed) | the paper's "keep everything" baseline |
| C3 | Text only (crops removed) | contribution of the crops |
| C4 | Crops only (summaries replaced by `[step k]`) | contribution of the text |
| C5 | Blank crops: each crop replaced by a uniform gray image of the same size; text kept | do the crop *pixels* matter, or just image presence? |
| C6 | Crops shuffled within the episode (crop i paired with entry j's text; derangement) | text–image alignment |
| C7 | Cross-episode swap: memory from another episode in the same category with a matched number of entries | format/presence control (same form, wrong content) |
| C8 | Targeted counterfactual (MD steps only): replace the needed string everywhere in memory with a same-type alternative (another name, number of the same digit count, etc.). In crops, cover the string's OCR box with a background-colored rectangle and render the alternative text in a similar font size. Keep everything else identical. | does the action follow the edited content? |

Use greedy decoding (temperature 0). Save raw outputs and parsed actions for every (episode, step, condition).

## 5. Metrics and statistics
- **AMS** (official) per condition, for all steps and for MD steps.
- **Text-argument accuracy on MD steps:** normalized edit similarity ≥ 0.8 between predicted and needed text.
- **Change rate vs C1:** fraction of steps where the parsed action (type + argument / target region) differs from C1.
- **Follow rate (C8):** fraction of MD steps where the predicted text matches the *counterfactual* string, vs matching the original vs neither. Also report it separately for the "edit text only" and "edit crop only" variants (run both on MD steps if compute allows).
- Bootstrap 95% CIs (resample episodes, 1,000×). Paired McNemar tests for accuracy differences: C1 vs each other condition, on MD steps.

## 6. Pre-registered predictions (state in the report whether each held)
- **H1 (memory helps):** C1 > C0 on MD steps by a clear margin.
- **H2 (content, not presence):** C7 ≈ C0 and C7 ≪ C1 on MD steps. If C7 is near C1, the gain is a presence or format effect.
- **H3 (pixels matter):** C5 < C1 on MD steps whose needed string is only in a crop.
- **H4 (causal following):** C8 follow rate is far above the rate at which the counterfactual string is output under C1 (should be ~0).
- Calibration only: compare C0 and C2 with the paper's 54.58 / 66.31. Their protocol is unknown, so expect differences. Report them and don't tune toward them.

## 7. Optional, if time remains (lower priority than a clean main result)
On 50 MD steps where C1 is correct and C8 follows the counterfactual: run activation patching with the HF model (not vLLM).
- Patch residual-stream activations from the C1 run into the C8 run at the token positions of the memory block (text tokens and crop image tokens separately), one layer at a time.
- Measure the log-prob of the original vs counterfactual first token of the typed string.
- Plot the recovery fraction by layer. Also report the attention mass from the action tokens onto memory text vs crop tokens.

## 8. Deliverables (in `/scratch/network/$USER/mementogui_causal/`)
- `README.md`: how to rerun everything (env setup, download script, sbatch scripts, analysis).
- `src/`: controller, memory builder, condition builder, backbone runner, converters, metrics. Add tests for the action converter and the counterfactual editor.
- `results/`: per-step parquet, `summary_table.csv` (condition × {all, MD} × metrics with CIs), figures (bar chart by condition; follow-rate chart), `provenance.json`.
- `reports/pilot.md` and `reports/final.md`. The final report covers the setup, deviations from MementoGUI, the label-precision check, the presence rate, the main table, H1–H4 verdicts, failure examples (10 annotated cases with images), limitations, and the compute used (GPU-hours).
- `LOG.md`: dated running log of what you did and what went wrong.

## 9. Ground rules
- Don't modify or fine-tune the backbone. The controller is prompted only.
- Don't download PSAI or any non-public data. Don't use API models.
- If a number looks too good or too bad, investigate before reporting it (e.g. a parsing bug, coordinate-convention mismatch, or leakage of the current screen into memory).
- Prefer a smaller, correct run over a large, broken one. If compute or time runs short, cut episodes, not conditions C0/C1/C5/C7/C8.
