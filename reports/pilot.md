# Pilot report (10 episodes)

_2026-10-03. Adroit job 3390145 on adroit-h11g3 (NVIDIA A40 48 GB), 15 min 13 s wall time. First attempt (job
3389614, A100) crashed at vLLM engine warm-up, see §4._

Pilot = the first 10 episodes of the stratified order (seed 0): 2 General_Tool, 2 Information_Management,
2 Multi_Apps, 2 Social_Sharing, 1 Media_Entertainment, 1 Web_Shopping. 146 steps; 7 MD steps (3 "strict", i.e. needed
string not in the instruction). **These numbers are far too small for inference; the pilot only checks the
pipeline.**

## 1. Setup checks

| Check | Result |
|---|---|
| vLLM 0.30 / torch 2.13+cu130 on Adroit GPU nodes (driver 610.57) | OK after disabling the FlashInfer sampler (§4) |
| Model load | Qwen3-VL-8B 16.8 GiB weights; 51.5 GiB KV cache on A100-80GB |
| Controller JSON validity | **146/146 valid on first try** (0 retries, 0 final failures); 21/21 merge calls valid |
| Controller behaviour | write rate 0.69, ROI box rate 0.66, mean summary 11.2 words; summaries carry concrete values forward (e.g. "…ready to type company name: Entertainment Retail Enterprises, LLC") |
| Presence of needed string in clean memory (MD steps) | text 7/7, crop 5/7, any 7/7 |
| C8 edits | text edited in 7/7, crop edited in 5/7 |
| Backbone output format | 0 unparseable outputs, 0 truncated (all contain `</action>`); mean 350–410 chars |
| Scroll convention | when the model predicts SCROLL on a GT SCROLL step, direction is right 13/14 (93%), so finger-trajectory semantics match |
| Condition availability | C6 n/a on 11/146 steps (single crop, no donor); C7 n/a on 22/146 (no same-category donor with a memory among only 10 pilot episodes; expected to vanish at n=600) |

## 2. Timing (A40)

* Controller: 146 calls + 21 merges in 254 s including model load (~0.6 calls/s effective at batch ≤ 10, step-synchronous).
* Condition build (CPU inside the job, incl. crop OCR): ~90 s.
* Backbone: 1,156 (condition × step) jobs → **1,013 unique prompts** after deduplication, 2.98 M prompt tokens
  (~2.9k per prompt), **552 s = 1.83 prompts/s**.
* Projection for n = 600 (60×): controller ≈ 3 GPU-h, backbone ≈ 61k prompts ≈ 9.3 GPU-h on A40 (less on A100);
  4 shards of each fit the 4 h gpu-short limit.

## 3. Pilot numbers (sanity only; CIs are huge)

AMS (official matcher), all 146 steps: C0 43.8, **C1 67.1**, C2 70.6, C3 64.4, C4 65.1, C5 61.6, C6 65.9 (n=135),
C7 54.0 (n=124). MD text accuracy (n=7): C0 2/7, C1 2/7, C2 3/7. C8 follow rate: 2/7 (both via the text edit;
C8t 2/7, C8c 0/7); C1 outputs the counterfactual string 0/7 times.

**C0 vs paper (54.58).** Our C0 is lower. Diagnosis: without history the model does not know it is mid-task. It predicts
PRESS_HOME on 17% of steps (GT 8%) and `Launch(app)` on 8%, which has no GUI-Odyssey counterpart and always scores
wrong. It also never predicts COMPLETE (GT 7%). This is a genuine no-history effect plus a protocol caveat (Launch
mapping), not a parsing bug. Per the protocol we do not tune toward the paper; the final report will give Launch/Wait/
PressEnter rates per condition.

## 4. Surprises and changes made after the pilot

1. **vLLM engine crash (first attempt).** vLLM's default FlashInfer top-k/top-p sampler JIT-compiles a CUDA kernel
   needing `curand.h`, which wasn't available, and wrote its JIT cache to `~/.cache` (bypassing our redirects).
   Fixed with `VLLM_USE_FLASHINFER_SAMPLER=0` (greedy = argmax, so outputs are unaffected),
   `FLASHINFER_WORKSPACE_BASE=$PROJ`, `module load cudatoolkit/13.0`, and `set -eo pipefail` in all job scripts (the failed
   job had been reported COMPLETED). Verified with a separate engine-check job (3390146) before the pilot reran.
2. **Analysis bugs** (no effect on model outputs): a DataFrame column named `gt` collided with pandas' `.gt()` method,
   and missing predicted text arrived as NaN. Both fixed in `src/analyze.py`.
3. **Counterfactual alternatives were often a different semantic type** (company name → "Veggie & Hummus Sandwich";
   "learn to grow herbs indoors" → "'Atomic Habits' by James Clear"). An implausible swap could under-estimate following.
   **Changed:** candidates are now ranked by same task template (`meta_task`, i.e. the same slot), then same category,
   then any; dissimilarity and never-seen-in-episode constraints unchanged. On the first 40 MD strings: 31 same-template,
   5 same-category, 4 digit perturbations (e.g. company → "Yoh, A Day & Zimmermann Company", Capcut → Microsoft Outlook,
   "learn to grow herbs indoors" → "learn to solve a Rubik's cube:"). Unit test added. The pilot's backbone outputs were
   moved to `results/pilot/` so the full run regenerates those 10 episodes with the new alternatives.
4. Clean causal-following example (non-strict MD; the string is also in the instruction): episode 7223485012107950, step 4.
   Memory mentions Duolingo; C0/C1/C7 type "Duolingo"; with the memory text edited to "Tokopedia" (C8, C8t) the model
   types "Tokopedia" ("To manage notifications for the Tokopedia app…"), although the instruction says Duolingo; the
   crop-only edit (C8c) does not change the action.

## 5. Sample prompt and outputs (episode 7223485012107950, step 4, MD)

C1 prompt, middle part (official UI-Venus-1.5 mobile prompt before and after it; `<IMAGE>` = memory crop; the current
screenshot follows `### Current Screenshot`):

```
### User Task
Manage the notifications for any app on your phone by turning them on or off, and then proceed to open the app. You can do this for apps like Duolingo by accessing the Settings app.
### Previous Actions
Memory of earlier steps:
[step 0] Home screen shows Duolingo app icon, which needs to be opened after managing its notifications via Settings app.
<IMAGE>
[step 1] Settings menu open, showing 'Network & internet' selected, with 'Apps' option visible for accessing app-specific settings.
<IMAGE>
[step 2] In Settings > Apps, 'All apps' shows 73 apps installed. Need to find Duolingo to manage notifications.
<IMAGE>
[step 3] In Settings > Apps > All apps, list shows apps like Advanced Security Lite, Android Auto, Android System WebView, AP News, Apartment List, Apartments.
<IMAGE>
```

C1 output → `TYPE: Duolingo`:
```
<think>To manage notifications for Duolingo, I need to locate the app in the list. Using the search function will help me quickly find Duolingo among the 73 installed apps.</think>
<action>Type(content='Duolingo')</action>
<conclusion>Type 'Duolingo' into the search bar to locate the app quickly.</conclusion>
```
C8 output → `TYPE: Tokopedia`:
```
<think>To manage notifications for the Tokopedia app, I need to locate it in the list of installed apps. Using the search function will help me quickly find Tokopedia without scrolling through all 73 apps.</think>
<action>Type(content='Tokopedia')</action>
<conclusion>Type 'Tokopedia' into the search bar to locate the app quickly.</conclusion>
```

## 6. Decision

Nothing is broken: controller and backbone formats are clean, the coordinate and scroll conventions check out, and
throughput fits the budget. **Proceed to the full run (n = 600)** with the change in §4.3.
