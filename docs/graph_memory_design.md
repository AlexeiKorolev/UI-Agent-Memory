# Design draft: within-episode task graph memory (follow-up 2)

Status: **draft, not implemented** (2026-10-06). Constraints unchanged: frozen UI-Venus-1.5-8B backbone, prompted
(not trained) Qwen3-VL-8B controller, public data only, Slurm jobs <= 4 h.

## Motivation (from our results)
* C1 notes describe screens, not actions; adding the action list fixes much of the state error (TEXT steps predicted
  as CLICK 39.5% -> 19.8%; AMS 66.3 -> 71.8). Action/event records should be first-class, not left to the note writer.
* Notes are vague and get merged away (salience saturated: ~89% of entries >= 0.8; writes on 98% of steps).
* Raw screenshots + actions (A2, 76.6 AMS) is the bar to beat; memory hurts copying from the current screen
  (A1 51.2 vs A0 68.6 on strings visible now, n=86), so the format should not crowd out the screen.
* The backbone types many strings from priors, so memory value must be measured against A0 (actions only) and A7.

## Graph schema (one graph per episode, built along the GT trajectory, state cached *before* step t as now)
| Node | Fields | Written by |
|---|---|---|
| `Task` | instruction | fixed |
| `Subgoal` | text, status (todo / active / done), order | controller (decomposed at step 0, status updated per step) |
| `Event` | step k, app, action string (GT, deterministic), one-line result | action: code; result: controller |
| `Value` | exact string, type (name/number/date/url/text), role ("company to type in Word"), source step, crop | controller (only when a task-relevant value is visible) |

Edges: `Event -advances-> Subgoal`, `Value -observed_at-> Event`, `Value -needed_for-> Subgoal`,
`Subgoal -next-> Subgoal`. Values are never merged or summarised (no cap; values are short); events beyond the last 20
collapse to their action strings only.

Controller output per step = a small list of JSON ops (`add_value`, `set_status`, `link`, `event_result`), validated
and applied by code (so the graph stays well-formed and every edit is auditable; retries as now).

## Serialisation into the "### Previous Actions" slot
```
Progress:  [done] find a graphic-designer job on Indeed  [active] write the company name in Word  [todo] ...
Facts:     company = "Innomark Communications"  (needed for: write the company name in Word; seen at step 5) <crop>
Actions:   [step 0] Click(box=(583,220)) -> opened Indeed  ...  [step 9] Click(box=(120,380)) -> blank document opened
```
(Crops attached only to Value nodes, max 4, chosen by `needed_for` the active subgoal, then recency.)

## Conditions
| ID | Content | Question |
|---|---|---|
| A0, A1, A2 | (existing) actions only; notes + actions; screenshots + actions | baselines |
| G1 | full graph | does structured memory beat A1? approach A2? |
| G-flat | same nodes as G1, no edges/roles/status (a flat list) | does *structure* matter, or just the extracted content? |
| G-noP / G-noF | G1 without Progress / without Facts | which part drives which error type |
| G+S | G1 + last 2 raw screenshots | hybrid |
| G7 | another episode's graph (same category, matched size) + the recipient's own actions | presence control |
| G8v | MD steps: value node string -> alternative (as C8) | does the agent follow edited values? |
| G8r | swap `needed_for` links between two Value nodes of the same type | does the agent use **relations**? (follow = types the re-linked value) |
| G8s | mark the active subgoal `done` (and the next one active) | does the agent use **state**? (does its action move on / finish early?) |

G8r and G8s are the new causal tests a graph makes possible: they edit structure, not strings.

## Metrics
All-step AMS (official), MD and strict-MD text accuracy **with A0 as the prior baseline**, follow rates for
G8v/G8r, action change rate and early-COMPLETE rate for G8s, TEXT-step click rate, COMPLETE accuracy, accuracy by step
index, and the exact-vs-fuzzy text split (`src/prior_knowledge.py`). Episode bootstrap CIs and paired McNemar as now.

## Protocol (avoid tuning on the reported set)
1. **Dev set** = episodes 600-899 of the same stratified order (`sample(900)[600:]`): needs OCR + MD labels (CPU),
   the graph controller (~1 GPU-h) and backbone runs (~0.5 GPU-h per condition). Iterate controller prompts here only.
2. Freeze prompts; pre-register hypotheses below; run once on the n=600 test episodes (graph controller ~2 GPU-h,
   ~1 GPU-h per backbone condition). Never rebuild the main-study specs/cache.
3. Pre-registered: **HG1** G1 > A1 on strict-MD text acc and all-step AMS. **HG2** G1 > G-flat (structure matters).
   **HG3** G8r follow rate > 0 and > C8-style spontaneous rate (~0). **HG4** G8s raises early-COMPLETE / skip rate vs G1.
   G1 vs A2 reported, not predicted.

## Open issues
* Teacher-forced: memory and actions come from the GT trajectory. A closed-loop variant (the agent's own predicted
  actions feed the graph, screens still GT) is needed before claiming deployment gains.
* Subgoal decomposition by a prompted 8B controller may be noisy; log and hand-check 30 graphs on dev (as for MD labels).
* Value extraction depends on the controller reading text correctly; reuse OCR to verify/copy exact strings.
