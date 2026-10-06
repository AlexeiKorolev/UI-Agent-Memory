# Related work: structured / graph memory for GUI agents (survey snapshot, 2026-10-06)

Scope: is there prior work that combines a **knowledge graph** with **within-task fact memory** and **state tracking**
for GUI agents, and does anyone test **causally** whether the agent uses memory content? Snapshot from web search +
arXiv abstracts (full texts not read unless noted); re-check before citing, the area moves monthly.

## 1. Graphs of *app knowledge* across many episodes (navigation)
Nodes = screens/UI states, edges = actions; built by offline exploration or from trajectories; retrieved at run time.
They answer "how do I get to X in this app", not "what value did I see three steps ago".

| Work | What it stores | Notes |
|---|---|---|
| KG-RAG (EMNLP 2025, arXiv 2509.00366) | UI transition graphs -> vector DB for retrieval | app-specific navigation knowledge |
| UI-KOBE (arXiv 2605.29534) | app KG: nodes UI states, edges executable transitions | small agent localises itself in the graph |
| GraphPilot (arXiv 2601.17418) | app KG for one-step planning | |
| Executable Agentic Memory (arXiv 2605.12294) | "GUI Logic KG" G=(S,A,E) from state-aware DFS + action-group mining | retrieval-and-execution planning |
| HyMEM (arXiv 2603.10291) | graph of symbolic nodes + trajectory embeddings, self-evolving | cross-task experience |
| Agent-SAMA (AAAI-26, arXiv 2505.23596) | finite-state machine: screens = states, actions = transitions | progress/recovery; Mobile-Eval-E 84.0%, AndroidWorld 63.7% |
| Hierarchical Memory Tree (arXiv 2603.07024, web) | intent -> stage (subgoals with entry/exit conditions) -> action patterns | cross-website generalisation (Mind2Web, WebArena) |

## 2. Within-task *state* memory (closest to our failure modes)
| Work | Representation | Graph? | Facts/values? | Causal test of use? |
|---|---|---|---|---|
| TSR (arXiv 2607.00502) | JSON: global task summary, progress tracker (done/remaining subgoals), transition-aware action check | no | partly (milestones, requirements) | no; API backbones; up to +12% SR on MobileWorld |
| ATMem (arXiv 2606.31612) | "active task-driving memory": hierarchical execution state linking values to roles/status | no (hierarchy) | yes | memory-on vs memory-off rollouts (presence, not content) |
| MemGUI-Agent (arXiv 2606.19926) | Context-as-Action: folded **action history**, folded UI state, recent step | no | "critical UI facts" | no; SFT 8B |
| MGA (arXiv 2510.24168) | verified state deltas -> state-transition chain | chain | yes (state facts) | no; OSWorld |
| AGMem (arXiv 2606.14106, ICML'26 workshop) | crops of regions tied to successful actions/recoveries | no | visual | failure taxonomy: full-image memory reduces state failures but worsens action failures; +33.3% SR over full-image memory on OSWorld |
| Mem-W (arXiv 2605.09317) | latent memory tokens (experiential + working) | no | latent | no |
| MementoGUI (arXiv 2605.18652) | text summaries + ROI crops, trained controller | no | yes | no (the paper this project tests) |

## 3. Graph memory with fact retrieval and temporal state (non-GUI, mostly chat)
* **Zep / Graphiti**: temporal KG; facts carry validity intervals (when true, when superseded) -> state over time.
* **Mem0 (graph variant)**: extract/update salient facts into a graph; **A-MEM**: Zettelkasten-style linked notes;
  **HippoRAG (v2)**: passage-entity-fact graph + Personalized PageRank; **MAGMA** (arXiv 2601.03236): multiple graphs.
* **MemPrism** (arXiv 2608.06745): event stream + task-conditioned *relational views* composed at decision time;
  embodied and web benchmarks.
* **StateMem / "Can Agent Memory Systems Track Evolving State?"** (arXiv 2608.19652): existing memory systems track
  changing state poorly in multi-session dialogue; a structural state wrapper gives +32 to +67 points.

## Gap (as far as this snapshot shows)
1. No GUI-agent work found that keeps a **within-episode graph** joining *values* (with roles), *actions/events* and
   *subgoal status*; GUI graphs are cross-episode navigation maps, and state trackers are flat JSON / hierarchies.
2. No GUI memory work found that **edits memory content** and measures whether actions follow (our C7/C8 design);
   ATMem's on/off contrast and MementoGUI's ablations measure presence, not content use.
3. Our own results add two points relevant to all of the above: (a) replacing the action history with memory notes
   hides "what was done" (A0 actions-only 68.2 AMS > C1 notes 66.3); (b) the backbone types many strings from priors
   (31.5% exact on strings that appear in no input), so "memory-dependent" accuracy needs a prior baseline (A0).
