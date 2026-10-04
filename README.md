# Do GUI agents causally use their memory?

A MementoGUI-style intervention study on GUI-Odyssey (v2): a prompted Qwen3-VL-8B controller builds a working memory
(text notes + ROI crops) along ground-truth trajectories; the frozen UI-Venus-1.5-8B backbone is evaluated under memory
interventions (none / clean / all screenshots / text-only / crops-only / blank crops / shuffled crops / other episode's
memory / targeted counterfactual edits) to test whether it uses memory *content* or merely benefits from its *presence*.

**Status (2026-10-03):** pilot done; full n=600 run in progress (controller and condition build done, backbone running). See [`CONTEXT.md`](CONTEXT.md) (§4 = pick-up point) for
the full design, decisions, verified facts, current status and the step-by-step run guide; [`LOG.md`](LOG.md) for the
dated log; [`docs/task_spec.md`](docs/task_spec.md) for the original specification.

## Quick start

```bash
bash scripts/setup_env.sh                 # login node: conda env (vllm 0.30)
bash scripts/download.sh                  # login node: models, data, screenshots for 600 sampled episodes
source env.sh && python -m pytest -q tests
mkdir -p logs && sbatch --mail-user=<you> slurm/pilot.sbatch     # adapt #SBATCH partition/GPU lines to your cluster
```

Full pipeline: `controller.sbatch 600` → `build_conditions.sbatch 600` → `backbone.sbatch 600` → `analyze.sbatch 600` →
`python -m src.figures` (details in CONTEXT.md §5).

Not an exact reproduction of MementoGUI (no released code/weights; prompted controller; no episodic memory).
