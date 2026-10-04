# CLAUDE.md

This is a thesis experiment: **do GUI agents causally use their memory?** (MementoGUI-style intervention study on
GUI-Odyssey; frozen UI-Venus-1.5-8B backbone, prompted Qwen3-VL-8B memory controller, Princeton Adroit/Slurm).

**Start by reading [`CONTEXT.md`](CONTEXT.md) §4 "PICK UP HERE"**: current job IDs, how to check them, recovery steps,
and the ordered list of remaining work. The design and all decisions are in CONTEXT.md §1–3, the original spec is in
`docs/task_spec.md`, and the dated history is in `LOG.md`.

Working rules for this project:
- Never invent numbers. Investigate numbers that look too good/bad (parsing, coordinate convention, leakage) before
  reporting them. Report failures as they are.
- Don't modify or fine-tune the backbone; the controller is prompted only; no API models; no non-public data.
- Heavy compute only through Slurm (`sbatch`, submit from the repo root, `--mail-user=<addresses from the princeton-adroit skill>` on the command line). Keep GPU
  jobs ≤ 4 h (gpu-short QOS) so they don't compete with the user's own long jobs. Login node = editing, downloads, light checks.
- Keep all caches inside the project (`source env.sh`); `/home` is near its quota.
- Don't re-run the controller or condition build for n=600: backbone outputs depend on those exact specs.
- Keep `LOG.md` updated (dated entries) and commit often. The GitHub remote is **public**: before pushing, `git grep` the tree for the
  user's NetID, email addresses and absolute home/scratch paths, and keep personal info out of tracked files.
- Commit attribution line: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
