"""Tally compute used by this project's Slurm jobs (job names mg_*), from sacct."""
import subprocess, sys, collections
since = sys.argv[1] if len(sys.argv) > 1 else "2026-10-02"
out = subprocess.check_output(["sacct", "-X", "-n", "-P", "--starttime", since, "-o",
                               "JobID,JobName,State,ElapsedRaw,AllocTRES,NodeList"], text=True)
gpu_h, cpu_h, rows = collections.Counter(), collections.Counter(), []
for line in out.strip().splitlines():
    jid, name, state, el, tres, node = line.split("|")
    if not name.startswith("mg_"):
        continue
    h = int(el or 0) / 3600
    t = dict(kv.split("=", 1) for kv in tres.split(",") if "=" in kv)
    ngpu = int(t.get("gres/gpu", 0)); ncpu = int(t.get("cpu", 0))
    gpu_h[name] += h * ngpu; cpu_h[name] += h * ncpu
    rows.append((jid, name, state, round(h, 3), ngpu, ncpu, node))
for r in rows:
    print(*r, sep="\t")
print("\nGPU-hours by stage:", {k: round(v, 2) for k, v in gpu_h.items()}, " total:", round(sum(gpu_h.values()), 2))
print("CPU-core-hours by stage:", {k: round(v, 1) for k, v in cpu_h.items()}, " total:", round(sum(cpu_h.values()), 1))
