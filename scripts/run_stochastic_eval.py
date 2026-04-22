"""Run stochastic evaluation for all 9 (seed, algo) in parallel."""
from __future__ import annotations
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PY = sys.executable
SEEDS = [0, 1, 2]
ALGOS = ["dqn", "ppo", "sac"]
RESULTS = Path("results")
MODELS = Path("models")
LOGS = RESULTS / "_logs"
LOGS.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    env[k] = "1"

jobs = []
for s in SEEDS:
    for a in ALGOS:
        cmd = [PY, "-m", "drone_rl.evaluate",
               "--algo", a, "--episodes", "30",
               "--seed", str(2000 + s),
               "--stochastic",
               "--model-path", str(MODELS / f"seed_{s}" / f"{a}_drone_navigation"),
               "--output-dir", str(RESULTS / f"seed_{s}")]
        jobs.append((f"s={s} {a}", cmd, LOGS / f"evalstoch_seed{s}_{a}.log"))

print(f"Launching {len(jobs)} stochastic eval jobs in parallel=9")
t0 = time.time()

def run(tag, cmd, log):
    with log.open("w", encoding="utf-8") as f:
        f.write(">>> " + " ".join(cmd) + "\n"); f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return tag, r.returncode, log

with ThreadPoolExecutor(max_workers=9) as pool:
    futs = {pool.submit(run, *j): j[0] for j in jobs}
    done = 0
    for f in as_completed(futs):
        tag, rc, log = f.result()
        done += 1
        st = "OK " if rc == 0 else f"FAIL({rc})"
        print(f"  [{done}/{len(jobs)}] {st} {tag} elapsed={time.time()-t0:.1f}s")
        if rc:
            print(log.read_text(encoding='utf-8')[-2000:])
print("done")
