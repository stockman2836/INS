"""Retrain with more timesteps (DQN/PPO long, SAC moderate) in parallel."""
from __future__ import annotations
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PY = sys.executable
SEEDS = [0, 1, 2]
RESULTS = Path("results")
MODELS = Path("models")
LOGS = RESULTS / "_logs"
LOGS.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    env[k] = "1"


def run_logged(tag, cmd, log):
    with log.open("w", encoding="utf-8") as f:
        f.write(">>> " + " ".join(cmd) + "\n"); f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return tag, r.returncode, log


def parallel(jobs, workers, label):
    print(f"\n[{label}] {len(jobs)} jobs, parallel={workers}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_logged, *j): j[0] for j in jobs}
        done = 0
        for f in as_completed(futs):
            tag, rc, log = f.result()
            done += 1
            st = "OK " if rc == 0 else f"FAIL({rc})"
            print(f"  [{done}/{len(jobs)}] {st} {tag} elapsed={time.time()-t0:.1f}s", flush=True)
            if rc:
                tail = log.read_text(encoding='utf-8', errors='replace')[-1500:]
                print(tail)
                raise SystemExit(f"FAIL {tag}")


ALGO_STEPS = {"dqn": 200000, "ppo": 200000, "sac": 30000}

# Train in parallel. Put fast algos first so slow SAC has full 3 slots.
train_jobs = []
for a in ("dqn", "ppo", "sac"):
    for s in SEEDS:
        cmd = [PY, "-m", "drone_rl.train",
               "--algo", a, "--timesteps", str(ALGO_STEPS[a]),
               "--seed", str(s),
               "--output-dir", str(MODELS / f"seed_{s}"),
               "--results-dir", str(RESULTS / f"seed_{s}")]
        train_jobs.append((f"train s={s} {a}", cmd, LOGS / f"train_seed{s}_{a}.log"))
parallel(train_jobs, workers=9, label="train")

# Deterministic eval.
eval_det = []
for s in SEEDS:
    for a in ("dqn", "ppo", "sac"):
        cmd = [PY, "-m", "drone_rl.evaluate",
               "--algo", a, "--episodes", "30",
               "--seed", str(1000 + s),
               "--model-path", str(MODELS / f"seed_{s}" / f"{a}_drone_navigation"),
               "--output-dir", str(RESULTS / f"seed_{s}")]
        eval_det.append((f"eval-det s={s} {a}", cmd, LOGS / f"eval_seed{s}_{a}.log"))
parallel(eval_det, workers=9, label="eval-deterministic")

# Stochastic eval.
eval_st = []
for s in SEEDS:
    for a in ("dqn", "ppo", "sac"):
        cmd = [PY, "-m", "drone_rl.evaluate",
               "--algo", a, "--episodes", "30",
               "--seed", str(2000 + s),
               "--stochastic",
               "--model-path", str(MODELS / f"seed_{s}" / f"{a}_drone_navigation"),
               "--output-dir", str(RESULTS / f"seed_{s}")]
        eval_st.append((f"eval-stoch s={s} {a}", cmd, LOGS / f"evalstoch_seed{s}_{a}.log"))
parallel(eval_st, workers=9, label="eval-stochastic")

# Per-seed comparison and plots.
first = RESULTS / "seed_0"
subprocess.run([PY, "-m", "drone_rl.compare", "--results-dir", str(first)], check=True)

import shutil
plots_src = first / "plots"
plots_dst = RESULTS / "plots"
if plots_src.exists():
    if plots_dst.exists():
        shutil.rmtree(plots_dst)
    shutil.copytree(plots_src, plots_dst)
for name in ("comparison.csv", "comparison.md"):
    src = first / name
    if src.exists():
        shutil.copy2(src, RESULTS / name)

# Aggregate both eval modes.
subprocess.run([PY, "-m", "drone_rl.compare_seeds", "--results-dir", str(RESULTS)], check=True)
subprocess.run([PY, "-m", "drone_rl.compare_seeds", "--results-dir", str(RESULTS),
                "--suffix", "_stochastic"], check=True)
print("all done")
