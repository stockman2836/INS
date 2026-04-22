"""Resume the experiment: evaluate already-trained DQN/PPO in parallel,
then train+evaluate SAC with higher threads/proc, then aggregate."""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SEEDS = [0, 1, 2]
RESULTS = Path("results")
MODELS = Path("models")
LOGS = RESULTS / "_logs"
LOGS.mkdir(parents=True, exist_ok=True)
PY = sys.executable


def run_logged(cmd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(">>> " + " ".join(cmd) + "\n"); f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return cmd, r.returncode, log_path


def run_serial(cmd):
    print(">>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        raise SystemExit(f"FAIL {cmd}")


def parallel(jobs, workers, env, label):
    print(f"\n[{label}] launching {len(jobs)} jobs, parallel={workers}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_logged, c, l, env): tag for (tag, c, l) in jobs}
        done = 0
        for fut in as_completed(futs):
            tag = futs[fut]
            cmd, rc, log = fut.result()
            done += 1
            el = time.time() - t0
            st = "OK " if rc == 0 else f"FAIL({rc})"
            print(f"  [{done}/{len(jobs)}] {st} {tag} elapsed={el:.1f}s", flush=True)
            if rc:
                raise SystemExit(f"FAIL {tag}, see {log}")


def env_with_threads(n):
    e = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        e[k] = str(n)
    return e


def main():
    # 1) DQN+PPO eval in parallel (6 tasks, 1 thread each).
    eval_jobs = []
    for seed in SEEDS:
        for algo in ("dqn", "ppo"):
            cmd = [PY, "-m", "drone_rl.evaluate",
                   "--algo", algo, "--episodes", "30",
                   "--seed", str(1000 + seed),
                   "--model-path", str(MODELS / f"seed_{seed}" / f"{algo}_drone_navigation"),
                   "--output-dir", str(RESULTS / f"seed_{seed}")]
            eval_jobs.append((f"eval s={seed} {algo}", cmd, LOGS / f"eval_seed{seed}_{algo}.log"))
    parallel(eval_jobs, workers=6, env=env_with_threads(1), label="DQN+PPO eval")

    # 2) SAC train x 3 seeds, parallel=3, 4 threads each.
    sac_train = []
    for seed in SEEDS:
        cmd = [PY, "-m", "drone_rl.train",
               "--algo", "sac", "--timesteps", "15000",
               "--seed", str(seed),
               "--output-dir", str(MODELS / f"seed_{seed}"),
               "--results-dir", str(RESULTS / f"seed_{seed}")]
        sac_train.append((f"train s={seed} sac", cmd, LOGS / f"train_seed{seed}_sac.log"))
    parallel(sac_train, workers=3, env=env_with_threads(4), label="SAC train")

    # 3) SAC eval x 3 seeds, parallel=3.
    sac_eval = []
    for seed in SEEDS:
        cmd = [PY, "-m", "drone_rl.evaluate",
               "--algo", "sac", "--episodes", "30",
               "--seed", str(1000 + seed),
               "--model-path", str(MODELS / f"seed_{seed}" / "sac_drone_navigation"),
               "--output-dir", str(RESULTS / f"seed_{seed}")]
        sac_eval.append((f"eval s={seed} sac", cmd, LOGS / f"eval_seed{seed}_sac.log"))
    parallel(sac_eval, workers=3, env=env_with_threads(2), label="SAC eval")

    # 4) compare + plots on seed_0.
    first = RESULTS / "seed_0"
    run_serial([PY, "-m", "drone_rl.compare", "--results-dir", str(first)])
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
    # 5) seed aggregation.
    run_serial([PY, "-m", "drone_rl.compare_seeds", "--results-dir", str(RESULTS)])
    print("\nDone.")


if __name__ == "__main__":
    main()
