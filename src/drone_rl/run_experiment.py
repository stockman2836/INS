"""Run full experiment: 3 algorithms x N seeds, train + evaluate, then aggregate.

Usage:
    python -m drone_rl.run_experiment --timesteps 40000 --seeds 0 1 2 --episodes 30 --parallel 6
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ALGORITHMS = ["dqn", "ppo", "sac"]


def _run_logged(cmd: list[str], log_path: Path, env: dict) -> tuple[list[str], int, Path]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(">>> " + " ".join(cmd) + "\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return cmd, result.returncode, log_path


def run_serial(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with code {result.returncode}: {cmd}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=40000)
    parser.add_argument(
        "--sac-timesteps",
        type=int,
        default=None,
        help="Override timesteps for SAC only (defaults to --timesteps).",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--clean", action="store_true", help="Wipe models/ and results/ first.")
    parser.add_argument(
        "--parallel",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of concurrent train/eval workers.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
        help="OMP/MKL/torch threads per child process.",
    )
    args = parser.parse_args()

    if args.clean:
        for path in (args.models_root, args.results_root):
            if path.exists():
                shutil.rmtree(path)
    args.models_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    logs_root = args.results_root / "_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    tpw = str(max(1, args.threads_per_worker))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        child_env[k] = tpw

    # Build training jobs.
    train_jobs = []
    for seed in args.seeds:
        models_dir = args.models_root / f"seed_{seed}"
        results_dir = args.results_root / f"seed_{seed}"
        for algo in ALGORITHMS:
            algo_timesteps = (
                args.sac_timesteps
                if (algo == "sac" and args.sac_timesteps is not None)
                else args.timesteps
            )
            cmd = [
                python, "-m", "drone_rl.train",
                "--algo", algo,
                "--timesteps", str(algo_timesteps),
                "--seed", str(seed),
                "--output-dir", str(models_dir),
                "--results-dir", str(results_dir),
            ]
            log_path = logs_root / f"train_seed{seed}_{algo}.log"
            train_jobs.append((f"train seed={seed} algo={algo}", cmd, log_path))

    print(f"Launching {len(train_jobs)} training jobs with parallel={args.parallel}, "
          f"threads-per-worker={tpw}.", flush=True)
    t0 = time.time()
    done_count = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {
            pool.submit(_run_logged, cmd, log, child_env): label
            for (label, cmd, log) in train_jobs
        }
        for fut in as_completed(futs):
            label = futs[fut]
            cmd, rc, log = fut.result()
            done_count += 1
            elapsed = time.time() - t0
            status = "OK " if rc == 0 else f"FAIL({rc})"
            print(f"  [{done_count}/{len(train_jobs)}] {status} {label} "
                  f"elapsed={elapsed:.1f}s log={log}", flush=True)
            if rc != 0:
                raise SystemExit(f"Training failed: {label} (see {log})")

    # Build evaluation jobs.
    eval_jobs = []
    for seed in args.seeds:
        models_dir = args.models_root / f"seed_{seed}"
        results_dir = args.results_root / f"seed_{seed}"
        for algo in ALGORITHMS:
            cmd = [
                python, "-m", "drone_rl.evaluate",
                "--algo", algo,
                "--episodes", str(args.episodes),
                "--seed", str(1000 + seed),
                "--model-path", str(models_dir / f"{algo}_drone_navigation"),
                "--output-dir", str(results_dir),
            ]
            log_path = logs_root / f"eval_seed{seed}_{algo}.log"
            eval_jobs.append((f"eval seed={seed} algo={algo}", cmd, log_path))

    print(f"\nLaunching {len(eval_jobs)} evaluation jobs with parallel={args.parallel}.",
          flush=True)
    t0 = time.time()
    done_count = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {
            pool.submit(_run_logged, cmd, log, child_env): label
            for (label, cmd, log) in eval_jobs
        }
        for fut in as_completed(futs):
            label = futs[fut]
            cmd, rc, log = fut.result()
            done_count += 1
            elapsed = time.time() - t0
            status = "OK " if rc == 0 else f"FAIL({rc})"
            print(f"  [{done_count}/{len(eval_jobs)}] {status} {label} "
                  f"elapsed={elapsed:.1f}s log={log}", flush=True)
            if rc != 0:
                raise SystemExit(f"Evaluation failed: {label} (see {log})")

    # Per-seed comparison and trajectory plots (uses first seed dir for plots).
    first_seed_dir = args.results_root / f"seed_{args.seeds[0]}"
    run_serial([python, "-m", "drone_rl.compare", "--results-dir", str(first_seed_dir)])

    # Copy the plots from the first seed up one level for README use.
    plots_src = first_seed_dir / "plots"
    plots_dst = args.results_root / "plots"
    if plots_src.exists():
        if plots_dst.exists():
            shutil.rmtree(plots_dst)
        shutil.copytree(plots_src, plots_dst)

    # Copy first-seed comparison tables to the top level.
    for name in ("comparison.csv", "comparison.md"):
        src = first_seed_dir / name
        if src.exists():
            shutil.copy2(src, args.results_root / name)

    # Seed aggregation.
    run_serial([python, "-m", "drone_rl.compare_seeds", "--results-dir", str(args.results_root)])

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
