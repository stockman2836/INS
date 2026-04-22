"""Aggregate metrics across seeds.

Expects directory layout::

    results/
        seed_<N>/
            <algo>_evaluation.json
            <algo>_training_metrics.json

Produces ``results/comparison_seeds.csv`` and ``results/comparison_seeds.md``
with mean and standard deviation per algorithm across seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ALGORITHMS = ["dqn", "ppo", "sac"]

EVAL_KEYS = [
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "average_steps",
    "average_steps_on_success",
    "average_path_length",
    "average_path_length_on_success",
    "average_reward",
    "path_length_excess_ratio_on_success",
]

TRAIN_KEYS = [
    "episodes",
    "collisions",
    "successes",
    "timeouts",
    "collision_rate",
    "success_rate",
    "timeout_rate",
]


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan")
    n = len(clean)
    mean = sum(clean) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)
    return mean, math.sqrt(var)


def collect(results_dir: Path, suffix: str = "") -> dict[str, dict]:
    seed_dirs = sorted([p for p in results_dir.glob("seed_*") if p.is_dir()])
    per_algo: dict[str, dict[str, list[float]]] = {
        a: {k: [] for k in EVAL_KEYS + [f"train_{k}" for k in TRAIN_KEYS]}
        for a in ALGORITHMS
    }
    per_algo_seeds: dict[str, list[str]] = {a: [] for a in ALGORITHMS}

    for seed_dir in seed_dirs:
        for algo in ALGORITHMS:
            eval_path = seed_dir / f"{algo}{suffix}_evaluation.json"
            train_path = seed_dir / f"{algo}_training_metrics.json"
            if not eval_path.exists():
                continue
            with eval_path.open("r", encoding="utf-8") as handle:
                eval_data = json.load(handle)
            for key in EVAL_KEYS:
                per_algo[algo][key].append(eval_data.get(key))
            if train_path.exists():
                with train_path.open("r", encoding="utf-8") as handle:
                    train_data = json.load(handle)
                for key in TRAIN_KEYS:
                    per_algo[algo][f"train_{key}"].append(train_data.get(key))
            per_algo_seeds[algo].append(seed_dir.name)

    summary: dict[str, dict] = {}
    for algo in ALGORITHMS:
        summary[algo] = {"seeds": per_algo_seeds[algo]}
        for key, values in per_algo[algo].items():
            mean, std = mean_std(values)
            summary[algo][f"{key}_mean"] = mean
            summary[algo][f"{key}_std"] = std
    return summary


def format_cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.3f}"
    return str(value)


def write_tables(summary: dict[str, dict], results_dir: Path, suffix: str = "") -> tuple[Path, Path]:
    columns = [
        ("algorithm", "Algorithm"),
        ("seeds", "Seeds"),
        ("train_collisions", "Train collisions"),
        ("train_collision_rate", "Train collision rate"),
        ("train_success_rate", "Train success rate"),
        ("success_rate", "Eval success rate"),
        ("collision_rate", "Eval collision rate"),
        ("timeout_rate", "Eval timeout rate"),
        ("average_steps_on_success", "Avg steps (success)"),
        ("average_path_length_on_success", "Avg path (success)"),
        ("path_length_excess_ratio_on_success", "Path / A*"),
        ("average_reward", "Avg reward"),
    ]

    rows: list[dict[str, str]] = []
    for algo in ALGORITHMS:
        if algo not in summary:
            continue
        algo_summary = summary[algo]
        row = {"algorithm": algo, "seeds": ",".join(algo_summary["seeds"]) or "-"}
        for key, _ in columns[2:]:
            mean = algo_summary.get(f"{key}_mean")
            std = algo_summary.get(f"{key}_std")
            if mean is None or (isinstance(mean, float) and math.isnan(mean)):
                row[key] = "-"
            else:
                row[key] = f"{format_cell(mean)} +/- {format_cell(std)}"
        rows.append(row)

    csv_path = results_dir / f"comparison_seeds{suffix}.csv"
    md_path = results_dir / f"comparison_seeds{suffix}.md"

    keys = [key for key, _ in columns]
    headers = [label for _, label in columns]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[k] for k in keys) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics across seeds.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing seed_<N>/ subdirectories.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix applied to evaluation filenames (e.g. '_stochastic').",
    )
    args = parser.parse_args()

    summary = collect(args.results_dir, suffix=args.suffix)

    # Write raw JSON for further analysis.
    raw_path = args.results_dir / f"comparison_seeds{args.suffix}.json"
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    csv_path, md_path = write_tables(summary, args.results_dir, suffix=args.suffix)
    print(f"Wrote {raw_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
