"""Aggregate training + evaluation metrics for DQN, PPO, SAC.

Reads ``results/<algo>_evaluation.json`` and
``results/<algo>_training_metrics.json`` for all three algorithms and
produces:

- ``results/comparison.csv`` and ``results/comparison.md`` tables,
- ``results/plots/trajectories_<algo>.png`` 3D plots showing the
  optimal A* path, obstacles and a sample of evaluated trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from drone_rl.envs import DroneNavigationEnv


ALGORITHMS = ["dqn", "ppo", "sac"]

COLUMNS = [
    ("algorithm", "Algorithm"),
    ("train_episodes", "Train episodes"),
    ("train_collisions", "Train collisions"),
    ("train_successes", "Train successes"),
    ("train_timeouts", "Train timeouts"),
    ("train_collision_rate", "Train collision rate"),
    ("eval_success_rate", "Eval success rate"),
    ("eval_collision_rate", "Eval collision rate"),
    ("eval_timeout_rate", "Eval timeout rate"),
    ("avg_steps_on_success", "Avg steps (success)"),
    ("avg_path_on_success", "Avg path (success)"),
    ("optimal_astar_path_length", "A* optimal path"),
    ("path_excess_ratio", "Path / A*"),
]


def load_row(algorithm: str, results_dir: Path) -> dict:
    eval_path = results_dir / f"{algorithm}_evaluation.json"
    train_path = results_dir / f"{algorithm}_training_metrics.json"

    if not eval_path.exists():
        raise FileNotFoundError(f"Missing {eval_path}. Run evaluate.py first.")

    with eval_path.open("r", encoding="utf-8") as handle:
        eval_data = json.load(handle)

    train_data = {}
    if train_path.exists():
        with train_path.open("r", encoding="utf-8") as handle:
            train_data = json.load(handle)

    return {
        "algorithm": algorithm,
        "train_episodes": train_data.get("episodes"),
        "train_collisions": train_data.get("collisions"),
        "train_successes": train_data.get("successes"),
        "train_timeouts": train_data.get("timeouts"),
        "train_collision_rate": train_data.get("collision_rate"),
        "eval_success_rate": eval_data.get("success_rate"),
        "eval_collision_rate": eval_data.get("collision_rate"),
        "eval_timeout_rate": eval_data.get("timeout_rate"),
        "avg_steps_on_success": eval_data.get("average_steps_on_success"),
        "avg_path_on_success": eval_data.get("average_path_length_on_success"),
        "optimal_astar_path_length": eval_data.get("optimal_astar_path_length"),
        "path_excess_ratio": eval_data.get("path_length_excess_ratio_on_success"),
    }


def format_cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.3f}"
    return str(value)


def write_tables(rows: list[dict], results_dir: Path) -> tuple[Path, Path]:
    csv_path = results_dir / "comparison.csv"
    md_path = results_dir / "comparison.md"

    keys = [key for key, _ in COLUMNS]
    headers = [label for _, label in COLUMNS]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row[k]) for k in keys) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, md_path


def plot_trajectories(algorithm: str, results_dir: Path, max_episodes: int = 5) -> Path | None:
    eval_path = results_dir / f"{algorithm}_evaluation.json"
    if not eval_path.exists():
        return None
    with eval_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    env = DroneNavigationEnv()
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Obstacles as wireframe boxes.
    for obstacle in env.obstacles:
        c = obstacle.center
        h = obstacle.half_size
        corners = np.array(
            [
                [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
                [c[0] + h[0], c[1] - h[1], c[2] - h[2]],
                [c[0] + h[0], c[1] + h[1], c[2] - h[2]],
                [c[0] - h[0], c[1] + h[1], c[2] - h[2]],
                [c[0] - h[0], c[1] - h[1], c[2] + h[2]],
                [c[0] + h[0], c[1] - h[1], c[2] + h[2]],
                [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
                [c[0] - h[0], c[1] + h[1], c[2] + h[2]],
            ]
        )
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                color="black",
                linewidth=0.8,
            )

    # Optimal A* waypoints.
    waypoints = data.get("optimal_waypoints") or []
    if waypoints:
        wp = np.array(waypoints)
        ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], color="green", linewidth=2.0, label="A* optimal")

    # Agent trajectories.
    trajectories = data.get("trajectories") or []
    shown = 0
    for i, trajectory in enumerate(trajectories):
        if shown >= max_episodes:
            break
        t = np.array(trajectory)
        if t.size == 0:
            continue
        ax.plot(t[:, 0], t[:, 1], t[:, 2], alpha=0.6, label=f"ep {i + 1}")
        shown += 1

    ax.scatter(*env.start, color="blue", s=50, label="start")
    ax.scatter(*env.goal, color="red", s=50, label="goal")

    ax.set_xlim(-env.world_size / 2, env.world_size / 2)
    ax.set_ylim(-env.world_size / 2, env.world_size / 2)
    ax.set_zlim(-env.world_size / 2, env.world_size / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{algorithm.upper()} trajectories vs A* optimum")
    ax.legend(loc="upper left", fontsize=8)

    plot_path = plots_dir / f"trajectories_{algorithm}.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Aggregate DQN/PPO/SAC results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory with per-algorithm JSON files.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=5,
        help="How many episode trajectories to draw per algorithm.",
    )
    args = parser.parse_args()

    rows = []
    for algorithm in ALGORITHMS:
        try:
            rows.append(load_row(algorithm, args.results_dir))
        except FileNotFoundError as error:
            print(f"Skipping {algorithm}: {error}")

    if not rows:
        print("No evaluation results found. Run train.py and evaluate.py first.")
        return

    csv_path, md_path = write_tables(rows, args.results_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    for algorithm in ALGORITHMS:
        plot_path = plot_trajectories(algorithm, args.results_dir, args.max_trajectories)
        if plot_path is not None:
            print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
