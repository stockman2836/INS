from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO, SAC

from drone_rl.envs import DroneNavigationEnv
from drone_rl.planning import compute_optimal_path


ALGORITHMS = {
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
}


def action_mode_for_algorithm(algorithm_name: str) -> str:
    if algorithm_name == "dqn":
        return "discrete"
    return "continuous"


def evaluate_model(
    algorithm_name: str,
    model_path: Path,
    episodes: int,
    seed: int,
    randomize: bool = False,
    deterministic: bool = True,
) -> dict:
    algorithm_cls = ALGORITHMS[algorithm_name]
    model = algorithm_cls.load(model_path)
    env = DroneNavigationEnv(
        action_mode=action_mode_for_algorithm(algorithm_name),
        randomize=randomize,
    )

    optimal = compute_optimal_path(env)

    episode_rows: list[dict] = []
    trajectories: list[list[list[float]]] = []

    for episode_index in range(1, episodes + 1):
        observation, info = env.reset(seed=seed + episode_index)
        terminated = False
        truncated = False
        total_reward = 0.0
        path_length = 0.0
        previous_position = info["position"].copy()
        trajectory = [previous_position.tolist()]

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            if algorithm_name == "dqn":
                env_action = int(action)
            else:
                env_action = action
            observation, reward, terminated, truncated, info = env.step(env_action)
            total_reward += float(reward)

            current_position = info["position"].copy()
            path_length += float(np.linalg.norm(current_position - previous_position))
            previous_position = current_position
            trajectory.append(current_position.tolist())

        reached_goal = bool(info.get("reached_goal", False))
        collision = bool(info.get("collision", False))
        timeout = bool(info.get("timeout", False)) or (
            truncated and not reached_goal and not collision
        )

        episode_rows.append(
            {
                "episode": episode_index,
                "success": float(reached_goal),
                "collision": float(collision),
                "timeout": float(timeout),
                "steps_taken": float(info["steps_taken"]),
                "distance_to_goal": float(info["distance_to_goal"]),
                "path_length": path_length,
                "reward": total_reward,
            }
        )
        trajectories.append(trajectory)

    env.close()
    return summarize_results(
        algorithm_name=algorithm_name,
        model_path=model_path,
        rows=episode_rows,
        trajectories=trajectories,
        optimal=optimal,
    )


def summarize_results(
    algorithm_name: str,
    model_path: Path,
    rows: list[dict],
    trajectories: list[list[list[float]]],
    optimal,
) -> dict:
    success_values = np.array([row["success"] for row in rows], dtype=np.float32)
    collision_values = np.array([row["collision"] for row in rows], dtype=np.float32)
    timeout_values = np.array([row["timeout"] for row in rows], dtype=np.float32)
    step_values = np.array([row["steps_taken"] for row in rows], dtype=np.float32)
    distance_values = np.array([row["distance_to_goal"] for row in rows], dtype=np.float32)
    path_values = np.array([row["path_length"] for row in rows], dtype=np.float32)
    reward_values = np.array([row["reward"] for row in rows], dtype=np.float32)

    success_mask = success_values.astype(bool)
    avg_steps_on_success = (
        float(step_values[success_mask].mean()) if success_mask.any() else float("nan")
    )
    avg_path_on_success = (
        float(path_values[success_mask].mean()) if success_mask.any() else float("nan")
    )

    optimal_length = optimal.astar_path_length
    excess_ratio = (
        float((path_values[success_mask] / optimal_length).mean())
        if success_mask.any() and optimal_length
        else float("nan")
    )

    return {
        "algorithm": algorithm_name,
        "model_path": str(model_path),
        "episodes": len(rows),
        "success_rate": float(success_values.mean()),
        "collision_rate": float(collision_values.mean()),
        "timeout_rate": float(timeout_values.mean()),
        "average_steps": float(step_values.mean()),
        "average_steps_on_success": avg_steps_on_success,
        "average_path_length": float(path_values.mean()),
        "average_path_length_on_success": avg_path_on_success,
        "average_remaining_distance": float(distance_values.mean()),
        "average_reward": float(reward_values.mean()),
        "optimal_straight_line_distance": optimal.straight_line_distance,
        "optimal_astar_path_length": optimal.astar_path_length,
        "optimal_astar_steps": optimal.astar_steps,
        "path_length_excess_ratio_on_success": excess_ratio,
        "episodes_data": rows,
        "trajectories": trajectories,
        "optimal_waypoints": optimal.astar_waypoints,
    }


def save_results(summary: dict, output_dir: Path, suffix: str = "") -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    algo = summary["algorithm"]
    json_path = output_dir / f"{algo}{suffix}_evaluation.json"
    csv_path = output_dir / f"{algo}{suffix}_episodes.csv"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary["episodes_data"][0].keys())
        writer.writeheader()
        writer.writerows(summary["episodes_data"])

    return json_path, csv_path


def default_model_path(algorithm_name: str, models_dir: Path) -> Path:
    return models_dir / f"{algorithm_name}_drone_navigation"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL drone model.")
    parser.add_argument(
        "--algo",
        choices=sorted(ALGORITHMS.keys()),
        required=True,
        help="RL algorithm that produced the model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to the saved model zip file. Defaults to models/<algo>_drone_navigation.zip",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=25,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base seed; episode i uses seed + i.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where evaluation outputs will be stored.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize start and goal positions on every reset.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy distribution instead of argmax/mean.",
    )
    args = parser.parse_args()

    model_path = args.model_path or default_model_path(args.algo, Path("models"))
    summary = evaluate_model(
        args.algo,
        model_path,
        args.episodes,
        args.seed,
        args.randomize,
        deterministic=not args.stochastic,
    )
    summary["deterministic"] = not args.stochastic
    suffix = "_stochastic" if args.stochastic else ""
    json_path, csv_path = save_results(summary, args.output_dir, suffix=suffix)

    print(f"Algorithm: {summary['algorithm']}")
    print(f"Episodes: {summary['episodes']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Collision rate: {summary['collision_rate']:.2%}")
    print(f"Timeout rate: {summary['timeout_rate']:.2%}")
    print(f"Average steps (all): {summary['average_steps']:.2f}")
    print(f"Average steps (success only): {summary['average_steps_on_success']:.2f}")
    print(f"Average path length (all): {summary['average_path_length']:.2f}")
    print(f"Average path length (success only): {summary['average_path_length_on_success']:.2f}")
    print(f"Optimal A* path length: {summary['optimal_astar_path_length']}")
    print(f"Excess ratio vs A* (success only): {summary['path_length_excess_ratio_on_success']}")
    print(f"Saved JSON summary to {json_path}")
    print(f"Saved episode table to {csv_path}")


if __name__ == "__main__":
    main()
