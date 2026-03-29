from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO, SAC

from drone_rl.envs import DroneNavigationEnv


ALGORITHMS = {
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
}


def evaluate_model(algorithm_name: str, model_path: Path, episodes: int) -> dict:
    algorithm_cls = ALGORITHMS[algorithm_name]
    model = algorithm_cls.load(model_path)
    env = DroneNavigationEnv()

    episode_rows: list[dict] = []

    for episode_index in range(1, episodes + 1):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        collisions = 0
        path_length = 0.0
        previous_position = env.position.copy()

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)

            current_position = env.position.copy()
            path_length += float(np.linalg.norm(current_position - previous_position))
            previous_position = current_position

            if info["collision"]:
                collisions += 1

        episode_rows.append(
            {
                "episode": episode_index,
                "success": float(not info["collision"] and terminated),
                "collision": float(info["collision"]),
                "steps_taken": float(info["steps_taken"]),
                "distance_to_goal": float(info["distance_to_goal"]),
                "path_length": path_length,
                "reward": total_reward,
            }
        )

    env.close()
    return summarize_results(algorithm_name, model_path, episode_rows)


def summarize_results(algorithm_name: str, model_path: Path, rows: list[dict]) -> dict:
    success_values = np.array([row["success"] for row in rows], dtype=np.float32)
    collision_values = np.array([row["collision"] for row in rows], dtype=np.float32)
    step_values = np.array([row["steps_taken"] for row in rows], dtype=np.float32)
    distance_values = np.array([row["distance_to_goal"] for row in rows], dtype=np.float32)
    path_values = np.array([row["path_length"] for row in rows], dtype=np.float32)
    reward_values = np.array([row["reward"] for row in rows], dtype=np.float32)

    return {
        "algorithm": algorithm_name,
        "model_path": str(model_path),
        "episodes": len(rows),
        "success_rate": float(success_values.mean()),
        "collision_rate": float(collision_values.mean()),
        "average_steps": float(step_values.mean()),
        "average_remaining_distance": float(distance_values.mean()),
        "average_path_length": float(path_values.mean()),
        "average_reward": float(reward_values.mean()),
        "episodes_data": rows,
    }


def save_results(summary: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{summary['algorithm']}_evaluation.json"
    csv_path = output_dir / f"{summary['algorithm']}_episodes.csv"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary["episodes_data"][0].keys())
        writer.writeheader()
        writer.writerows(summary["episodes_data"])

    return json_path, csv_path


def default_model_path(algorithm_name: str, models_dir: Path) -> Path:
    return models_dir / f"{algorithm_name}_drone_navigation.zip"


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
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where evaluation outputs will be stored.",
    )
    args = parser.parse_args()

    model_path = args.model_path or default_model_path(args.algo, Path("models"))
    summary = evaluate_model(args.algo, model_path, args.episodes)
    json_path, csv_path = save_results(summary, args.output_dir)

    print(f"Algorithm: {summary['algorithm']}")
    print(f"Episodes: {summary['episodes']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Collision rate: {summary['collision_rate']:.2%}")
    print(f"Average steps: {summary['average_steps']:.2f}")
    print(f"Average path length: {summary['average_path_length']:.2f}")
    print(f"Average remaining distance: {summary['average_remaining_distance']:.2f}")
    print(f"Average reward: {summary['average_reward']:.2f}")
    print(f"Saved JSON summary to {json_path}")
    print(f"Saved episode table to {csv_path}")


if __name__ == "__main__":
    main()
