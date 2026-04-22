from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_rl.callbacks import TrainingMetricsCallback
from drone_rl.envs import DroneNavigationEnv


ALGORITHMS = {
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
}


# Reasonable per-algorithm defaults for a small 9-D observation MLP task.
# Not optimal, but not copy-pasted between fundamentally different algorithms.
ALGO_KWARGS: dict[str, dict] = {
    "dqn": {
        "learning_rate": 1e-3,
        "buffer_size": 50_000,
        "learning_starts": 1_000,
        "batch_size": 128,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 500,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
    },
    "sac": {
        "learning_rate": 3e-4,
        "buffer_size": 50_000,
        "learning_starts": 1_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
    },
}


def action_mode_for_algorithm(algorithm_name: str) -> str:
    if algorithm_name == "dqn":
        return "discrete"
    return "continuous"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(algorithm_name: str, seed: int | None = None, randomize: bool = False):
    env = DroneNavigationEnv(
        action_mode=action_mode_for_algorithm(algorithm_name),
        randomize=randomize,
    )
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


def train(
    algorithm_name: str,
    total_timesteps: int,
    output_dir: Path,
    seed: int,
    results_dir: Path,
    randomize: bool = False,
) -> Path:
    algorithm_cls = ALGORITHMS[algorithm_name]
    env = DummyVecEnv([lambda: make_env(algorithm_name, seed, randomize)])
    env.seed(seed)

    model = algorithm_cls(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(output_dir / "tensorboard"),
        **ALGO_KWARGS[algorithm_name],
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_callback = TrainingMetricsCallback(
        output_path=results_dir / f"{algorithm_name}_training_metrics.json",
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=metrics_callback,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{algorithm_name}_drone_navigation"
    model.save(model_path)
    env.close()
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train an RL agent for 3D drone navigation.")
    parser.add_argument(
        "--algo",
        choices=sorted(ALGORITHMS.keys()),
        default="ppo",
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Number of training timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where trained models will be saved.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where training-time metrics will be saved.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run Gymnasium environment validation before training.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize start and goal positions on every reset.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.check_env:
        check_env(
            DroneNavigationEnv(
                action_mode=action_mode_for_algorithm(args.algo),
                randomize=args.randomize,
            ),
            warn=True,
        )

    model_path = train(
        algorithm_name=args.algo,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        seed=args.seed,
        results_dir=args.results_dir,
        randomize=args.randomize,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
