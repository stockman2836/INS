from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_rl.envs import DroneNavigationEnv


ALGORITHMS = {
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
}


def make_env():
    return Monitor(DroneNavigationEnv())


def train(algorithm_name: str, total_timesteps: int, output_dir: Path) -> Path:
    algorithm_cls = ALGORITHMS[algorithm_name]
    env = DummyVecEnv([make_env])

    model = algorithm_cls(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

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
        default=20_000,
        help="Number of training timesteps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where trained models will be saved.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run Gymnasium environment validation before training.",
    )
    args = parser.parse_args()

    if args.check_env:
        check_env(DroneNavigationEnv(), warn=True)

    model_path = train(args.algo, args.timesteps, args.output_dir)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
