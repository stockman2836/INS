"""Training callbacks for metrics the assignment explicitly requires."""

from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """Counts collisions, successes and timeouts across all envs during training.

    The environment terminates the episode on collision, so each collision
    corresponds to exactly one terminated episode. This callback therefore
    reports, at the end of training:
    - total collisions during training,
    - total successful goal reaches,
    - total timeouts,
    - total finished episodes and collision rate per episode.
    """

    def __init__(self, output_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.output_path = Path(output_path)
        self.collisions = 0
        self.successes = 0
        self.timeouts = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            self.episodes += 1
            if info.get("collision"):
                self.collisions += 1
            elif info.get("reached_goal"):
                self.successes += 1
            elif info.get("timeout"):
                self.timeouts += 1
        return True

    def _on_training_end(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "episodes": self.episodes,
            "collisions": self.collisions,
            "successes": self.successes,
            "timeouts": self.timeouts,
            "collision_rate": (self.collisions / self.episodes) if self.episodes else 0.0,
            "success_rate": (self.successes / self.episodes) if self.episodes else 0.0,
            "timeout_rate": (self.timeouts / self.episodes) if self.episodes else 0.0,
        }
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        if self.verbose:
            print(f"Training metrics saved to {self.output_path}")
