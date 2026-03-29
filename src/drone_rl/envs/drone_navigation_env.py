from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Obstacle:
    center: np.ndarray
    half_size: np.ndarray


class DroneNavigationEnv(gym.Env):
    """Simple 3D navigation task with discrete motion and box obstacles."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, world_size: float = 10.0, max_steps: int = 150):
        super().__init__()
        self.world_size = world_size
        self.max_steps = max_steps
        self.step_size = 1.0
        self.goal_threshold = 0.75

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        self._moves = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        self.obstacles = [
            Obstacle(
                center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                half_size=np.array([1.5, 1.5, 1.5], dtype=np.float32),
            ),
            Obstacle(
                center=np.array([3.0, -2.0, 2.0], dtype=np.float32),
                half_size=np.array([1.0, 2.0, 1.0], dtype=np.float32),
            ),
        ]

        self.position = np.zeros(3, dtype=np.float32)
        self.start = np.array([-4.0, -4.0, -4.0], dtype=np.float32)
        self.goal = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        self.steps_taken = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.position = self.start.copy()
        self.steps_taken = 0
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        self.steps_taken += 1
        move = self._moves[action] * self.step_size
        proposed_position = self.position + move
        previous_distance = self._distance_to_goal(self.position)

        collision = self._is_out_of_bounds(proposed_position) or self._hits_obstacle(
            proposed_position
        )
        if collision:
            reward = -100.0
            terminated = True
            truncated = False
            info = self._get_info(collision=True)
            return self._get_obs(), reward, terminated, truncated, info

        self.position = proposed_position
        current_distance = self._distance_to_goal(self.position)
        progress_reward = (previous_distance - current_distance) * 5.0
        step_penalty = -0.1
        reward = progress_reward + step_penalty

        reached_goal = current_distance <= self.goal_threshold
        terminated = reached_goal
        truncated = self.steps_taken >= self.max_steps and not terminated

        if reached_goal:
            reward += 100.0
        elif truncated:
            reward -= 10.0

        info = self._get_info(collision=False)
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(
            f"step={self.steps_taken} position={self.position.round(2)} "
            f"goal={self.goal.round(2)}"
        )

    def close(self):
        return None

    def _get_obs(self) -> np.ndarray:
        nearest_obstacle = min(
            self.obstacles,
            key=lambda obstacle: np.linalg.norm(self.position - obstacle.center),
        )
        obs = np.concatenate(
            [
                self.position / self.world_size,
                self.goal / self.world_size,
                (nearest_obstacle.center - self.position) / self.world_size,
            ]
        )
        return obs.astype(np.float32)

    def _get_info(self, collision: bool = False) -> dict:
        return {
            "distance_to_goal": self._distance_to_goal(self.position),
            "collision": collision,
            "steps_taken": self.steps_taken,
        }

    def _distance_to_goal(self, position: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal - position))

    def _is_out_of_bounds(self, position: np.ndarray) -> bool:
        return bool(np.any(np.abs(position) > self.world_size / 2.0))

    def _hits_obstacle(self, position: np.ndarray) -> bool:
        for obstacle in self.obstacles:
            lower = obstacle.center - obstacle.half_size
            upper = obstacle.center + obstacle.half_size
            if np.all(position >= lower) and np.all(position <= upper):
                return True
        return False

