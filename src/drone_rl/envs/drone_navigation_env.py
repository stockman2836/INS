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

    def __init__(
        self,
        world_size: float = 10.0,
        max_steps: int = 150,
        action_mode: str = "continuous",
        randomize: bool = False,
        randomize_start: bool = True,
        randomize_goal: bool = True,
        min_start_goal_distance: float = 6.0,
    ):
        super().__init__()
        self.world_size = world_size
        self.max_steps = max_steps
        self.step_size = 1.0
        self.goal_threshold = 0.75
        self.action_mode = action_mode
        self.randomize = randomize
        self.randomize_start = randomize_start
        self.randomize_goal = randomize_goal
        self.min_start_goal_distance = float(min_start_goal_distance)

        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(6)
        elif self.action_mode == "continuous":
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

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
        if self.randomize:
            self._sample_start_goal()
        self.position = self.start.copy()
        self.steps_taken = 0
        return self._get_obs(), self._get_info()

    def _sample_free_point(self, max_tries: int = 200) -> np.ndarray:
        half = self.world_size / 2.0 - self.step_size
        rng = self.np_random
        for _ in range(max_tries):
            candidate = rng.uniform(-half, half, size=3).astype(np.float32)
            if not self._hits_obstacle(candidate):
                return candidate
        return np.array([-half, -half, -half], dtype=np.float32)

    def _sample_start_goal(self) -> None:
        if self.randomize_start:
            start = self._sample_free_point()
        else:
            start = self.start.copy()
        for _ in range(200):
            if self.randomize_goal:
                goal = self._sample_free_point()
            else:
                goal = self.goal.copy()
            if float(np.linalg.norm(goal - start)) >= self.min_start_goal_distance:
                self.start = start.astype(np.float32)
                self.goal = goal.astype(np.float32)
                return
        self.start = start.astype(np.float32)
        self.goal = (
            self.goal.copy() if not self.randomize_goal else self._sample_free_point()
        ).astype(np.float32)

    def step(self, action: int | np.ndarray):
        self.steps_taken += 1
        move = self._decode_action(action)
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

    def _decode_action(self, action: int | np.ndarray) -> np.ndarray:
        if self.action_mode == "discrete":
            return self._moves[int(action)] * self.step_size

        continuous_action = np.asarray(action, dtype=np.float32).reshape(3)
        clipped_action = np.clip(continuous_action, -1.0, 1.0)
        norm = np.linalg.norm(clipped_action)
        if norm > 1.0:
            clipped_action = clipped_action / norm
        return clipped_action * self.step_size

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
        reached_goal = self._distance_to_goal(self.position) <= self.goal_threshold
        timeout = (not collision) and (not reached_goal) and self.steps_taken >= self.max_steps
        return {
            "distance_to_goal": self._distance_to_goal(self.position),
            "collision": collision,
            "reached_goal": bool(reached_goal and not collision),
            "timeout": bool(timeout),
            "steps_taken": self.steps_taken,
            "position": self.position.copy(),
            "start": self.start.copy(),
            "goal": self.goal.copy(),
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
