"""Baseline path planning for drone navigation.

Provides:
- straight-line optimal distance (ignoring obstacles),
- A* shortest path on a 3D axis-aligned grid with the same obstacles as
  ``DroneNavigationEnv``. A* uses the 6-neighbour discrete move set
  (matching the DQN action space) and therefore gives the optimal
  collision-free path length for the discrete version of the task.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from drone_rl.envs import DroneNavigationEnv


@dataclass(frozen=True)
class OptimalPath:
    straight_line_distance: float
    astar_path_length: float | None
    astar_steps: int | None
    astar_waypoints: list[tuple[float, float, float]]


def _in_obstacle(position: np.ndarray, env: DroneNavigationEnv) -> bool:
    for obstacle in env.obstacles:
        lower = obstacle.center - obstacle.half_size
        upper = obstacle.center + obstacle.half_size
        if np.all(position >= lower) and np.all(position <= upper):
            return True
    return False


def _in_bounds(position: np.ndarray, env: DroneNavigationEnv) -> bool:
    return bool(np.all(np.abs(position) <= env.world_size / 2.0))


def compute_optimal_path(env: DroneNavigationEnv) -> OptimalPath:
    start = env.start.astype(np.float32)
    goal = env.goal.astype(np.float32)
    step = float(env.step_size)
    threshold = float(env.goal_threshold)

    straight = float(np.linalg.norm(goal - start))

    # A* on a regular 6-neighbour grid aligned with start and step size.
    # State is the integer offset from start in units of step.
    moves = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    def to_pos(state: tuple[int, int, int]) -> np.ndarray:
        return start + np.array(state, dtype=np.float32) * step

    def heuristic(state: tuple[int, int, int]) -> float:
        return float(np.linalg.norm(goal - to_pos(state)))

    start_state = (0, 0, 0)
    open_heap: list[tuple[float, tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start_state), start_state))
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    g_score: dict[tuple[int, int, int], float] = {start_state: 0.0}

    # Bound the search box to world_size / step to avoid unbounded growth.
    bound = int(np.ceil(env.world_size / step)) + 1

    goal_state: tuple[int, int, int] | None = None

    while open_heap:
        _, current = heapq.heappop(open_heap)
        current_pos = to_pos(current)
        if np.linalg.norm(goal - current_pos) <= threshold:
            goal_state = current
            break

        for dx, dy, dz in moves:
            neighbour = (current[0] + dx, current[1] + dy, current[2] + dz)
            if any(abs(c) > bound for c in neighbour):
                continue
            neighbour_pos = to_pos(np.array(neighbour))
            if not _in_bounds(neighbour_pos, env):
                continue
            if _in_obstacle(neighbour_pos, env):
                continue
            tentative = g_score[current] + step
            if tentative < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                f_score = tentative + heuristic(neighbour)
                heapq.heappush(open_heap, (f_score, neighbour))

    if goal_state is None:
        return OptimalPath(
            straight_line_distance=straight,
            astar_path_length=None,
            astar_steps=None,
            astar_waypoints=[],
        )

    waypoints_states: list[tuple[int, int, int]] = [goal_state]
    while waypoints_states[-1] in came_from:
        waypoints_states.append(came_from[waypoints_states[-1]])
    waypoints_states.reverse()

    waypoints = [tuple(float(v) for v in to_pos(s)) for s in waypoints_states]
    path_length = float(g_score[goal_state])

    return OptimalPath(
        straight_line_distance=straight,
        astar_path_length=path_length,
        astar_steps=len(waypoints_states) - 1,
        astar_waypoints=waypoints,
    )
