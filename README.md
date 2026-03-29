# INS

Simple starter project for the assignment "Drone control in 3D space with RL".

## Project idea

This repository starts with a simplified 3D environment:

- the drone moves in 3D space,
- the goal is a fixed point in 3D space,
- collisions with obstacles end the episode,
- the reward encourages progress toward the target and penalizes crashes.

`DQN` uses a discrete action version of the environment, while `PPO` and `SAC` use a continuous action version compatible with actor-critic control methods.

## Setup

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

## Train an agent

Run a short training session:

```powershell
$env:PYTHONPATH="src"
python -m drone_rl.train --algo ppo --timesteps 20000 --check-env
```

Try other algorithms:

```powershell
python -m drone_rl.train --algo dqn --timesteps 20000
python -m drone_rl.train --algo sac --timesteps 20000
```

## Evaluate a trained model

After training, run evaluation episodes and save metrics:

```powershell
$env:PYTHONPATH="src"
python -m drone_rl.evaluate --algo ppo --episodes 25
```

Evaluation outputs are saved to `results/`:

- JSON summary with aggregate metrics,
- CSV table with one row per episode.

These metrics are a good starting point for your report:

- success rate,
- collision rate,
- average number of steps,
- average path length,
- average remaining distance to goal,
- average reward.

## Next steps

- add evaluation scripts for comparing metrics,
- log collisions, path length, and success rate,
- visualize trajectories,
- tune reward shaping and obstacle layouts.
