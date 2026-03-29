# INS

## 1. Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Set source path

```powershell
$env:PYTHONPATH="src"
```

## 3. Train

PPO:

```powershell
python -m drone_rl.train --algo ppo --timesteps 100000 --check-env
```

DQN:

```powershell
python -m drone_rl.train --algo dqn --timesteps 100000 --check-env
```

SAC:

```powershell
python -m drone_rl.train --algo sac --timesteps 100000 --check-env
```

## 4. Evaluate

PPO:

```powershell
python -m drone_rl.evaluate --algo ppo --episodes 25
```

DQN:

```powershell
python -m drone_rl.evaluate --algo dqn --episodes 25
```

SAC:

```powershell
python -m drone_rl.evaluate --algo sac --episodes 25
```

## 5. Outputs

Trained models:

```text
models/
```

Evaluation results:

```text
results/
```

## 6. Run without activation

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m drone_rl.train --algo ppo --timesteps 100000 --check-env
.\.venv\Scripts\python.exe -m drone_rl.evaluate --algo ppo --episodes 25
```
