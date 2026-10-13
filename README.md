# Brick Breaker AI

Brick Breaker AI is a small reinforcement-learning project where a Deep Q-Network
(DQN) learns to control a paddle in a browser-rendered Brick Breaker game. The
Python environment simulates the game, the agent trains with replay memory and a
target network, and the FastAPI server streams gameplay to `index.html` over a
WebSocket.

## Overview

The goal is simple: keep the ball alive, hit bricks, and clear the board. The
agent sees a compact numeric state, chooses one of three paddle actions, receives
reward feedback, and gradually improves its Q-value estimates.

### Deep Q-Learning

DQN learns a function:

```text
Q(state, action) -> expected future reward
```

During training, the agent balances:

- Exploration: try random actions using epsilon-greedy selection.
- Exploitation: choose the action with the highest predicted Q-value.
- Replay memory: train on random batches of past transitions to reduce
  correlation between consecutive frames.
- Target network: use a slower-moving copy of the model to stabilize Q targets.

In this project, epsilon decays once per completed episode with a default rate of
`0.995`. That keeps exploration active for hundreds of games instead of collapsing
after a few optimizer steps.

### State Space

Each observation is a 7-value vector:

```text
[paddle_x, ball_x, ball_y, ball_dx, ball_dy, remaining_bricks, lives]
```

These values give the model enough information to track where the paddle is, where
the ball is moving, how many bricks remain, and how close the game is to ending.

### Action Space

The action space has three discrete actions:

```text
0 = stay still
1 = move left
2 = move right
```

### Reward Structure

The reward signal in `env.py` is shaped to give the model useful feedback every
frame:

- Dense alignment reward:
  `+0.1 * (1.0 - abs(paddle_center_x - ball_x) / screen_width)`
- Paddle hit bonus: `+5.0`
- Brick hit reward: `+10.0`
- Dropped ball penalty: `-20.0`
- Small survival reward: `+0.01` while the episode is still active

The dense reward helps the agent learn paddle tracking even before it becomes good
enough to hit many bricks.

## Architecture

### `env.py`

Defines `BrickBreakerEnv`, a Gymnasium-compatible environment. It owns the game
state, physics, collision detection, rewards, observations, episode resets, and
browser render dictionaries.

### `agent.py`

Defines:

- `DQN`: the neural network that predicts Q-values for each action.
- `ReplayBuffer`: fixed-size storage for past transitions.
- `DQNAgent`: action selection, replay training, epsilon decay, target-network
  updates, checkpoint saving, and checkpoint loading.
- `train_agent`: a standalone helper for script-based training.

### `server.py`

Runs the FastAPI app and training entry points:

- `GET /`: serves the browser dashboard.
- `GET /health`: reports whether `dqn_checkpoint.pth` exists.
- `WS /ws/play`: loads the checkpoint and streams greedy AI gameplay at 60 FPS.
- `WS /ws/train`: runs training episodes and streams metrics.
- `python server.py --train`: runs console pretraining and saves checkpoints.
- `python server.py --serve`: launches the web dashboard.

During training, the server logs episode reward, rolling average reward, loss,
epsilon, score, remaining bricks, replay-buffer size, and optimizer-step count.

### `index.html`

Provides the browser dashboard. It draws the game on a canvas, connects to
`/ws/play`, and displays score, lives, bricks, best score, episodes, and the last
action.

## Setup & Usage

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

The requirements pin PyTorch to its CPU-only wheel, so a CUDA toolkit and NVIDIA
runtime packages are not installed. On Linux, use a fresh virtual environment to
avoid retaining CUDA packages from an older environment:

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip cache purge
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
```

### 3. Pre-train the agent

Run 300 training episodes:

```powershell
python server.py --train --episodes 300
```

You should see console output like:

```text
[train] episode=25 reward=... avg_reward_10=... loss=... epsilon=... optimizer_steps=...
[train] saved checkpoint after episode 25 -> ...\dqn_checkpoint.pth
```

The checkpoint is saved to:

```text
dqn_checkpoint.pth
```

By default, checkpoints save every 5 episodes. You can change that with:

```powershell
$env:CHECKPOINT_EVERY_EPISODES = "10"
python server.py --train --episodes 300
```

### 4. Launch the dashboard

```powershell
python server.py --serve
```

Open:

```text
http://127.0.0.1:8000
```

Click **Start AI Play**. The `/ws/play` endpoint loads `dqn_checkpoint.pth` if it
exists, then runs the model greedily with exploration disabled.

### 5. Train from the WebSocket endpoint

The server also supports live training through:

```text
ws://127.0.0.1:8000/ws/train
```

Each message contains training metrics:

```json
{
  "episode": 10,
  "total_reward": 123.4,
  "avg_reward_10": 98.7,
  "avg_loss": 0.0123,
  "epsilon": 0.951,
  "optimizer_steps": 18234
}
```

## Troubleshooting

### The agent repeats the same movements

Check whether a checkpoint exists:

```powershell
python -c "from pathlib import Path; print(Path('dqn_checkpoint.pth').exists())"
```

If it prints `False`, train first:

```powershell
python server.py --train --episodes 300
```

### The checkpoint is not saving

During training, look for log lines like:

```text
[train] saved checkpoint after episode 5 -> ...\dqn_checkpoint.pth
```

Also confirm the file exists and has a recent timestamp:

```powershell
Get-Item .\dqn_checkpoint.pth
```

### The checkpoint is not loading in play mode

Start the server:

```powershell
python server.py --serve
```

Then click **Start AI Play** and watch the server console. A successful load prints:

```text
[server] Loaded checkpoint from ...\dqn_checkpoint.pth
```

If you see:

```text
[server] No checkpoint found ...
```

make sure you trained from the project root and that `dqn_checkpoint.pth` is in the
same folder as `server.py`.

### Loss is always zero

Loss is `0.0` until replay memory has at least one full batch. With the default
batch size of `64`, this is normal during the first few dozen frames. After enough
steps, `optimizer_steps` should increase and loss should become non-zero.

### Rewards are not improving

Watch `avg_reward_10` rather than a single episode. Brick Breaker has noisy
episodes, so the rolling average is a better signal. Also train for at least
`200-300` episodes because epsilon now decays gradually at episode boundaries.

### The dashboard still looks untrained

The dashboard uses greedy playback from the latest checkpoint. Run more training,
restart the server, then reconnect the dashboard so `/ws/play` reloads the updated
weights.
