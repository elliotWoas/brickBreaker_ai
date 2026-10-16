import asyncio
import argparse
import json
import os
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from env import BrickBreakerEnv
from agent import DQNAgent


# -----------------------------
# Configuration
# -----------------------------
FPS = 60
FRAME_INTERVAL = 1.0 / FPS  # ~17ms per frame

CHECKPOINT_PATH = Path(__file__).parent / "dqn_checkpoint.pth"
CHECKPOINT_EVERY_EPISODES = int(os.getenv("CHECKPOINT_EVERY_EPISODES", "5"))
TRAIN_MAX_STEPS = int(os.getenv("TRAIN_MAX_STEPS", "2000"))
TRAIN_CPU_THREADS = max(1, int(os.getenv("TRAIN_CPU_THREADS", "1")))
TRAIN_UPDATE_INTERVAL = int(os.getenv("TRAIN_UPDATE_INTERVAL", "4"))

# Brick colors per row (top -> bottom)
BRICK_COLORS = [
    "#ef4444",  # red
    "#f97316",  # orange
    "#eab308",  # yellow
    "#22c55e",  # green
]


# -----------------------------
# FastAPI App Setup
# -----------------------------
app = FastAPI(title="Brick Breaker AI", description="DQN-trained Brick Breaker game")

BASE_DIR = Path(__file__).parent.resolve()


@app.get("/")
def read_root():
    """Serve the main HTML page."""
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok", "checkpoint_exists": CHECKPOINT_PATH.exists()}


def _create_agent_and_env(training=False):
    """Build the environment and DQN agent used for playback or training."""
    if training and not torch.cuda.is_available():
        torch.set_num_threads(TRAIN_CPU_THREADS)

    env = BrickBreakerEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        epsilon_start=1.0 if training else 0.0,
        epsilon_end=0.01 if training else 0.0,
        epsilon_decay=0.995,
    )

    # Load trained weights if available
    if CHECKPOINT_PATH.exists():
        try:
            agent.load(str(CHECKPOINT_PATH))
            print(f"[server] Loaded checkpoint from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"[server] Failed to load checkpoint: {e}")
    else:
        print(f"[server] No checkpoint found at {CHECKPOINT_PATH}; using random weights.")

    return env, agent


def _save_checkpoint(agent, episode):
    """Save the current DQN state and report the result in the server console."""
    try:
        agent.save(str(CHECKPOINT_PATH))
        print(
            f"[train] saved checkpoint after episode {episode} -> {CHECKPOINT_PATH}",
            flush=True,
        )
    except Exception as e:
        print(f"[train] failed to save checkpoint after episode {episode}: {e}", flush=True)


def _run_training_episode(
    env,
    agent,
    max_steps=TRAIN_MAX_STEPS,
    train_every=TRAIN_UPDATE_INTERVAL,
):
    """Run one full training episode, including replay optimization steps."""
    if train_every < 1:
        raise ValueError("train_every must be at least 1")

    state, _ = env.reset()
    total_reward = 0.0
    losses = []
    steps = 0
    final_info = {}

    for step in range(max_steps):
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train_step() if (step + 1) % train_every == 0 else None
        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward
        steps += 1
        final_info = info

        if done:
            break

    agent.finish_episode()

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "steps": steps,
        "total_reward": float(total_reward),
        "avg_loss": float(avg_loss),
        "epsilon": float(agent.epsilon),
        "score": int(final_info.get("score", 0)),
        "remaining_bricks": int(final_info.get("remaining_bricks", env.TOTAL_BRICKS)),
        "buffer_size": len(agent.memory),
        "optimizer_steps": agent.train_step_count,
    }


def run_training_loop(
    num_episodes=300,
    max_steps=TRAIN_MAX_STEPS,
    train_every=TRAIN_UPDATE_INTERVAL,
):
    """Run pretraining from the console and persist checkpoints regularly."""
    if train_every < 1:
        raise ValueError("train_every must be at least 1")

    env, agent = _create_agent_and_env(training=True)
    recent_rewards = []

    print(
        f"[train] starting training for {num_episodes} episodes | "
        f"epsilon={agent.epsilon:.3f}, decay={agent.epsilon_decay}, "
        f"checkpoint_every={CHECKPOINT_EVERY_EPISODES}, "
        f"cpu_threads={TRAIN_CPU_THREADS if not torch.cuda.is_available() else 'n/a'}, "
        f"train_every={train_every}",
        flush=True,
    )

    training_started = time.perf_counter()
    total_steps = 0
    first_episode = agent.episodes_trained + 1
    for episode in range(first_episode, first_episode + num_episodes):
        stats = _run_training_episode(
            env,
            agent,
            max_steps=max_steps,
            train_every=train_every,
        )
        total_steps += stats["steps"]
        recent_rewards.append(stats["total_reward"])
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)

        avg_reward = sum(recent_rewards) / len(recent_rewards)
        elapsed = time.perf_counter() - training_started
        completed_episodes = episode - first_episode + 1
        print(
            f"[train] episode={episode} reward={stats['total_reward']:.2f} "
            f"avg_reward_10={avg_reward:.2f} loss={stats['avg_loss']:.5f} "
            f"epsilon={stats['epsilon']:.3f} score={stats['score']} "
            f"bricks_left={stats['remaining_bricks']} steps={stats['steps']} "
            f"optimizer_steps={stats['optimizer_steps']} buffer={stats['buffer_size']} "
            f"elapsed={elapsed:.1f}s "
            f"ep/s={completed_episodes / max(elapsed, 1e-9):.2f} "
            f"steps/s={total_steps / max(elapsed, 1e-9):.1f}",
            flush=True,
        )

        if episode % CHECKPOINT_EVERY_EPISODES == 0:
            _save_checkpoint(agent, episode)

    _save_checkpoint(agent, agent.episodes_trained)


@app.websocket("/ws/play")
async def websocket_play(websocket: WebSocket):
    """
    WebSocket endpoint that streams game frames at ~60 FPS while the AI plays.
    Each frame is a JSON object with full render state.
    """
    await websocket.accept()
    print("[server] WebSocket client connected")

    env, agent = _create_agent_and_env(training=False)
    state, _ = env.reset()

    # Session stats
    episodes_played = 0
    total_score = 0
    best_score = 0

    try:
        while True:
            # Agent selects an action (greedy, no exploration)
            action = agent.select_action(state, training=False)

            # Step the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Build render frame (game state + meta info)
            frame = env.get_render_dict()
            frame["meta"] = {
                "reward": float(reward),
                "action": int(action),
                "episodes_played": int(episodes_played),
                "best_score": int(best_score),
                "total_score_this_session": int(total_score),
                "using_checkpoint": bool(CHECKPOINT_PATH.exists()),
            }

            # Send frame as JSON
            await websocket.send_text(json.dumps(frame))

            state = next_state

            # Reset if episode finished
            if done:
                episodes_played += 1
                if info["score"] > best_score:
                    best_score = info["score"]
                total_score += info["score"]

                # Brief "game over" pause so user notices
                await asyncio.sleep(0.5)
                state, _ = env.reset()

            # Throttle to ~60 FPS
            await asyncio.sleep(FRAME_INTERVAL)

    except WebSocketDisconnect:
        print("[server] WebSocket client disconnected")
    except Exception as e:
        print(f"[server] WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    """
    Optional: stream training progress over a WebSocket.
    Runs 1 episode at a time, reports results after each.
    """
    await websocket.accept()
    print("[server] Training WebSocket client connected")

    env, agent = _create_agent_and_env(training=True)
    recent_rewards = []

    try:
        episode = 1
        while True:
            stats = _run_training_episode(env, agent, max_steps=TRAIN_MAX_STEPS)
            recent_rewards.append(stats["total_reward"])
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            if episode % CHECKPOINT_EVERY_EPISODES == 0:
                _save_checkpoint(agent, agent.episodes_trained)

            print(
                f"[train/ws] episode={agent.episodes_trained} "
                f"reward={stats['total_reward']:.2f} avg_reward_10={avg_reward:.2f} "
                f"loss={stats['avg_loss']:.5f} epsilon={stats['epsilon']:.3f} "
                f"score={stats['score']} bricks_left={stats['remaining_bricks']} "
                f"steps={stats['steps']} optimizer_steps={stats['optimizer_steps']}",
                flush=True,
            )

            msg = {
                "episode": agent.episodes_trained,
                "steps": stats["steps"],
                "total_reward": stats["total_reward"],
                "avg_reward_10": float(avg_reward),
                "avg_loss": stats["avg_loss"],
                "epsilon": stats["epsilon"],
                "score": stats["score"],
                "remaining_bricks": stats["remaining_bricks"],
                "buffer_size": stats["buffer_size"],
                "optimizer_steps": stats["optimizer_steps"],
                "checkpoint_path": str(CHECKPOINT_PATH),
            }
            await websocket.send_text(json.dumps(msg))
            episode += 1

    except WebSocketDisconnect:
        print("[server] Training WebSocket client disconnected")
    except Exception as e:
        print(f"[server] Training WS error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brick Breaker DQN server and trainer")
    parser.add_argument("--train", action="store_true", help="Run the DQN training loop")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes for --train")
    parser.add_argument("--max-steps", type=int, default=TRAIN_MAX_STEPS, help="Max steps per episode")
    parser.add_argument(
        "--train-every",
        type=int,
        default=TRAIN_UPDATE_INTERVAL,
        help="Run one optimizer update every N environment steps",
    )
    parser.add_argument("--serve", action="store_true", help="Launch the FastAPI dashboard server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.train:
        run_training_loop(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            train_every=args.train_every,
        )

    if args.serve or not args.train:
        uvicorn.run(
            "server:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
