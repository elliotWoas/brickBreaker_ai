import asyncio
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from env import BrickBreakerEnv
from agent import DQNAgent


# -----------------------------
# Configuration
# -----------------------------
FPS = 30
FRAME_INTERVAL = 1.0 / FPS  # ~33ms per frame

CHECKPOINT_PATH = Path(__file__).parent / "dqn_checkpoint.pth"

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


def _create_agent_and_env():
    """Helper to build env + agent pair used during playback."""
    env = BrickBreakerEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        epsilon_start=0.0,  # No exploration when playing back
        epsilon_end=0.0,
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


@app.websocket("/ws/play")
async def websocket_play(websocket: WebSocket):
    """
    WebSocket endpoint that streams game frames at ~30 FPS while the AI plays.
    Each frame is a JSON object with full render state.
    """
    await websocket.accept()
    print("[server] WebSocket client connected")

    env, agent = _create_agent_and_env()
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

            # Throttle to ~30 FPS
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

    env, agent = _create_agent_and_env()

    # Reset to training exploration values
    agent.epsilon = 1.0
    max_steps = 2000

    try:
        episode = 1
        while True:
            state, _ = env.reset()
            total_reward = 0.0
            losses = []
            steps = 0
            brick_count_history = []

            for step in range(max_steps):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

                brick_count_history.append(info["remaining_bricks"])

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            # Occasional checkpoints
            if episode % 25 == 0:
                try:
                    agent.save(str(CHECKPOINT_PATH))
                except Exception as e:
                    print(f"[server] Failed to save checkpoint: {e}")

            avg_loss = sum(losses) / len(losses) if losses else 0.0
            final_info = env.get_render_dict()
            msg = {
                "episode": episode,
                "steps": steps,
                "total_reward": float(total_reward),
                "avg_loss": float(avg_loss),
                "epsilon": float(agent.epsilon),
                "score": int(final_info["score"]),
                "remaining_bricks": int(final_info["remaining_bricks"]),
                "buffer_size": len(agent.memory),
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
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
