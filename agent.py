import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQN(nn.Module):
    """
    Simple feedforward Deep Q-Network architecture.
    Input: state vector (7 values)
    Hidden: two fully-connected layers with ReLU
    Output: Q-values for each action (3 actions)
    """

    def __init__(self, state_dim=7, action_dim=3, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Fixed-size Experience Replay buffer using NumPy arrays for efficient storage.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity=100000, state_dim=7):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        # Preallocate NumPy arrays for speed
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer.
        """
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of transitions uniformly at random.
        Returns (states, actions, rewards, next_states, dones) as NumPy arrays.
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


class DQNAgent:
    """
    DQN Agent with:
    - Target network for stable training
    - Experience Replay buffer
    - Epsilon-greedy exploration policy
    """

    def __init__(
        self,
        state_dim=7,
        action_dim=3,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=100,
        device=None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step_count = 0

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks: policy net (trained each step) and target net (updated periodically)
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Experience Replay
        self.memory = ReplayBuffer(buffer_capacity, state_dim)

    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.
        If training=False, always take the greedy action (no exploration).
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: action with highest Q-value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        """
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one gradient update step using a batch from the replay buffer.
        Returns the loss value if a training step was performed, else None.
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to PyTorch tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Q-values for current states (gather the chosen actions)
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Target Q-values using target network (double DQN style: policy selects, target evaluates)
        with torch.no_grad():
            # Select best next actions using policy net
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            # Evaluate those actions with target net
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            # Compute target: r + gamma * Q(s', a') * (1 - done)
            target_q = rewards_t + (self.gamma * next_q * (1.0 - dones_t))

        # Compute loss (MSE between current Q and target Q)
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Increment counter and decay epsilon
        self.train_step_count += 1
        self._decay_epsilon()

        # Update target network periodically
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def _decay_epsilon(self):
        """
        Exponentially decay epsilon from start to end.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """
        Save policy network weights to a file.
        """
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step_count": self.train_step_count,
            },
            path,
        )

    def load(self, path):
        """
        Load policy network weights from a file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.train_step_count = checkpoint.get("train_step_count", 0)


def train_agent(env, agent, num_episodes=1000, max_steps_per_episode=2000, verbose=True):
    """
    Helper function to train the DQN agent on the Brick Breaker environment.

    Args:
        env: BrickBreakerEnv instance
        agent: DQNAgent instance
        num_episodes: number of episodes to train for
        max_steps_per_episode: maximum steps per episode
        verbose: print progress

    Returns:
        list of episode rewards
    """
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        losses = []

        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state, training=True)

            # Take action in env
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        if verbose and episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(losses) if losses else 0.0
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Reward: {total_reward:.1f} | "
                f"Avg Reward (10 ep): {avg_reward:.1f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer Size: {len(agent.memory)}"
            )

    return episode_rewards
