import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import random
from collections import deque

# Hyperparameters
GAMMA = 0.99  # Discount factor
TAU = 0.005   # Soft target update rate
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
ALPHA = 0.2   # Entropy regularization coefficient
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1))

    def size(self):
        return len(self.buffer)

# Actor (Policy Network) with Reparameterization Trick
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)  # Log standard deviation clipping
        std = log_std.exp()
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z) * self.max_action  # Squash to action range
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

# Critic (Twin Q-Networks)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

# Soft Actor-Critic Agent
class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.alpha = ALPHA
        self.max_action = max_action

    def select_action(self, state, eval_mode=False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor)
        return action.cpu().numpy()[0] if not eval_mode else action.mean().cpu().numpy()[0]

    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return  # Not enough samples

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Compute target Q-value
        with torch.no_grad():
            next_actions, log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_next, q2_next) - self.alpha * log_probs
            target_q = rewards + GAMMA * (1 - dones) * q_target

        # Compute critic loss and update
        q1, q2 = self.critic(states, actions)
        critic_loss = ((q1 - target_q).pow(2) + (q2 - target_q).pow(2)).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss and update
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1_new, q2_new)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target Q-networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

# Training Loop
def train_sac(env_name="Pendulum-v1", episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC(state_dim, action_dim, max_action)

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.add_experience(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {episode+1}, Total Reward: {total_reward}")

    env.close()

# Run SAC training
train_sac()
