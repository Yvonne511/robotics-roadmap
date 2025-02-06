import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Define the Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.softmax = nn.Softmax(dim=-1)  # Convert logits to probabilities

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

# REINFORCE Algorithm
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def update_policy(self, episode_rewards, log_probs):
        returns = []
        G = 0

        # Compute discounted returns (Monte Carlo return)
        for r in reversed(episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize returns

        # Compute policy gradient loss
        loss = torch.stack([-log_prob * G for log_prob, G in zip(log_probs, returns)]).sum()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Train REINFORCE on OpenAI Gym CartPole-v1
def train_reinforce(env_name="CartPole-v1", num_episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, action_dim)

    for episode in range(num_episodes):
        state = env.reset()[0]  # Gym 0.26+ returns (obs, info)
        episode_rewards = []
        log_probs = []

        done = False
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            episode_rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

        # Update policy at the end of the episode
        agent.update_policy(episode_rewards, log_probs)

        # Logging
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {sum(episode_rewards)}")

    env.close()

# Run the training
if __name__ == "__main__":
    train_reinforce()
