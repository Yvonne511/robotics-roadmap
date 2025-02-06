import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Define Actor (Policy) Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # Output probability distribution
        )

    def forward(self, state):
        return self.fc(state)

# Define Critic (Value Function) Network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Outputs a scalar value estimate V_w(s)
        )

    def forward(self, state):
        return self.fc(state)

# Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)  # Return action and log probability

    def update(self, state, action_log_prob, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        done_tensor = torch.tensor(done, dtype=torch.float32)

        # Compute TD(0) Error
        with torch.no_grad():
            V_next = self.critic(next_state_tensor) * (1 - done_tensor)  # 0 if terminal state
        V_current = self.critic(state_tensor)
        delta = reward_tensor + self.gamma * V_next - V_current  # TD error

        # Update Critic (Value Function)
        critic_loss = delta.pow(2).mean()  # Mean squared TD error
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor (Policy Gradient)
        actor_loss = -action_log_prob * delta.detach()  # Policy gradient
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# Training Loop
def train_actor_critic(env_name="CartPole-v1", episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCriticAgent(state_dim, action_dim)

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        
        while True:
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.update(state, action_log_prob, reward, next_state, done)

            state = next_state
            total_reward += reward
            
            if done:
                break

        print(f"Episode {episode+1}, Total Reward: {total_reward}")

    env.close()

# Run training
train_actor_critic()

