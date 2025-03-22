import gymnasium as gym
import numpy as np

class OptionCriticAgent:
    def __init__(self, env, num_options=2, alpha=0.1, gamma=0.99, epsilon=0.1, bins=10):
        self.env = env
        self.num_options = num_options
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.bins = bins

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Discretization bins for continuous state
        self.state_bins = [np.linspace(-1, 1, self.bins) for _ in range(self.state_dim)]

        # Initialize Q-value function for options
        self.Q_option = np.zeros((self.bins,) * self.state_dim + (num_options,))

        # Initialize intra-option policies using Dirichlet distribution (ensures sum to 1)
        self.policy = np.random.dirichlet(np.ones(self.action_dim), 
                                          size=(num_options,) + (self.bins,) * self.state_dim)

        # Initialize termination functions
        self.beta = np.random.rand(num_options, *(self.bins,) * self.state_dim)

    def discretize_state(self, state):
        """ Convert a continuous state into a discrete index """
        state_idx = tuple(np.digitize(state[i], self.state_bins[i]) - 1 for i in range(self.state_dim))
        return state_idx

    def choose_option(self, state):
        """ Epsilon-greedy option selection """
        state_idx = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_options)
        return np.argmax(self.Q_option[state_idx])

    def choose_action(self, option, state):
        """ Sample action from intra-option policy """
        state_idx = self.discretize_state(state)
        return np.random.choice(self.action_dim, p=self.policy[option][state_idx])

    def update(self, state, option, action, reward, next_state, done):
        """ Update Q-values and intra-option policy """
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)

        # Compute U-value
        Q_U = reward + self.gamma * np.max(self.Q_option[next_state_idx]) if not done else reward
        
        # Update Q-values
        self.Q_option[state_idx][option] += self.alpha * (Q_U - self.Q_option[state_idx][option])
        
        # Update intra-option policy using Q-learning
        EPSILON = 1e-8  # To avoid zero probabilities
        self.policy[option][state_idx][action] += self.alpha * (1 - self.policy[option][state_idx][action])

        # Renormalize policy
        self.policy[option][state_idx] = np.maximum(self.policy[option][state_idx], EPSILON)  
        self.policy[option][state_idx] /= np.sum(self.policy[option][state_idx])  # Normalize

        # Update termination function
        self.beta[option][state_idx] += self.alpha * (Q_U - self.Q_option[state_idx][option])

def run(is_training=True, render=False, num_episodes=50000):
    env = gym.make('Acrobot-v1', render_mode='human' if render else None)

    num_options = 2
    lr =   0.1
    gamma = 0.99
    epsilon = 0.1
    bins = 10
    agent = OptionCriticAgent(env, alpha=lr, num_options=num_options, gamma=gamma, epsilon=epsilon, bins=bins)

    for episode in range(num_episodes):
        state, _ = env.reset()
        option = agent.choose_option(state)

        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(option, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if is_training:
                agent.update(state, option, action, reward, next_state, done)

            # Option termination condition
            if np.random.rand() < agent.beta[option][agent.discretize_state(state)]:
                option = agent.choose_option(next_state)

            state = next_state
            total_reward += reward

            if render:
                env.render()
        if episode % 500 == 0:
            print(f"Episode {episode+1}: Reward = {total_reward}")

    env.close()

if __name__ == '__main__':
    run(is_training=True, render=False)  # Train
    run(is_training=False, render=True)  # Test and visualize


