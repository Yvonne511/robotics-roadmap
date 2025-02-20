# Deep Deterministic Policy Gradient
The paper this concept originates from is [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971) (2015)

## Paper Review (Key insights)
- combines DQN and Deterministic Policy Gradient (DPG)
- works for high-dimensional continuous control tasks
- **Off-policy learning**: Uses a replay buffer for stability.
- **Deterministic policy**: Unlike stochastic policies in policy gradient methods
- **Exploration with noise**: Uses Ornstein-Uhlenbeck noise for exploration
- **Target networks**: Helps in stabilizing training
- <ins>unstable to train</ins>

## Algorithm
**Input:**  
- A **parameterized actor** policy $\mu(s|\theta^{\mu})$
- A **parameterized critic** $Q(s, a | \theta^Q)$
- **Replay buffer** $\mathcal{D}$
- **Target networks** $\mu'$ and $Q'$ with parameters $\theta^{\mu'}$ and $\theta^{Q'}$
- **Noise process** $\mathcal{N}$ for exploration

1. Initialize
    - Initialize **actor** $\mu(s|\theta^\mu)$ and **critic** $Q(s, a | \theta^Q)$ networks with random parameters.
    - Initialize **target networks**: $\theta^{\mu'} \leftarrow \theta^{\mu}, \quad \theta^{Q'} \leftarrow \theta^{Q}$
    - Initialize an **empty replay buffer** $\mathcal{D}$
2. For $episode=1$ to convergence:
    - **Reset the environment** and get the initial state $s_0$
    - **For each time step $t$ in the episode**:
        - Select action with exploration noise: $a_t = \mu(s_t | \theta^\mu) + \mathcal{N}_t$
        - Execute action $a_t$, observe **reward** $r_t$ and **next state** $s_{t+1}$
        - Store **transition** $(s_t, a_t, r_t, s_{t+1})$ in replay buffer $\mathcal{D}$
        ---
3. After collecting enough experiences (**inside episode**):
    1. Sample **mini-batch** of $N$ transitions $(s_i, a_i, r_i, s'_i)$ from buffer $\mathcal{D}$.
    2. Compute **target Q-value** using target networks:  
    $y_i = r_i + \gamma Q'(s'_i, \mu'(s'_i | \theta^{\mu'}) | \theta^{Q'})$
    3. **Critic Loss**: Minimize the **Mean Squared Error (MSE)** loss:  
    $L(\theta^Q) = \frac{1}{N} \sum_i (y_i - Q(s_i, a_i | \theta^Q))^2$
    4. **Actor Update** (using policy gradient):  
    $$
    \nabla_{\theta^\mu} J \approx \frac{1}{N} \sum_i \nabla_a Q(s, a | \theta^Q) |_{a = \mu(s)} \nabla_{\theta^\mu} \mu(s | \theta^\mu)
    $$
    5. **Update target networks** with soft update:  
    $\theta^{Q'} \leftarrow \tau \theta^Q + (1 - \tau) \theta^{Q'}$  
    $\theta^{\mu'} \leftarrow \tau \theta^\mu + (1 - \tau) \theta^{\mu'}$  
    where $\tau \ll 1$ (e.g., 0.001) is a small update rate.
