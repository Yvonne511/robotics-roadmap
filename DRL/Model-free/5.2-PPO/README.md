## Proximal Policy Optimization (PPO) Algorithms
- On-policy, no replay buffer
Proximal Policy Optimization (PPO) is introduced by OpenAI in their paper, [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). A detailed explanation and implementation guide can also be found on the [Spinning Up PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

PPO builds on the theoretical foundations of [Trust Region Policy Optimization (TRPO)](https://spinningup.openai.com/en/latest/algorithms/trpo.html#background), aiming to achieve stable and reliable policy updates with a simpler implementation.


### :sparkles: Motivation
In RL, policies are not trained on static datasets. Instead, they generate their own data via interaction with the environment. If the policy is poor, it collects low-quality data, which in turn worsens the policy to never recover.

PPO addresses this by **constraining policy updates**, avoiding large destructive changes during training.

#### 1. Vanilla Policy Gradient
- Updates the policy directly in the direction that increases expected return.
- Prone to instability due to large policy updates.
#### 2. TRPO
- Adds a trust region constraint to prevent large policy shifts using KL divergence.
- Involves solving a constrained optimization problem, which is complex and computationally expensive.
#### 3. PPO
- Simplifies TRPO by using a **clipped surrogate objective**.
- Avoids the complexity of second-order optimization.
- Allows multiple epochs of minibatch updates using the same data.