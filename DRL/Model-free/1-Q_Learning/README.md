# Q-Learning

- Model-free
- Off-policy

## Assumptions
**Markov Decision Process (MDP)** with
- a state space $S$
- an action space $A$
- a transition probability $P(s' | s, a)$
- a reward function $R(s, a)$

## Q-value
Expected cumulative reward an agent can obtain from a given state-action pair while following an optimal policy  
What reward can obtain by taking action $a$ at state $s$ under policy $\pi$?
```math
Q^\pi(s, a) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a, \pi \right]
$$
```
## Q-Value Update Equation
```math
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
```
### Derivation
#### Bellman Equation for Q-values
The recursive relationship between the Q-values  
Optimal Q-value of a state-action pair is equal to the immediate reward plus the best possible discounted future reward
```math
Q^*(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]
```
where:
- $Q^*(s, a)$ is the optimal Q-value function,
- $r$ is the immediate reward,
- s' is the next state reached after taking action $a$ in state $s$,
- $\max_{a'} Q^*(s', a')$ is the maximum estimated future reward if we act optimally in the next state

Converting to an Update Rule
```math
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
```
The **temporal difference (TD) error** measures how far our current estimate of $Q(s, a)$ is from the true expected value.
```math
\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)
```
Apply a learning rate $\alpha$ to update $Q(s, a)$, so that update is stable
```math
Q(s, a) \leftarrow Q(s, a) + \alpha \delta
```

# Gymnasium Environment
conda create --name q_learning python=3.9
pip install gymnasium
pip install "gymnasium[toy-text]"

# Reference
The code is based on [youtube channel](https://www.youtube.com/watch?v=ZhoIgo3qqLU) and its [code](https://github.com/johnnycode8/gym_solutions)