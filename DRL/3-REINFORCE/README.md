# REINFORCE
The paper this concept originates from is [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696) (1992)
- Monte Carlo Policy Gradient (MCPG) algorithm
- optimize a stochastic policy using policy gradient methods
- directly optimize the policy $(\pi\theta(a|s))$
- pure policy, hence **Actor-only**

## Paper Review (Key insights)
- immediate reinforcement (learn by most recent input-output pair only)
- use of connectionist network
- episodic, update after every episode

## Algorithm
### REINFORCE Formulas
**Objective functions:**  
```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
```  
Since the expectation is taken over all trajectories sampled from the policy, the **expected return** depends on the probability of sampling trajectories:
```math
J(\theta) = \sum_{\tau} P_{\theta}(\tau) R(\tau)
```  
where:  
- $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$ is the total discounted reward for the trajectory.  

**Gradient of the objective function:**  
```math
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \left( \sum_{k=t}^{T} \gamma^{k-t} r_k \right) \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) \right]
```
where:
- $r_k$ is the reward received at time step $k$  

**REINFORCE Update Rule**
```math
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \left( \sum_{k=t}^{T} \gamma^{k-t} r_k \right) \nabla_{\theta} \log \pi_{\theta}(a_t | s_t)
```

### REINFORCE Algorithm (Monte Carlo Policy Gradient)
#### **Input:**
- A differentiable policy $\pi_{\theta}(a | s)$ with parameters $\theta$
- Learning rate $\alpha$
- Discount factor $\gamma$
- Number of episodes $N$

#### **Algorithm:**
- **Initialize** policy parameters $\theta$ randomly.
- **For** episode $i = 1$ to $N$ **do**:
    1. **Generate an episode** $(s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)$ following policy $\pi_{\theta}$.
    2. **Compute returns** for each timestep $t$:
      $R_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$
    3. **For each timestep $t$ in the episode**:
        - Compute the gradient of the policy:
         $\nabla_{\theta} \log \pi_{\theta}(a_t | s_t)$
        - Compute the policy gradient update:
         $\theta \leftarrow \theta + \alpha R_t \nabla_{\theta} \log \pi_{\theta}(a_t | s_t)$
- **End For**
- **Return** optimized policy $\pi_{\theta}$.
