# Actor-Critic
The paper this concept originates from is [Neuronlike adaptive elements that can solve difficult learning control problems](https://ieeexplore.ieee.org/document/6313077) (1983) by Sutton, Barto, and Anderson.

## Paper Review (Key insights)
- combine value-based and policy-based
    - Actor: the policy network
    - Critic: the value function
- low variance
    - compared to REINFORCE due to reliance on Monte Carlo estimates
- sample efficiency 
    - compared to Q-learning, since it update only state-action values
- leading to:
    - Advantage Actor-Critic (A2C)
    - Asynchronous Advantage Actor-Critic (A3C)
    - Deep Deterministic Policy Gradient (DDPG)

## Actor-Critic Algorithm
Updating the **policy (actor)** using feedback from the **value function (critic)**

### **Algorithm:**
- **Initialize**  
    - policy parameters $\theta$ (actor)  
    - value function parameters $V_w$ (critic)
- **for $episode=1$ to $N$**:
    - **Initialize** state $s_0$
    - **Run episode** for time $t$ until the episode ends:
      1. **Select action** $a_t \sim \pi_\theta(a_t | s_t)$
      2. **Execute action** $a_t$ and observe reward $r_t$ and next state $s_{t+1}$
      3. **Compute Temporal Difference (TD) Error**:
         $\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$
      4. **Update Critic (Value Function)** using TD error: $w \leftarrow w + \alpha_c \delta_t \nabla_w V_w(s_t)$
      5. **Update Actor (Policy Parameters)** using policy gradient: $\theta \leftarrow \theta + \alpha_a \delta_t \nabla_\theta \log \pi_\theta(a_t | s_t)$
      6. **Move to the next state** $s_t \leftarrow s_{t+1}$
    - **End episode** when terminal state is reached.

