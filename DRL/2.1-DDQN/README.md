# Double Deep Q-network
The paper this concept originates from is [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461)

## Paper Review (Key insights)
Instead of using the standard DQN update:
```math
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
```  
DDQN separates action selection and evaluation:
1. **Action Selection** using the online network: $a^* = \arg\max_{a'} Q_{\theta}(s', a')$
2. **Action Evaluation** using the target network: $Q_{\theta'}(s', a^*)$
Thus, the DDQN update rule becomes:
```math
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma Q_{\theta'}(s', \arg\max_{a'} Q_{\theta}(s', a')) - Q(s, a) \right)
```
where:
- $Q_{\theta}$ is the online Q-network
- $Q_{\theta'}$ is the target Q-network
- $a^*$ is the action chosen by the online network
- $\gamma$ is the discount factor
- $r$ is the immediate reward