# Diffusion Model
Diffusion model is introduced in paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by J Ho et al. (2020)

## [Notes on Diffusion Policy Controlling Robots](https://www.youtube.com/watch?v=zc5NTeJbk-k&t=3s)

# [Notes on score functions (gradients of log probability density functions)](https://yang-song.net/blog/2021/score/) by Yang Song
### Model
Difficulty of directly model the probability density function (likelihood-based models):  
- Define p.d.f. as $p_{\theta}(\mathbf{x}) = \frac{e^{-f_{\theta}(\mathbf{x})}}{Z_{\theta}}$
    - use $Z_{\theta}$ to normalize, such that $\int p_{\theta}(\mathbf{x}) d\mathbf{x} = 1$
    - $e^{-f_{\theta}(\mathbf{x})}$ guarantee $p>0$  
    - since $Z_{\theta}$ is parameterized by $\theta$, need to make it tractable

Modeling the score function instead of the density function (solve the problem):
- score function of distribution $p(\mathbf{x})$ is defined as $\nabla_{\mathbf{x}} \log p(\mathbf{x})$
- score-based model is learned through:
```math
\mathbf{s}_\theta(\mathbf{x}) 
= \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x})
= -\nabla_{\mathbf{x}} f_\theta(\mathbf{x}) 
   \;-\; \underbrace{\nabla_{\mathbf{x}} \log Z_\theta}_{=0}
= -\nabla_{\mathbf{x}} f_\theta(\mathbf{x})
```
### Langevin dynamics
- a MCMC procedure to sample from distribution $p(\mathbf{x})$
    - initialize from random prior $\mathbf{x}_0 \sim \pi(\mathbf{x})$
    - iterate  
    $\mathbf{x}_{i+1} \leftarrow \mathbf{x}_i + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \sqrt{2\epsilon} \mathbf{z}_i, \quad i = 0,1, \dots, K$  
- Training objective:
```math
\mathbb{E}_{p(\mathbf{x})} \left[ \|\nabla_{\mathbf{x}} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x})\|_2^2 \right] = \int p(\mathbf{x}) \|\nabla_{\mathbf{x}} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x})\|_2^2 d\mathbf{x}
```
### Multiple noise perturbations
For naive score-based generative modeling, estimated score functions are inaccurate in low density regions since there are little data in the area.
- noise-perturbed data distribution: $p_{\sigma_i}(\mathbf{x}) = \int p(\mathbf{y}) \mathcal{N}(\mathbf{x}; \mathbf{y}, \sigma_i^2 \mathbf{I}) d\mathbf{y}$
    - Instead of using $p(\mathbf{x})$, we add Gaussian noise with standard deviation $\sigma_i$ to create a set of smoothed distributions $p_{\sigma_i}(\mathbf{x})$.
    - Multiple noise levels (discrete) $\sigma_1 < \sigma_2 < \dots < \sigma_L$, so we have multiple versions of the data distribution, each corresponding to a different level of noise.
- having multiple perturbed distributions, we define a score function for each:
```math
s_\theta(\mathbf{x}, i) \approx \nabla_{\mathbf{x}} \log p_{\sigma_i}(\mathbf{x}).
```
- Instead of learning a single score function for $p(\mathbf{x})$, we now train a *Noise Conditional Score-Based Model (NCSN)* that learns the score for each noise level $\sigma_i$. This function estimates the gradient of the log-likelihood for each noise-perturbed distribution. Hence, the new objective:
```math
\sum_{i=1}^{L} \lambda(i) \mathbb{E}_{p_{\sigma_i}(\mathbf{x})} \left[ \|\nabla_{\mathbf{x}} \log p_{\sigma_i}(\mathbf{x}) - s_\theta(\mathbf{x}, i)\|_2^2 \right],
```
## Score-based generative modeling with stochastic differential equations (SDEs)
### Perturbing data with an SDE
- Perturb the data distribution with continuously (emphasize on ***continuous***) growing levels of noise: $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}$
    - $\mathbf{f}(\mathbf{x}, t)$: (vector valued) drift coefficient, which determines the deterministic evolution of $\mathbf{x}(t)$ over time.
    - $g(t)$: (scalar valued) diffusion coefficient, which controls the amount of noise injected into the system.
    - $d\mathbf{w}$: **infinitesimal white noise**, which introduces randomness at each small time step
- At $t=0$, $p_0(x) = p(x)$ since no perturbation is applied. After sufficiently long time $T$, $p_T(x)$ is close to a tractable noise distribution $\pi(x)$, which is the prior
### Reversing the SDE for sample generation
- Apply annealed Langevin dynamics
- Any SDE has a corresponding reverse SDE, given by:
```math
d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + g(t) d\mathbf{w}
```