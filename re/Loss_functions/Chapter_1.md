# Loss Functions — Chapter 1: Foundations & Regression Losses

> **Interview Prep Notes** | Covers: What is a loss, MSE, MAE, Huber, NLL, KL Divergence

---

## 1. What is a Loss Function?

A **loss function** (also called a **cost function** or **objective function**) measures how far a model's predictions are from the true values. Training a neural network is an optimization problem: minimize `L(θ)` over parameters `θ`.

### Key Distinctions

| Term | Meaning |
|---|---|
| **Loss** | Error on a single sample |
| **Cost** | Average loss over the entire dataset |
| **Objective** | The quantity being optimized (can be loss + regularization) |

### Properties of a Good Loss Function

- **Differentiable** — needed for gradient-based optimization
- **Convex** (ideally) — guarantees a global minimum
- **Sensitive to the right errors** — penalizes what matters for the task
- **Numerically stable** — no overflow/underflow in practice

### The Optimization Loop

```
Forward pass → Compute loss → Backprop (∂L/∂θ) → Update θ via optimizer
```

The loss gradient tells each parameter how much it contributed to the error.

---

## 2. Mean Squared Error (MSE) / L2 Loss

### Formula

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Intuition

MSE penalizes errors **quadratically**. A prediction that's 2 units off gets a penalty of 4; a prediction that's 4 units off gets 16. This makes MSE **very sensitive to large errors** (outliers).

### Gradient

$$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)$$

The gradient is **linear in the error** — large errors produce large gradients, which is why training converges fast when far from the target but can overshoot near it.

### Properties

| Property | Detail |
|---|---|
| **Convex** | Yes — has a single global minimum |
| **Differentiable** | Everywhere |
| **Outlier sensitivity** | High — squared term amplifies outliers |
| **Units** | Squared units of the target (hard to interpret directly) |
| **Probabilistic basis** | MLE under Gaussian noise assumption |

### When to Use

- Regression tasks where large errors should be penalized heavily
- Output distribution is assumed to be Gaussian
- Data is relatively clean (no significant outliers)

### When NOT to Use

- Data has outliers → prefer Huber or MAE
- You care about absolute error magnitude, not squared

### MSE vs RMSE

**RMSE** (Root MSE) = √MSE. Same mathematical behavior, but units match the target, making it more interpretable. Preferred for reporting; MSE preferred for training (avoids the sqrt's gradient complexity).

---

## 3. Mean Absolute Error (MAE) / L1 Loss

### Formula

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Intuition

MAE penalizes all errors **equally** (linearly). Every unit of error contributes the same amount regardless of magnitude — hence much more **robust to outliers**.

### Gradient

$$\frac{\partial \text{MAE}}{\partial \hat{y}_i} = -\frac{1}{n} \cdot \text{sign}(y_i - \hat{y}_i)$$

The gradient is **constant** (±1/n), regardless of error size. This means:
- Training doesn't speed up for large errors (unlike MSE)
- Training doesn't slow down near the optimum either
- **Not differentiable at zero** — requires a subgradient at that point

### Properties

| Property | Detail |
|---|---|
| **Convex** | Yes |
| **Differentiable** | Not at zero |
| **Outlier sensitivity** | Low — linear penalty |
| **Units** | Same units as the target |
| **Probabilistic basis** | MLE under Laplace noise assumption |

### When to Use

- Regression with outliers in the data
- When you want a model to predict the **median** (vs. MSE which predicts the **mean**)
- Robust regression tasks

### MSE vs MAE Summary

| | MSE | MAE |
|---|---|---|
| Penalty type | Quadratic | Linear |
| Outlier sensitivity | High | Low |
| Optimal prediction | Mean | Median |
| Differentiable at 0 | Yes | No |
| Gradient near zero | Approaches 0 (smooth) | Constant (can oscillate) |

---

## 4. Huber Loss (Smooth L1)

### Motivation

Huber loss is the **best of both worlds**: behaves like MSE for small errors (smooth, fast convergence) and like MAE for large errors (robust to outliers). It solves the non-differentiability of MAE and the outlier sensitivity of MSE.

### Formula

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

### Intuition

- For errors within `δ`: quadratic (MSE behavior) → smooth gradients near zero
- For errors beyond `δ`: linear (MAE behavior) → large errors don't dominate
- `δ` is a **hyperparameter** that defines the transition point

### Gradient

$$\frac{\partial L_\delta}{\partial \hat{y}} = \begin{cases} -(y - \hat{y}) & \text{if } |y - \hat{y}| \leq \delta \\ -\delta \cdot \text{sign}(y - \hat{y}) & \text{otherwise} \end{cases}$$

### Key Insight: δ as a Threshold

- **Small δ**: Loss behaves more like MAE (robust but slow near optimum)
- **Large δ**: Loss behaves more like MSE (fast but outlier-sensitive)
- Choosing δ ≈ the expected noise level in your data is a good heuristic

### Properties

| Property | Detail |
|---|---|
| **Differentiable** | Everywhere (including at ±δ) |
| **Outlier sensitivity** | Low |
| **Tuneable** | δ controls MSE-MAE tradeoff |
| **Use case** | Regression with noisy data, reinforcement learning, object detection |

### Where It's Used

- **Object detection** (e.g., bounding box regression in Faster R-CNN, SSD)
- **Reinforcement learning** (TD error clipping in DQN)
- Any robust regression task

### Common Interview Question

**"When would you choose Huber over MSE?"**
> When your regression target has outliers or the data is noisy. Huber gives you MSE's fast convergence for typical errors while capping the influence of outliers.

---

## 5. Negative Log-Likelihood (NLL) Loss

### Foundation: Maximum Likelihood Estimation

NLL is the backbone of most probabilistic loss functions. The idea: given a model that outputs a probability distribution, we want to **maximize the probability it assigns to the true data**.

For a dataset with i.i.d. samples:

$$\mathcal{L}(\theta) = \prod_{i=1}^{n} p_\theta(y_i | x_i)$$

Taking the **negative log** (converts product → sum, log is monotonic so same optimum):

$$\text{NLL}(\theta) = -\sum_{i=1}^{n} \log p_\theta(y_i | x_i)$$

### For Classification

If the model outputs a probability distribution over C classes, the NLL for one sample is:

$$\text{NLL} = -\log p_\theta(y = c | x)$$

where `c` is the true class. In practice, the model outputs logits, converted to probabilities via softmax. The loss then becomes **Cross-Entropy** (covered in Chapter 2).

### Connection to Other Losses

| Noise Assumption | MLE → Loss |
|---|---|
| Gaussian | MSE |
| Laplace | MAE |
| Bernoulli | Binary Cross-Entropy |
| Categorical | Cross-Entropy / NLL |

**Key insight:** Most loss functions are NLL under some distributional assumption. This is why they work — we're doing maximum likelihood estimation.

### NLL in PyTorch

```python
import torch
import torch.nn as nn

# NLL expects log-probabilities as input (use log_softmax before)
log_probs = torch.log_softmax(logits, dim=1)
loss = nn.NLLLoss()(log_probs, targets)

# Equivalent (and more numerically stable):
loss = nn.CrossEntropyLoss()(logits, targets)  # applies log_softmax internally
```

---

## 6. KL Divergence

### Definition

KL Divergence measures **how different one probability distribution is from another**. Specifically, KL(P || Q) measures the information lost when using Q to approximate P.

$$D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

For continuous distributions:

$$D_{KL}(P \| Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx$$

### Properties

| Property | Detail |
|---|---|
| **Non-negative** | KL(P\|\|Q) ≥ 0 always (Gibbs' inequality) |
| **Zero iff P = Q** | KL = 0 exactly when distributions are identical |
| **Asymmetric** | KL(P\|\|Q) ≠ KL(Q\|\|P) |
| **Not a metric** | No triangle inequality, not symmetric |

### Asymmetry — Critical Intuition

This is **the most important thing** to understand about KL divergence.

**Forward KL: KL(P || Q)** — "mode-covering"
- We minimize the cost of using Q to approximate P
- Q is penalized heavily when P(x) > 0 but Q(x) ≈ 0 (Q must cover all of P's mass)
- Result: Q tends to spread out and cover all modes of P
- Also called "inclusive KL"

**Reverse KL: KL(Q || P)** — "mode-seeking"
- We minimize the cost of using P to approximate Q
- Q is penalized heavily when Q(x) > 0 but P(x) ≈ 0
- Result: Q concentrates on one mode of P, ignoring others
- Also called "exclusive KL"
- Used in Variational Inference (ELBO), VAEs

```
P = multimodal (two peaks)
Forward KL → Q is a broad Gaussian covering both peaks
Reverse KL → Q is a narrow Gaussian sitting on one peak
```

### Relation to Cross-Entropy

$$D_{KL}(P \| Q) = H(P, Q) - H(P)$$

where H(P,Q) is cross-entropy and H(P) is the entropy of P. If P is fixed (true labels), minimizing cross-entropy = minimizing KL divergence. This is why cross-entropy is the standard classification loss.

### Where KL Divergence Is Used

1. **VAEs (Variational Autoencoders)**: Regularization term in ELBO — KL(q(z|x) || p(z)) pushes the learned posterior toward the prior
2. **Knowledge Distillation**: KL(p_teacher || p_student) — trains student to match teacher's soft distribution
3. **Reinforcement Learning**: PPO uses KL constraint to prevent policy updates from being too large
4. **Language Models**: KL divergence between current and reference policy (RLHF, DPO)
5. **Distributional Shift**: Measuring dataset drift

### KL in Practice (VAE Example)

For a Gaussian posterior q(z|x) = N(μ, σ²) against prior p(z) = N(0,1):

$$D_{KL}(q \| p) = \frac{1}{2} \sum_j \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)$$

This has a **closed form** — no sampling needed.

### Jensen-Shannon Divergence (JSD)

A symmetrized, bounded version of KL:

$$\text{JSD}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{P+Q}{2}$$

- Always between 0 and log(2)
- Symmetric: JSD(P||Q) = JSD(Q||P)
- Used in **GANs** (original GAN objective minimizes JSD between real and generated distributions)

---

## 7. Summary Table — Regression & Probabilistic Losses

| Loss | Formula | Outlier Robust | Differentiable | Use Case |
|---|---|---|---|---|
| MSE | (y - ŷ)² | No | Yes | Clean regression |
| MAE | |y - ŷ| | Yes | Robust regression |
| Huber | MSE if small, MAE if large | Yes | Yes | Noisy regression, detection |
| NLL | -log p(y|x) | Depends | Yes | Probabilistic models |
| KL Divergence | Σ P log(P/Q) | N/A | Yes | VAEs, distillation, RL |

---

## 8. Common Interview Questions

**Q: Why do we use log in loss functions?**
> Log converts products of probabilities into sums (numerically stable), and log is monotonically increasing so it doesn't change the location of the maximum. It also maps (0,1] → (-∞, 0], giving well-behaved gradients.

**Q: What does it mean to minimize MSE vs. MAE?**
> MSE minimization finds the conditional **mean** of the target distribution. MAE minimization finds the conditional **median**. This matters when the data has skewed or heavy-tailed distributions.

**Q: MSE and MAE — which converges faster?**
> MSE converges faster far from the optimum (large gradients for large errors) but can struggle near the optimum (gradients also shrink). MAE has constant gradient magnitude, which can cause oscillation near the minimum but is steady far away.

**Q: Is KL divergence a distance metric?**
> No. It's not symmetric (KL(P||Q) ≠ KL(Q||P)) and doesn't satisfy the triangle inequality. It's a divergence, not a distance. JSD is a symmetrized variant.

**Q: Why is Huber used in object detection?**
> Bounding box regression targets can vary wildly in scale, and outlier anchors with large localization errors would dominate MSE training. Huber loss caps the gradient magnitude for large errors, leading to more stable training.

**Q: Explain the connection between MLE and cross-entropy.**
> Maximizing the likelihood of data under a model = maximizing Σ log p(yᵢ|xᵢ) = minimizing -Σ log p(yᵢ|xᵢ) = minimizing NLL. For a categorical distribution, NLL = cross-entropy. So cross-entropy loss IS maximum likelihood estimation.