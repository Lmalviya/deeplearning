# Chapter 1 — Appendix: Deep Dives on Tricky Concepts

---

## A. Interpretability of Loss Values (MSE vs MAE)

### The Interview Question

*"Is the MSE loss value interpretable? What does it tell you?"*

This question is really asking: **can you look at the loss number and understand what it means in the real world?**

### Why MAE is Interpretable

MAE is in the **same units as your target variable**. If you're predicting house prices in dollars:

```
Target: $300,000    Prediction: $320,000    Error: $20,000
MAE = $20,000  → "On average, my model is off by $20,000"
```

That sentence makes sense to a business stakeholder. It has a direct, concrete meaning.

### Why MSE is NOT Interpretable

MSE squares the errors, so its units are **squared units of the target**.

```
Target: $300,000    Prediction: $320,000    Error: $20,000
MSE = 20,000² = 400,000,000 dollars²
```

"My model has a loss of 400 million squared dollars." That means nothing to anyone.

### The Fix: RMSE

RMSE = √MSE brings units back to the original scale.

```
RMSE = √400,000,000 = $20,000
```

Now interpretable again. But RMSE ≠ MAE even though they're in the same units. RMSE is always ≥ MAE because squaring before averaging gives more weight to large errors.

### How to Answer This in an Interview

Frame it around **three dimensions of interpretability**:

**1. Unit interpretability** — Can you explain the number to a non-ML person?
- MAE: Yes. "Average absolute error is X units."
- MSE: No. Units are squared.
- RMSE: Yes. Same units as target.

**2. Scale sensitivity** — Does the number change meaning with data scale?
- MSE: Yes, dramatically. Predicting temperatures in Celsius vs Kelvin gives wildly different MSE even for the same model.
- MAE/RMSE: Linear scaling — if you multiply targets by 10, loss multiplies by 10 (MSE multiplies by 100).

**3. Error magnitude intuition** — Does a higher loss always mean a worse model?
- Both yes in relative terms, but absolute MSE value is hard to compare across datasets.

### Complete Answer Template

> "MSE is not directly interpretable because squaring creates units of target² which have no real-world meaning. MAE is interpretable because it represents the average absolute deviation in the same units as the target — you can directly say 'the model is off by X on average.' For MSE, we typically report RMSE to recover interpretability, though RMSE ≠ MAE even in the same units since RMSE penalizes large errors more. In practice, I use MSE/RMSE for training (smooth gradients near zero) but evaluate and communicate results using MAE or RMSE depending on whether outliers should be emphasized."

---

## B. Subgradients — What, When, and Why

### The Problem That Requires Subgradients

Standard gradient descent requires the loss to be **differentiable everywhere**. But MAE has the term |error|, and the absolute value function has a **kink at zero** — it's not differentiable there.

```
f(x) = |x|

Derivative:
  x > 0 → f'(x) = +1
  x < 0 → f'(x) = -1
  x = 0 → f'(x) = ??? (left derivative = -1, right derivative = +1 → they don't agree)
```

Classical calculus says: the derivative at x=0 **does not exist**. So you can't run gradient descent?

### The Subgradient — Extending the Idea

A **subgradient** is a generalization of the gradient for non-differentiable (but convex) functions.

**Formal definition:** A value `g` is a subgradient of `f` at point `x₀` if for **all** x:

$$f(x) \geq f(x_0) + g \cdot (x - x_0)$$

In plain English: `g` defines a **supporting hyperplane** — a line that touches the function at x₀ and lies entirely *below* (or on) the function everywhere else.

### Visualizing It

```
f(x) = |x|

        |  /
        | /
        |/          ← The "kink" is here at x = 0
        |\
        | \
   ─────┼──────→ x
       0

At x = 0, you can draw ANY line with slope g ∈ [-1, +1]
that "supports" the function from below.
All of these are valid subgradients.
```

So the **subdifferential** (set of all subgradients) at x=0 is the interval [-1, +1].

At any other point where f is differentiable, the subdifferential contains exactly one value — the regular gradient.

### The Subgradient Rule for |x|

$$\partial |x| = \begin{cases} \{-1\} & x < 0 \\ [-1, +1] & x = 0 \\ \{+1\} & x > 0 \end{cases}$$

In practice, frameworks like PyTorch just **pick one subgradient** at the kink — typically 0. This is an arbitrary but valid choice.

```python
x = torch.tensor(0.0, requires_grad=True)
loss = torch.abs(x)
loss.backward()
print(x.grad)  # → 0.0  (PyTorch picks the midpoint subgradient)
```

### When Do You Need Subgradients?

Any time a loss or activation has a **non-differentiable point**:

| Function | Non-differentiable At | Subgradient Used |
|---|---|---|
| MAE / L1 loss | error = 0 | Any value in [-1, +1] |
| ReLU | x = 0 | Typically 0 (by convention) |
| Hinge loss | margin = 0 | Any value in [0, 1] |
| Huber loss | \|error\| = δ | Continuous by design (no kink) |
| L1 regularization | weight = 0 | Any value in [-λ, +λ] |

### Why Subgradients Still Work

Subgradient methods **converge** for convex functions even with non-differentiable points because:

1. You're still moving in a **descent direction** (the subgradient points uphill, so negating it goes downhill)
2. The non-differentiable point (e.g., error = 0) is the **minimum** — you'd stop there anyway
3. For neural networks, hitting exactly x = 0 has **measure zero probability** — it almost never happens in practice

### Subgradients vs Smooth Approximations

Sometimes instead of dealing with subgradients, people **smooth out the kink**:

- Huber loss smooths the MAE kink at zero → fully differentiable, no subgradient needed
- Softplus smooths ReLU: `log(1 + eˣ)` → differentiable everywhere
- Pseudo-Huber loss: another smooth approximation of MAE

**Trade-off:** Smooth approximations are mathematically cleaner but change the loss slightly. Subgradients keep the exact loss but require more care in optimization.

### Complete Answer Template

> "A subgradient is a generalization of the gradient for functions that are convex but not differentiable everywhere — like MAE at zero error. At non-differentiable points, instead of a single gradient value, there's a set of valid subgradients — any slope that defines a supporting hyperplane below the function. For MAE at zero, any value in [-1, +1] is a valid subgradient; PyTorch picks 0 by convention. Subgradient descent still converges for convex functions because these kink points are typically minima anyway. In practice, hitting exactly zero has near-zero probability during training, so it rarely matters. Alternatively, Huber loss avoids the issue entirely by smoothing the kink, trading exactness for differentiability."

---

## C. Probabilistic Basis — Why It Matters

### The Question Behind the Question

When interviewers ask about the "probabilistic basis" of a loss function, they're really asking:

> *"Why did you choose this loss? Is it principled, or did you just guess?"*

A loss with a probabilistic basis means it **follows from first principles** — specifically, from Maximum Likelihood Estimation (MLE). This makes it defensible and connects the loss to the underlying assumption about your data.

### The Framework: MLE

You have a model that outputs a prediction. You assume the **residuals** (errors) follow some probability distribution. You then ask: *what parameters make the observed data most probable under this model?*

$$\hat{\theta} = \arg\max_\theta \prod_{i=1}^n p(y_i | x_i; \theta)$$

Taking the negative log (flips max to min, converts product to sum):

$$\hat{\theta} = \arg\min_\theta -\sum_{i=1}^n \log p(y_i | x_i; \theta)$$

This is the **NLL (Negative Log-Likelihood)**. Different distributional assumptions give different loss functions.

### Case 1: Gaussian Noise → MSE

Assume the true label is your prediction plus Gaussian noise:

$$y_i = \hat{y}_i + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

Probability of observing $y_i$:

$$p(y_i | x_i) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}\right)$$

Log-likelihood:

$$\log p = -\frac{(y_i - \hat{y}_i)^2}{2\sigma^2} - \text{const}$$

Minimizing negative log-likelihood = minimizing $(y_i - \hat{y}_i)^2$ = **MSE**.

**Implication:** Using MSE means you are implicitly assuming your residuals are Gaussian. If they're not, MSE is not the optimal loss.

### Case 2: Laplace Noise → MAE

Assume residuals follow a Laplace (double exponential) distribution:

$$p(y_i | x_i) = \frac{1}{2b} \exp\left(-\frac{|y_i - \hat{y}_i|}{b}\right)$$

Log-likelihood:

$$\log p = -\frac{|y_i - \hat{y}_i|}{b} - \text{const}$$

Minimizing NLL = minimizing $|y_i - \hat{y}_i|$ = **MAE**.

**Implication:** MAE assumes Laplace-distributed residuals. The Laplace distribution has heavier tails than Gaussian — it puts more probability mass on large deviations. This is why MAE is more robust: it's the *right* loss when you expect occasional large errors (outliers).

### Case 3: Bernoulli → Binary Cross-Entropy

For binary classification, the outcome follows a Bernoulli distribution:

$$p(y_i | x_i) = \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1-y_i}$$

Log-likelihood:

$$\log p = y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)$$

Minimizing NLL = **Binary Cross-Entropy**.

### Comparing the Distributions — The Real Insight

```
          Gaussian         Laplace
          (MSE)            (MAE)

Tails:    Light            Heavy
          (outliers very   (outliers more
          improbable)       probable)

Penalty:  Quadratic        Linear
          (outlier         (outlier
          explodes loss)    contained)
```

**The key insight:** Choosing a loss = choosing a belief about the noise in your data.

- Believe your data is clean with rare flukes? → Gaussian → MSE
- Believe your data regularly has large outliers? → Laplace → MAE
- Not sure? → Huber (somewhere between the two, with δ as the dial)

### Why This Matters in Interviews

It lets you give **principled answers** to design questions:

*"Why did you choose MSE for this regression task?"*
> "MSE follows from MLE under a Gaussian noise assumption. Our residuals from a validation set roughly followed a normal distribution with few outliers, so MSE is the principled choice."

*"The model performs poorly on test data with occasional extreme values. What loss would you switch to?"*
> "The extreme values suggest the error distribution has heavy tails — more consistent with a Laplace distribution than Gaussian. I'd switch to MAE or Huber loss, which are the MLE losses under Laplace and a mixture assumption respectively."

### Beyond Regression — The Universal Principle

Every standard loss is an MLE loss under some assumption:

| Loss | Distribution | Why It's Natural |
|---|---|---|
| MSE | Gaussian | Residuals are small and symmetric |
| MAE | Laplace | Residuals have heavy tails (outliers) |
| BCE | Bernoulli | Binary outcomes |
| Cross-Entropy | Categorical | Multi-class outcomes |
| KL Divergence | Any pair of distributions | Comparing/fitting distributions |
| Poisson NLL | Poisson | Count data (integer targets ≥ 0) |

This is the unifying principle: **loss functions are not arbitrary — they encode your belief about how your data was generated**.

---

## D. Convergence Behavior: MSE vs MAE Near the Optimum

### The Setup

This is a question about the shape of the **loss landscape** near the minimum and what that means for gradient descent.

Let's think about a single parameter case: we're minimizing L(w) and we're close to the optimal w*.

### MSE Near the Optimum — Why It Can Overshoot

**Gradient of MSE:** $\frac{\partial \text{MSE}}{\partial w} = -\frac{2}{n} \sum (y_i - \hat{y}_i) \cdot x_i$

As the model approaches the optimum, the **errors get small**. Since the MSE gradient scales with the error itself, the gradient **naturally shrinks**.

```
Far from optimum:   error = 10  → gradient ∝ 10  (large step)
Close to optimum:   error = 0.1 → gradient ∝ 0.1 (small step)
At optimum:         error = 0   → gradient = 0   (stops)
```

This looks ideal. So where does overshooting come from?

**The problem is the learning rate.** With a fixed learning rate η:

```
Step size = η × gradient

Far from optimum:  step = η × 10  (appropriate large step)
Close to optimum:  step = η × 0.1 (still appropriate, just smaller)
```

MSE is actually well-behaved near the optimum in this regard. The shrinking gradient provides **natural momentum reduction**.

But here's the real issue: the same quadratic nature that helps near the optimum causes **explosive gradients far from it**. If your learning rate is tuned for being near the optimum, a large early error can produce a massive gradient update that **overshoots** the minimum completely — possibly to the other side.

```
Parabolic loss landscape:
    
         \      /
          \    /      ← With high lr, gradient step from far left
           \  /         can shoot past the minimum to the far right
            \/
            w*

Large error → large gradient → large step → might land far past w*
```

The gradient magnitude is proportional to the distance from the optimum. If your learning rate is too large for that distance, you overshoot.

### MAE Near the Optimum — Why It Oscillates Instead of Slowing

**Gradient of MAE:** $\frac{\partial \text{MAE}}{\partial w} = -\frac{1}{n} \sum \text{sign}(y_i - \hat{y}_i) \cdot x_i$

The gradient is **constant in magnitude** regardless of error size. The sign tells you direction; the magnitude is always ±1/n.

```
Far from optimum:   error = 10  → gradient ∝ ±1  (step = η × 1)
Close to optimum:   error = 0.1 → gradient ∝ ±1  (step = η × 1, SAME SIZE!)
Very close:         error = 0.001 → gradient ∝ ±1 (STILL same size!)
```

This is the problem. Near the minimum, you're still taking the same size steps. The loss landscape is **V-shaped** — steep constant slopes meeting at a point.

```
V-shaped loss landscape (MAE):

      \        /
       \      /
        \    /       ← Near the bottom, gradient = ±1 (constant)
         \  /          Each step of size η bounces you left or right
          \/
          w*

If you're slightly left of w*, gradient says "go right" with magnitude 1
You step right by η, now you're slightly right of w*
Gradient says "go left" with magnitude 1
You step left by η...
→ Oscillates around w* indefinitely
```

MAE training can **oscillate** near the minimum unless the learning rate is decayed.

### Head-to-Head Comparison

```
Distance from   | MSE gradient  | MAE gradient  | Who wins?
optimum         | magnitude     | magnitude     |
─────────────────────────────────────────────────────────────
Far (error=10)  | ∝ 10          | constant ∝ 1  | MSE (faster)
Medium (error=1)| ∝ 1           | constant ∝ 1  | Tie
Close (error=.1)| ∝ 0.1         | constant ∝ 1  | MAE overshoots, MSE naturally slows
Very close      | ∝ 0.01        | constant ∝ 1  | MSE converges cleanly, MAE oscillates
```

### Why Huber Gets Both Right

Huber is quadratic (MSE-like) for small errors and linear (MAE-like) for large errors.

```
Far from optimum: Huber behaves like MAE → bounded gradients, no explosion
Close to optimum: Huber behaves like MSE → shrinking gradients, smooth convergence
```

This is the deeper reason Huber is preferred when you want reliable training:
- It won't explode from outliers (MAE-like)
- It won't oscillate near the minimum (MSE-like)

### The Learning Rate Decay Connection

Both oscillation (MAE) and overshooting (MSE) are mitigated by **learning rate decay**:

- Decaying lr → smaller steps as training progresses → MAE can converge
- Adaptive optimizers (Adam, RMSProp) effectively provide per-parameter learning rate scaling, which helps both

But the fundamental difference remains: with a **fixed learning rate**, MSE self-corrects near the optimum (gradients shrink naturally), while MAE does not.

### Complete Answer Template

> "MSE's gradient is proportional to the prediction error, so as the model converges and errors shrink, the gradient naturally shrinks too — this is self-correcting. However, far from the optimum, large errors produce large gradients that can overshoot the minimum if the learning rate is too high for the current scale of errors. MAE's gradient is constant in magnitude (just the sign of the error), so it takes the same size step regardless of how close you are to the minimum — this leads to oscillation near the optimum since you keep bouncing over the minimum at the same step size. Huber combines the best of both: constant gradient far away (no explosion) and linearly shrinking gradient near the optimum (smooth convergence). In practice, learning rate schedulers and adaptive optimizers like Adam mitigate both issues, which is why the theoretical difference matters less with modern training setups."