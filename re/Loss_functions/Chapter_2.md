# Loss Functions — Chapter 2: Classification Losses

> **Interview Prep Notes** | Covers: Binary Cross-Entropy, Cross-Entropy, Categorical Cross-Entropy, NLL Loss, Focal Loss, Weighted Loss

---

## 1. Entropy — The Foundation

Before cross-entropy, understand **entropy** itself.

### Shannon Entropy

$$H(P) = -\sum_{x} P(x) \log P(x)$$

Entropy measures the **average uncertainty** (or information content) of a distribution.

- High entropy → distribution is spread out, uncertain (e.g., uniform)
- Low entropy → distribution is concentrated, certain (e.g., one-hot)

```
P = [0.5, 0.5]    → H = 1 bit  (maximum uncertainty for 2 classes)
P = [1.0, 0.0]    → H = 0 bits (no uncertainty)
P = [0.9, 0.1]    → H = 0.47 bits
```

---

## 2. Cross-Entropy

### Definition

Cross-entropy measures the **average number of bits needed to encode samples from distribution P using a code optimized for Q**.

$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$

- P = true distribution (ground truth)
- Q = predicted distribution (model output)

### Relation to KL Divergence

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

Since H(P) is fixed (doesn't depend on model parameters), **minimizing cross-entropy = minimizing KL divergence from true to predicted distribution**.

### For One-Hot Labels

In classification, P is typically a one-hot vector (label = 1 for the true class, 0 otherwise). Cross-entropy simplifies to:

$$H(P, Q) = -\log Q(y_\text{true})$$

We only care about the probability assigned to the **correct class**. All other terms are zero.

---

## 3. Binary Cross-Entropy (BCE)

### Use Case

Binary classification: one output neuron, sigmoid activation, predicting P(y=1|x).

### Formula

For a single sample with true label y ∈ {0, 1} and predicted probability ŷ = σ(z):

$$\text{BCE} = -[y \log \hat{y} + (1 - y) \log(1 - \hat{y})]$$

Over a batch of n samples:

$$\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)]$$

### Intuition

The two terms handle each case:
- When `y = 1`: loss = `-log(ŷ)` → penalize if ŷ is close to 0
- When `y = 0`: loss = `-log(1 - ŷ)` → penalize if ŷ is close to 1

```
Prediction ŷ = 0.9, True label y = 1  → loss = -log(0.9) ≈ 0.105   (small, correct)
Prediction ŷ = 0.1, True label y = 1  → loss = -log(0.1) ≈ 2.303   (large, wrong)
Prediction ŷ = 0.5, True label y = 1  → loss = -log(0.5) ≈ 0.693   (uncertain)
```

### Gradient

$$\frac{\partial \text{BCE}}{\partial z} = \hat{y} - y \quad \text{(where z is the logit, ŷ = σ(z))}$$

Beautiful result: the gradient is simply the **prediction error**. This is why sigmoid + BCE is the natural pairing — the sigmoid's derivative cancels with the BCE gradient, giving a clean linear gradient.

### Numerical Stability

Never compute `log(ŷ)` directly from sigmoid output — floating point issues when ŷ ≈ 0.

```python
# BAD — numerical instability
loss = -(y * torch.log(sigmoid(z)) + (1-y) * torch.log(1 - sigmoid(z)))

# GOOD — use BCEWithLogitsLoss (applies log-sum-exp trick internally)
loss = nn.BCEWithLogitsLoss()(z, y)  # Takes raw logits, not probabilities
```

The stable formula:
$$\text{BCE}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$$

### Multi-Label Classification

BCE extends naturally to multi-label problems (multiple classes can be 1 simultaneously):
- Each output neuron gets its own sigmoid + BCE
- No competition between classes (unlike softmax)

```python
# Multi-label: each class is independent binary prediction
loss = nn.BCEWithLogitsLoss()(logits, targets)  # targets shape: [B, num_classes]
```

---

## 4. Categorical Cross-Entropy (Softmax + Cross-Entropy)

### Use Case

Multi-class classification: C classes, one correct class per sample, softmax output.

### Softmax

$$\hat{y}_c = \text{softmax}(z_c) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$

Softmax converts raw logits into a probability distribution: outputs are positive and sum to 1.

### Cross-Entropy Loss

$$\text{CE} = -\sum_{c=1}^{C} y_c \log \hat{y}_c$$

With one-hot labels (only one y_c = 1):

$$\text{CE} = -\log \hat{y}_\text{true\_class}$$

Over a batch:

$$\text{CE} = -\frac{1}{n} \sum_{i=1}^{n} \log \hat{y}_{i, y_i}$$

### Gradient

$$\frac{\partial \text{CE}}{\partial z_c} = \hat{y}_c - y_c$$

Again, the gradient is simply **predicted probability minus true label** — for the correct class it's `(ŷ - 1)`, for all other classes it's `ŷ`. This elegance is why softmax + cross-entropy is the standard.

### Numerical Stability: Log-Sum-Exp

Computing softmax then log is numerically unstable (exp can overflow).

```python
# NEVER do this:
probs = torch.softmax(logits, dim=1)
loss = -torch.log(probs[range(n), targets])

# ALWAYS use this (applies log-sum-exp trick internally):
loss = nn.CrossEntropyLoss()(logits, targets)  # Takes raw logits
```

The stable computation: log(softmax(z)) = z - log(Σ exp(z)) = z - (max(z) + log(Σ exp(z - max(z))))

### CrossEntropyLoss vs NLLLoss in PyTorch

This is a **very common interview/practical confusion**.

```python
# These are equivalent:

# Option 1: CrossEntropyLoss (preferred)
loss = nn.CrossEntropyLoss()(logits, targets)
# Internally: applies log_softmax → then NLLLoss

# Option 2: Manual
log_probs = nn.LogSoftmax(dim=1)(logits)
loss = nn.NLLLoss()(log_probs, targets)
```

| Function | Input | What it does |
|---|---|---|
| `CrossEntropyLoss` | Raw logits | log_softmax + NLL |
| `NLLLoss` | Log-probabilities | -log_prob[true_class] |
| `BCEWithLogitsLoss` | Raw logits (binary) | Sigmoid + BCE |
| `BCELoss` | Probabilities | BCE only (unstable) |

> **Rule of thumb**: Always feed raw logits to PyTorch loss functions. Let the library handle the softmax/sigmoid + numerical stability.

---

## 5. Label Smoothing

### Problem with Hard Labels

One-hot labels make the model overconfident — it's encouraged to push the correct class logit to +∞.

### Label Smoothing

Replace hard one-hot labels with soft labels:

$$y_c^{\text{smooth}} = \begin{cases} 1 - \varepsilon & \text{if } c = \text{true class} \\ \varepsilon / (C - 1) & \text{otherwise} \end{cases}$$

Typical ε = 0.1 (10% mass distributed to incorrect classes).

### Effect

- Prevents the model from becoming overconfident
- Acts as regularization
- Improves calibration
- Used in: Transformer training (original paper used ε=0.1), image classifiers

```python
loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, targets)
```

### Cross-Entropy with Label Smoothing

$$\text{CE}_{\text{smooth}} = (1 - \varepsilon) \cdot \text{CE}_\text{hard} + \frac{\varepsilon}{C} \cdot \text{CE}_\text{uniform}$$

---

## 6. Weighted Loss

### Problem: Class Imbalance

In many real-world datasets, classes are **not equally represented**:
- Fraud detection: 0.1% fraud, 99.9% normal
- Medical imaging: 5% positive cases, 95% negative
- Object detection: many background anchors, few object anchors

A naive model that always predicts the majority class gets high accuracy but is useless.

### Solution: Weighted Cross-Entropy

Assign a **higher weight** to the minority class to make the loss penalize those misclassifications more.

$$\text{Weighted CE} = -\frac{1}{n} \sum_{i=1}^{n} w_{y_i} \log \hat{y}_{i, y_i}$$

where $w_c$ is the weight for class c.

### Choosing Weights

**Option 1: Inverse frequency**
$$w_c = \frac{1}{\text{count}(c)}$$

**Option 2: Balanced (sklearn-style)**
$$w_c = \frac{n}{C \cdot \text{count}(c)}$$

**Option 3: Effective number of samples (He et al., 2019)**
$$w_c = \frac{1 - \beta}{1 - \beta^{n_c}}, \quad \beta \in [0, 1)$$

Better than inverse frequency — accounts for feature overlap between samples.

### In PyTorch

```python
# Binary classification
pos_weight = torch.tensor([neg_count / pos_count])
loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)

# Multi-class classification
class_weights = torch.tensor([w0, w1, w2, ...])
loss = nn.CrossEntropyLoss(weight=class_weights)(logits, targets)
```

### Weighted Loss vs. Oversampling/Undersampling

| Approach | Pros | Cons |
|---|---|---|
| Weighted loss | No data duplication, simple | Only adjusts loss signal, not data distribution |
| Oversampling (minority) | Model sees more minority examples | Risk of overfitting minority |
| Undersampling (majority) | Faster training | Loses majority class information |
| SMOTE (synthetic) | More minority diversity | Can generate unrealistic samples |

> In practice: try weighted loss first (easiest). If insufficient, combine with oversampling.

---

## 7. Focal Loss

### Problem: Hard vs. Easy Examples

Weighted loss assigns static weights based on class frequency. But even within the minority class, some examples are **easy** (model already confident) and some are **hard** (model is confused). Standard cross-entropy spends gradient budget on easy examples.

**Example in object detection:**
- 98% of anchors are background (easy negatives: model quickly learns to say "not object")
- 2% are objects (hard positives the model struggles with)
- These easy negatives dominate training, drowning the signal from hard examples

### Focal Loss (Lin et al., 2017 — RetinaNet)

Focal loss adds a **modulating factor** `(1 - p_t)^γ` to cross-entropy that down-weights easy examples and focuses training on hard examples.

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t = \hat{y}$ if y=1, else $p_t = 1 - \hat{y}$
- $\gamma \geq 0$ is the **focusing parameter** (typically γ = 2)

### Intuition

- **Well-classified example** (high $p_t$, e.g., 0.9): factor = (1-0.9)² = 0.01 → loss is down-weighted by 100×
- **Poorly-classified example** (low $p_t$, e.g., 0.1): factor = (1-0.1)² = 0.81 → loss is barely reduced

The model focuses its learning on the examples it's **currently getting wrong**.

```
γ = 0  → Standard cross-entropy (no focusing)
γ = 2  → Original paper's default, very effective in practice
γ = 5  → Very aggressive focusing on hardest examples
```

### With Class Balancing (αt)

In practice, focal loss is combined with a class-balance factor α:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- α balances positive/negative classes (addresses class imbalance)
- γ focuses on hard examples (addresses easy/hard imbalance)

They solve **different** problems and work better together.

### Visual Comparison

```
              ┌────────────────────────────────────────┐
Loss          │                                        │
              │  CE (γ=0) ────────────────────\        │
              │                                \       │
              │  FL γ=0.5 ─────────────────\    \      │
              │                              \    \    │
              │  FL γ=2 ──────────────────\   \    \   │
              │                            \   \    \  │
              └────────────────────────────────────────┘
              p_t: 0.0   0.2   0.4   0.6   0.8   1.0
                   (wrong)                    (correct)

Easy examples (high p_t) → FL drastically reduces their contribution
```

### In PyTorch (Manual Implementation)

```python
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)
    focal_weight = (1 - p_t) ** gamma
    loss = focal_weight * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()
```

### Where Focal Loss Is Used

- **RetinaNet**: The paper that introduced it — one-stage object detection
- **Dense prediction tasks**: Semantic segmentation with class imbalance
- **Medical imaging**: Rare pathology detection
- **Any heavily imbalanced classification problem**

### Focal Loss vs. Weighted Loss

| | Weighted Loss | Focal Loss |
|---|---|---|
| What it addresses | Static class imbalance | Dynamic easy/hard imbalance |
| Weight assignment | Fixed per class | Adaptive per sample per step |
| Based on | Class frequency | Current model confidence |
| Combine them? | Yes — use both α (class balance) + γ (hard mining) |

---

## 8. Dice Loss & IoU Loss (Bonus — Common in Segmentation)

### Dice Loss

Used in **semantic segmentation** — directly optimizes the Dice coefficient (overlap metric).

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

$$\text{Dice Loss} = 1 - \frac{2 \sum \hat{y}_i y_i}{\sum \hat{y}_i + \sum y_i}$$

- Naturally handles class imbalance (doesn't require explicit weighting)
- Commonly combined with cross-entropy: `L = CE + Dice`
- Used in: U-Net, medical image segmentation

### IoU Loss

$$\text{IoU Loss} = 1 - \frac{|A \cap B|}{|A \cup B|}$$

Used for bounding box regression and segmentation.

---

## 9. Putting It All Together — Which Loss to Use?

### Decision Tree

```
Is it regression?
  ├─ Clean data, no outliers → MSE
  ├─ Outliers present → MAE or Huber
  └─ Want robust + smooth → Huber (default choice for regression)

Is it binary classification?
  ├─ Balanced data → BCE
  ├─ Imbalanced (static) → Weighted BCE
  └─ Severe imbalance, dense predictions → Focal Loss

Is it multi-class classification?
  ├─ Balanced → CrossEntropyLoss
  ├─ Imbalanced → Weighted CrossEntropyLoss
  └─ Overconfident model → Label Smoothing

Is it multi-label classification?
  └─ BCEWithLogitsLoss (per class, sigmoid, not softmax)

Is it segmentation?
  ├─ Class imbalance → Dice + CE
  └─ Hard examples → Focal + Dice

Is it a generative model / distribution matching?
  └─ KL Divergence (VAE, knowledge distillation)
```

---

## 10. Summary Table

| Loss | Task | Handles Imbalance | Key Hyperparameter |
|---|---|---|---|
| BCE | Binary classification | No | — |
| Weighted BCE | Binary, imbalanced | Yes (static) | w_pos |
| CE (Softmax) | Multi-class | No | — |
| Weighted CE | Multi-class, imbalanced | Yes (static) | w_c per class |
| NLL | Same as CE (different API) | With weights | — |
| Focal Loss | Dense prediction, severe imbalance | Yes (dynamic) | γ (focus), α (balance) |
| Label Smoothing CE | Multi-class, overconfidence | No | ε |
| Dice Loss | Segmentation | Yes (implicitly) | smooth (numerical stability) |

---

## 11. Common Interview Questions

**Q: What is the difference between BCE and CE?**
> BCE is for binary classification — one sigmoid output, one class boundary. CE is for multi-class classification — C softmax outputs, one true class. BCE can also handle multi-label (multiple simultaneous classes), while CE (softmax) cannot — softmax forces probabilities to sum to 1, implying mutual exclusivity.

**Q: Why use CrossEntropyLoss with logits rather than applying softmax first?**
> Numerical stability. Softmax involves exponentiation and can overflow (large logits) or produce zeros (very negative logits). PyTorch's CrossEntropyLoss applies the log-sum-exp trick internally, which is numerically stable. Also, the fused computation is faster.

**Q: How does focal loss solve the class imbalance problem differently from weighted loss?**
> Weighted loss addresses static class frequency imbalance — minority classes always get higher weight regardless of how well the model currently predicts them. Focal loss addresses the hard/easy imbalance dynamically — examples the model already predicts confidently receive down-weighted gradients, regardless of class. They address different aspects of imbalance and are complementary.

**Q: What is γ = 0 in focal loss?**
> Focal loss with γ = 0 is identical to standard cross-entropy. As γ increases, the modulating factor increasingly down-weights easy examples.

**Q: Why softmax for multi-class but sigmoid for multi-label?**
> Softmax forces all class probabilities to sum to 1 — it models a competition between classes where exactly one is correct. Sigmoid applies independently to each class — each class is a separate binary decision, allowing multiple classes to be active simultaneously.

**Q: What's label smoothing and when would you use it?**
> Label smoothing replaces hard one-hot targets with soft labels that distribute a small probability mass ε to all classes. It prevents overconfidence, improves generalization, and improves calibration. Used when training deep models on clean datasets where the model tends to become overconfident (Transformers, large ResNets).

**Q: You have a dataset with 1000 positive and 100,000 negative samples. How do you handle this?**
> Multiple options: (1) Weighted BCE with `pos_weight = 100` to upweight positives; (2) Focal loss to focus on hard examples; (3) Oversample positives or undersample negatives; (4) Combine: weighted + focal. I'd start with weighted BCE, evaluate precision/recall (not accuracy), and iterate.

**Q: What evaluation metrics should you use with imbalanced data? Why not accuracy?**
> Accuracy is misleading — a model predicting "always negative" gets 99% accuracy on a 1:99 split. Better metrics: Precision, Recall, F1-score, AUC-ROC, AUC-PR (especially AUC-PR for severe imbalance). The loss function handles training; the metric handles evaluation — they should both account for imbalance.