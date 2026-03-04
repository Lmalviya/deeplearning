# Module 2: Training & Optimization — Execution Plan

## File Split

| File | Sections | Notebook |
|------|----------|----------|
| File 1 | 1–3: Learning Loop, Loss Functions, Backprop | `02_training_loss_backprop.ipynb` |
| File 2 | 4–5: Optimizers, LR Scheduling | `03_optimizers_lr_scheduling.ipynb` |
| File 3 | 6: Regularization | `04_regularization.ipynb` |
| File 4 | 7–8: Normalization Layers, Weight Init | `05_normalization_initialization.ipynb` |
| File 5 | 9–10, 12: NumPy Scratch, PyTorch Loop, Q&A | `06_training_loops_and_qna.ipynb` |

> Section 11 (Manim video) is skipped for now.

---

## Detailed Index

### File 1 — `02_training_loss_backprop.ipynb`
- **Sec 1** — The Big Picture: The Learning Loop
- **Sec 2** — Loss Functions
  - 2.1 MSE Loss
  - 2.2 MAE Loss & Robustness to Outliers
  - 2.3 Huber Loss — Best of Both Worlds
  - 2.4 Binary Cross-Entropy (BCE)
  - 2.5 Categorical Cross-Entropy & NLL Loss
  - 2.6 Loss Landscape Visualization
- **Sec 3** — Backpropagation
  - 3.1 Forward Pass vs Backward Pass
  - 3.2 Chain Rule — The Core Mechanism
  - 3.3 Manual Backprop: Worked Example (2-Layer Net)
  - 3.4 Computational Graphs & Reverse-Mode Autodiff
  - 3.5 Vanishing & Exploding Gradients in Backprop
  - 3.6 Gradient Clipping
- **Sec 4** — NumPy: Manual forward+backward pass
- **Sec 5** — PyTorch: autograd demo
- **Sec 6** — Interview Q&A (Beginner / Mid / Senior)

### File 2 — `03_optimizers_lr_scheduling.ipynb`
- **Sec 1** — Gradient Descent Variants (Batch, Mini-batch, SGD)
- **Sec 2** — Momentum — Escaping Ravines
- **Sec 3** — AdaGrad — Per-parameter Learning Rates
- **Sec 4** — RMSProp — Fixing AdaGrad's Fading Memory
- **Sec 5** — Adam — The Industry Default
- **Sec 6** — AdamW — Weight Decay Done Right
- **Sec 7** — Optimizer Trajectory Visualization
- **Sec 8** — LR Scheduling (Step, Cosine, Warmup, LR Finder)
- **Sec 9** — PyTorch Optimizer & Scheduler Demo
- **Sec 10** — Interview Q&A

### File 3 — `04_regularization.ipynb`
- **Sec 1** — Bias-Variance Tradeoff
- **Sec 2** — L2 Regularization (Weight Decay)
- **Sec 3** — L1 Regularization & Sparsity
- **Sec 4** — Elastic Net (L1 + L2)
- **Sec 5** — Dropout — Random Neuron Silencing
- **Sec 6** — Inverted Dropout & Test-time Behaviour
- **Sec 7** — Early Stopping
- **Sec 8** — Weighted Loss (Class Imbalance)
- **Sec 9** — PyTorch Regularization Demo
- **Sec 10** — Interview Q&A

### File 4 — `05_normalization_initialization.ipynb`
- **Sec 1** — Why Normalization Matters (Internal Covariate Shift)
- **Sec 2** — Batch Normalization (Math + Forward + Backward)
- **Sec 3** — Batch Norm at Test Time (Running Stats)
- **Sec 4** — Layer Normalization (Transformers)
- **Sec 5** — Instance Normalization (Style Transfer)
- **Sec 6** — Group Normalization
- **Sec 7** — Comparison Table: Which Norm, When?
- **Sec 8** — Why Initialization Matters
- **Sec 9** — Zero Init & Symmetry Breaking
- **Sec 10** — Xavier / Glorot Init
- **Sec 11** — He (Kaiming) Init
- **Sec 12** — Orthogonal Init
- **Sec 13** — PyTorch Demo
- **Sec 14** — Interview Q&A

### File 5 — `06_training_loops_and_qna.ipynb`
- **Sec 1** — Full NumPy Training Loop from Scratch
- **Sec 2** — PyTorch Training Loop (nn.Module + torch.optim)
- **Sec 3** — Master Interview Q&A Cheatsheet (All of Module 2)

---

## Loss Function Scope Decision
**In Module 2:** MSE, MAE, Huber, BCE, Categorical Cross-Entropy / NLL
**Deferred to architecture modules:**
- Focal Loss → Object Detection
- Contrastive / Triplet Loss → Metric Learning / Self-supervised
- KL Divergence → VAEs
- Wasserstein → GANs
- IoU / Dice → Segmentation
