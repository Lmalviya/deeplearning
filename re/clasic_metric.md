# Classification Metrics: Precision, Recall, F1, AUC-ROC, PR-AUC

## 1. The Confusion Matrix (foundation for everything below)

|                     | Predicted Positive | Predicted Negative |
|---------------------|---------------------|----------------------|
| **Actually Positive** | True Positive (TP)  | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN)  |

**Running example:** Disease detection model tested on 100 patients (20 actually have the disease).

- TP = 16 (correctly caught disease)
- FN = 4 (missed disease)
- FP = 6 (false alarms)
- TN = 74 (correctly cleared)

---

## 2. Precision

**Formula:**
```
Precision = TP / (TP + FP)
```

**Example:** 16 / (16 + 6) = **0.727**

**Definition:** Of everything the model flagged as positive, what fraction was actually positive.

**Use when:** False positives are costly — e.g. spam filters (don't want to lose real email), fraud alerts that trigger manual review, wrongly flagging legitimate transactions.

---

## 3. Recall (Sensitivity / True Positive Rate)

**Formula:**
```
Recall = TP / (TP + FN)
```

**Example:** 16 / (16 + 4) = **0.80**

**Definition:** Of everything that was actually positive, what fraction did the model catch.

**Use when:** False negatives are costly — e.g. missed cancer diagnosis, missed security threats, missed fraud.

**The trade-off:** Precision and recall pull in opposite directions. A model that flags everything as positive gets 100% recall but terrible precision. A model that only flags when extremely confident gets high precision but misses real cases.

---

## 4. F1 Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example:** 2 × (0.727 × 0.80) / (0.727 + 0.80) = **0.762**

**Definition:** The harmonic mean of precision and recall. Unlike a simple average, it punishes imbalance between the two — a model with precision 1.0 and recall 0.1 gets F1 ≈ 0.18, not 0.55.

**Use when:** You care about false positives and false negatives roughly equally, and want a single number to rank models — common in search/retrieval and general classification benchmarks.

**Caveat:** F1 hides *which direction* the errors lean. Two models can share an F1 score with very different precision/recall balances. In high-stakes domains, report precision and recall alongside F1, not instead of it.

---

## 5. ROC Curve & AUC-ROC

Precision, recall, and F1 all depend on a fixed **classification threshold** (often 0.5). Change the threshold and every one of them changes. ROC and AUC evaluate the model **across all thresholds at once**.

**ROC curve axes:**
- x-axis: False Positive Rate = FP / (FP + TN)
- y-axis: True Positive Rate (Recall) = TP / (TP + FN)

Plot this pair at every possible threshold from 0 to 1 to trace the curve.

**AUC (Area Under the Curve):** single number from 0 to 1 summarizing the whole curve.
- AUC = 1.0 → perfect classifier at every threshold
- AUC = 0.5 → equivalent to random guessing (diagonal line)
- AUC = 0.8 → if you pick one random actual-positive and one random actual-negative, the model ranks the positive higher 80% of the time

**Use when:** Comparing models before committing to an operating threshold; balanced or moderately balanced datasets.

**Caveat:** ROC-AUC can look deceptively good on **imbalanced datasets**, because FPR's denominator (TN + FP) is dominated by the huge negative class — many false positives barely move FPR.

---

## 6. Precision-Recall Curve & PR-AUC

**PR curve axes:**
- x-axis: Recall = TP / (TP + FN)
- y-axis: Precision = TP / (TP + FP)

Traced the same way — sweep the threshold from 0 to 1 and plot each (Recall, Precision) pair.

**PR-AUC:** area under this curve. Summarizes how well the model finds positives without dragging in too many false ones, across all thresholds.

**Baseline is not fixed at 0.5** — a random classifier's PR-AUC equals the positive class rate in the data (e.g. 0.01 for a 1%-positive dataset). This means PR-AUC values aren't directly comparable across datasets with different imbalance ratios; always check what a no-skill baseline would score first.

---

## 7. ROC-AUC vs PR-AUC — the key difference

**Worked example (1,000 cases: 990 negative, 10 positive — 1% positive rate):**

Model at a given threshold: TP = 8, FN = 2, FP = 50, TN = 940

| Metric | Calculation | Result | Read |
|---|---|---|---|
| FPR | 50 / 990 | 0.051 | Looks tiny — dataset dominated by negatives |
| TPR (Recall) | 8 / 10 | 0.80 | Looks great on ROC |
| Precision | 8 / (8 + 50) | 0.138 | Looks bad — 86% of flagged cases are false alarms |

**Same model, same threshold, two very different verdicts.** ROC-AUC is optimistic here because FPR is diluted by the large negative class. PR-AUC is harsher and more operationally honest, because precision directly reflects how much of your "positive" bucket is noise.

### Summary table

| | ROC-AUC | PR-AUC |
|---|---|---|
| Axes | TPR vs FPR | Precision vs Recall |
| Baseline (random model) | Always 0.5 | Equals the positive class rate |
| Sensitive to class imbalance | No — can look good even when precision is poor | Yes — directly exposes false-alarm burden |
| Best for | Balanced classes, or symmetric interest in both classes | Imbalanced classes, or when the positive class is what matters most |
| Common domains | General ML benchmarking, balanced diagnostic cohorts | Fraud detection, rare-disease screening, spam/anomaly detection, information retrieval |

---

## Quick reference: all formulas

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
FPR       = FP / (FP + TN)
TPR       = Recall
```

## Python (scikit-learn) quick start

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve
)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Threshold-independent, need predicted probabilities (not hard labels)
roc_auc = roc_auc_score(y_true, y_proba)
pr_auc = average_precision_score(y_true, y_proba)

fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
prec, rec, pr_thresholds = precision_recall_curve(y_true, y_proba)
```

Note: `average_precision_score` computes PR-AUC using a step-function interpolation, which can differ slightly from a naive trapezoidal `auc(rec, prec)` — this is the standard, recommended way to compute it in scikit-learn.