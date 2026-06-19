# Lesson 2.1 — The Training Objective for Language Models
### Chapter 2: What Fine-Tuning Actually Does to a Model

---

## The Problem Story

Vikram was three months into his first ML role. His fine-tuned model had a training loss of 0.43. His manager asked: "Is that good?"

Vikram said yes, because 0.43 is less than 1.0, and lower loss is better. His manager asked: "Good compared to what? What does 0.43 mean in absolute terms?"

Vikram had no answer. He had been minimizing a number without understanding what that number represented — what the model was being asked to do at each training step, what a good value actually looks like, or why this particular objective was chosen for language modeling.

A week later, a different problem: his model was producing fluent-sounding but factually wrong answers. Loss was still low. He did not understand why low loss did not mean correct outputs.

Both failures trace back to the same root: not understanding what the training objective actually is and what it guarantees — and what it does not.

---

## The Concept

### What is a Training Objective?

A training objective is a mathematical function that measures how wrong the model's predictions are. The training loop minimizes this function by adjusting model weights. Whatever you minimize, the model gets better at — and only at that specific thing.

This sounds obvious, but the implication is deep: **the model becomes good at predicting the next token in your training data. It does not inherently become factually correct, helpful, safe, or coherent. Those properties emerge only if your training data reliably associates them with low next-token prediction loss.**

This is not a flaw in language modeling. It is a fundamental fact that shapes every decision you make in fine-tuning.

---

### The Language Modeling Objective: Next Token Prediction

Given a sequence of tokens `[t₁, t₂, t₃, ..., tₙ]`, the model learns to predict each token given all preceding tokens:

```
P(tₙ | t₁, t₂, ..., tₙ₋₁)
```

In plain English: "Given everything you have seen so far, what is the probability of the next token?"

The model assigns a probability to every token in its vocabulary. The probability of the correct next token should be high. The probability of wrong tokens should be low.

---

### Cross-Entropy Loss: The Full Mathematics

Cross-entropy measures the difference between the model's predicted probability distribution and the true distribution (which, in supervised learning, is a one-hot distribution — the correct token has probability 1, all others have probability 0).

**For a single token prediction:**

```
loss = -log(P(correct_token))
```

That is it. The loss is the negative log probability of the correct next token.

**Why negative log?**

If the model assigns probability 0.9 to the correct token:
```
loss = -log(0.9) = 0.105  (low loss — model was confident and right)
```

If the model assigns probability 0.1 to the correct token:
```
loss = -log(0.1) = 2.303  (high loss — model was uncertain or wrong)
```

If the model assigns probability 0.01 to the correct token:
```
loss = -log(0.01) = 4.605  (very high loss — model was confident about something else)
```

The log function turns multiplication of probabilities into addition of losses (useful mathematically), and the negative sign flips it so we minimize instead of maximize.

**For a full sequence:**

The loss over a sequence of N tokens is the average cross-entropy at each position:

```
L = -(1/N) × Σᵢ log P(tᵢ | t₁, ..., tᵢ₋₁)
```

Each position contributes a loss term. The model is simultaneously trying to predict every next token in the sequence correctly.

**In code (the actual computation):**

```python
import torch
import torch.nn.functional as F

# logits: shape (batch_size, seq_len, vocab_size)
# labels: shape (batch_size, seq_len) — the actual token IDs

# Shift: position i predicts token at position i+1
# So we drop the last logit and the first label
shift_logits = logits[:, :-1, :].contiguous()   # (batch, seq-1, vocab)
shift_labels = labels[:, 1:].contiguous()         # (batch, seq-1)

# Flatten and compute
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),  # (batch*(seq-1), vocab)
    shift_labels.view(-1),              # (batch*(seq-1),)
    ignore_index=-100                   # ignore padding / masked tokens
)
```

The `ignore_index=-100` is critical for fine-tuning. We set label positions we do not want to train on (padding tokens, prompt tokens) to -100, and the loss function skips those positions entirely. This is **loss masking** — covered in full in Lesson 2.5.

---

### What the Model Outputs: Logits vs Probabilities

The model outputs **logits** — raw, unnormalized scores for each token in the vocabulary. These are not probabilities. They can be any real number, positive or negative.

To convert logits to probabilities, you apply softmax:

```
P(token_i) = exp(logit_i) / Σⱼ exp(logit_j)
```

Softmax does two things:
1. Makes all values positive (via exp)
2. Makes them sum to 1 (via division by the sum)

The result is a valid probability distribution over the entire vocabulary.

**Why use logits instead of probabilities internally?**

Numerical stability. Working with raw log-probabilities (before taking exp) avoids floating-point underflow. Very small probabilities (like 0.000001) become manageable when you stay in log space. The `F.cross_entropy` function in PyTorch takes logits (not probabilities) and computes the log-softmax internally in a numerically stable way.

---

### Perplexity: What It Is, What It Tells You, When It Misleads

Perplexity is defined as:

```
Perplexity = exp(cross_entropy_loss)
```

If loss = 2.3, perplexity = exp(2.3) ≈ 10.

**The intuition behind perplexity:**

Perplexity tells you how many choices the model effectively considers at each step. A perplexity of 10 means the model is as uncertain as if it were choosing uniformly from 10 equally likely options at every position.

- Perplexity = 1: The model predicts every token with certainty. It has memorized the test set (or it is cheating).
- Perplexity = 10: On average, 10 tokens seem equally plausible at each step.
- Perplexity = 100: 100 options seem equally plausible. The model is very uncertain.
- Perplexity = vocab_size (e.g., 32,000): Random guessing — the model knows nothing.

**Reference points for well-known models:**

| Model | Dataset | Perplexity |
|-------|---------|-----------|
| GPT-2 Small | WikiText-103 | ~35 |
| GPT-2 Large | WikiText-103 | ~22 |
| GPT-3 175B | WikiText-103 | ~12 |
| Fine-tuned model on domain data | Domain test set | Typically 2–8 |

When you fine-tune and perplexity drops from 35 to 5 on your domain test set, it means the model has learned the statistical patterns of your domain text very well.

**When perplexity misleads you:**

Perplexity measures how well the model predicts the *specific text in your test set*. It does not measure:

1. **Factual correctness.** A model that confidently predicts "Paris is the capital of Germany" has low loss on that sentence if it appeared in training data. It is wrong, but loss would not tell you that.

2. **Usefulness.** A model trained on high-quality instruction data might have higher perplexity than one trained on repetitive boilerplate — because boilerplate is easy to predict, not because it is more useful.

3. **Generalization.** Low perplexity on the training set means the model fit the training data. It says nothing about performance on out-of-distribution inputs.

4. **Task performance.** A model can have low perplexity but fail at specific tasks (summarization, code generation) if those tasks require capabilities not directly measured by next-token prediction accuracy.

This is why Chapter 9 exists: evaluation beyond loss. Perplexity is a proxy. It is necessary but not sufficient.

---

### Teacher Forcing: How Training Differs from Inference

During inference, the model generates token by token, feeding its own output back as input:

```
Step 1: Input = "The cat sat"   → Model predicts "on"
Step 2: Input = "The cat sat on" → Model predicts "the"
Step 3: Input = "The cat sat on the" → Model predicts "mat"
```

If the model makes a mistake at step 1 (predicts "in" instead of "on"), then step 2 sees a wrong context: "The cat sat in." The error propagates.

During training, the model is never allowed to make this mistake matter. At every position, it sees the *correct* previous tokens from the training data, regardless of what it would have predicted:

```
Position 1: Input = "The"         → Predict "cat"     (see true "cat" next)
Position 2: Input = "The cat"     → Predict "sat"     (see true "sat" next)
Position 3: Input = "The cat sat" → Predict "on"      (see true "on" next)
```

This is **teacher forcing**. The "teacher" forces the model to see correct context at every step.

**Why teacher forcing is used:**

- Training is much more stable. Without it, early in training when predictions are poor, errors compound rapidly, making the gradient signal useless.
- Every position provides a gradient signal independently, making training more data-efficient.
- It is computationally efficient — you compute predictions for all positions in one forward pass.

**The exposure bias problem:**

At inference time, the model faces a distribution it never saw during training: its own (potentially wrong) previous outputs. This mismatch between training and inference is called **exposure bias**.

In practice, this means:
- Models sometimes get stuck in repetitive loops (because they have never trained on recovering from their own repetitive outputs)
- Errors early in generation can cascade in ways the training loss would never have shown
- Evaluation during training (where teacher forcing is used) can look better than actual generation quality

Techniques like **scheduled sampling** (randomly replacing true tokens with model predictions during training) and **reinforcement learning from feedback** (RLHF/DPO, covered in Chapter 10) address exposure bias. For standard SFT fine-tuning, we accept this limitation.

---

### What Loss Values Mean in Practice

You now know the math. But what does a loss of 0.43 actually mean when you see it on your training dashboard?

**Calibrating your expectations:**

The loss value depends heavily on what you are training on and what kind of task it is.

For **instruction fine-tuning** (training model to follow instructions in chat format):
- Starting loss (before any training): usually 1.5–3.0 (the base model has no idea what your chat format is)
- Converged loss with good data: usually 0.8–1.5
- Loss below 0.5: often a sign of overfitting or that your dataset is small/repetitive

For **domain adaptation** (continued training on domain text):
- Loss depends heavily on how different your domain is from the pre-training data
- A medical text corpus might start at loss ≈ 2.5 and converge to ≈ 1.5
- A general English corpus might start at ≈ 1.8 and converge to ≈ 1.4

For **task-specific fine-tuning** (e.g., always output JSON in a specific schema):
- Very structured outputs can reach very low loss (< 0.3) because the model learns a rigid pattern
- Low loss here does not mean the model generalizes — it may just be memorizing the pattern

**The most important number is not the absolute value but the trajectory:**
- Loss dropping steadily: training is working
- Loss plateaued: need to check LR, data diversity, number of epochs
- Loss spiking: instability (LR too high, bad batch, numerical issue)
- Train loss drops but val loss rises: overfitting

We cover loss curve reading in full in Chapter 8. For now, internalize this: loss is a signal, not a verdict.

---

## The Intuition Bridge

Think of the cross-entropy loss as a test score on a fill-in-the-blank exam.

Every position in every training sequence is one question: "What comes next?" The model writes down its probability distribution (its guesses with confidence levels). The correct answer is the actual next token.

If the model says "I am 90% sure the next word is 'cat'" and the correct answer is "cat" — low loss, almost full marks.

If the model says "I think it might be any of these 50 words equally" and the correct answer is "cat" — high loss.

Perplexity is the average number of options the model is considering across all the questions. Training minimizes average uncertainty across every fill-in-the-blank question in your dataset.

Teacher forcing means on question 5, you always show the correct answer to questions 1–4, even if the model got questions 1–4 wrong. The exam always starts from a clean slate for each question.

---

## Why This Matters for Fine-Tuning

**Reason 1: You are training on the wrong things if you do not mask**

If your training data is:
```
[INSTRUCTION]: Summarize this article.
[ARTICLE]: ...1000 tokens of article text...
[SUMMARY]: ...50 tokens of summary...
```

And you compute loss over the entire sequence — then 1050 of your 1100 loss contributions come from predicting the instruction and article tokens. Only 50 come from predicting the summary. The model is primarily being trained to predict the instruction format and article text, not to summarize.

Loss masking (setting label=-100 for instruction/article tokens) fixes this. The model only trains on the summary tokens. This is fundamental and gets its own lesson (2.5).

**Reason 2: Low training loss does not mean your model is good**

The objective is next-token prediction on your training data. If your training data has errors, the model learns those errors with low loss. If your training data is not diverse enough, the model overfits to your specific examples with low loss. If your training data is instruction-response pairs where responses are factually wrong — the model learns those wrong responses.

The quality of your objective is only as good as the quality of your data. We return to this in Chapter 3.

**Reason 3: Loss on training data vs validation data tells you something specific**

Train loss = how well the model predicts your training data.
Val loss = how well the model predicts held-out data from the same distribution.

If train loss < val loss and the gap is growing: overfitting. The model is memorizing training data rather than learning generalizable patterns.

If both losses are high and not decreasing: underfitting. The model is not learning. Check LR, data quality, model capacity.

If val loss is lower than train loss: something unusual is happening. Check if your splits are correct and there is no data leakage.

---

## The Code

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ── 1. Manual cross-entropy loss computation ────────────────────

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"]

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits  # (1, seq_len, vocab_size)
vocab_size = logits.shape[-1]

# Shift: position i predicts token at position i+1
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()

# Manual cross-entropy
manual_loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1)
)

# Model's built-in loss
with torch.no_grad():
    outputs_with_loss = model(**inputs, labels=input_ids)

print(f"Manual cross-entropy loss:  {manual_loss.item():.6f}")
print(f"Model's computed loss:       {outputs_with_loss.loss.item():.6f}")
print(f"Match: {abs(manual_loss.item() - outputs_with_loss.loss.item()) < 0.01}")

# ── 2. Per-token loss — see which tokens are hard to predict ────

print("\n── Per-token loss ──")
with torch.no_grad():
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none"  # do not average — get individual losses
    )

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(f"{'Token':<15} {'Predicting this':<20} {'Loss':<10} {'Perplexity'}")
print("-" * 60)
for i, (loss_val) in enumerate(per_token_loss):
    context_token = tokens[i]
    predicted_token = tokens[i + 1]
    ppl = torch.exp(loss_val).item()
    bar = "█" * min(int(loss_val.item() * 3), 20)
    print(f"'{context_token:<12}' → '{predicted_token:<15}' {loss_val.item():.4f}  {ppl:.2f}  {bar}")

# ── 3. Perplexity computation ───────────────────────────────────

print(f"\nAverage loss:       {manual_loss.item():.4f}")
print(f"Perplexity:         {torch.exp(manual_loss).item():.2f}")

# ── 4. Compare loss on in-domain vs out-of-domain text ─────────

texts = {
    "Natural English": "The weather is nice today and I enjoy walking in the park.",
    "Technical ML":    "The gradient descent algorithm minimizes the cross-entropy loss function.",
    "Random tokens":   "purple quantum seventeen banana electricity hospital fork",
    "Repetitive":      "the the the the the the the the the the the the the",
}

print("\n── Loss on different text types ──")
for label, t in texts.items():
    inp = tokenizer(t, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, labels=inp["input_ids"])
    ppl = torch.exp(out.loss).item()
    print(f"  {label:<20} loss={out.loss.item():.3f}  perplexity={ppl:.1f}")

# ── 5. What does the probability distribution look like? ────────

print("\n── Probability distribution at one position ──")
text2 = "The capital of France is"
inp2 = tokenizer(text2, return_tensors="pt").to(model.device)
with torch.no_grad():
    out2 = model(**inp2)

last_logits = out2.logits[0, -1, :]   # last position
probs = F.softmax(last_logits, dim=-1)

top10 = torch.topk(probs, 10)
print(f"Top 10 predictions after '{text2}':")
for token_id, prob in zip(top10.indices, top10.values):
    token = tokenizer.decode([token_id])
    print(f"  '{token:<15}' {prob.item():.4f} ({prob.item()*100:.2f}%)")

# ── 6. Teacher forcing illustration ─────────────────────────────

print("\n── Teacher Forcing vs Autoregressive (first 5 tokens) ──")
prompt = "The cat sat on"
inp3 = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids3 = inp3["input_ids"]

# Teacher forcing: compute predictions for all positions at once
with torch.no_grad():
    tf_output = model(**inp3)
tf_logits = tf_output.logits[0]  # (seq_len, vocab)

print("Teacher forcing predictions at each position:")
tokens3 = tokenizer.convert_ids_to_tokens(input_ids3[0])
for i in range(len(tokens3)):
    top_pred_id = tf_logits[i].argmax()
    top_pred = tokenizer.decode([top_pred_id])
    actual_next = tokens3[i+1] if i+1 < len(tokens3) else "[end]"
    print(f"  After '{tokens3[i]}': predict '{top_pred}', actual next: '{actual_next}'")
```

---

## The Experiment

**EXP-2.1.A — Loss Calibration Study**

Goal: Build intuition for what different loss values mean.

```python
# Extend the loss comparison above with your own texts.
# For each category below, write 3 examples and measure loss + perplexity.

categories = {
    "Very predictable English":     ["...", "...", "..."],
    "Unpredictable but valid":      ["...", "...", "..."],
    "Domain text (your target domain)": ["...", "...", "..."],
    "Grammatically wrong":          ["...", "...", "..."],
    "Factually wrong":              ["...", "...", "..."],
}
```

Fill in your experiment log:

```
════════════════════════════════════════════════════════
EXPERIMENT LOG
════════════════════════════════════════════════════════
ID:       EXP-2.1.A
Lesson:   2.1 — Training Objective
Goal:     Calibrate intuition for what loss values mean
          across different text types and domains

SETUP
Model: [your model]
Input: Custom text strings across 5 categories

RAW OBSERVATIONS
[Fill: loss + perplexity for each of your 15 text examples]
[Fill: which category had highest/lowest loss?]
[Fill: was the factually wrong text distinguishable from
       the correct text by loss alone?]

WHAT SURPRISED ME
[Fill honestly]

INTERPRETATION
[Why does repetitive text have very low loss?]
[Why does random text have very high loss?]
[Can loss detect factual errors? What does that tell you?]

IMPLICATIONS FOR FINE-TUNING
[If my training data has factually wrong examples,
 will the model learn them? How would I know?]
[What loss range should I expect for my specific task?]

OPEN QUESTIONS
[Fill]

NEXT STEP
[Fill]
════════════════════════════════════════════════════════
```

---

## Interview Checkpoint

**Q: What is the language modeling objective and why was it chosen?**

> A: The language modeling objective is next-token prediction — given a sequence of tokens, predict the probability of each next token. The loss function is cross-entropy between the model's predicted distribution and the true distribution (which is a one-hot vector on the correct next token). It was chosen because it requires no labeled data — any text corpus is automatically a training signal, just predict what comes next. This enables training on internet-scale data. The remarkable finding is that a model that becomes very good at predicting text across all domains must have learned language structure, world knowledge, and reasoning patterns as instrumental skills.

**Q: Why is perplexity not sufficient as an evaluation metric for fine-tuned models?**

> A: Perplexity measures how well the model predicts the specific tokens in a test set. It does not measure factual correctness, task performance, or helpfulness. A model that has memorized factually wrong statements will have low perplexity on a test set containing those same wrong statements. A model producing fluent but wrong summaries will have low perplexity if the test summaries are similarly wrong. Perplexity measures fit to the test distribution, not quality of that distribution. For fine-tuned models, task-specific metrics (ROUGE, exact match, pass@k for code, human preference ratings) are necessary alongside perplexity.

**Q: What is teacher forcing and what problem does it cause?**

> A: Teacher forcing is a training technique where the model always receives the correct previous tokens as input at each position, rather than its own predictions. This makes training stable and efficient. The problem it causes is exposure bias: during inference, the model sees its own (potentially wrong) previous outputs, which is a distribution it never encountered during training. This mismatch can cause error cascades during generation — a mistake early in the sequence creates an unfamiliar context that leads to more mistakes. Techniques like DPO and RLHF (Chapter 10) address this by training on model-generated outputs as well.

**Q: What does a loss of 0.8 mean for your fine-tuned model?**

> A: On its own, 0.8 is not interpretable without context. You need to know: what was the loss before fine-tuning? What is the val loss? What is the domain? For instruction fine-tuning, 0.8–1.2 is a reasonable converged range for a well-trained model. If the pre-fine-tuning loss was 2.5 and it converged to 0.8, the model has clearly adapted to your data. But if train loss is 0.8 and val loss is 2.3, you are severely overfitting. The absolute value matters far less than the ratio between train and val loss, and the trajectory of both over training.

---

## Common Mistakes & Misconceptions

❌ **"Lower loss always means a better model."**
Lower training loss means the model fits the training data better. It does not mean the model is more accurate, more useful, or more generalizable. A model with loss=0.2 that is overfitting to a small dataset is far worse than a model with loss=1.0 that generalizes well.

❌ **"Perplexity is the ground truth metric for LLMs."**
Perplexity is one metric, appropriate for measuring language modeling quality. For fine-tuned task-specific models, task metrics almost always matter more. The AI research community has largely moved away from perplexity as a primary metric for practical model evaluation.

❌ **"The model is minimizing the probability of wrong answers."**
The model is minimizing the negative log probability of the correct next token. These are mathematically equivalent but the mental model matters. The model is never explicitly penalized for being confident about wrong answers — it is only rewarded for being confident about right ones. This is why calibration (being uncertain when wrong) is a separate concern that requires additional training.

❌ **"Teacher forcing means the model never makes mistakes during training."**
Teacher forcing means the model always receives correct context as input. The model still makes wrong predictions at every position — it just does not feed those wrong predictions back as input. The loss measures exactly how wrong those predictions are. The difference from inference is that the wrong predictions do not accumulate.