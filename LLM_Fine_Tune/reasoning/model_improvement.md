# How to Actually Improve a Model
> Diagnosis → Interpretation → Targeted Fix. Not random hyperparameter search.

---

## The Core Mental Model

Most people approach improvement wrong:

```
WRONG approach:
Train → bad results → randomly change hyperparameters → retrain → hope

CORRECT approach:
Train → read signals → diagnose root cause → change ONE thing → retrain → compare
```

There are only 4 root causes of a bad model. Everything maps to one of them:

```
1. DATA PROBLEM      — wrong distribution, low quality, bad format, not enough
2. OPTIMIZATION PROBLEM — learning rate, batch size, schedule causing instability
3. CAPACITY PROBLEM  — model too small for task, or LoRA rank too low
4. OBJECTIVE MISMATCH — your loss/reward doesn't actually measure what you want
```

Before touching any hyperparameter, you must identify which bucket you are in.

---

## Part 1 — Reading Training Signals

### 1.1 SFT Loss Curve Shapes

Every shape tells you something specific:

```
Shape 1: Healthy training
Loss
│
│\
│ \
│  \___
│      ‾‾‾___
│            ‾‾‾___
└────────────────────── steps
Diagnosis: Learning normally. Loss decreases, plateaus, stays stable.
Action: None. Let it finish.

─────────────────────────────────────────────────────

Shape 2: Loss spikes
Loss
│\    /\
│ \  /  \    /\
│  \/    \  /  \___
│         \/
└────────────────────── steps
Diagnosis: Learning rate too HIGH, or batch size too small causing noisy gradients.
Action: Reduce LR by 2-5x, or increase gradient_accumulation_steps.

─────────────────────────────────────────────────────

Shape 3: Loss barely moves
Loss
│\
│ \___________________________
│
└────────────────────── steps
Diagnosis: LR too LOW, or model already knows this data (base model pre-trained on it).
Action: Increase LR 3-5x. If still flat, check your data — it may be in pretraining.

─────────────────────────────────────────────────────

Shape 4: Train loss down, eval loss UP
Loss
│
│  train \____
│              ‾‾‾___
│  eval  \___________/‾‾‾‾‾
│
└────────────────────── steps
Diagnosis: OVERFITTING. Model memorizing training data.
Action: (a) more data, (b) reduce epochs, (c) reduce LoRA rank,
        (d) increase dropout, (e) add weight_decay=0.01.

─────────────────────────────────────────────────────

Shape 5: Loss goes to 0 very fast
Loss
│\
│ \___
│    ‾‾ (near 0 by step 50)
└────────────────────── steps
Diagnosis: Data is too simple OR you forgot to mask prompt tokens.
           Model is memorizing, not learning to generalize.
Action: Check DataCollatorForCompletionOnlyLM is masking prompts correctly.
```

### 1.2 The Most Important SFT Metrics to Log

```python
# Add this to your SFTConfig to log everything useful
SFTConfig(
    logging_steps=10,
    # TRL logs these automatically:
    # train/loss          ← primary signal
    # train/grad_norm     ← tells you about instability
    # train/learning_rate ← confirm LR schedule is working
    # eval/loss           ← overfitting detector
)
```

**grad_norm interpretation:**

| grad_norm value | Meaning | Action |
|---|---|---|
| 0.1 – 2.0 | Healthy | None |
| 2.0 – 5.0 | Slightly high | Watch, may be ok |
| > 5.0 consistently | Unstable gradients | Reduce LR or add `max_grad_norm=1.0` |
| Sudden spike to 50+ | Gradient explosion | Add gradient clipping immediately |

```python
# Gradient clipping — add to any trainer config
SFTConfig(max_grad_norm=1.0)   # clips gradients > 1.0
```

---

### 1.3 GRPO Signal Interpretation

GRPO has richer signals because you have multiple reward components:

```python
# TRL GRPOTrainer logs these automatically:
# rewards/correctness_reward    ← did the answer improve?
# rewards/format_reward         ← is structure being learned?
# rewards/mean                  ← combined reward
# rewards/std                   ← diversity in generations (want this > 0)
# kl                            ← KL divergence from reference model
# loss/policy                   ← policy gradient loss
```

**The most critical GRPO diagnostic — rewards/std:**

```
If rewards/std ≈ 0:
    All G generations are getting the same reward.
    GRPO advantage = (r_i - mean) / std → division by near-zero → NaN or no signal.

    This means either:
    (a) All outputs are correct → problems too easy, use harder subset
    (b) All outputs are wrong  → problems too hard, use easier subset
    (c) Temperature too low    → increase temperature (0.7 → 1.0)
    (d) num_generations too low → increase G (4 → 8)

Target: rewards/std should be > 0.1 throughout training
```

**GRPO reward curve — what to expect over time:**

```
Phase 1 (early steps): Format reward rises BEFORE accuracy reward
    → model learns to use <think> tags before it learns to reason correctly
    → this is normal and expected

Phase 2 (mid training): Accuracy reward starts rising
    → model is learning to reach correct answers

Phase 3 (late training): Both plateau
    → you've extracted what this data can give you
    → either add harder problems or stop

If accuracy reward NEVER rises (stays at 0 after 1000 steps):
    → reward function has a bug (most common cause)
    → or task is too hard for this model size
```

**KL divergence in GRPO:**

```
kl < 0.5   → model close to reference, still conservative
kl 0.5–2.0 → healthy exploration
kl > 5.0   → model drifting too far from reference
             → increase beta (KL penalty) from 0.01 → 0.1

If kl collapses to 0:
    → model stopped changing
    → LR too low, or reward variance too low
```

---

### 1.4 DPO Signal Interpretation

```python
# TRL DPOTrainer logs:
# rewards/chosen         ← implicit reward on chosen responses
# rewards/rejected       ← implicit reward on rejected responses
# rewards/margins        ← chosen - rejected (THE key metric)
# rewards/accuracies     ← fraction of samples where chosen > rejected
# logps/chosen           ← log prob of chosen under policy
# logps/rejected         ← log prob of rejected under policy
# beta                   ← your beta value (constant)
```

**The margin is everything:**

```
rewards/margins = rewards/chosen - rewards/rejected

Starting value: typically small positive (0.1 – 0.5)
Healthy training: margin grows steadily
End of training: margin 1.0 – 3.0 is typical

If margin never grows (stays near 0):
    → beta is too high (penalizing deviation from ref too much)
    → reduce beta: 0.1 → 0.05

If margin grows too fast (jumps to > 5 early):
    → overfitting to preference data
    → increase beta: 0.1 → 0.2
    → or reduce epochs

If rewards/chosen goes DOWN:
    → This is normal and expected in DPO.
    → DPO increases margin by lowering rejected MORE than chosen.
    → Do not panic unless chosen drops below rejected.
```

**rewards/accuracies target:**

```
< 55%  → model barely learning preferences → check data quality, reduce beta
55-65% → learning but slow → normal for early training
65-80% → healthy
> 85%  → check for length exploitation (see below)
```

**DPO failure mode — length exploitation:**

```
Symptom: rewards/accuracies very high, but model outputs are getting longer and longer
Cause:   Chosen responses in your dataset happen to be longer than rejected.
         Model learns: "longer = better" instead of "higher quality = better"
Fix:     Add length normalization to DPO loss:
```

```python
# Use SimPO instead of DPO (built-in length normalization)
from trl import SimPOTrainer, SimPOConfig

trainer = SimPOTrainer(
    model=model,
    args=SimPOConfig(
        beta=2.0,         # SimPO uses different beta scale
        gamma=0.5,        # length penalty
    ),
    ...
)
```

---

## Part 2 — Hyperparameter Decision Framework

### 2.1 Learning Rate — The Most Impactful Parameter

**How to choose the starting LR by training type:**

| Training type | Safe starting LR | Typical range |
|---|---|---|
| SFT (LoRA) | `2e-4` | `1e-4` to `5e-4` |
| SFT (full fine-tune) | `2e-5` | `1e-5` to `5e-5` |
| GRPO | `5e-6` | `1e-6` to `1e-5` |
| DPO | `5e-5` | `1e-5` to `1e-4` |

> Rule of thumb: GRPO and DPO use 10x lower LR than SFT because you're making small
> adjustments to an already-capable model. Too high → catastrophic forgetting.

**LR schedule matters more than the value:**

```python
# Cosine with warmup — best for almost all cases
SFTConfig(
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,    # 5% of steps for warmup
)

# Linear — simpler, less likely to over-anneal
SFTConfig(lr_scheduler_type="linear")

# Constant — use only for quick experiments
SFTConfig(lr_scheduler_type="constant")
```

**When to adjust LR based on loss shape:**

```
Loss spikes at start           → increase warmup_ratio (0.05 → 0.1)
Loss spikes in middle          → reduce LR by 3x
Loss too flat                  → increase LR by 3x
Loss good but eval diverges    → reduce LR by 2x AND reduce epochs
```

---

### 2.2 LoRA Rank — Capacity Control

**What rank controls:**

```
Low rank (r=4,8)   → few learnable parameters → underfits complex tasks
High rank (r=64,128) → many parameters → can overfit on small datasets

Rule: rank × 2 = lora_alpha is the default (r=16, alpha=32)
```

**How to choose rank based on task:**

| Task | Dataset size | Recommended rank |
|---|---|---|
| Simple format learning | 1k–5k | r=8 |
| Instruction following | 5k–50k | r=16 |
| Domain-specific reasoning | 10k–100k | r=32 |
| Full capability addition | > 100k | r=64 |

**Diagnosis: underfitting vs overfitting with LoRA:**

```
Train loss high, Eval loss high  → Underfitting → increase rank
Train loss low, Eval loss high   → Overfitting  → decrease rank, add more data
Train loss low, Eval loss low    → Good fit      → done
```

**Which layers to target:**

```python
# Conservative (attention only) — less params, safer
target_modules=["q_proj", "v_proj"]

# Standard (full attention) — good default
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# Aggressive (attention + MLP) — more capacity
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# Rule: if model is underfitting, expand target_modules before increasing rank
```

---

### 2.3 Batch Size and Gradient Accumulation

**Effective batch size = per_device_batch_size × gradient_accumulation_steps × num_GPUs**

```
Kaggle 2x T4:
per_device=2, grad_accum=8, gpus=2 → effective=32

Effective batch size effects:
Small (8–16):   Noisy gradients, faster to overfit, higher LR needed
Medium (32–64): Good default
Large (128+):   Stable but may need longer warmup and lower LR
```

**The batch size–LR linear scaling rule:**

```
If you double effective batch size → multiply LR by sqrt(2) ≈ 1.4x
If you halve effective batch size  → divide LR by sqrt(2) ≈ 0.7x

Example:
  Original: batch=32, LR=2e-4
  Changed:  batch=64, LR=2e-4 × 1.4 = ~3e-4
```

---

### 2.4 GRPO-Specific Hyperparameters

```python
GRPOConfig(
    num_generations=8,      # G — more = better signal, more memory
    temperature=0.9,        # diversity of generations — do not go below 0.7
    beta=0.01,              # KL penalty — start low, increase if KL explodes
    epsilon=0.2,            # PPO clip range — usually leave at default
    max_completion_length=512,  # cap generation length
)
```

**num_generations (G) trade-off:**

```
G=2  → Very noisy advantage estimates, fast but low signal
G=4  → Minimum useful signal, good for Kaggle memory constraints
G=8  → Standard, good signal quality
G=16 → Best signal, needs big GPU

If accuracy reward not improving: try G=8 before changing anything else
```

**temperature in GRPO:**

```
temperature=0.3  → Too deterministic → rewards/std ≈ 0 → no learning signal
temperature=0.7  → Good diversity while staying coherent
temperature=1.0  → High diversity → noisy but sometimes needed for hard tasks
temperature=1.2+ → Too random → gibberish generations → reward always 0

Start at 0.9. If rewards/std is low, increase to 1.0.
```

---

## Part 3 — Data Strategy

### 3.1 Quality vs Quantity Decision Tree

```
Question: Should I add more data or improve existing data?

Is eval loss still decreasing at end of training?
    YES → You haven't saturated the data. Add more.
    NO  → You've saturated. Improving data quality is more impactful now.

Is train loss >> eval loss? (underfitting)
    YES → Add more data OR increase model capacity (rank)
    NO  → go to next check

Is train loss << eval loss? (overfitting)
    YES → Add more diverse data, or reduce rank/epochs
    NO  → Data is not the problem. Look at hyperparameters.
```

### 3.2 Data Quality Filters (What to Actually Do)

**Filter 1: Length distribution**

```python
import matplotlib.pyplot as plt

lengths = [len(tokenizer(ex['text'])['input_ids']) for ex in dataset]
plt.hist(lengths, bins=50)
plt.axvline(x=512, color='r', label='your max_seq_length')
plt.show()

# If > 30% of examples are being truncated → increase max_seq_length
# If most examples are < 100 tokens → data is too simple for the task
```

**Filter 2: Deduplication**

```python
# Fuzzy dedup — removes near-duplicate examples that corrupt training
from datasketch import MinHash, MinHashLSH

# Or simple exact dedup:
seen = set()
def dedup(example):
    key = example['messages'][0]['content'][:100]  # first 100 chars of prompt
    if key in seen:
        return False
    seen.add(key)
    return True

dataset = dataset.filter(dedup)
```

**Filter 3: Response quality check using model-as-judge**

```python
from transformers import pipeline

judge = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device=0)

def quality_score(example):
    prompt = f"""Rate this response quality from 1-5.
Question: {example['messages'][0]['content'][:200]}
Response: {example['messages'][1]['content'][:200]}
Score (just a number 1-5):"""
    
    out = judge(prompt, max_new_tokens=5)[0]['generated_text']
    score_match = re.search(r'[1-5]', out.split("Score")[-1])
    return int(score_match.group()) if score_match else 3

# Keep only quality >= 3
dataset = dataset.filter(lambda x: quality_score(x) >= 3)
```

**Filter 4: For GRPO — difficulty filtering**

```python
# Problems where model ALWAYS gets correct → too easy (no learning signal)
# Problems where model NEVER gets correct → too hard (no learning signal)
# Target: 20-80% accuracy range on initial model

def difficulty_filter(example, model, tokenizer, n_samples=4):
    correct = 0
    for _ in range(n_samples):
        output = generate(model, tokenizer, example['prompt'])
        if extract_answer(output) == example['answer']:
            correct += 1
    pass_rate = correct / n_samples
    return 0.1 < pass_rate < 0.9   # keep problems in the sweet spot

# This is called "difficulty filtering" or "curriculum selection"
```

### 3.3 Data Mix Ratios

When mixing multiple datasets for SFT:

```python
from datasets import concatenate_datasets, interleave_datasets

# Option 1: Equal mix (simple)
mixed = concatenate_datasets([dataset_a, dataset_b])
mixed = mixed.shuffle(seed=42)

# Option 2: Weighted mix (better)
# If you want 70% general, 30% domain-specific:
mixed = interleave_datasets(
    [general_dataset, domain_dataset],
    probabilities=[0.7, 0.3],
    seed=42,
    stopping_strategy="all_exhausted"
)
```

**Rule of thumb for mixing:**

```
If domain data < 10% of mix → model forgets domain capability
If domain data > 80% of mix → model loses general capability (catastrophic forgetting)
Sweet spot: 20–40% domain data in the mix
```

---

## Part 4 — Systematic Improvement Process

### 4.1 The Diagnostic Checklist (run this after every training)

```
After SFT:
□ Plot train vs eval loss — which shape is it?
□ Check grad_norm — any spikes > 5?
□ Run format adherence check — is output structured correctly?
□ Generate 20 outputs manually — do they look right?
□ Check token length distribution — are examples getting truncated?

After GRPO:
□ Check rewards/std — is it > 0.1 throughout?
□ Check reward component curves separately — format before accuracy?
□ Check KL — is it < 5?
□ Check sample outputs at step 100, 500, 1000 — is reasoning getting longer?
□ Count answers extracted — is extraction rate > 90%?

After DPO:
□ Check rewards/margins — growing steadily?
□ Check rewards/accuracies — 65-85% range?
□ Check output lengths — getting longer? (length exploitation)
□ Compare outputs on same prompt before/after DPO
```

### 4.2 What to Change First — Priority Order

```
For SFT underperforming:
Priority 1: Data quality (filter, deduplicate, inspect manually)
Priority 2: Learning rate (most impactful single hyperparameter)
Priority 3: Number of epochs (add 1 epoch at a time)
Priority 4: LoRA rank (double it)
Priority 5: Target modules (add MLP layers)

For GRPO not improving accuracy:
Priority 1: Reward function bug check (unit test it)
Priority 2: Temperature (increase to 1.0)
Priority 3: num_generations (increase to 8)
Priority 4: Dataset difficulty filter (remove always-right and always-wrong)
Priority 5: Learning rate (reduce by 2x)

For DPO not learning preferences:
Priority 1: Data quality (inspect chosen vs rejected manually — are they actually different?)
Priority 2: Beta (reduce by 2x)
Priority 3: More diverse preference pairs
Priority 4: Extend training (add 1 epoch)
```

### 4.3 Ablation Template — Change One Thing at a Time

```python
# Track all experiments. Never "remember" results.

experiments = {
    "baseline": {
        "lr": 2e-4, "rank": 16, "epochs": 1, "data": "ultrachat_10k",
        "eval_loss": 1.43, "format_adherence": 0.72, "notes": "first run"
    },
    "exp_01_higher_rank": {
        "lr": 2e-4, "rank": 32, "epochs": 1, "data": "ultrachat_10k",
        "eval_loss": 1.38, "format_adherence": 0.78, "notes": "improved"
    },
    "exp_02_more_data": {
        "lr": 2e-4, "rank": 32, "epochs": 1, "data": "ultrachat_20k",
        "eval_loss": 1.31, "format_adherence": 0.81, "notes": "best so far"
    },
}

# Rule: only change ONE variable between experiments
# Exception: if you change dataset, reset hyperparameters to baseline
```

---

## Part 5 — What You Are Missing (New Topics)

### 5.1 Evaluation — You Need Multiple Lenses

Loss and reward are not enough. You need task-specific evaluation:

```python
# 1. ROUGE score — for summarization/instruction following
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, prediction)

# 2. Exact match — for math/code
exact_match = predicted_answer == ground_truth

# 3. LLM-as-judge — for open-ended quality
judge_prompt = f"""
Rate the response on:
- Helpfulness: 1-5
- Accuracy: 1-5  
- Format: 1-5

Question: {question}
Response: {response}
Return JSON only.
"""

# 4. Benchmark eval using lm-evaluation-harness
# !pip install lm-eval
# lm_eval --model hf --model_args pretrained=./my_model --tasks gsm8k,arc_easy
```

### 5.2 Catastrophic Forgetting

When you fine-tune, the model can forget general capabilities. Test for this:

```python
# Before training, record baseline on general tasks
general_tasks = [
    "What is the capital of France?",
    "Write a Python function to reverse a string.",
    "Summarize: The quick brown fox...",
]
baseline_responses = [generate(base_model, t) for t in general_tasks]

# After training, compare
finetuned_responses = [generate(finetuned_model, t) for t in general_tasks]

# If responses degrade significantly → you have catastrophic forgetting
# Fix: add a small % of general data back into your training mix (5-10%)
```

### 5.3 NEFTune — Free Quality Boost for SFT

Adding noise to embeddings during training consistently improves instruction following with zero cost:

```python
SFTConfig(
    neftune_noise_alpha=5,  # add this one line → often +1-2% on benchmarks
)
```

### 5.4 Learning Rate Finder (Stop Guessing LR)

```python
# Before long training runs, do a short LR range test
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt

# Train for 100 steps with LR increasing from 1e-6 to 1e-2
# Plot loss vs LR
# Choose LR where loss decreases fastest (steepest slope)
# That's your optimal LR, use it ÷ 10 for safety

lrs = []
losses = []
lr = 1e-6
for step, batch in enumerate(dataloader):
    if step > 100: break
    loss = train_step(batch, lr=lr)
    lrs.append(lr)
    losses.append(loss)
    lr *= 1.3  # increase by 30% each step

plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
# Optimal LR = steepest downward slope
```

### 5.5 Merge Strategies — Often Overlooked

After training, how you merge LoRA back matters:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "./my_lora")

# Option 1: Simple merge (standard)
merged = model.merge_and_unload()

# Option 2: DARE merge — drops low-magnitude LoRA weights before merging
# Often better than simple merge for generalization
from peft import TaskType
merged = model.merge_and_unload(safe_merge=True, progressbar=True)

# Option 3: Model merging (combining multiple fine-tuned models)
# Combine SFT model + reasoning model without retraining
# Uses TIES, DARE, SLERP interpolation
# Library: mergekit (pip install mergekit)
```

### 5.6 The Missing Loop — Iterative Data Improvement

```
The actual workflow professionals use:

Step 1: Train on initial data
Step 2: Generate outputs on held-out test set
Step 3: Identify failure categories (not just failure rate)
         e.g. "fails on multi-step math" not just "50% accuracy"
Step 4: Collect/generate more data specifically for failure categories
Step 5: Retrain with augmented data
Step 6: Repeat

This targeted data collection beats:
- Random data scaling
- Blind hyperparameter tuning
- Switching model architectures
```

---

## Quick Reference Card

```
SIGNAL                         DIAGNOSIS                     FIRST FIX
────────────────────────────────────────────────────────────────────────
SFT loss spikes                LR too high                   LR × 0.3
SFT loss flat                  LR too low                    LR × 3
Train loss ↓ eval loss ↑       Overfitting                   More data
Loss → 0 very fast             Prompt not masked             Fix collator
grad_norm > 5                  Unstable                      max_grad_norm=1.0

GRPO rewards/std ≈ 0           Bad temperature or G          temp → 1.0, G → 8
GRPO accuracy flat at 0        Reward function bug           Unit test reward fn
GRPO KL > 5                    Drifting from ref             beta × 3
GRPO format before accuracy    ✓ Normal                      Do nothing

DPO margin not growing         Beta too high                 beta × 0.5
DPO margin too fast            Overfitting prefs             beta × 2
DPO outputs getting longer     Length exploitation           Switch to SimPO
DPO accuracy > 90%             Suspicious                    Check data quality
────────────────────────────────────────────────────────────────────────
```