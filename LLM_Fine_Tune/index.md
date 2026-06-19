# LLM Fine-Tuning Mastery Roadmap
### For ML Engineer / AI Engineer Interview Preparation
**Target:** Both ML Engineer + AI Engineer roles  
**Weekly commitment:** 8–15 hours  
**Hardware:** 16GB Intel GPU (local) + Google Colab + Kaggle  
**Estimated duration:** ~5–6 months to full interview-ready depth

---

> **How to use this roadmap:**  
> We go chapter by chapter. Each chapter has multiple lessons.  
> Every lesson has theory + code + an experiment you run yourself.  
> You do not move to the next chapter until you can *explain* the current one out loud.

---

## The Core Philosophy

Fine-tuning mastery is not about running scripts. It is about:
1. **Understanding what is happening at every step** — mathematically and intuitively
2. **Knowing why you made each decision** — model choice, hyperparameters, dataset design
3. **Being able to read training behavior** — loss curves, metrics, failures
4. **Being able to debug** — what broke, why, how you fixed it
5. **Being able to tell a story** — "I tried X, observed Y, changed Z, got result W"

Every chapter feeds into this ability.

---

## Roadmap Overview

| Chapter | Title | Estimated Time |
|---------|-------|---------------|
| 0 | Environment Setup & Tooling | 1 week |
| 1 | How Language Models Actually Work (Under the Hood) | 1.5 weeks |
| 2 | What Fine-Tuning Actually Does to a Model | 1 week |
| 3 | Dataset Preparation — The Most Underrated Skill | 1.5 weeks |
| 4 | Full Fine-Tuning — Theory and Practice | 1 week |
| 5 | Parameter Efficient Fine-Tuning (PEFT) — Deep Mastery | 2 weeks |
| 6 | QLoRA — The Method You Already Used, Now Deeply | 1 week |
| 7 | Hyperparameters — Reasoning, Not Guessing | 1.5 weeks |
| 8 | Reading Training Behavior — Loss, Metrics, Curves | 1.5 weeks |
| 9 | Evaluation — Beyond Just Loss | 1.5 weeks |
| 10 | Modern Alignment Techniques — SFT, RLHF, DPO | 2 weeks |
| 11 | Debugging Failed Runs — The Engineer's Skill | 1 week |
| 12 | Inference, Deployment & Production Concerns | 1 week |
| 13 | Interview Preparation — Telling Your Story | 1 week |

---

## Chapter 0: Environment Setup & Tooling
*"Your workbench must be solid before you build anything."*

### Why this chapter exists
Most beginners skip environment setup and suffer later. A properly configured environment saves you hours of debugging unrelated to your actual learning.

### Lessons

**Lesson 0.1 — Python Environment & Dependency Management**
- `conda` vs `venv` vs `uv` — what to use and why
- Understanding `requirements.txt` vs `pyproject.toml`
- How to isolate experiments so they don't break each other
- *Practical:* Set up a clean environment for this entire course

**Lesson 0.2 — Understanding Your Hardware Constraints**
- How to profile your Intel 16GB GPU (what is it actually? Arc? Iris Xe?)
- VRAM vs RAM — what each limits
- CPU offloading — what it means and when it helps
- What model sizes you can actually run locally vs need Colab for
- *Practical:* Run `torch.cuda.get_device_properties()` and understand every field

**Lesson 0.3 — Google Colab & Kaggle for ML Work**
- Colab Pro vs Free — GPU types, session limits, RAM limits
- Kaggle GPU limits (30 hrs/week on T4s and P100s)
- How to persist work across sessions (Google Drive mounting, Kaggle datasets)
- How to avoid losing runs (checkpointing strategy)
- *Practical:* Set up your Colab + Kaggle workspace, connect Drive, test GPU availability

**Lesson 0.4 — Core Libraries You Must Know Cold**
- `transformers` — the heart of everything (Hugging Face)
- `datasets` — how HF manages data
- `peft` — parameter efficient fine-tuning library
- `trl` — for SFT, RLHF, DPO training
- `bitsandbytes` — quantization
- `accelerate` — multi-GPU and mixed precision training
- `wandb` / `tensorboard` — experiment tracking (critical, not optional)
- *Practical:* Install all, run hello-world with each, understand what each one does

**Lesson 0.5 — Experiment Tracking from Day One**
- Why you MUST log every run — memory is not enough
- Setting up Weights & Biases (free tier is fine)
- What to log: loss, learning rate, gradient norm, eval metrics, config
- How to compare runs
- *Practical:* Run a toy training loop and log everything to W&B

---

## Chapter 1: How Language Models Actually Work (Under the Hood)
*"You said you have strong theory. This chapter tests that and fills the gaps."*

### Why this chapter exists
Fine-tuning changes specific parts of the model. If you don't know what those parts are and what they do, you cannot reason about what fine-tuning is doing.

### Lessons

**Lesson 1.1 — Tokenization Deep Dive**
- What a tokenizer actually does (BPE, WordPiece, SentencePiece)
- Vocabulary size and why it matters for fine-tuning
- Special tokens: `[BOS]`, `[EOS]`, `[PAD]`, `[SEP]`, chat templates
- How tokenization affects your training data
- *Common interview question:* "Why do some tokens cost more than others?"
- *Practical:* Tokenize 10 different strings, inspect input IDs, attention masks, understand padding

**Lesson 1.2 — Transformer Architecture Internals (The Real Depth)**
- Embedding layer — what it stores and learns
- Multi-head self-attention — the mechanics, not just the concept
- What Q, K, V matrices are and what they represent
- Feedforward layers — their role (often underestimated)
- Layer normalization — pre-norm vs post-norm, why it matters for training stability
- Residual connections — why models get deeper without breaking
- *Practical:* Load a small model (GPT-2), print every layer, understand what each weight matrix does

**Lesson 1.3 — The Forward Pass Step by Step**
- Input → Embedding → Attention → FFN → Output logits
- What logits are and what they represent
- Softmax, temperature, and probability distributions over vocab
- How the model "chooses" the next token
- *Practical:* Write a minimal forward pass manually, compare output to `model.generate()`

**Lesson 1.4 — Autoregressive Generation**
- How text generation actually works token by token
- Greedy decoding vs sampling vs beam search
- Temperature, top-k, top-p (nucleus sampling) — what each does mathematically
- Why generation is slow and what affects it
- *Practical:* Generate text with different decoding strategies, observe and explain differences

**Lesson 1.5 — Pre-training vs Fine-tuning vs In-Context Learning**
- What pre-training learns (statistical patterns across internet-scale text)
- What the model "knows" before you touch it
- The difference between fine-tuning (changing weights) vs prompting (same weights)
- Why few-shot prompting works and when fine-tuning beats it
- *Common interview question:* "When would you fine-tune vs just prompt?"

---

## Chapter 2: What Fine-Tuning Actually Does to a Model
*"This is the chapter you were missing when you did QLoRA the first time."*

### Why this chapter exists
Fine-tuning is gradient descent on a pre-trained model. You must understand what that means at the weight level.

### Lessons

**Lesson 2.1 — The Training Objective for Language Models**
- Cross-entropy loss over next-token prediction — the full math
- What the model is minimizing and why
- Perplexity — what it is and why it is not always what you want
- Teacher forcing — how training differs from inference
- *Practical:* Compute cross-entropy loss manually on a small sequence, match it to `model()` output

**Lesson 2.2 — Gradient Descent & Backpropagation Review**
- The optimization loop: forward pass → loss → backward pass → weight update
- Why learning rate is the most important hyperparameter
- Gradient accumulation — why and when to use it (critical for small GPU setups)
- Mixed precision training (fp16, bf16) — what it does and why you need it
- *Practical:* Write a manual training step, inspect gradients, verify they flow

**Lesson 2.3 — What Weights Actually Change During Fine-Tuning**
- Which layers are typically fine-tuned and which are sometimes frozen
- The knowledge stored in early vs late layers
- Catastrophic forgetting — what it is, why it happens, how to mitigate it
- Weight initialization — why starting from a pre-trained model is so powerful
- *Practical:* Fine-tune a small model, compare weights before and after, measure which layers changed most

**Lesson 2.4 — Types of Fine-Tuning Objectives**
- Causal LM fine-tuning (predicting next token) — the standard approach
- Masked LM fine-tuning (BERT-style) — when and why
- Instruction fine-tuning vs continued pre-training vs task-specific fine-tuning
- What changes in the data format for each type
- *Practical:* Prepare the same dataset in two formats and understand the difference in loss computation

**Lesson 2.5 — Chat Templates and Instruction Formatting**
- Why chat models need special formatting (system, user, assistant turns)
- `apply_chat_template()` in HuggingFace
- Different templates: Llama chat, ChatML, Alpaca, ShareGPT formats
- Loss masking on prompt tokens — why you do NOT want to train on the input
- *This is a major interview topic — most people get this wrong*
- *Practical:* Format data correctly for a chat model, verify loss is only computed on assistant turns

---

## Chapter 3: Dataset Preparation — The Most Underrated Skill
*"Bad data kills good models. This chapter is where most ML engineers fail silently."*

### Why this chapter exists
Interviewers ask about this because it reveals whether you actually ran experiments or just copied a tutorial. The dataset is the most impactful thing you control.

### Lessons

**Lesson 3.1 — What Makes a Good Fine-Tuning Dataset**
- Quality vs quantity — the fundamental tradeoff
- Signal-to-noise ratio in training data
- Diversity — why repetitive data hurts
- Length distribution — padding and packing effects
- Dataset size guidelines for different tasks (rules of thumb + why they are rules of thumb)
- *Common interview question:* "How much data do you need to fine-tune?"

**Lesson 3.2 — Data Collection and Sourcing**
- Public datasets: Alpaca, ShareGPT, OpenHermes, Dolly, FLAN — when to use each
- Synthetic data generation using stronger models (GPT-4 distillation)
- Web scraping and cleaning considerations
- Legal/license issues with training data (important for real roles)
- *Practical:* Download and inspect 3 different open fine-tuning datasets, understand their structure

**Lesson 3.3 — Data Cleaning and Filtering**
- Deduplication — why it matters more than you think
- Quality filtering — length filters, language filters, content filters
- Heuristic-based filtering (perplexity filtering, regex filters)
- Model-based filtering (using a classifier to score quality)
- *Practical:* Take a raw dataset, apply cleaning pipeline, measure what percentage is removed and why

**Lesson 3.4 — Data Formatting and Prompt Engineering for Training**
- Input-output pairs vs conversational format vs completion format
- How format affects what the model learns
- Designing prompts for training vs inference consistency
- Common formatting mistakes that hurt training
- *Practical:* Take a task, design 3 different dataset formats, train on each and compare

**Lesson 3.5 — Train / Validation / Test Splits for Fine-Tuning**
- Why the split strategy matters more than in classical ML
- Data contamination — what it is and how to avoid it
- How to create a meaningful evaluation set (not just random 10%)
- Held-out test set vs real-world evaluation
- *Practical:* Build a proper split for a task-specific dataset, verify no contamination

**Lesson 3.6 — Tokenization and Packing Strategies**
- Max sequence length — how to choose it
- Packing (concatenating examples) — why it improves GPU utilization
- Padding — when unavoidable, how to handle it correctly
- Truncation — what to truncate and what to throw away
- `DataCollatorForLanguageModeling` vs `DataCollatorForSeq2Seq`
- *Practical:* Compare training speed and memory use with packing vs padding on same dataset

---

## Chapter 4: Full Fine-Tuning — Theory and Practice
*"You need to understand this before PEFT makes sense."*

### Why this chapter exists
PEFT methods like LoRA exist to solve problems with full fine-tuning. You must understand those problems first.

### Lessons

**Lesson 4.1 — The Full Fine-Tuning Process End to End**
- What "full fine-tuning" means: every weight is updated
- Memory requirements: model weights + gradients + optimizer states + activations
- Why full fine-tuning is often impossible on consumer hardware
- When full fine-tuning is the right choice (you have the resources + need max performance)
- *Practical:* Calculate memory requirements for full fine-tuning a 7B model, verify on Colab

**Lesson 4.2 — Optimizers for Fine-Tuning**
- AdamW — the default choice and why
- SGD — when and why you might use it
- Adafactor — for memory-constrained training
- 8-bit Adam from bitsandbytes — how it reduces optimizer memory
- *Practical:* Train the same setup with AdamW and 8-bit Adam, compare memory and results

**Lesson 4.3 — Learning Rate Schedulers**
- Why a constant learning rate is almost never optimal
- Warmup — why it matters for fine-tuning (especially with pre-trained models)
- Cosine decay, linear decay — differences in behavior
- How to choose: number of warmup steps, total steps, final LR
- *Practical:* Plot 3 different LR schedules, observe their effect on training loss curves

**Lesson 4.4 — Batch Size, Gradient Accumulation, and Effective Batch Size**
- Why batch size affects generalization, not just speed
- Gradient accumulation: simulating large batch sizes on small GPUs
- The math: `effective_batch_size = batch_size × gradient_accumulation_steps`
- Linear scaling rule and when it breaks down
- *Practical:* Run same training with batch_size=4 vs batch_size=1 + grad_accum=4, compare

**Lesson 4.5 — Mixed Precision Training in Depth**
- FP32 vs FP16 vs BF16 — numerical ranges and precision tradeoffs
- Why BF16 is preferred over FP16 for modern LLMs
- How Automatic Mixed Precision (AMP) works
- When mixed precision causes instability and how to fix it
- *Practical:* Train in FP32 vs BF16, compare speed, memory, and loss curves

---

## Chapter 5: Parameter Efficient Fine-Tuning (PEFT) — Deep Mastery
*"This is the heart of modern LLM fine-tuning. Master this chapter."*

### Why this chapter exists
PEFT is why fine-tuning is accessible without massive compute. Every serious ML role expects you to know this deeply — not just "I used LoRA."

### Lessons

**Lesson 5.1 — Why PEFT Exists: The Problem It Solves**
- The memory and compute cost of full fine-tuning
- The problem of storing one model per task (catastrophic for deployment)
- The insight: pre-trained weights contain most of the knowledge; we just need small adaptations
- Overview of PEFT landscape: LoRA, prefix tuning, prompt tuning, IA3, adapters
- *Interview framing:* Be able to explain why PEFT was necessary historically

**Lesson 5.2 — Prompt Tuning and Prefix Tuning**
- Soft prompts — what they are and how they are trained
- Prefix tuning — prepending trainable vectors to attention keys and values
- What parameters are trained and how many
- When these work well and when they don't
- Why LoRA is generally preferred over both
- *Practical:* Implement prefix tuning on a small model, compare with LoRA

**Lesson 5.3 — Adapter Layers**
- What an adapter is: a small bottleneck module inserted into each transformer layer
- The original Houlsby adapter architecture
- Parameter count vs performance tradeoff
- Why adapters are less popular than LoRA today
- *Practical:* Implement a simple adapter, understand how it plugs into the transformer

**Lesson 5.4 — LoRA: Low-Rank Adaptation — The Mathematics**
- The core insight: weight updates during fine-tuning are low-rank
- What low-rank decomposition means: `ΔW = A × B` where A is (d × r) and B is (r × k)
- Why this works: intrinsic dimensionality of fine-tuning tasks
- What happens to the original weights (they are FROZEN)
- Rank r — what it means and how it affects parameter count
- Alpha (scaling factor) — why it exists and what it does: `scale = alpha / r`
- *This is a guaranteed interview topic*
- *Practical:* Implement LoRA from scratch manually (no PEFT library), apply to a linear layer

**Lesson 5.5 — LoRA: Choosing Which Modules to Target**
- Why we apply LoRA to attention matrices (Q, K, V, O projections)
- Should you also target FFN layers? (Gate, up, down projections)
- How module targeting affects parameter count and performance
- The `target_modules` parameter in PEFT
- *Practical:* Fine-tune same model with LoRA on Q+V only vs all attention vs all linear layers

**Lesson 5.6 — LoRA: Rank Selection (Critical Interview Topic)**
- What happens with rank=1 vs rank=4 vs rank=16 vs rank=64 vs rank=256
- Higher rank = more parameters = more capacity = more memory = more risk of overfitting
- Rule of thumb starting points and how to verify with ablations
- When low rank is enough (most task-specific fine-tuning)
- When you need higher rank (diverse capabilities, domain adaptation)
- *Practical:* Train with r=4, r=16, r=64 on same dataset, plot loss and eval metrics

**Lesson 5.7 — LoRA: Alpha and the Scaling Factor**
- Why alpha was introduced (to avoid retuning LR when changing rank)
- The convention: `alpha = 2 × rank` as a starting point
- How alpha and rank together determine the effective update magnitude
- What happens if alpha is too high or too low
- *Practical:* Fix rank=16, vary alpha (8, 16, 32, 64), observe training dynamics

**Lesson 5.8 — LoRA Weight Merging**
- What merging means: `W_merged = W_original + (alpha/r) × B × A`
- Why you merge for deployment (no overhead during inference)
- When NOT to merge (when you want to swap adapters)
- `merge_and_unload()` in PEFT
- Multi-adapter scenarios — using different adapters for different tasks
- *Practical:* Train a LoRA adapter, merge it, verify output is identical to unmerged

**Lesson 5.9 — IA3 and Other Modern PEFT Methods**
- IA3: scaling activations instead of adding low-rank matrices
- Why IA3 uses even fewer parameters than LoRA
- When IA3 works well (few-shot style fine-tuning)
- Brief survey of newer methods: DoRA, VeRA, LoftQ
- *Practical:* Run IA3 on same task as LoRA, compare parameter count and performance

---

## Chapter 6: QLoRA — The Method You Already Used, Now Deeply
*"You ran this as a fresher. Now you understand every line of it."*

### Why this chapter exists
QLoRA is your most direct interview experience. You must be able to explain every component in depth.

### Lessons

**Lesson 6.1 — Quantization Fundamentals**
- What quantization is: representing weights in fewer bits
- INT8, INT4, FP4, NF4 — the formats and their tradeoffs
- Symmetric vs asymmetric quantization
- Quantization error — where it comes from and how to minimize it
- Why quantization matters for fine-tuning on consumer hardware

**Lesson 6.2 — bitsandbytes and 4-bit Quantization**
- How bitsandbytes implements 4-bit quantization
- NF4 (Normal Float 4) — why it was designed for normally-distributed weights
- Double quantization — quantizing the quantization constants to save even more memory
- How to load a model in 4-bit and what you get
- *Practical:* Load Llama-3-8B in FP16 vs 4-bit, compare memory, compare output quality

**Lesson 6.3 — How QLoRA Combines Quantization and LoRA**
- The full QLoRA setup: frozen 4-bit base model + trainable FP16 LoRA adapters
- Why the base model can be 4-bit but adapters must be higher precision
- The computation graph: how gradients flow through quantized weights to LoRA adapters
- What you are actually training (only A and B matrices in LoRA)
- Memory savings breakdown: where each saving comes from
- *Interview question:* "Walk me through exactly what QLoRA does step by step"

**Lesson 6.4 — QLoRA Setup in Code (Every Parameter Explained)**
- `BitsAndBytesConfig` — every parameter and what it does
- `prepare_model_for_kbit_training()` — what this does and why it is required
- `get_peft_model()` — how the adapter is attached
- `LoraConfig` — every parameter explained in context of QLoRA
- Common mistakes in QLoRA setup
- *Practical:* Write QLoRA setup from scratch without copying a tutorial, explain each line

**Lesson 6.5 — QLoRA Limitations and When to Use Something Else**
- Quality gap vs full fine-tuning — how large is it really?
- Speed tradeoffs — quantized models are slower per token
- When QLoRA is the right choice vs full fine-tuning vs other PEFT
- Newer alternatives: GPTQ + LoRA, AWQ + LoRA
- *Practical:* Compare QLoRA vs LoRA (FP16) on same task, measure quality vs memory

---

## Chapter 7: Hyperparameters — Reasoning, Not Guessing
*"The question you couldn't answer before. Now you will have a full answer."*

### Why this chapter exists
"Why did you choose those hyperparameters?" is the most common fine-tuning interview question. You need a structured reasoning process, not "I copied from a tutorial."

### Lessons

**Lesson 7.1 — The Hyperparameter Reasoning Framework**
- The right mental model: start from what you know, reason about constraints, experiment to verify
- First principles vs empirical tuning — when to use each
- How to document your reasoning (important for interviews)
- The experiment-log mindset: every run teaches you something

**Lesson 7.2 — Learning Rate: The Most Important Hyperparameter**
- Why fine-tuning LR is typically 10–100x lower than pre-training LR
- The intuition: large LR destroys pre-trained knowledge (catastrophic forgetting)
- Typical ranges: 1e-5 to 1e-3 for full fine-tuning; 1e-4 to 1e-3 for LoRA
- Learning rate finders — how they work
- Signs your LR is too high vs too low in the loss curve
- *Practical:* Run 5 experiments varying LR only, document what you observe in each

**Lesson 7.3 — Batch Size and Gradient Accumulation**
- How batch size affects gradient noise (small = noisy = sometimes regularizing)
- GPU memory limit → physical batch size constraint → gradient accumulation
- Effective batch size: how to match it across different setups
- When to increase vs decrease batch size
- *Practical:* Map your Colab T4 memory limit to effective batch sizes for different model sizes

**Lesson 7.4 — Number of Epochs and Early Stopping**
- How many epochs to train? The honest answer: it depends, and here's how to decide
- Overfitting signals in fine-tuning (different from classical ML overfitting)
- Early stopping — how to implement it and why it is essential
- Checkpoint selection — don't always use the last checkpoint
- *Practical:* Train for too many epochs deliberately, observe and document the overfitting pattern

**Lesson 7.5 — LoRA-Specific Hyperparameters**
- Rank (r): covered in Chapter 5, but now in the context of hyperparameter search
- Alpha: how to set it relative to rank
- Dropout in LoRA adapters — when and why
- Target modules — as a hyperparameter decision
- Bias: `none` vs `all` vs `lora_only` — what each does

**Lesson 7.6 — Regularization in Fine-Tuning**
- Weight decay — what it does in the fine-tuning context
- Dropout — where to apply it and typical values
- Label smoothing — when it helps
- How regularization interacts with PEFT (often less needed with LoRA)

**Lesson 7.7 — Designing a Hyperparameter Search Strategy**
- Grid search vs random search vs Bayesian optimization — practical tradeoffs
- What to fix first and what to sweep
- Resource-efficient HP search on limited compute
- Reading W&B parallel coordinates plots to understand HP interactions
- *Practical:* Design and run a 10-experiment HP search, write a report on what you found

---

## Chapter 8: Reading Training Behavior — Loss, Metrics, Curves
*"This is what separates engineers from script-runners."*

### Why this chapter exists
"Did anything go wrong?" — if you can't read training curves, you can't answer this. This chapter makes training dynamics readable.

### Lessons

**Lesson 8.1 — Training Loss vs Validation Loss: The Fundamental Diagnostic**
- What each loss curve tells you about model behavior
- The four scenarios: both falling, training falls but val plateaus, training falls but val rises, both plateau early
- Healthy loss curves — what they look like and what they mean
- Overfitting — the pattern, the causes, the fixes
- Underfitting — the pattern, the causes, the fixes
- *Practical:* Create labeled diagrams of each scenario from real training runs

**Lesson 8.2 — Loss Spikes and Instability**
- Why loss spikes happen (LR too high, bad batch, gradient explosion)
- Gradient clipping — what it does and why it prevents spikes
- `max_grad_norm` — what value to use and why
- How to tell a temporary spike from a diverging run
- Recovering from spikes: rolling back to checkpoint vs continuing
- *Practical:* Deliberately cause a loss spike by increasing LR, observe and document

**Lesson 8.3 — Gradient Norms and What They Tell You**
- What the gradient norm represents
- High gradient norms — what they indicate
- Gradient explosion vs vanishing — how each manifests
- Logging gradient norms in W&B and reading them
- *Practical:* Log gradient norms across a full training run, annotate the plot

**Lesson 8.4 — Learning Rate and Loss Relationship**
- How to read the LR schedule in your loss curve
- The warmup effect — what it looks like
- Cosine decay — how it affects loss in late training
- When the loss stops decreasing: LR too low vs data exhausted vs model saturated

**Lesson 8.5 — Perplexity: What It Is and When It Matters**
- Perplexity = exp(loss) — the intuition (model's "surprise" at test data)
- Why perplexity is used for language modeling tasks
- When perplexity is the right metric and when it is misleading
- Baseline perplexity for common models on common benchmarks

**Lesson 8.6 — GPU Utilization and Training Efficiency**
- Reading GPU utilization metrics
- Why low GPU utilization means your dataloader is the bottleneck
- `num_workers` in DataLoader — how to tune it
- Memory fragmentation and why your GPU crashes at step 100 but not step 1
- *Practical:* Profile a training run with `torch.profiler`, identify bottlenecks

**Lesson 8.7 — Recognizing Common Failure Patterns**
- Loss goes to 0 immediately (data leakage or label leakage)
- Loss never moves (LR too low, frozen layers, optimizer bug)
- Loss oscillates wildly (LR too high)
- Validation loss diverges early (overfitting, dataset quality issue)
- Training crashes with OOM (memory planning failure)
- *Practical:* Deliberately reproduce each failure, document what you see and how to fix it

---

## Chapter 9: Evaluation — Beyond Just Loss
*"You trained a model. Now prove it is actually good."*

### Why this chapter exists
Loss on the training set tells you the model fit the data. It does NOT tell you the model is useful. Real evaluation is what interviewers want to hear about.

### Lessons

**Lesson 9.1 — The Evaluation Philosophy**
- Why training loss is a proxy, not a goal
- Task-specific evaluation vs general capability evaluation
- Automated metrics vs human evaluation — when to use each
- The evaluation-driven development mindset

**Lesson 9.2 — Automatic Metrics for Generation Tasks**
- BLEU — what it measures, its weaknesses, when to use it
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) — for summarization tasks
- METEOR — improvement over BLEU
- BERTScore — semantic similarity using embeddings
- Why none of these are perfect and how to combine them
- *Practical:* Compute all four on same model outputs, compare and interpret

**Lesson 9.3 — Perplexity-Based Evaluation**
- Using perplexity on a held-out set as evaluation
- Why lower perplexity does not always mean better task performance
- Domain-specific perplexity benchmarks

**Lesson 9.4 — Task-Specific Evaluation**
- Classification tasks: accuracy, F1, precision, recall
- QA tasks: Exact Match (EM), F1
- Code generation: pass@k
- Instruction following: how to evaluate compliance
- Designing your own eval set for a custom task
- *Practical:* Build a 50-example eval set for a specific task, compute multiple metrics

**Lesson 9.5 — LM-Evaluation-Harness**
- What `lm-evaluation-harness` (EleutherAI) is and how to use it
- Common benchmarks: MMLU, HellaSwag, TruthfulQA, GSM8K
- Running benchmarks on your fine-tuned model
- Interpreting benchmark results — what they actually measure
- *Practical:* Run at least 2 benchmarks on a fine-tuned vs base model, compare

**Lesson 9.6 — LLM-as-Judge Evaluation**
- Using a stronger model (GPT-4, Claude) to evaluate outputs
- Designing evaluation prompts for LLM judges
- Bias in LLM-as-judge (position bias, verbosity bias)
- When LLM-as-judge is reliable vs unreliable
- *Practical:* Build a simple LLM-as-judge pipeline using API calls

**Lesson 9.7 — Qualitative Evaluation and Error Analysis**
- Why you must read model outputs, not just look at numbers
- Building an error taxonomy: what kinds of mistakes does the model make?
- Slice-based evaluation: where does the model fail specifically?
- Writing an evaluation report (interview-ready format)
- *Practical:* Take 50 model outputs, read every one, categorize failures, write a report

---

## Chapter 10: Modern Alignment Techniques — SFT, RLHF, DPO
*"This is what every 2024/2025 ML/AI role expects you to know."*

### Why this chapter exists
Fine-tuning a model to follow instructions and behave helpfully requires more than just SFT. Modern LLMs use alignment pipelines. You must understand this pipeline conceptually and practically.

### Lessons

**Lesson 10.1 — The Alignment Problem**
- Why a well-pre-trained model is not automatically helpful or safe
- What alignment means: making models follow human intent
- The RLHF pipeline overview: SFT → Reward Model → PPO
- Why alignment became a major focus (brief history)

**Lesson 10.2 — Supervised Fine-Tuning (SFT) for Instruction Following**
- How SFT is used to teach instruction following behavior
- SFT dataset formats: Alpaca style, ShareGPT style, system+user+assistant
- The `SFTTrainer` from TRL library
- Loss masking on instructions (train only on responses)
- *Practical:* Fine-tune a 1B model with SFT on an instruction dataset using TRL

**Lesson 10.3 — Reward Modeling**
- What a reward model is and what it learns
- Pairwise comparison data: chosen vs rejected responses
- How the Bradley-Terry model converts comparisons to scores
- Training a reward model (classification head on top of LLM)
- Reward hacking — what it is and why it is dangerous
- *Practical:* Understand the reward model structure, inspect a public reward model

**Lesson 10.4 — RLHF with PPO**
- The full RLHF loop: SFT model + Reward model + PPO optimization
- What PPO is at a high level (policy optimization)
- The KL divergence penalty — why it exists (prevents going too far from SFT)
- Why RLHF is unstable and expensive
- Practical limitations that led to DPO being developed
- *Note:* Full PPO implementation is complex; understand conceptually, focus on DPO practically

**Lesson 10.5 — Direct Preference Optimization (DPO)**
- The key insight: skip the reward model, optimize preferences directly
- DPO loss function — the mathematics and the intuition
- What "chosen" and "rejected" data means and how to source it
- The reference model — what it is and why it is needed
- DPO vs RLHF: tradeoffs and when each is preferred
- *Practical:* Run DPO on a small model using TRL's `DPOTrainer`

**Lesson 10.6 — Building a Preference Dataset**
- Sources of preference data: human annotation, AI annotation, self-play
- UltraFeedback, Anthropic HH, OpenAssistant datasets
- Designing good chosen/rejected pairs for your task
- Data quality issues in preference data
- *Practical:* Build 20 chosen/rejected pairs for a specific task manually

**Lesson 10.7 — Other Alignment Methods (Survey)**
- ORPO (Odds Ratio Preference Optimization) — DPO without reference model
- SimPO — simplified DPO
- Constitutional AI — rule-based alignment
- When to use which method
- *Goal:* Understand what each one is; practical depth on DPO is enough

**Lesson 10.8 — Full Alignment Pipeline End to End**
- SFT → DPO: the practical modern pipeline
- How to structure a project that includes both stages
- Evaluating alignment: MT-Bench, AlpacaEval
- Common pitfalls in alignment training
- *Practical:* Run a complete SFT + DPO pipeline on a small model end to end

---

## Chapter 11: Debugging Failed Runs — The Engineer's Skill
*"This chapter is what makes you an engineer, not a tutorial follower."*

### Why this chapter exists
Interviewers ask "did anything go wrong?" because debugging reveals competence. Every real training run has problems. This chapter prepares you to face them systematically.

### Lessons

**Lesson 11.1 — The Debugging Mindset**
- Treat every failed run as an experiment with information
- The debugging loop: observe → hypothesize → isolate → fix → verify
- Why you should never change more than one thing at a time when debugging
- Keeping a debug log (just as important as your training log)

**Lesson 11.2 — Out of Memory (OOM) Errors**
- Why OOM happens: the full memory breakdown during training
- Systematic checklist: reduce batch size → enable gradient checkpointing → reduce sequence length → use quantization → use gradient accumulation
- Gradient checkpointing: the memory-compute tradeoff explained
- How to estimate required memory before running
- *Practical:* Deliberately cause OOM, then fix it using each technique in sequence

**Lesson 11.3 — NaN and Inf in Loss or Gradients**
- Common causes: LR too high, bad data, overflow in mixed precision, unstable activations
- Debug checklist: check your data first, then LR, then precision
- Using `torch.autograd.detect_anomaly()`
- Gradient clipping as prevention
- *Practical:* Introduce a NaN-causing bug, reproduce it, trace it to root cause

**Lesson 11.4 — Training Loss Not Moving**
- LR too small or zero — how to verify
- Optimizer not updating — common code bugs
- Data pipeline issue — model sees same batch repeatedly
- All parameters frozen — checking `requires_grad`
- Label leakage — model has trivial solution
- *Practical:* Reproduce each of the above, document the debugging process

**Lesson 11.5 — Overfitting Debugging**
- Confirming it is overfitting (training loss falls, val loss rises)
- Systematic fixes: more data → add regularization → reduce model capacity → early stopping
- When to call it "good enough" vs when to fix it
- The quality of your eval set matters — is val loss rising or is your eval set bad?

**Lesson 11.6 — Checkpoint Management**
- Saving checkpoints: how often, which ones to keep
- Resuming from checkpoint correctly (optimizer state, scheduler state, RNG state)
- Evaluating checkpoints to pick the best one (not always the last)
- *Practical:* Set up proper checkpointing, resume a run mid-training, verify everything is correct

**Lesson 11.7 — Reproducibility**
- Why ML experiments are hard to reproduce
- Seeds: `torch.manual_seed`, `numpy.random.seed`, `transformers.set_seed`
- Deterministic operations in PyTorch
- Logging your full config to W&B or a file
- *Practical:* Run same experiment twice with and without seeds, measure variance

---

## Chapter 12: Inference, Deployment & Production Concerns
*"Training is only half the job. The model must run efficiently in production."*

### Why this chapter exists
AI Engineer roles especially care about this. How you serve the model after training is a complete skill domain.

### Lessons

**Lesson 12.1 — Inference Basics**
- How inference differs from training (no gradient computation, no optimizer)
- Batched inference vs single-example inference
- Inference memory: why it is lower than training but still matters
- KV cache — what it is and why it makes generation fast

**Lesson 12.2 — Quantization for Inference**
- GPTQ — post-training quantization for faster inference
- AWQ — activation-aware weight quantization
- INT8 inference vs INT4 inference — quality and speed tradeoffs
- How to load quantized models for inference with transformers
- *Practical:* Load a model in GPTQ format, compare inference speed vs FP16

**Lesson 12.3 — Inference Frameworks**
- vLLM — the standard for production LLM serving (PagedAttention)
- llama.cpp — for CPU and low-resource inference
- Ollama — for local model serving
- Text Generation Inference (TGI) by HuggingFace
- When to use which framework
- *Practical:* Serve your fine-tuned model with Ollama or vLLM, hit it with a REST request

**Lesson 12.4 — Adapter Serving Strategies**
- Serving base model + multiple adapters (hot-swapping)
- Merging adapter vs keeping it separate — the deployment decision
- LoRAX — serving multiple LoRA adapters efficiently
- *Practical:* Serve the same base model with two different LoRA adapters, swap between them

**Lesson 12.5 — Evaluating Inference Quality vs Training Quality**
- Why inference output can differ from training-time expectations
- Temperature and sampling in production — how to set them
- Prompt formatting at inference — must match training format exactly
- Regression testing: building a golden set for deployment verification

**Lesson 12.6 — Cost and Latency Estimation**
- Estimating inference cost before deploying
- Tokens per second — how to measure it and what affects it
- Time to first token vs total generation time
- How to communicate these tradeoffs to stakeholders (important for senior roles)

---

## Chapter 13: Interview Preparation — Telling Your Story
*"All the knowledge in the world fails if you can't communicate it."*

### Why this chapter exists
This chapter synthesizes everything into interview-ready communication. Technical knowledge + communication = the offer.

### Lessons

**Lesson 13.1 — The "Walk Me Through Your Fine-Tuning Project" Answer**
- How to structure a story: context → data → model choice → training decisions → results → learnings
- What to lead with and what to save for follow-up
- The specific QLoRA project from your past: now reframe it with depth
- *Exercise:* Record yourself giving a 3-minute answer, review it critically

**Lesson 13.2 — Deep Dive Questions: Model and Method**
- "Why QLoRA specifically?" — the full answer
- "Why did you choose that rank?" — the reasoning answer
- "How did you choose your learning rate?" — the process answer
- "What is the difference between LoRA and QLoRA?" — the precise answer
- *Exercise:* Answer each with and without notes, until fluent

**Lesson 13.3 — Deep Dive Questions: Training and Debugging**
- "Did anything go wrong? How did you fix it?"
- "What did your loss curves look like?"
- "How did you know when to stop training?"
- "What would you do differently now?"
- *Exercise:* For each question, write a 200-word answer, then shorten to 5 bullet points

**Lesson 13.4 — Deep Dive Questions: Evaluation and Results**
- "How did you evaluate your model?"
- "What metrics did you use and why?"
- "What were the model's failure cases?"
- "How did you compare it to a baseline?"
- *Exercise:* Build an evaluation story for a real experiment you ran

**Lesson 13.5 — Breadth Questions (AI Engineering Scope)**
- "What are the tradeoffs between RAG and fine-tuning?"
- "How would you decide whether to fine-tune or prompt-engineer?"
- "What is DPO and when would you use it?"
- "What is the difference between SFT and RLHF?"
- *Exercise:* Answer each question in under 90 seconds

**Lesson 13.6 — Coding Interview Prep for Fine-Tuning**
- Implementing a LoRA layer from scratch
- Writing a training loop manually (without Trainer)
- Computing cross-entropy loss for next-token prediction
- Implementing a basic evaluation pipeline
- *Exercise:* Complete each implementation without looking at references

**Lesson 13.7 — Building Your Portfolio**
- What to put on GitHub: clean code, good README, W&B report link, results
- How to write a technical blog post about a fine-tuning experiment
- Project ideas that will impress interviewers (listed below)

#### Project Ideas for Portfolio (in order of impact)
1. Fine-tune a 1B model on a specific task (code, medical QA, customer support), run full eval pipeline, write detailed W&B report
2. Compare LoRA ranks (ablation study) on same task, publish results
3. Run SFT + DPO pipeline end to end on small model with self-created preference dataset
4. Build a domain-specific chatbot with QLoRA, serve with Ollama, document everything
5. Reproduce a paper result (e.g., QLORA on Alpaca dataset), compare your run to paper numbers

---

## Appendix A: Experiments Tracker Template

For every experiment you run, fill this in:

```
Experiment ID:
Date:
Objective (what are you testing):

Model:
Dataset:
Task:

Config:
  - Method (QLoRA / LoRA / SFT / DPO):
  - Rank:
  - Alpha:
  - Learning Rate:
  - LR Schedule:
  - Batch Size:
  - Gradient Accumulation Steps:
  - Effective Batch Size:
  - Epochs:
  - Max Sequence Length:
  - Target Modules:
  - Precision (bf16/fp16/fp32):

Results:
  - Training Loss (final):
  - Validation Loss (final):
  - Eval Metric:
  - GPU Memory Used:
  - Training Time:

Observations:
  - What did the loss curve look like?
  - Any spikes or instability?
  - Any OOM or errors?

Conclusion:
  - What did I learn from this run?
  - What would I change next time?

W&B run link:
```

---

## Appendix B: Terminology Cheat Sheet

| Term | One-line definition |
|------|---------------------|
| SFT | Supervised Fine-Tuning: training on input-output pairs with teacher forcing |
| PEFT | Parameter Efficient Fine-Tuning: updating a small subset of parameters |
| LoRA | Low-Rank Adaptation: adding trainable low-rank matrices to frozen weights |
| QLoRA | LoRA on a 4-bit quantized base model |
| RLHF | Reinforcement Learning from Human Feedback: PPO with a reward model |
| DPO | Direct Preference Optimization: alignment without a reward model |
| Perplexity | exp(cross_entropy_loss): model's uncertainty on held-out data |
| KV Cache | Cached key-value attention outputs to speed up autoregressive generation |
| Gradient Accumulation | Simulating large batch sizes by accumulating gradients over multiple steps |
| Gradient Checkpointing | Trading compute for memory by recomputing activations during backward pass |
| BF16 | Brain Float 16: 16-bit format with wider dynamic range than FP16 |
| NF4 | Normal Float 4: 4-bit format optimized for normally distributed weights |
| Catastrophic Forgetting | Model losing pre-trained knowledge when fine-tuned too aggressively |
| Loss Masking | Setting loss to 0 on prompt tokens so model only learns to predict responses |
| Teacher Forcing | Training by feeding ground truth tokens as input (not model's own predictions) |
| Intrinsic Rank | The actual dimensionality of the fine-tuning update (motivation for LoRA) |

---

## Appendix C: Hardware-Based Model Selection Guide

| Model | Parameters | Colab T4 (16GB) | Kaggle T4 (16GB) | Local 16GB Intel GPU |
|-------|-----------|-----------------|-------------------|----------------------|
| Phi-3 Mini | 3.8B | ✅ QLoRA + Full | ✅ QLoRA + Full | ✅ QLoRA |
| Gemma 2B | 2B | ✅ Full FP16 | ✅ Full FP16 | ✅ Full FP16 |
| Qwen2 1.5B | 1.5B | ✅ Full FP16 | ✅ Full FP16 | ✅ Full FP16 |
| Llama 3 8B | 8B | ✅ QLoRA only | ✅ QLoRA only | ⚠️ Very tight |
| Mistral 7B | 7B | ✅ QLoRA only | ✅ QLoRA only | ⚠️ Very tight |
| Llama 3 70B | 70B | ❌ | ❌ | ❌ |

*Start with Phi-3 Mini or Gemma 2B locally. Use Colab/Kaggle for 7B experiments.*

---

## Appendix D: Recommended Resources

**Books**
- "Natural Language Processing with Transformers" (Tunstall et al.) — HuggingFace team, covers fine-tuning deeply
- "Designing Machine Learning Systems" (Chip Huyen) — for the production and evaluation mindset

**Papers to read (in order)**
1. "Attention Is All You Need" (Vaswani et al., 2017) — transformer architecture
2. "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021) — the LoRA paper
3. "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023) — your core paper
4. "Training language models to follow instructions with human feedback" (Ouyang et al., 2022) — InstructGPT / RLHF
5. "Direct Preference Optimization" (Rafailov et al., 2023) — DPO paper

**Courses**
- HuggingFace NLP Course (free) — covers transformers and fine-tuning
- fast.ai Practical Deep Learning — for intuition about training dynamics
- Andrej Karpathy's "Let's build GPT from scratch" — for deep mechanistic understanding

**Tools you must be proficient with**
- Weights & Biases (wandb.ai) — experiment tracking
- HuggingFace Hub — models, datasets
- PEFT library — LoRA, QLoRA implementation
- TRL library — SFT, DPO, RLHF training
- LM Evaluation Harness — benchmarking

---

*Last updated: June 2025*  
*Next step: Start with Chapter 0, Lesson 0.1. Do not skip. Do not rush.*