# ML Training Mastery: From Pretraining to Alignment — A Deep-Dive + Interview Prep Curriculum

**Goal:** Go from "I can describe it at a high level" to "I can derive it, code it, debug it, and defend every design choice under interview pressure."

Every chapter ends with two lessons that are *not* optional:
- **Diagnosis & Mental Models** — how to think when something is broken or ambiguous
- **Interview Lab** — likely questions, strong vs. weak answer patterns, and follow-up traps interviewers use to test depth

---

## Chapter 0 — Foundations You're Assumed to Know Cold
*(Quick-fire chapter — if any lesson here feels shaky, that's a gap worth closing before Chapter 1)*

1. Tokenization: BPE, WordPiece, Unigram — how they're trained, vocab size tradeoffs
2. Embeddings & positional encoding: absolute, sinusoidal, learned, RoPE, ALiBi — why RoPE won
3. Attention mechanics from first principles: QKV derivation, scaling factor (why √d_k), multi-head vs. multi-query vs. grouped-query attention
4. The Transformer block, end to end: pre-LN vs. post-LN, residual streams, why architecture choices affect trainability
5. Loss functions and optimization basics: cross-entropy derivation, why Adam/AdamW over SGD for transformers
6. Interview Lab: "Walk me through attention on a whiteboard" — the five ways candidates trip on this

---

## Chapter 1 — Pretraining: Data
1. Data sourcing and mixture design (web, code, books, papers) — why mixture ratios matter
2. Deduplication: exact vs. fuzzy (MinHash/LSH), why dedup improves downstream performance
3. Quality filtering: heuristic filters, classifier-based filtering, perplexity filtering
4. Tokenizer training in practice — code: train a BPE tokenizer on a custom corpus
5. Packing sequences, document boundaries, and the `<eos>` handling problem
6. Data contamination: what it is, how it's detected, why interviewers ask about it
7. Diagnosis & Mental Models: "My eval numbers look too good" → contamination checklist
8. Interview Lab: explaining data pipeline decisions without sounding like you just ran a script someone else wrote

## Chapter 2 — Pretraining: Objectives & Architectures
1. Causal LM (decoder-only) objective — code: implement next-token loss with teacher forcing
2. Masked LM + NSP (BERT-style) and why NSP was later dropped (RoBERTa's findings)
3. Prefix-LM and encoder-decoder objectives (T5-style span corruption) — when each architecture is chosen
4. Mixture-of-Experts pretraining: routing, load balancing loss, why MoE changes the hyperparameter story
5. Long-context pretraining: context extension methods, position interpolation, YaRN
6. Diagnosis & Mental Models: choosing an architecture family for a given constraint (latency, data size, downstream task)
7. Interview Lab: "Explain decoder training" redo — building an answer that's structured, not stream-of-consciousness

## Chapter 3 — Pretraining: The Training Loop, Systems & Scale
1. Forward/backward pass at code level — build a minimal decoder block in PyTorch from scratch
2. Mixed precision training: FP16 vs BF16, loss scaling, why BF16 dominates for LLMs
3. Distributed training strategies: Data Parallel, Tensor Parallel, Pipeline Parallel, FSDP/ZeRO stages 1-3
4. Gradient accumulation, gradient checkpointing — memory/compute tradeoffs, when to use each
5. Learning rate schedules: warmup, cosine decay, WSD (warmup-stable-decay) — why warmup is non-negotiable
6. Batch size, tokens-per-step, and the relationship to learning rate (linear scaling rule)
7. Scaling laws: Chinchilla, compute-optimal training, how to size a model for a given compute budget
8. Diagnosis & Mental Models: reading loss curves — spikes, plateaus, divergence, and what each usually means
9. Diagnosis & Mental Models: training instability playbook (loss spikes, NaN gradients, gradient clipping thresholds)
10. Interview Lab: "How do you train the model" — building a complete forward+backward+systems answer (this is where you got caught short)

## Chapter 4 — Pretraining Hyperparameters, In Depth
1. The full hyperparameter list for pretraining: LR, warmup steps, weight decay, β1/β2/ε for AdamW, batch size, sequence length, gradient clipping norm
2. Typical ranges by model scale (small ~100M–1B, mid ~1B–10B, large 10B+) with citations to published training recipes
3. Hyperparameter search methods for small models: grid search, random search, Bayesian optimization, ASHA/Hyperband
4. Hyperparameter transfer for large models: μP (maximal update parametrization), scaling hyperparameters from small proxy models
5. Practical tuning workflow: what to sweep first, what to fix, cost-aware tuning order
6. Diagnosis & Mental Models: "LR too high vs too low" symptom table; weight decay under/over-regularization signs
7. Interview Lab: "How did you tune hyperparameters for your fine-tune?" — building a credible, specific answer (not "I used default values")

## Chapter 5 — Fine-Tuning: Full and Parameter-Efficient Methods
1. Full fine-tuning: when it's justified, catastrophic forgetting risk, cost math
2. Instruction tuning: dataset formats (Alpaca-style, ChatML, conversation templates), loss masking on prompts
3. LoRA: math derivation, rank selection, target modules, code — implement LoRA on a small model from scratch
4. QLoRA, quantization-aware fine-tuning: 4-bit/8-bit basics, when precision loss matters
5. Other PEFT methods: prefix tuning, adapters, (IA)³ — comparison table of when each wins
6. Data preparation for instruction tuning — code: build an instruction dataset with proper masking
7. Diagnosis & Mental Models: is my model underfit, overfit, or the data itself is the problem? Triage flowchart
8. Diagnosis & Mental Models: full fine-tune vs. PEFT vs. RAG vs. better prompting — the decision tree you were missing
9. Interview Lab: "Give me your mental model for fine-tuning" — rebuilding your answer with a diagnostic layer before the fix layer

## Chapter 6 — Fine-Tuning Hyperparameters, In Depth
1. LoRA-specific hyperparameters: rank, alpha, dropout, target modules — ideal ranges and how they interact
2. Learning rate for fine-tuning vs. pretraining — why it's typically 10-100x smaller, and exceptions
3. Epochs, effective batch size, and overfitting risk on small fine-tuning sets
4. Hyperparameter tuning for small fine-tunes (few GPUs) vs. large-scale fine-tunes (multi-node)
5. Early stopping, validation strategy, and eval-set design for fine-tuning
6. Diagnosis & Mental Models: reading fine-tuning loss curves vs. pretraining loss curves — different failure signatures
7. Interview Lab: mock Q&A — defending specific hyperparameter choices with reasoning, not memorized numbers

## Chapter 7 — Alignment Tuning
1. RLHF pipeline end to end: reward model training, PPO mechanics, why it's expensive and unstable
2. DPO: derivation from the RLHF objective, why it skips the reward model, practical implementation
3. GRPO and other PPO-alternatives (RLOO, ReMax) — what problem each variant solves
4. Reward modeling: data collection (pairwise preferences), reward hacking, and how to detect it
5. Constitutional AI / RLAIF: using model feedback instead of human feedback
6. Alignment hyperparameters: KL penalty coefficient, reward scaling, clip ranges — what breaks when these are wrong
7. Diagnosis & Mental Models: signs of reward hacking, mode collapse, alignment tax on capability
8. Interview Lab: "Which alignment method would you pick and why" — comparative reasoning under constraints (compute, data, latency to ship)

## Chapter 8 — Evaluation (the chapter most candidates skip, and interviewers exploit that)
1. Pretraining evals: perplexity, held-out loss, why they're insufficient alone
2. Benchmark evals: MMLU, HellaSwag, GSM8K, etc. — known contamination and gaming issues
3. Fine-tuning/alignment evals: win-rate judging, LLM-as-judge pitfalls, human eval design
4. Building a custom eval set for a domain-specific fine-tune — code: a small eval harness
5. Diagnosis & Mental Models: "my eval says it's better but users say it's worse" — common causes
6. Interview Lab: "How would you know your fine-tune actually worked?" — this question catches high-level answers every time

## Chapter 9 — Applied System Design (RAG, Hybrid Retrieval, Agentic Pipelines)
1. Structured + unstructured hybrid retrieval architecture, redone in depth (your last interview question)
2. Text-to-SQL: intent classification before generation, injection risks, guardrails
3. Vector retrieval in depth: chunking strategies, embedding model choice, hybrid (BM25 + dense) search, reranking
4. GraphRAG and memory architectures (Mem0-style): when graph structure beats flat vector retrieval
5. Clarifying-question discipline: a checklist to run before answering any system design prompt
6. Production concerns checklist: latency budgets, caching, fallback paths, observability
7. Diagnosis & Mental Models: retrieval quality debugging — is it a chunking problem, embedding problem, or ranking problem?
8. Interview Lab: full mock system-design interview, structured Q&A with follow-up pressure-testing

## Chapter 10 — Communication: Answering Like a Senior Engineer
1. The structure interviewers are actually grading: framing → tradeoffs → decision → depth on demand
2. How to signal depth without over-explaining — calibrating answer length to the question
3. Turning a one-line answer into a layered answer: the "headline, then expand on request" technique
4. Handling ambiguous questions: when and how to ask clarifying questions without stalling
5. Recovering gracefully when you realize mid-answer you've made an error (you won't always catch it before speaking)
6. Full mock interview: a complete session covering pretraining, fine-tuning, alignment, and system design back to back, with a scored debrief

---

## How to Use This
- Work chapter by chapter — each has code where relevant, so plan to actually run things, not just read.
- Every "Diagnosis & Mental Models" lesson is the part that was thin in your actual interview transcript — don't skip these even if the technical lesson feels familiar.
- Every "Interview Lab" lesson will include a rewritten, stronger version of the type of answer you gave, plus the follow-up questions interviewers use to probe past a rehearsed answer.

Let me know which chapter to start with, or if you want me to reorder based on your upcoming interview timeline.