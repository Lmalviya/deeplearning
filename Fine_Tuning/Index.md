# LLM Fine-Tuning: From Data to Production

> **Two-axis mental model that runs through this entire course:**
> - **HOW axis** — the optimization strategy (PEFT vs Full Fine-Tuning). This is about *compute, memory, and parameter efficiency*.
> - **WHAT axis** — the training objective and goal (instruction following, alignment, reasoning, tool use, domain adaptation). This is about *what capability you are teaching the model*.
>
> These two axes are independent. You can use LoRA (HOW) to train an instruction-following model (WHAT), or you can use full fine-tuning (HOW) to do the same. Understanding this separation is the key to reasoning through any fine-tuning decision in an interview.

---

## Part 1 — Foundations & Mental Models

**Lesson 1.1** — What fine-tuning is and why it exists: pre-training vs fine-tuning vs prompting vs RAG — when each wins \
**Lesson 1.2** — The two axes decoded: HOW (PEFT vs Full FT) and WHAT (objectives/goals) — and how they combine \
**Lesson 1.3** — The full fine-tuning spectrum: continued pre-training → SFT → RLHF/alignment — what happens at each stage and why the order matters \
**Lesson 1.4** — The fine-tuning decision framework: a practical guide to when to fine-tune, which HOW to pick, and which WHAT to target

---

## Part 2 — Data Preparation (The Foundation Everything Else Depends On)

**Lesson 2.1** — Dataset types and their anatomy: instruction datasets, preference datasets, domain corpora — what each looks like and why format matters \
**Lesson 2.2** — Instruction dataset construction: prompt/response formatting, chat templates (Alpaca, ChatML, Llama-3, ShareGPT formats), system prompt design \
**Lesson 2.3** — Preference dataset construction: chosen/rejected pairs, how to collect human annotations, synthetic preference generation with LLM-as-judge \
**Lesson 2.4** — Data quality: filtering strategies, deduplication (exact and near-duplicate), quality scoring, removing harmful content \
**Lesson 2.5** — Data mixing and ratios: how to blend multiple datasets, domain upsampling, preventing catastrophic forgetting through data mixing \
**Lesson 2.6** — Tokenization internals: how text becomes token IDs, attention masks, labels masking (why you mask the prompt tokens during SFT loss)

---

## Part 3 — The HOW Axis: PEFT Methods

**Lesson 3.1** — Why PEFT exists: the full fine-tuning memory problem, catastrophic forgetting, and the case for parameter efficiency \
**Lesson 3.2** — Adapter Tuning: architecture (where adapters are inserted), forward pass mechanics, latency overhead, when to use it \
**Lesson 3.3** — Prompt Tuning and Prefix Tuning: soft prompts vs hard prompts, prefix tokens in attention, why they struggle on smaller models \
**Lesson 3.4** — LoRA (Low-Rank Adaptation): the low-rank decomposition math, rank `r` and `alpha` explained, which modules to target, merge vs separate weights \
**Lesson 3.5** — QLoRA: 4-bit quantization (NF4), double quantization, paged optimizers — how it fits 65B parameter training on a single GPU \
**Lesson 3.6** — Beyond LoRA: DoRA (weight decomposition), IA³ (scaling activations), LoftQ (quantization-aware initialization) — what they improve and trade-offs \
**Lesson 3.7** — PEFT method comparison matrix: memory footprint, training speed, inference overhead, task performance — choosing the right one

---

## Part 4 — The HOW Axis: Full Fine-Tuning

**Lesson 4.1** — Supervised Fine-Tuning (SFT) internals: the training loop, cross-entropy loss on completions only, learning rate schedules, warmup \
**Lesson 4.2** — Continued Pre-Training (CPT): when to do it, domain corpus construction, next-token prediction objective, how it differs from SFT \
**Lesson 4.3** — Memory anatomy of full fine-tuning: model weights + gradients + optimizer states (Adam moments) — why 7B model needs 112GB+ \
**Lesson 4.4** — Gradient checkpointing, mixed precision (FP16 vs BF16 vs FP32), and gradient accumulation — the tools to make full FT feasible \
**Lesson 4.5** — When full fine-tuning beats PEFT: the evidence, the scenarios, and the honest trade-off analysis

---

## Part 5 — The WHAT Axis: Training Objectives & Capabilities

**Lesson 5.1** — Domain adaptation: teaching the model a new vocabulary and knowledge domain, CPT + SFT combined strategy (medical, legal, code, finance) \
**Lesson 5.2** — Instruction following: what it means to teach a model to follow instructions, the role of diversity in instruction datasets \
**Lesson 5.3** — Reasoning capability training: Chain-of-Thought distillation, process reward models (PRMs), outcome reward models (ORMs), and the DeepSeek-R1 / o1 approach (GRPO) \
**Lesson 5.4** — Tool use and function calling: how models learn to select and invoke tools, training data format for tool calls, multi-turn tool trajectories \
**Lesson 5.5** — Coding capability training: what makes code datasets special, execution feedback, fill-in-the-middle (FIM) objective \
**Lesson 5.6** — SLM training strategies: how Small Language Models (Phi, Gemma, Qwen) are trained differently — knowledge distillation, data curation over scale, capability-focused training \
**Lesson 5.7** — LLM vs SLM: architecture differences, training philosophy differences, where each wins in production — the honest comparison

---

## Part 6 — The WHAT Axis: Alignment Methods

> **This part is fully self-contained.** It builds from first principles — no prior knowledge of RL or statistics is assumed. Read the lessons in order.

**Lesson 6.1** — Why alignment is needed: the gap between "predicts next tokens" and "follows human intent" — what SFT cannot solve and why a reward signal is necessary \
**Lesson 6.2** — RL foundations for alignment: policy, state, action, reward, episode, trajectory — the exact RL vocabulary used in every alignment algorithm, scoped to LLMs \
**Lesson 6.3** — KL divergence in alignment: what it measures, forward vs reverse KL, why the KL penalty is non-optional, the β hyperparameter — the math every alignment paper assumes you know \
**Lesson 6.4** — Reward models: how human preferences become a trainable signal — preference pairs, the Bradley-Terry model, reward model training pipeline, reward score distribution, and model limitations \
**Lesson 6.5** — Reward hacking: Goodhart's Law in ML, the over-optimization curve, how models exploit reward models, and the mitigation strategies (KL penalty, ensemble reward models, constitutional AI) \
**Lesson 6.6** — RLHF with PPO: the full 4-phase training loop (rollout → score → advantage → clipped update), actor-critic setup, KL constraint, why PPO is powerful but complex and unstable \
**Lesson 6.7** — DPO (Direct Preference Optimization): the analytical derivation from the PPO objective, implicit reward as log ratio, how it bypasses the reward model, training setup, stability vs PPO \
**Lesson 6.8** — GRPO (Group Relative Policy Optimization): group sampling, normalized group advantage, eliminating the critic network, verifiable reward signals, the DeepSeek-R1 connection \
**Lesson 6.9** — ORPO (Odds Ratio Preference Optimization): combining SFT loss + odds ratio preference loss in a single training pass, no reference model, when to prefer over DPO \
**Lesson 6.10** — Other alignment methods: SimPO (reference-free DPO with length normalization), KTO (binary feedback without preference pairs), IPO (identity preference optimization, avoiding over-fitting to margins) — when each is the right choice \
**Lesson 6.11** — Alignment method comparison: data requirements, compute cost, number of models needed, training stability, performance — the full decision matrix for choosing between PPO / DPO / GRPO / ORPO / SimPO / KTO

---

## Part 7 — Multimodal Model Training

**Lesson 7.1** — How multimodal models work: modality-specific encoders, projection layers, and how different input types enter the LLM token space \
**Lesson 7.2** — Vision-Language Models (VLMs): CLIP encoder + LLM, LLaVA architecture, training stages (alignment pre-training → instruction tuning) \
**Lesson 7.3** — How the model decides which modality to output: decoder heads, special tokens, output routing for text vs image vs audio generation \
**Lesson 7.4** — Training a multimodal model: dataset types (image-caption, VQA, interleaved), training stages, what gets frozen and what gets trained at each stage \
**Lesson 7.5** — Audio and video in LLMs: how speech and video frames are tokenized and fed to the model (high-level orientation)

---

## Part 8 — Model Evaluation

**Lesson 8.1** — Evaluation during training: training loss vs validation loss, perplexity, learning curves — what they tell you and what they hide \
**Lesson 8.2** — Benchmark evaluation: MMLU, HellaSwag, TruthfulQA, HumanEval, MT-Bench, IFEval — what each measures and when each is the right signal \
**Lesson 8.3** — LLM-as-judge evaluation: win rate, pairwise comparison, scoring rubrics, bias issues (position bias, verbosity bias) \
**Lesson 8.4** — Alignment-specific evaluation: preference win rates, reward score tracking, KL divergence from base model \
**Lesson 8.5** — Detecting and diagnosing fine-tuning failures: overfitting, catastrophic forgetting, reward hacking, mode collapse — symptoms and fixes

---

## Part 9 — Case Studies (Interview: "Walk me through your training pipeline")

**Case Study 1** — Training an instruction-following chat model end-to-end: dataset curation → SFT with QLoRA → DPO alignment → evaluation → model merging and publishing \
**Case Study 2** — Domain adaptation for a medical Q&A model: domain corpus CPT → instruction fine-tuning → safety alignment → evaluation on medical benchmarks \
**Case Study 3** — Training a function-calling / tool-use model: tool call dataset construction → SFT → evaluation on tool-use benchmarks (BFCL) \
**Case Study 4** — Training a reasoning model (DeepSeek-R1 style): cold-start SFT on CoT data → GRPO with outcome rewards → rejection sampling → distillation to smaller model

---

## Part 10 — Deployment on AWS (Interview: "Walk me through your deployment pipeline")

**Lesson 10.1** — Model serialization formats: SafeTensors, GGUF, GPTQ, AWQ, TensorRT — what each is for and which to pick \
**Lesson 10.2** — AWS services landscape for LLM deployment: SageMaker, EC2, ECS/Fargate, Lambda, Bedrock — what each is, when each is the right choice \
**Lesson 10.3** — SageMaker deep dive: Model Registry, real-time endpoints, async inference endpoints, batch transform — choosing the right inference mode \
**Lesson 10.4** — Deploying with vLLM on EC2/ECS: container setup, SageMaker custom inference container, environment config, health checks \
**Lesson 10.5** — Auto-scaling and load balancing: Application Load Balancer, SageMaker auto-scaling policies, target tracking, handling cold starts \
**Lesson 10.6** — Cost optimization on AWS: spot instances for batch inference, reserved capacity for real-time, request batching, caching repeated queries \
**Lesson 10.7** — CI/CD for model deployment: model versioning, A/B testing with SageMaker shadow deployments, canary rollouts, rollback strategy

---

## Part 11 — Model Inference Optimization

**Lesson 11.1** — The inference bottleneck: why LLM inference is memory-bandwidth bound, not compute bound — the fundamental insight that drives all optimizations \
**Lesson 11.2** — KV cache: what it is, how it works, why it is essential, memory cost of KV cache, paged attention (the vLLM insight) \
**Lesson 11.3** — Continuous batching and in-flight batching: how vLLM serves multiple requests without waiting for the batch to finish \
**Lesson 11.4** — Quantization for inference: INT8 (LLM.int8()), GPTQ, AWQ, INT4 — quality vs speed vs memory trade-offs \
**Lesson 11.5** — Speculative decoding: draft model + verifier model, how it reduces latency without quality loss \
**Lesson 11.6** — Flash Attention: why standard attention is slow, the memory-efficient kernel, where it applies \
**Lesson 11.7** — Inference serving frameworks compared: vLLM vs TGI (Text Generation Inference) vs TensorRT-LLM vs Ollama — when to use each

---

## Part 12 — Distributed Training (Advanced)

**Lesson 12.1** — Why distributed training: the math of model size vs single GPU memory — when you are forced to go distributed \
**Lesson 12.2** — Data Parallelism: DDP (DistributedDataParallel), gradient synchronization, communication overhead, when it is sufficient \
**Lesson 12.3** — ZeRO Optimization (DeepSpeed): ZeRO Stage 1, 2, 3 — what gets sharded at each stage, FSDP (PyTorch's ZeRO-3 equivalent) \
**Lesson 12.4** — Model Parallelism: tensor parallelism (splitting weight matrices), pipeline parallelism (splitting layers across GPUs), trade-offs of each \
**Lesson 12.5** — Mixed precision training in depth: FP32 vs BF16 vs FP16, master weights, loss scaling, numerical stability issues \
**Lesson 12.6** — Practical distributed training setup: DeepSpeed config, FSDP config, gradient checkpointing + ZeRO + mixed precision — the full stack
