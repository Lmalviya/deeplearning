# ML Training Mastery — Revised Index (v3), Chapters 5 Onward

> **What changed this time, and why:** Three structural problems surfaced on cross-check, not just the reordering you asked for.
> 1. **Evaluation was stranded after the fix-chapters (old Chapter 9)** — but every diagnosis lesson needs "how do I test for this" already in hand. Evaluation now directly follows Diagnosis.
> 2. **The decision tree collapsed "continued pretraining / domain-adaptive pretraining (DAPT)" and "tokenizer extension" into "fine-tune"** — these are distinct, larger interventions for severe distribution/vocab mismatch, not the same fix. Now split out explicitly, with DAPT given its own lesson in the fine-tuning chapter.
> 3. **Two capability dimensions were missing:** long-context/retrieval faithfulness (distinct from reasoning), and safety/refusal calibration (over-refusal vs. under-refusal) — both added to Chapter 5.

Chapters 1-4 unchanged.

---

## Chapter 5 — Diagnosing the Model Before Choosing an Intervention

1. The diagnostic mental model: symptom → root cause → intervention
2. Data distribution & vocabulary mismatch: domain shift, tokenizer fertility, out-of-vocabulary degradation
3. Instruction-following gaps vs. knowledge gaps vs. formatting gaps — isolating which one you're looking at
4. Tool-use / function-calling capability: where it comes from, "never learned it" vs. "learned it but unreliable"
5. Reasoning capability: chain-of-thought, inference-time compute, shallow pattern-matching vs. genuine multi-step reasoning
6. Structured output capability: JSON mode, grammar-constrained decoding, schema adherence vs. content correctness
7. Code generation capability: pretraining data composition effects, syntax vs. logic vs. tool-integration failures
8. Multilingual capability: tokenizer fertility per language, pretraining data mixture, cross-lingual transfer
9. **Long-context & retrieval-faithfulness capability (NEW):** does the model actually use information buried mid-context, distinct from reasoning depth — needle-in-haystack-style diagnosis, ties back to Chapter 2 Lesson 5's long-context extension methods
10. **Safety & refusal calibration (NEW):** over-refusal (safe requests declined) vs. under-refusal (unsafe requests answered) as a distinct, measurable capability axis, not a side effect of other tuning
11. The full decision tree: symptom → root cause → correct intervention — now explicitly distinguishing **continued pretraining/DAPT**, **tokenizer extension**, fine-tuning, instruction-tuning, alignment, RAG, better prompting, a different base model, or no intervention needed
12. Interview Lab: full rebuild of "give me your mental model for fine-tuning," using the complete diagnostic framework

## Chapter 6 — Evaluation (moved up — this is the measurement toolkit Chapter 5 assumes you already have)

1. Pretraining evals: perplexity, held-out loss, why they're insufficient alone
2. Benchmark evals: MMLU, HellaSwag, GSM8K — contamination and gaming issues
3. Capability-specific eval design: testing tool-use reliability, reasoning depth, structured-output adherence, code correctness, multilingual quality, long-context faithfulness, and refusal calibration — the direct measurement layer for every Chapter 5 capability
4. Fine-tuning/alignment evals: win-rate judging, LLM-as-judge pitfalls, human eval design
5. Building a custom eval set for a domain-specific fine-tune — code: a small eval harness
6. Diagnosis & Mental Models: "my eval says it's better but users say it's worse"
7. Interview Lab: "How would you know your fine-tune actually worked?"

## Chapter 7 — Fine-Tuning Methods: Full, Domain-Adaptive, and Parameter-Efficient

1. **Continued pretraining / domain-adaptive pretraining (DAPT) (NEW, standalone):** when vocabulary/distribution mismatch is severe enough that fine-tuning alone can't fix it; tokenizer extension mechanics; how this differs structurally from every method that follows
2. Full fine-tuning: when it's justified given Chapter 5's diagnosis, catastrophic forgetting risk, cost math
3. Instruction tuning: dataset formats, loss masking on prompts
4. LoRA: math derivation, rank selection, target modules, code
5. QLoRA and quantization-aware fine-tuning
6. Other PEFT methods: prefix tuning, adapters, (IA)³
7. Data preparation for instruction tuning — code
8. Diagnosis & Mental Models: underfit vs. overfit vs. data-problem triage (fine-tuning-specific)
9. Choosing among methods once fine-tuning is confirmed: full vs. DAPT vs. LoRA vs. QLoRA vs. other PEFT, matched to the diagnosed gap
10. Interview Lab: choosing and defending a specific fine-tuning method live

## Chapter 8 — Fine-Tuning Hyperparameters, In Depth

1. LoRA-specific hyperparameters: rank, alpha, dropout, target modules
2. Learning rate for fine-tuning vs. pretraining — why 10-100x smaller, and exceptions
3. Epochs, effective batch size, overfitting risk on small fine-tuning sets
4. Hyperparameter tuning for small vs. large-scale fine-tunes
5. Early stopping, validation strategy, eval-set design for fine-tuning
6. Diagnosis & Mental Models: fine-tuning loss curves vs. pretraining loss curves
7. Interview Lab: defending specific hyperparameter choices with reasoning

## Chapter 9 — Alignment Tuning

1. RLHF pipeline: reward model training, PPO mechanics
2. DPO: derivation, practical implementation
3. GRPO and other PPO-alternatives (RLOO, ReMax)
4. Reward modeling: preference data, reward hacking
5. Constitutional AI / RLAIF
6. Alignment hyperparameters: KL penalty, reward scaling, clip ranges
7. Diagnosis & Mental Models: reward hacking, mode collapse, alignment tax
8. Interview Lab: choosing an alignment method under constraints

## Chapter 10 — Applied System Design (RAG, Hybrid Retrieval, Agentic Pipelines)

1. Structured + unstructured hybrid retrieval architecture (your last interview's actual question)
2. Text-to-SQL: intent classification, injection risks, guardrails
3. Vector retrieval: chunking, embedding choice, hybrid search, reranking
4. GraphRAG and memory architectures (Mem0-style)
5. Clarifying-question discipline before answering system design prompts
6. Production concerns: latency, caching, fallback paths, observability
7. Diagnosis & Mental Models: chunking vs. embedding vs. ranking problem
8. Interview Lab: full mock system-design interview with follow-up pressure-testing

## Chapter 11 — Communication: Answering Like a Senior Engineer

1. The structure interviewers actually grade: framing → tradeoffs → decision → depth on demand
2. Calibrating answer length to the question
3. The "headline, then expand on request" technique
4. Clarifying questions without stalling
5. Recovering gracefully from a mid-answer error
6. Full mock interview spanning diagnosis, fine-tuning, alignment, and system design, with a scored debrief

---

## Still Worth Flagging, Not Yet Resolved

Two open questions I don't think should be silently decided for you:

- **Should Chapter 9 (Alignment) come before or after Chapter 7/8 (Fine-tuning)?** Current order assumes SFT-then-alignment, the conventional pipeline — but if your interviews are testing whether you know *when* alignment might be skipped or reordered (e.g., some recent recipes blend SFT and preference optimization more tightly), that's worth a callout lesson rather than a silent assumption baked into the chapter order.
- **Chapter 5 is now 12 lessons** — long for one chapter. I kept it unsplit because the capabilities genuinely belong together conceptually, but if it starts feeling bloated once we're generating content, splitting it into "Diagnosis Foundations" (1-3, 11-12) and "Capability Audit" (4-10) is a clean split point.

Let me know if this structure holds, or if either open question above needs resolving before we start on Chapter 5.