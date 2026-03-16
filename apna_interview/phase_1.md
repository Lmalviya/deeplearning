# Day 1 — Phase 1: LLM Training Stack Overview

> **Revision time:** 25-30 minutes  
> **Priority:** Highest — every other phase references this

---

## The One Unifying Concept

Before anything else, internalize this.  
Every stage of LLM training does **the same fundamental thing**:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)$$

**In plain English:**  
Given everything seen so far, predict the next token.  
Minimize the difference between predicted and actual next token.

**What changes across stages is only $x$ and $y$:**

| Stage | Input $x$ | Target $y$ |
|---|---|---|
| Pretraining | `"The Eiffel Tower is located in"` | `"Paris"` |
| SFT | `"<instruction> Translate to Hindi </instruction> Hello"` | `"नमस्ते"` |
| Alignment (DPO) | Same instruction | Preferred response ranked above rejected response |

> **Interview anchor:** *"All three stages use the same cross-entropy loss mechanism.  
> The pipeline exists because what we train on — not how we train — changes the model's behavior."*

---

## The Three Stages — Full Picture

### Stage 1 — Pretraining

**WHAT:**  
Train a model on trillions of tokens of raw text.  
Wikipedia, books, code, web crawl, news — everything.  
No labels. No instructions. Just raw text.

**WHY:**  
Teaches the model language, facts, reasoning patterns,  
and world knowledge. Without this, the model knows nothing.

**WHEN:**  
Done once by large organizations (Meta, Google, Mistral).  
Costs millions of dollars and thousands of GPUs.  
**You almost never do this from scratch.**

**Output:** Base model — knows everything, talks to nobody.

```
# What a base model actually does:
Input:  "What is the capital of France?"
Output: "What is the capital of Germany?
         What is the capital of Spain?
         What is the capital of Italy?..."

# It completes text — it does not answer questions.
# It has no concept of being an assistant.
```

**What breaks if skipped:**  
Model knows nothing about language or the world.  
All subsequent stages fail completely.

**Cost:** $1M — $100M+  
**Who does it:** Meta, Google, Mistral, AI labs  
**Your role:** Rarely — only continued pretraining (see below)

---

### Stage 1.5 — Continued Pretraining ⚠️ Often Missed

**WHAT:**  
Take an existing base model and keep pretraining it  
on a domain-specific or language-specific corpus.  
Same objective as pretraining — just more of it,  
on targeted data.

**WHY:**  
Base models like Llama 3 have seen some Hindi  
during pretraining — but Hindi is severely  
underrepresented compared to English.

```
Llama 3 pretraining data (approximate):
  English  → ~89% of tokens
  Hindi    → ~0.1% of tokens
  Hinglish → effectively 0%
```

Continued pretraining on Indic corpus improves  
Hindi/Hinglish fluency **before** SFT begins.  
SFT on top of a model that barely knows Hindi  
produces poor Hindi instruction following.

**WHEN:**  
- Target language is underrepresented in base model
- Domain-specific vocabulary needed (medical, legal)
- Code-switching (Hinglish) not seen during pretraining

**What breaks if skipped:**  
SFT still works — but Hindi quality is noticeably lower.  
Model struggles with Devanagari script and Hinglish patterns.  
This is the step most engineers skip and later regret.

**Cost:** $10K — $100K (much cheaper than full pretraining)  
**Who does it:** YOU — this is part of your role  
**Datasets:** IndicCorp, Sangraha, CulturaX-Hindi, OSCAR-Hindi

> **Interview answer:**  
> *"Before SFT, I run continued pretraining on Indic language corpora  
> because base models are English-dominant. Llama 3 has seen less than  
> 0.1% Hindi tokens. SFT on top of that gives suboptimal Hindi quality.  
> Continued pretraining is a one-time cost that significantly improves  
> all downstream stages for Indic languages."*

---

### Stage 2 — Supervised Fine-Tuning (SFT)

**WHAT:**  
Train on (instruction, response) pairs.  
Show the model thousands of examples of  
good conversations — it learns to replicate the pattern.

**WHY:**  
Base model can complete text but cannot follow instructions.  
SFT teaches the model **how to be an assistant** —  
what format to respond in, how to stay on topic,  
how to handle different types of requests.

**WHEN:**  
Always. SFT is mandatory before alignment.  
Alignment on a base model produces garbage.

```
# SFT data format example (ChatML template):
<|im_start|>system
You are a helpful Hindi customer support assistant.
<|im_end|>
<|im_start|>user
Mera order kab aayega?
<|im_end|>
<|im_start|>assistant
Aapka order 3-5 business days mein aa jayega.
<|im_end|>
```

**Output:** Instruct model — can follow instructions,  
responds in correct format, but tone and safety  
may still be inconsistent.

**What breaks if skipped:**  
Alignment has no well-behaved model to work with.  
You cannot preference-optimize a model that  
cannot even follow instructions yet.

**Cost:** $1K — $50K depending on model size and data  
**Who does it:** YOU — primary responsibility in this role  
**Covered in depth:** Day 1, Phase 2

---

### Stage 3 — Alignment / Preference Optimization

**WHAT:**  
Train on human preferences.  
Given two responses to the same instruction,  
humans label which one is better.  
Model learns to generate responses closer to  
human-preferred outputs.

**WHY:**  
SFT teaches format and instruction following  
but does NOT guarantee:
- Consistent tone (polite vs aggressive)
- Honesty (no hallucinations)  
- Safety (refusing harmful requests)
- Helpfulness (actually useful answers)

Alignment fixes all of these by teaching  
the model what **good** looks like vs **bad**.

**WHEN:**  
After SFT. For any production deployment.  
Can be skipped for quick internal experiments  
but never for user-facing systems.

**Methods overview** (deep dive in Day 2):

| Method | Complexity | Needs Reward Model | Best For |
|---|---|---|---|
| PPO | Very High | Yes | When you have reward model already |
| DPO | Low | No | Subjective quality (tone, fluency) |
| GRPO | Medium | No (rule-based) | Verifiable outputs (code, math, JSON) |
| ORPO | Low | No | When SFT data is limited |

**GRPO vs DPO decision rule** (critical for interview):
```
Use GRPO when output is automatically verifiable:
  ✅ Math answers, code execution, JSON validity, SQL

Use DPO when output needs human judgment:
  ✅ Tone, fluency, helpfulness, creativity

Hybrid: Use GRPO for structured parts + DPO for quality parts
  Example: Tool calling output must be valid JSON (GRPO)
           AND be helpful and well-worded (DPO)
```

**What breaks if skipped:**  
Model follows instructions but may be inconsistent,  
harmful, or produce low-quality responses.  
Not safe for production deployment.

**Cost:** Similar to SFT  
**Who does it:** YOU  
**Covered in depth:** Day 2

---

## Llama 3 Base vs Instruct — Precise Definition

```
Llama 3 Base:
  = Pretraining only
  Cannot follow instructions
  Completes text, does not answer questions
  Starting point for YOUR pipeline

Llama 3 Instruct:
  = Pretraining + SFT + some alignment (DPO/RLHF)
  Can follow instructions
  Has basic safety guardrails
  BUT can still generate unsafe answers —
  alignment quality is never perfect
  Good starting point if you want to skip
  continued pretraining (trade-off: lower Indic quality)
```

> **Interview answer:**  
> *"Llama 3 Base is a pure language model — it completes text  
> but has no concept of being an assistant. Llama 3 Instruct  
> has gone through SFT and alignment, so it can follow instructions  
> and has basic safety. However even instruct models can generate  
> unsafe or inconsistent responses — alignment is not a guarantee,  
> it is a significant improvement. For Indic language work,  
> I prefer starting from Base because Instruct's alignment  
> may conflict with Indic-specific instruction styles,  
> and I want full control over the SFT stage."*

---

## Full Pipeline for Indic LLM Role

```
[Llama 3 Base]
      ↓
Stage 1.5: Continued Pretraining
  → Hindi Wikipedia, IndicCorp, Sangraha
  → Hinglish data (Roman script Hindi + English mix)
  → Duration: days to weeks on multi-GPU
      ↓
Stage 2: SFT
  → Instruction datasets in Hindi, Hinglish, English
  → Tool calling examples
  → Customer support / domain-specific conversations
  → LoRA or QLoRA on single/multi GPU
      ↓
Stage 3: Alignment
  → DPO for tone and quality (subjective)
  → GRPO for tool calling / structured output (verifiable)
  → Or ORPO if data is limited (combines SFT + alignment)
      ↓
[Production Model]
      ↓
Inference Optimization
  → Quantization (AWQ/GPTQ)
  → Serve via vLLM or TGI
  → Latency and throughput targets
```

---

## Challenges If Building This Yourself (Individual Contributor)

```
Challenge 1 → Continued pretraining data quality
  Indic web data is noisy — OCR errors, transliteration
  inconsistencies, mixed scripts in same document.
  Cleaning pipeline for Devanagari is non-trivial.

Challenge 2 → Catastrophic forgetting
  Continued pretraining on Hindi can degrade English.
  Full pretraining on Indic data → model forgets English.
  Need careful data mixing ratios (English + Indic blend).

Challenge 3 → SFT data for Hinglish is scarce
  Almost no publicly available high-quality
  Hinglish instruction datasets exist.
  Synthetic data generation is required.

Challenge 4 → Alignment data collection
  DPO needs preference pairs — human annotations.
  For Hinglish, annotators who can judge quality
  in both Hindi and English simultaneously are rare.

Challenge 5 → GPU cost on single machine
  Continued pretraining even on 7B model
  requires multi-GPU or very long single-GPU runs.
  Managing this as IC without cluster access is hard.

Challenge 6 → Evaluation for Indic languages
  No standard Hinglish benchmark exists.
  Must build custom evaluation sets.
  Hard to know if model improved without good evals.
```

---

## Quick Reference Summary

```
STAGE              PURPOSE                    YOU DO IT?
────────────────────────────────────────────────────────
Pretraining        World knowledge + language  Rarely
                   Base model output

Continued          Improve Indic/Hinglish      Yes — important
Pretraining        before SFT                  step to mention

SFT                Teach instruction           Yes — primary job
                   following and format

Alignment          Consistency, safety,        Yes — DPO/GRPO
(DPO/GRPO/ORPO)   tone, helpfulness

Inference          Latency, throughput,        Yes — quantization
Optimization       cost reduction              + vLLM/TGI serving

────────────────────────────────────────────────────────
DECISION RULES
────────────────────────────────────────────────────────
Start from Base or Instruct?
  → Base: full control, recommended for Indic work
  → Instruct: faster, skip if alignment conflicts

GRPO or DPO?
  → Verifiable output  → GRPO
  → Subjective quality → DPO
  → Both               → Hybrid (GRPO + DPO together)

Skip continued pretraining?
  → Only if base model has strong Indic representation
  → Llama 3: do NOT skip for Hindi/Hinglish work

────────────────────────────────────────────────────────
THE ONE UNIFYING EQUATION
  All stages minimize: -log P(y_t | y_<t, x)
  Only x and y change across stages.
────────────────────────────────────────────────────────
```

---

*Next: Day 1 — Phase 2: SFT Deep Dive*  
*Covers: instruction templates, LoRA vs QLoRA vs full fine-tune,*  
*loss function details, Hinglish-specific SFT considerations*