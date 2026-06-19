# Lesson 1.5 — Pre-training vs Fine-tuning vs In-Context Learning

---

## 1. The Problem Story

At an interview, the question was: "We have a task where we need a model to answer technical questions about our internal documentation. Would you fine-tune a model or use RAG with prompting? Explain your reasoning."

The candidate said: "I would fine-tune because that is more powerful."

The interviewer asked: "What specifically would fine-tuning learn that prompting cannot provide? And what are the downsides of fine-tuning in this scenario?"

The candidate had no answer. They had never thought about the *decision* — they just assumed fine-tuning is always better.

This is the wrong mental model. The right mental model is a decision framework. This lesson gives you that.

---

## 2. The Concept

### Pre-training: Learning Everything from Scratch

Pre-training is the initial, massive training of a language model on internet-scale text.

**What happens:**
- Trillions of tokens of text (Common Crawl, Wikipedia, books, code, etc.)
- Months of compute on thousands of GPUs
- The model learns the statistical structure of language, world knowledge, reasoning patterns, coding patterns, and more
- Objective: predict the next token at every position

**What the model learns:**
- Grammar and language structure
- Factual knowledge (capitals, historical events, scientific facts)
- Reasoning patterns (if A then B...)
- Coding patterns (function signatures, algorithms)
- Writing styles, domains, etc.

**What it does NOT learn:**
- How to follow instructions ("do X")
- How to have a conversation (user/assistant format)
- How to refuse harmful requests
- Any organization-specific knowledge
- Any task that was underrepresented in training data

**Cost:** Hundreds of millions of dollars. Not something you do.

**Result:** A base model (e.g., LLaMA-3-8B base). Technically capable but not instruction-following.

### Fine-tuning: Specialization from a Good Starting Point

Fine-tuning takes the pre-trained weights and continues training on a smaller, targeted dataset. The goal is to adapt the model's behavior without losing its general knowledge.

**Supervised Fine-Tuning (SFT):**
The most common form. You train on (input, output) pairs. The model learns to produce the right output for your specific inputs.

**What fine-tuning changes:**
- Behavioral patterns (how to respond to instructions)
- Style and format (always respond in bullet points, use formal language)
- Domain knowledge enhancement (if your domain was underrepresented in pre-training)
- Task-specific capabilities (following a specific output schema, medical reasoning)

**What fine-tuning does NOT easily change:**
- Deep factual knowledge (fine-tuning on 1000 examples doesn't replace knowledge from trillion-token pre-training)
- Fundamental capabilities (you cannot fine-tune a small model to reason like GPT-4)

**Cost:** Hours to days on a single GPU (with PEFT). Feasible.

### Instruction Fine-tuning

A specific type of SFT where the goal is to teach the model to follow instructions in general, not just one task. Examples: the original InstructGPT, Alpaca, Vicuna.

This is what turns a base model into a chat model.

### In-Context Learning (Prompting)

No weight updates. You put examples, instructions, or context in the prompt, and the model uses them to produce better outputs.

**Few-shot prompting:**
```
Q: What is the capital of France? A: Paris.
Q: What is the capital of Germany? A: Berlin.
Q: What is the capital of Japan? A: 
```
The model learns the pattern from examples in the prompt and applies it.

**Why it works:** Pre-training exposed the model to so many examples of Q&A patterns that it learned to "recognize" and "complete" such patterns in context.

**Limits:** The context window is finite. You cannot put 10,000 examples in the prompt. The model cannot fundamentally change its behavior from context alone.

### RAG: Retrieval Augmented Generation

Not exactly fine-tuning, but important to understand in comparison.

RAG retrieves relevant documents and adds them to the prompt:
```
[Retrieved doc]: "Our return policy is 30 days for all products..."
[User]: "What is the return policy?"
[Model]: "Based on the information provided, our return policy is 30 days..."
```

The model's weights are not changed. The knowledge comes from the retrieved documents.

### The Decision Framework

This is what the interviewer wanted to hear:

```
┌─────────────────────────────────────────────────────────────┐
│  Should I fine-tune or prompt/RAG?                         │
│                                                             │
│  Fine-tune when:                                           │
│  • You need a specific output FORMAT/STYLE consistently    │
│  • The task requires behavior that prompting cannot achieve│
│  • You need the model to "internalize" a skill             │
│  • Latency matters (shorter prompts = faster inference)    │
│  • You have enough task-specific training data (100+)      │
│  • The task is well-defined and stable                     │
│                                                             │
│  Prompt/RAG when:                                          │
│  • You need up-to-date or organization-specific knowledge  │
│  • The task changes frequently                             │
│  • You have no training data                               │
│  • You need to cite sources                                │
│  • The information exists in documents (RAG is ideal)      │
│  • Fine-tuning cost/time is not justified                  │
│                                                             │
│  Often the best answer: both together                      │
│  Fine-tune for behavior/style + RAG for knowledge          │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. The Intuition Bridge

**Pre-training** is like a person going through school, reading millions of books, learning languages, science, history — they develop general intelligence and broad knowledge.

**Fine-tuning** is like that person then getting a job-specific training program. They already know how to read, reason, and communicate. Now they learn the specific workflows, terminology, and behaviors needed for this specific role. They don't forget what they learned in school.

**In-context learning** is like giving that trained professional a reference card before each task: "Remember to sign off with 'Best regards'. Here are three examples of good responses." They don't need to be retrained — they just use the card.

**RAG** is like giving them access to a search engine and a company wiki. They look things up before answering.

Each has its place. The experienced professional (fine-tuned model) with a search engine (RAG) and a reference card (well-designed prompts) is the ideal setup for production.

---

## 4. Why This Matters for Fine-Tuning

**Fine-tuning is not always the answer**

In your next interview, when you describe a fine-tuning project, you should be able to say why fine-tuning was the right choice — not just "we fine-tuned because we fine-tuned."

Good answer: "We fine-tuned because we needed the model to consistently output structured JSON in a specific schema. We tried prompting, but it failed in about 20% of cases. Fine-tuning reduced this to <1%."

Bad answer: "We fine-tuned because we wanted better performance."

**Fine-tuning for behavior vs knowledge**

Fine-tuning is excellent for changing how the model behaves (output format, response style, instruction following behavior). It is less effective for teaching new factual knowledge — a model trained on 1000 QA pairs about your internal docs learns those 1000 patterns, but cannot generalize to questions not represented in training. RAG handles that case better.

---

## 5. The Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

# ── Zero-shot (no examples) ─────────────────────────────────────
zero_shot_prompt = "Classify the sentiment of this review as POSITIVE or NEGATIVE.\nReview: The food was amazing and the service was wonderful!\nSentiment:"
print("Zero-shot:")
print(generate(zero_shot_prompt))

# ── Few-shot (2 examples in context) ────────────────────────────
few_shot_prompt = """Classify the sentiment of this review as POSITIVE or NEGATIVE.

Review: The room was dirty and the staff was rude.
Sentiment: NEGATIVE

Review: Excellent product, highly recommend it!
Sentiment: POSITIVE

Review: The food was amazing and the service was wonderful!
Sentiment:"""
print("\nFew-shot:")
print(generate(few_shot_prompt))

# ── In-context learning with instruction ────────────────────────
instruction_prompt = """You are a sentiment classifier. You ONLY output one word: POSITIVE or NEGATIVE.
Never explain. Never add extra text.

Review: The food was amazing and the service was wonderful!
Sentiment:"""
print("\nInstruction (zero-shot):")
print(generate(instruction_prompt))

# ── Demonstrating knowledge limit (in-context vs fine-tuning) ───
# A model cannot reliably "know" private information from a prompt
# unless it is in the context (RAG) or baked in via fine-tuning
private_info_prompt = "What is Anthropic's internal code name for their next model?"
print("\nQuery about private info (no context):")
print(generate(private_info_prompt))

rag_style_prompt = """Context: According to Anthropic's internal Q3 planning document, 
the next model is codenamed 'Project Helios' and targets a Q4 2025 release.

Question: What is Anthropic's internal code name for their next model?
Answer:"""
print("\nSame query with RAG-style context:")
print(generate(rag_style_prompt))
```

---

## 6. The Experiment

**Experiment 1.5.A — Fine-tune vs Prompt Comparison**

Pick a structured output task (e.g., "extract person names from text and return as JSON array").

1. First try zero-shot prompting. Record success rate on 20 examples.
2. Try few-shot prompting with 3 examples. Record success rate.
3. Try with a very detailed system prompt. Record success rate.
4. Record which cases fail and why.

Then answer: Would fine-tuning help here? What would the training data look like? What success rate would you target?

You do not actually need to fine-tune for this experiment — you need to think through it. Write a one-page "decision analysis" as if you were presenting it to your tech lead.

---

## 7. Interview Checkpoint

**Q: When would you fine-tune a model instead of using prompting?**

> A: I use fine-tuning when prompting cannot reliably achieve the target behavior. Specifically: when I need a consistent output format (like strict JSON schema) and prompting fails in 10–20% of cases; when the task requires internalizing domain-specific patterns that cannot be described in a prompt; when I need to reduce prompt length for latency reasons (a fine-tuned model knows the behavior without a long system prompt); or when I have 100+ examples of the correct behavior. I use prompting or RAG when the information is in external documents, when the task evolves frequently, or when I don't have enough training data.

**Q: Can fine-tuning teach a model new facts?**

> A: Fine-tuning can introduce knowledge that was not in pre-training, but it is not the ideal mechanism for this. A model fine-tuned on 500 Q&A pairs about your company will "learn" those answers, but it will also confabulate (hallucinate) confidently for questions not covered by training. RAG is almost always better for knowledge retrieval because the information is sourced explicitly at inference time, it can be updated without retraining, and the model can be prompted to say "based on the provided context" rather than confabulating. The best pattern for production is: fine-tune for behavior and style, use RAG for knowledge.

---

## 8. Common Mistakes & Misconceptions

❌ **"Fine-tuning always beats prompting."**
False. For many tasks, a well-engineered prompt with a capable base model outperforms a poorly fine-tuned smaller model. Fine-tuning adds complexity and cost. Use it when it solves a real problem that prompting cannot.

❌ **"Fine-tuning is just continued pre-training."**
Fine-tuning and continued pre-training have different data formats, objectives, and learning rates. Continued pre-training uses raw text and trains on next-token prediction at scale. Fine-tuning uses structured (input, output) pairs and uses much lower learning rates to avoid overwriting pre-trained knowledge.

❌ **"A fine-tuned model doesn't need prompting."**
Fine-tuning changes the model's weights. You still need to format your inputs correctly and provide appropriate context. Fine-tuning reduces the need for elaborate few-shot examples, but it does not eliminate the need for good prompts.

---

# Chapter 1 — Summary and What to Do Before Chapter 2

---

## What You Now Know

| Lesson | Core takeaway |
|--------|---------------|
| 1.1 Tokenization | Text → integer IDs via BPE/SentencePiece. Token count depends on vocabulary coverage. Special tokens define structure. |
| 1.2 Architecture | Embedding → N × (Attention + FFN + LayerNorm) → LM Head. Q/K/V/O are the attention weights LoRA targets. FFN stores factual knowledge. |
| 1.3 Forward Pass | model(input_ids) → logits (batch, seq, vocab). Loss = cross-entropy on shifted labels. Perplexity = exp(loss). |
| 1.4 Generation | Autoregressive token-by-token loop. Temperature/top-p control randomness. KV cache makes it fast. |
| 1.5 Decision Framework | Fine-tune for behavior/style. RAG for knowledge. Prompting first if sufficient. Fine-tuning when prompting fails or is too slow. |

---

## Checklist Before Moving to Chapter 2

Complete all of these before proceeding. Do not skip.

- [ ] You can tokenize a string, inspect token IDs, and explain what each token is
- [ ] You can load a HuggingFace model and print its full architecture
- [ ] You know what `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` are and what they do
- [ ] You have run the forward pass code and verified that manual loss matches model's computed loss
- [ ] You have run generation with at least 3 different temperature values and observed the difference
- [ ] You have completed Experiment 1.1.A (token efficiency audit)
- [ ] You have completed Experiment 1.3.A (surprisal analysis)
- [ ] You have completed Experiment 1.4.A (generation sensitivity study)
- [ ] You can answer all 10 Interview Checkpoint questions out loud, without notes

---

## The Test: Explain It Out Loud

Before Chapter 2, do this test:

Set a 5-minute timer. Explain out loud, as if to a friend who knows Python but not ML:

> "When I type a question into a language model and it gives me an answer, what is actually happening from the moment I hit enter to when the first word appears?"

A complete answer covers: tokenization → embedding → forward pass through transformer layers → logits → sampling → generating token by token → stopping condition.

If you cannot do this fluently, re-read the relevant lessons. Do not skip to Chapter 2.

---

## Coming Up: Chapter 2

Chapter 2 — *What Fine-Tuning Actually Does to a Model* — builds directly on Chapter 1.

Now that you understand what the model is doing, Chapter 2 answers: what exactly changes in those weights during fine-tuning, how the training loop works, and the critical concept of loss masking that most fine-tuning beginners get wrong.

---

*Chapter 1 complete. You earned Chapter 2.*