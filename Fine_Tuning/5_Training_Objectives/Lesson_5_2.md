# Lesson 5.2 — Instruction Following: Teaching a Model to Do What You Ask

---

## What Instruction Following Actually Means

A base language model predicts the next token. It is trained to continue text. If you prompt it with "What is the capital of France?", it will continue the text in whatever direction seems most probable given its training data — which might be another question, a list of European capitals, or the beginning of a geography quiz.

An instruction-following model understands that a question is a request for an answer. It knows when to be brief vs detailed, when to use lists vs prose, when to add caveats vs state facts directly. It can write code when asked to code, summarize when asked to summarize, and decline when asked to do something harmful.

This is not a trivial capability. It is a fundamentally different mode of behavior that must be explicitly trained. The training data, format, and diversity choices are what determine whether instruction following actually works.

---

## The Instruction Tuning Revolution

The Stanford Alpaca paper (Taori et al., 2023) showed something remarkable: fine-tuning LLaMA 7B on just **52,000 instruction-response pairs** — generated automatically using GPT-3.5 — produced a model that, in human evaluations, was often preferred over the full GPT-3.5 for general conversation.

This was a watershed moment because it demonstrated:
1. The base LLM already has the knowledge and language capability
2. A relatively small number of well-formatted examples can unlock instruction-following behavior
3. Synthetic data (model-generated) can be good enough for a strong signal

The LIMA paper (Zhou et al., 2023) pushed this further: **1,000 carefully curated examples** outperformed 50,000 randomly collected ones. The quality and diversity of examples matter far more than sheer volume.

> **Interview note:** "How many examples do you need for instruction tuning?" Wrong answer: "As many as possible." Right answer: "Quality and diversity matter more than quantity. The LIMA paper showed 1000 curated examples can beat 50K low-quality ones. For a specific narrow task, 500–2000 high-quality examples are often sufficient. For broad general instruction following, 10K–100K diverse examples are needed to cover the variety of instruction types and formats the model will encounter."

---

## Diversity: The Most Important Property of Instruction Data

The model can only generalize to instruction types it has seen during training. If all your training examples are "answer this factual question", the model will be poor at "write me a persuasive essay" or "debug this code".

**Task diversity** — the range of different things you are asking the model to do:

| Category | Example tasks |
|---|---|
| Information extraction | Summarize, extract entities, classify sentiment |
| Generation | Write an email, generate a story, write code |
| Transformation | Translate, reformat, paraphrase, expand/compress |
| Analysis | Compare X and Y, critique this argument, explain the pros and cons |
| Reasoning | Solve this math problem, debug this logic error |
| Dialogue | Roleplay as, answer follow-up questions, maintain consistency |
| Refusal | Identify and decline harmful requests |

**Format diversity** — how the instruction is framed:
- Direct command: "Write a Python function that..."
- Question form: "Can you explain what a transformer is?"
- Multi-turn: the instruction builds over several turns
- System-prompt guided: behavior conditioned on a system prompt persona

**Output diversity** — the range of response styles:
- Short, direct answers
- Long, detailed explanations
- Structured (JSON, tables, lists)
- Conversational prose
- Code with explanations

A model trained on only long-form explanations will pad all responses unnecessarily. A model trained only on short answers will fail to elaborate when needed.

---

## The Self-Instruct Bootstrap Problem

How do you get diverse instruction-response data? The self-instruct approach (Wang et al., 2022) solved this:

1. Start with a small seed set of human-written instruction-response pairs (~175 examples)
2. Feed these to a capable LLM (GPT-3.5 or GPT-4) and ask it to generate new, diverse instructions
3. Filter out near-duplicates and low-quality generations
4. Use the same LLM to generate high-quality responses for the new instructions
5. Repeat to scale

This generates data that is diverse (the LLM generates many types of instructions) and reasonably high-quality (the LLM knows what good responses look like). The Alpaca dataset was built this way. So was Evol-Instruct, which additionally asked GPT-4 to "evolve" simple instructions into harder, more complex ones — creating a richer training distribution.

**The quality filter is critical.** Without filtering, generated data includes near-duplicates (same instruction phrased slightly differently), toxic content, factually wrong responses, and format inconsistencies. Standard filters:
- Deduplication by cosine similarity of instruction embeddings (remove examples with similarity > 0.85)
- Length filter: remove very short or extremely long responses
- Keyword filter: remove responses containing specific failure patterns
- LLM-as-judge filter: score each example for quality and discard bottom 20%

---

## Prompt Templates: The Consistency Requirement

Instruction-following behavior is tied to the specific template format used during training. The model learns that tokens in the template format signal "instruction following mode."

**Common templates:**

```
# Alpaca template
### Instruction:
{instruction}

### Response:
{response}
```

```
# ChatML template (used by OpenAI API, many open models)
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

```
# LLaMA-3 template
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>
```

**Critical rule:** The tokenizer must apply exactly the same template at inference time as was used during training. Mismatched templates cause degraded performance — the model's conditioning signals do not match what it learned to respond to.

Most HuggingFace model repos include a `tokenizer_config.json` with a `chat_template` field (Jinja2 format) that applies the correct template automatically via `tokenizer.apply_chat_template()`. Always use this rather than manually constructing prompts.

---

## Loss Masking: Only Learn from the Response

During SFT training, the loss should be computed only on **response tokens** — not on the prompt, system message, or user turn tokens.

Why? The model should learn to generate good responses. It should not learn to predict the next token of the instruction (which is fixed/deterministic). Computing loss on instruction tokens wastes model capacity and can cause the model to "memorize" prompts.

```python
# In a data collator for instruction tuning
# labels=-100 means "ignore this token in the loss computation"

def apply_loss_masking(example, tokenizer):
    messages = example["messages"]
    
    # Tokenize the full conversation
    full_tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    
    # Find where the assistant turn starts
    # Set labels=-100 for all tokens before the assistant response
    labels = full_tokens.copy()
    
    # Tokenize everything up to the assistant response
    prompt_tokens = tokenizer.apply_chat_template(
        messages[:-1],  # All turns except the last assistant turn
        tokenize=True,
        add_generation_prompt=True
    )
    
    # Mask prompt tokens: set to -100 (ignored by cross-entropy loss)
    labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
    
    return {"input_ids": full_tokens, "labels": labels}
```

A common bug in instruction tuning pipelines is forgetting loss masking — computing loss on the full sequence including the prompt. The result: lower training loss (more tokens to fit) but a model that learns to "continue" prompts rather than respond to them.

---

## The System Prompt: Training the Default Behavior

The system prompt defines the model's default persona and behavior constraints. Training examples with diverse system prompts teach the model to adapt its behavior based on system-level instructions.

```python
# Include diverse system prompts in training data
system_prompts = [
    "You are a helpful assistant.",
    "You are an expert Python programmer. Always provide working code.",
    "You are a concise assistant. Keep all responses under 3 sentences.",
    "You are a formal business writing assistant.",
    None  # No system prompt — model should still behave helpfully
]
```

A model trained with no system prompt variation will either ignore system prompts entirely or be confused by them at inference time. Include examples with varied system prompts AND examples with no system prompt.

---

## Summary

- Instruction following is not a natural property of base LLMs — it is explicitly trained behavior. The base model predicts text; the instruction-following model understands requests and responds appropriately.
- The LIMA paper established the core principle: 1,000 high-quality, diverse examples beat 50,000 low-quality ones. Quality and diversity are the primary levers, not scale.
- Diversity must cover: task type (generation, extraction, analysis, refusal), format (direct, question, multi-turn), and output style (short, long, structured, prose).
- Self-instruct allows bootstrapping large instruction datasets from seed examples using an LLM. Quality filtering (dedup, length, LLM-judge) is essential — unfiltered synthetic data degrades performance.
- Prompt template consistency is mandatory: use the exact same template at inference that was used during training. Use `tokenizer.apply_chat_template()` to guarantee this.
- Loss masking is critical: set labels=-100 for all prompt/instruction tokens, compute loss only on response tokens.

---
