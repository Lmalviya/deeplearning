# Lesson 1.4 — Autoregressive Generation

---

## 1. The Problem Story

Suresh built a customer service chatbot. During testing, it worked perfectly. But in production, users complained that sometimes the bot gave completely different answers to the same question.

The problem: during testing, Suresh used `temperature=0` (greedy, deterministic). In production, someone set `temperature=0.9` and `top_p=0.95` for "more natural" responses. These settings changed the randomness of generation completely — the model was sampling different tokens every time.

He did not know what temperature or top_p actually did. He thought they were minor settings. They are not.

This lesson teaches you the full generation process and every sampling parameter, so you never have a Suresh situation.

---

## 2. The Concept

### The Generation Loop

Language models are trained to predict one token at a time. To generate a full response, you run the model repeatedly:

```
Input: "What is 2 + 2?"

Step 1: Model sees "What is 2 + 2?" → predicts next token → picks "4"
Step 2: Model sees "What is 2 + 2? 4" → predicts next token → picks "."
Step 3: Model sees "What is 2 + 2? 4." → predicts next token → picks "<EOS>"
Step 4: "<EOS>" detected → stop
```

Each step is one forward pass. Generating 100 tokens = 100 forward passes. This is why generation is slow.

### Greedy Decoding

At each step, pick the token with the highest probability. Simple. Deterministic. Often produces repetitive or boring text.

```python
next_token = logits.argmax(dim=-1)  # always pick the max
```

### Sampling

Instead of always picking the top token, sample from the probability distribution. This introduces controlled randomness.

```python
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)  # sample
```

### Temperature

Temperature controls how "peaked" or "flat" the probability distribution is before sampling.

```python
probs = F.softmax(logits / temperature, dim=-1)
```

- `temperature=1.0`: original distribution (no change)
- `temperature<1.0` (e.g., 0.3): sharper distribution → top tokens get even higher prob → more deterministic, more repetitive
- `temperature>1.0` (e.g., 1.5): flatter distribution → more uniform → more random, more creative but less coherent

**Rule of thumb:**
- Code generation: 0.0 – 0.2 (deterministic, correctness matters)
- Factual QA: 0.2 – 0.5
- Creative writing: 0.7 – 1.0
- Exploratory sampling: > 1.0 (usually not recommended)

### Top-K Sampling

Before sampling, restrict to only the top K tokens and renormalize:

```python
top_k_logits, top_k_indices = logits.topk(k=50)
probs = F.softmax(top_k_logits, dim=-1)
next_token_idx = torch.multinomial(probs, num_samples=1)
next_token = top_k_indices[next_token_idx]
```

`top_k=50` means: at each step, only the 50 highest-probability tokens are candidates. This prevents the model from ever choosing a very unlikely token.

**Problem with top-K:** K is fixed, but the natural "head" of the distribution varies. Sometimes the top 5 tokens are essentially all the viable options; top-50 would include clearly wrong choices. Other times, 200 tokens are all reasonable.

### Top-P (Nucleus) Sampling

Instead of a fixed K, include the smallest set of tokens whose cumulative probability exceeds P:

```python
# Sort by probability
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = sorted_probs.cumsum(dim=-1)
# Remove tokens above the threshold
sorted_indices_to_remove = cumulative_probs > top_p
sorted_probs[sorted_indices_to_remove] = 0
# Renormalize and sample
sorted_probs /= sorted_probs.sum()
```

`top_p=0.9` means: at each step, take the fewest tokens whose probabilities sum to 90%, then sample from those.

When the distribution is peaked (the model is confident), nucleus might be just 3–5 tokens. When flat (model uncertain), it might be 50+. Top-P adapts to the distribution; Top-K doesn't.

**Top-P is generally preferred over Top-K for language generation.**

### Beam Search

Instead of generating one token at a time, keep the top B candidate sequences:

```
beam_width = 3
Step 1: Model generates top 3 next tokens → 3 candidate sequences
Step 2: For each candidate, generate top 3 → 9 sequences
         Keep only the 3 with highest cumulative probability
Step 3: Continue until all 3 reach <EOS>
Step 4: Return the sequence with highest overall probability
```

Beam search often produces more "correct" text but can sound robotic. Used heavily in translation and summarization. Less used for chat.

### KV Cache (Key-Value Cache)

Without caching: at each generation step, the model recomputes attention for the entire sequence so far. If the sequence is 100 tokens, step 100 processes all 100 tokens.

With KV cache: the K and V matrices for all previous tokens are stored. Each new step only needs to process the new token.

This is the most important optimization for generation speed. HuggingFace enables it by default when using `model.generate()`.

### Repetition Penalty

Generation can get stuck in loops. A repetition penalty reduces the probability of tokens that have already appeared:

```python
# Before sampling, divide logits of already-seen tokens by repetition_penalty
for token_id in generated_so_far:
    logits[token_id] /= repetition_penalty  # > 1.0 reduces probability
```

---

## 3. The Intuition Bridge

**Temperature as confidence level:**

Imagine asking someone "what is 2+2?" They answer confidently: "4." That is low temperature — one clear answer.

Now ask "what should I do with my life?" They are uncertain, many answers seem valid. That is high temperature — a spread-out distribution.

Temperature does not make the model smarter. It makes it less committed to its top choice. If the top choice was wrong, higher temperature can help. But if the top choice was right, higher temperature introduces errors.

**Top-P as "reasonable candidates only":**

If you are predicting the next word after "The weather is very ___", the reasonable candidates are: "hot", "cold", "nice", "beautiful", "bad", etc. "elephant" is technically possible but absurd.

Top-P draws a line: include all reasonable options (whose total probability is 90%), exclude everything else. This prevents absurd outputs while preserving diversity.

---

## 4. Why This Matters for Fine-Tuning

**Your eval during training should match your production settings**

If you evaluate your model during training with greedy decoding but deploy with temperature=0.8, you will see a quality gap. Use the same generation settings for training eval and production.

**Format tokens affect generation**

If you fine-tune a model on data with `<|assistant|>` as the start-of-response token, you must prompt it with `<|assistant|>` at inference to trigger the response. The generation settings interact with the prompt format.

**Stopping criteria**

You must configure what token stops generation. For chat models, this is the `<EOS>` token or a special token like `<|im_end|>`. Without proper stopping criteria, the model will continue generating after the response ends.

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

prompt = "The best programming language is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("=" * 60)
print(f"Prompt: '{prompt}'")
print("=" * 60)

# ── Greedy decoding ─────────────────────────────────────────────
output = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=False,      # greedy
    pad_token_id=tokenizer.eos_token_id
)
text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\nGreedy:               '{text}'")

# ── Sampling with different temperatures ────────────────────────
for temp in [0.3, 0.7, 1.0, 1.5]:
    torch.manual_seed(42)  # same seed for fair comparison
    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=temp,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Temperature={temp}:    '{text}'")

# ── Top-P sampling ───────────────────────────────────────────────
for top_p in [0.5, 0.9, 0.99]:
    torch.manual_seed(42)
    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=1.0,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Top-P={top_p}:        '{text}'")

# ── Beam search ─────────────────────────────────────────────────
output = model.generate(
    **inputs,
    max_new_tokens=30,
    num_beams=5,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\nBeam search (5):      '{text}'")

# ── Measure generation speed ─────────────────────────────────────
import time
long_prompt = "Tell me about machine learning: "
long_inputs = tokenizer(long_prompt, return_tensors="pt").to(model.device)

start = time.time()
output = model.generate(**long_inputs, max_new_tokens=100, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id)
elapsed = time.time() - start
new_tokens = output.shape[1] - long_inputs["input_ids"].shape[1]
print(f"\nGeneration speed: {new_tokens/elapsed:.1f} tokens/sec")
print(f"Generated {new_tokens} tokens in {elapsed:.2f}s")
```

---

## 6. The Experiment

**Experiment 1.4.A — Generation Sensitivity Study**

Take one factual prompt and one creative prompt. For each, generate 5 outputs at each of: temperature 0.1, 0.5, 1.0, 1.5. Measure:
1. Are the outputs deterministic or varied within same temperature?
2. At what temperature do factual errors start appearing?
3. At what temperature does creative writing become incoherent?

Write your findings as if you were writing a 1-paragraph report for a colleague.

This is exactly the kind of "I experimented with this" story you want to tell in an interview.

---

## 7. Interview Checkpoint

**Q: What is the difference between temperature and top-p?**

> A: Both control the randomness of generation but in different ways. Temperature scales the logits before softmax — low temperature sharpens the distribution (more confident), high temperature flattens it (more random). Top-p (nucleus sampling) restricts the candidate tokens to the smallest set that covers P% of the probability mass, adapting to the shape of the distribution at each step. Temperature controls how peaked the distribution is; top-p controls which part of the distribution you sample from. In practice, top-p handles varying distribution shapes better than top-k and is generally preferred.

**Q: What is the KV cache and why does it matter?**

> A: The KV cache stores the computed key and value matrices for all previously generated tokens. Without it, generating token N would require recomputing attention over all N-1 previous tokens. With it, only the new token needs to be processed, since all previous KV pairs are cached. This reduces generation time from O(N²) per token to O(N) per token and is the primary optimization that makes autoregressive generation practical for long sequences.

---

## 8. Common Mistakes & Misconceptions

❌ **"Temperature = 0 means no creativity."**
Temperature controls randomness, not intelligence. At temperature 0 (greedy), the model is more predictable and consistent but not necessarily less "creative" — it is producing its highest-probability output. Whether that is creative depends on training, not generation settings.

❌ **"Higher beam search = always better."**
Beam search with high beam width often produces overly "safe," generic, sometimes repetitive text. For open-ended generation (chat, creative writing), sampling (top-p) almost always beats beam search in human preference evaluations.

❌ **"`do_sample=False` always gives the same output."**
`do_sample=False` with `num_beams=1` gives greedy decoding — always the same output. But `do_sample=False` with `num_beams>1` gives beam search — also deterministic, but not greedy. If you set `do_sample=True` without a seed, results are random each run.

---