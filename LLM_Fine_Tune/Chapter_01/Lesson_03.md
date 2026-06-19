# Lesson 1.3 — The Forward Pass Step by Step

---

## 1. The Problem Story

Priya was fine-tuning a summarization model. The training loss was good. But during inference, the outputs were completely different from what she saw during training evaluation.

She eventually found the bug: she was calling `model(input_ids)` during training but using `model.generate()` during inference, with no thought about the difference. The generation settings (temperature, max_new_tokens) were defaulting to values that produced very different behavior.

More fundamentally, she did not understand what the model was *doing* in each case. The forward pass during training and the forward pass during inference are used differently, and confusing them led to this gap.

This lesson removes that confusion entirely.

---

## 2. The Concept

### What is a forward pass?

A forward pass is a single computation of the model: input goes in, output comes out. No weight updates happen during a forward pass. Weight updates happen in the backward pass (backpropagation).

During training: you do forward → compute loss → backward → update weights.
During inference: you do forward → read the output. That is all.

### The Forward Pass in Detail

Let's trace exactly what happens when you pass text through a language model.

**Input: raw text**
```
"The cat sat on the"
```

**Step 1: Tokenize**
```
token_ids = [450, 6635, 7960, 373, 278]
# Shape: (batch_size=1, seq_len=5)
```

**Step 2: Embedding lookup**
```python
embeddings = embedding_table[token_ids]
# Shape: (1, 5, 4096)  — seq_len=5, hidden_dim=4096
```

Each of the 5 token IDs is replaced by its 4096-dimensional embedding vector.

**Step 3: Pass through N transformer layers**

For each layer l = 1 to N:
```
hidden_states = LayerNorm(hidden_states)
attn_output = MultiHeadSelfAttention(hidden_states)
hidden_states = hidden_states + attn_output        # residual

hidden_states = LayerNorm(hidden_states)
ffn_output = FeedForwardNetwork(hidden_states)
hidden_states = hidden_states + ffn_output         # residual
```

At the end of 32 layers (for a 7B model), `hidden_states` has shape `(1, 5, 4096)`.

**Step 4: Final layer norm**
```
hidden_states = FinalLayerNorm(hidden_states)
```

**Step 5: LM head projection**
```python
logits = hidden_states @ lm_head_weight.T
# Shape: (1, 5, 32000)  — vocab_size=32000
```

For each of the 5 token positions, we now have a probability distribution over all 32,000 vocabulary tokens.

**Step 6: What logits mean**

`logits[0, 4, :]` — the logits at position 4 (the 5th token "the") — represents the model's prediction for the next token. This is what the model "thinks" comes after "The cat sat on the".

The token with the highest logit is the model's best guess for the next word.

### The Output: `logits`

When you call `model(input_ids)`, you get back `logits` of shape `(batch_size, seq_len, vocab_size)`.

**Critically: each position predicts the NEXT token.**

```
Position 0 ("The")    → predicts what comes after "The"
Position 1 ("cat")    → predicts what comes after "The cat"
Position 2 ("sat")    → predicts what comes after "The cat sat"
Position 3 ("on")     → predicts what comes after "The cat sat on"
Position 4 ("the")    → predicts what comes after "The cat sat on the"
```

This is how training works: you feed the model a sequence, and at every position it predicts the next token. You compare those predictions against the actual next tokens and compute loss. This is called teacher forcing — the model always sees the true previous tokens, even if its own predictions were wrong.

### From Logits to Probabilities

Logits are raw scores. To get probabilities, apply softmax:

```python
import torch.nn.functional as F
probs = F.softmax(logits, dim=-1)  # shape: (batch, seq_len, vocab_size)
```

Each position's probability distribution sums to 1.

### The Loss: Cross-Entropy

Training loss computes how wrong the model's predictions are at each position:

```python
# shift: predictions at position i should match the token at position i+1
shift_logits = logits[:, :-1, :]      # all positions except the last
shift_labels = input_ids[:, 1:]       # all tokens except the first

loss = F.cross_entropy(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1)
)
```

**This is the most important equation in this entire course.** Everything in fine-tuning is about minimizing this loss on your chosen data.

---

## 3. The Intuition Bridge

Think of the model as a very sophisticated "guess the next word" player. You show it "The cat sat on the ___" and it assigns probabilities to every word in the dictionary. "mat" gets high probability. "universe" gets low probability.

During training: you show it the actual next word, measure how surprised it was (cross-entropy loss), and update it to be less surprised next time.

During inference: you let it pick a word (based on its probabilities), append that word to the sequence, show it the new sequence, let it pick the next word, and so on. This is generation.

The forward pass is the "guessing" step. The training loop adds the "learning from the guess" step.

---

## 4. Why This Matters for Fine-Tuning

**Loss is computed over ALL token positions by default**

If your training example is:
```
[SYSTEM]: You are a helpful assistant.
[USER]: What is the capital of France?
[ASSISTANT]: The capital of France is Paris.
```

And you compute loss over all tokens, the model is being trained to predict the system message and user message tokens too — not just the assistant response. This is usually wrong. You want the model to learn to *produce* good responses, not to predict instructions.

This is called **prompt masking** or **loss masking** — setting the loss to 0 for prompt tokens. We cover this in detail in Chapter 2.

**Teacher forcing vs autoregressive generation mismatch**

During training (teacher forcing), the model sees the correct previous tokens at every step. During inference, it sees its own generated tokens. If the model makes a mistake early in generation, it has to work with that mistake for the rest of the sequence — a situation it never saw during training. This is called "exposure bias" and is one reason fine-tuned models can degrade at inference.

---

## 5. The Code

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
model.eval()  # disable dropout, BatchNorm running stats updates

text = "The cat sat on the"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# ── The forward pass ────────────────────────────────────────────

with torch.no_grad():  # no gradients needed for inference
    outputs = model(**inputs)

print("Output type:", type(outputs))
print("Output keys:", outputs.keys())

logits = outputs.logits
print(f"\nLogits shape: {logits.shape}")
print("→ (batch_size, seq_len, vocab_size)")

# ── What does the model predict as the next token? ──────────────

# We want the predictions at the LAST position
last_position_logits = logits[0, -1, :]  # shape: (vocab_size,)
print(f"\nLogits at last position: {last_position_logits.shape}")

# Top 10 predicted next tokens
top_k = torch.topk(last_position_logits, k=10)
print("\nTop 10 predictions for next token after 'The cat sat on the':")
for i, (token_id, score) in enumerate(zip(top_k.indices, top_k.values)):
    token_str = tokenizer.decode([token_id])
    prob = F.softmax(last_position_logits, dim=-1)[token_id].item()
    print(f"  {i+1}. '{token_str}' (logit: {score:.2f}, prob: {prob:.4f})")

# ── Compute loss manually ────────────────────────────────────────

text_with_next = "The cat sat on the mat"
inputs2 = tokenizer(text_with_next, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs2 = model(**inputs2)

logits2 = outputs2.logits
input_ids = inputs2["input_ids"]
vocab_size = logits2.shape[-1]

# Shift: position i predicts token at position i+1
shift_logits = logits2[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()

loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1)
)
perplexity = torch.exp(loss)

print(f"\nManual cross-entropy loss: {loss.item():.4f}")
print(f"Perplexity: {perplexity.item():.2f}")

# Compare with the model's own loss computation
with torch.no_grad():
    outputs3 = model(**inputs2, labels=input_ids)
print(f"Model's computed loss:     {outputs3.loss.item():.4f}")
print("(Should match manual computation — small differences due to shift handling)")

# ── Probability distribution at each position ───────────────────

probs = F.softmax(logits2[0], dim=-1)  # shape: (seq_len, vocab_size)
print(f"\nProbability distribution shape: {probs.shape}")
print("Sum of probs at position 0:", probs[0].sum().item())  # should be ~1.0
```

---

## 6. The Experiment

**Experiment 1.3.A — Surprisal Analysis**

Surprisal is `-log2(prob)` for a token given its context. It measures how unexpected a token was. High surprisal = the model was very surprised.

Take 5 different sentences. For each one, compute the per-token surprisal and find which tokens the model was most surprised by. Then reason about why those tokens are surprising.

```python
def compute_surprisal(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]  # (seq_len, vocab_size)
    probs = F.softmax(logits, dim=-1)
    input_ids = inputs["input_ids"][0]

    results = []
    for i in range(1, len(input_ids)):  # start from 1 (position 0 has no prediction before it)
        actual_token_id = input_ids[i]
        actual_token = tokenizer.decode([actual_token_id])
        # Probability of actual token given previous context
        prob = probs[i-1, actual_token_id].item()
        surprisal = -torch.log2(torch.tensor(prob)).item()
        results.append({
            "token": actual_token,
            "prob": prob,
            "surprisal": surprisal
        })
    return results

sentences = [
    "The cat sat on the mat.",
    "The cat sat on the quantum.",
    "Paris is the capital of France.",
    "Paris is the capital of banana.",
]

for sentence in sentences:
    results = compute_surprisal(model, tokenizer, sentence)
    print(f"\n'{sentence}'")
    for r in results:
        bar = "█" * min(int(r["surprisal"]), 20)
        print(f"  '{r['token']:15s}' surprisal={r['surprisal']:5.2f}  {bar}")
```

**Write down:** Which tokens are most surprising? Do they match your intuition? Why does "quantum" have higher surprisal than "mat" after "The cat sat on the"?

---

## 7. Interview Checkpoint

**Q: What is teacher forcing and why is it used?**

> A: Teacher forcing is a training technique where the model's input at each position is always the true previous token from the training data, not the model's own prediction. This makes training more stable and faster — the model always gets correct context. The downside is "exposure bias": during inference, the model sees its own (possibly wrong) previous outputs, which is a different distribution from what it trained on.

**Q: What is the shape of the output logits and what does each dimension represent?**

> A: Logits have shape `(batch_size, sequence_length, vocab_size)`. The first dimension is how many examples are in the batch. The second is the number of tokens in the input sequence. The third is the vocabulary size — for each position, the model outputs a score for every possible next token. The token with the highest score at position i is the model's prediction for the token at position i+1.

**Q: What is perplexity?**

> A: Perplexity is `exp(average cross-entropy loss)`. It measures how "surprised" the model is on average by the test data. A perplexity of 10 means the model is as uncertain as if it were choosing uniformly from 10 equally likely options at each step. Lower perplexity = better language model. For reference, GPT-2 has perplexity ~35 on WikiText-103; GPT-3 has ~20; a model that memorized the test set would have perplexity ≈1.

---

## 8. Common Mistakes & Misconceptions

❌ **"When I call model(input_ids), I get the model's output text."**
You get logits — raw scores over the vocabulary. To get text, you need to take the argmax, decode the token ID, and in practice use `model.generate()` which handles the autoregressive loop for you.

❌ **"Loss is computed on the input sequence."**
Loss is computed on the prediction of the next token at each position. The labels are the input sequence shifted by one position. You are measuring: "given the first i tokens, how well did the model predict token i+1?"

❌ **"model.eval() is just for safety/convention."**
No. `model.eval()` disables dropout (sets all dropout probabilities to 0) and switches BatchNorm to use running statistics. If you forget this during inference, you get different (and wrong) outputs every time due to dropout randomness.

---