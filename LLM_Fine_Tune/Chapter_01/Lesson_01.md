# Chapter 1: How Language Models Actually Work (Under the Hood)
### LLM Fine-Tuning Mastery — ML/AI Engineering Interview Preparation

---

## How Every Lesson in This Chapter Is Structured

Each lesson follows this exact framework. Learn to expect it:

```
┌─────────────────────────────────────────────────────────────┐
│  1. THE PROBLEM STORY                                       │
│     A real situation where not knowing this hurt someone.   │
│     This answers: "Why should I care about this?"          │
│                                                             │
│  2. THE CONCEPT (Theory)                                    │
│     What it is, how it works, the math where needed.       │
│     No hand-waving. No "don't worry about the details."    │
│                                                             │
│  3. THE INTUITION BRIDGE                                    │
│     An analogy or mental model that makes it stick.        │
│     Theory without intuition evaporates by morning.        │
│                                                             │
│  4. WHY THIS MATTERS FOR FINE-TUNING                       │
│     Every concept connects back to our goal.               │
│     If it didn't affect fine-tuning, it wouldn't be here. │
│                                                             │
│  5. THE CODE (Hands-On)                                    │
│     Working, runnable code. Not pseudocode.                │
│     You run this. You read the output. You understand it.  │
│                                                             │
│  6. THE EXPERIMENT                                         │
│     A deliberate task to deepen understanding.             │
│     Not optional. This is where the learning happens.      │
│                                                             │
│  7. INTERVIEW CHECKPOINT                                   │
│     The exact questions interviewers ask about this topic. │
│     Model answers you should be able to give fluently.     │
│                                                             │
│  8. COMMON MISTAKES & MISCONCEPTIONS                       │
│     What people get wrong. What you should not say.        │
└─────────────────────────────────────────────────────────────┘
```

---

## Chapter 1 Overview

**What this chapter is about:**
You said you have strong ML/DL theory. This chapter does not review basic ML. It goes into the specific internals of language models that fine-tuning touches directly. After this chapter, when an interviewer asks "what is the model doing during fine-tuning?" — you will have a precise, layered answer.

**What you need before starting:**
- Comfortable with Python and PyTorch basics
- Understand what a neural network is
- Know what gradient descent is conceptually

**What you will be able to do after this chapter:**
- Explain exactly what happens from raw text to model output, step by step
- Load any HuggingFace model and inspect its internals with confidence
- Understand why tokenization decisions affect fine-tuning quality
- Explain autoregressive generation with precision
- Answer "when would you fine-tune vs just prompt?" with a real argument

**Lessons in this chapter:**
- Lesson 1.1 — Tokenization Deep Dive
- Lesson 1.2 — Transformer Architecture Internals
- Lesson 1.3 — The Forward Pass Step by Step
- Lesson 1.4 — Autoregressive Generation
- Lesson 1.5 — Pre-training vs Fine-tuning vs In-Context Learning

**Estimated time:** 1.5 weeks at 8–15 hrs/week

---

---

# Lesson 1.1 — Tokenization Deep Dive

---

## 1. The Problem Story

Arjun fine-tuned a model for customer support in Hindi. The results were terrible — the model kept producing garbled outputs and mixing languages randomly. His loss looked fine during training. His code had no bugs.

The actual problem? He used a tokenizer trained entirely on English text. In that tokenizer, a common Hindi word like "नमस्ते" (namaste) was split into 6–8 individual byte-level tokens, each meaningless on its own. The model had no efficient way to represent Hindi concepts — it was trying to learn a language through broken, fragmented pieces.

He wasted two weeks on architecture changes and hyperparameter tuning before someone pointed at his tokenizer.

**The lesson:** Tokenization is the first thing that touches your data. A mismatch here poisons everything downstream — and it is invisible in your training loss until it is too late.

---

## 2. The Concept

### What is a tokenizer?

Raw text is a sequence of characters. Neural networks work with numbers. A tokenizer is the bridge — it converts a string of characters into a sequence of integer IDs, where each ID represents a "token."

A token is not necessarily a word. It could be:
- A whole word: `"hello"` → `[15339]`
- A subword: `"fine-tuning"` → `["fine", "-", "tun", "ing"]` → `[2986, 12, 83193, 278]`
- A character: `"a"` → `[64]`
- A byte: for characters outside the vocabulary

### The Three Main Tokenization Algorithms

**1. Byte Pair Encoding (BPE)**

Used by: GPT-2, GPT-3, GPT-4, LLaMA, Mistral, most modern LLMs

How it works:
- Start with individual characters as the vocabulary
- Count the most frequent pair of adjacent tokens in the corpus
- Merge that pair into a single new token
- Repeat until you reach your target vocabulary size (e.g., 32,000 or 128,000 tokens)

Example process:
```
Start: ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
Most frequent pair: ("l", "l") → merge to "ll"
Now:   ["h", "e", "ll", "o", " ", "w", "o", "r", "l", "d"]
Most frequent pair: ("o", " ") → merge to "o "
... and so on for thousands of iterations
Final: "hello world" → ["hello", " world"]  (if these became tokens)
```

The vocabulary is built from the training corpus. Tokens that appear frequently in the training data get their own ID. Rare words get broken into subwords.

**2. WordPiece**

Used by: BERT, DistilBERT, ALBERT

Very similar to BPE but with a different merge criterion. Instead of merging the most frequent pair, it merges the pair that maximizes the likelihood of the training data. In practice, results are similar to BPE.

WordPiece selects the pair with the highest likelihood-inspired score:

$$
\text{score}(a,b) =
\frac{\text{freq}(a,b)}
{\text{freq}(a)\,\text{freq}(b)}
$$

This favors pairs that occur together much more often than would be expected from their individual frequencies.

Therefore, WordPiece prefers token pairs that have a strong statistical association rather than simply appearing frequently.

WordPiece marks subword pieces with `##` prefix:
```
"fine-tuning" → ["fine", "-", "##tun", "##ing"]
```

**3. SentencePiece**

Used by: T5, LLaMA, Gemma, many multilingual models

Key difference: treats the input as a raw byte stream, not pre-tokenized words. It handles spaces as explicit characters (represented as `▁`). This makes it language-agnostic and handles any language or script without word-level pre-tokenization.

```
"hello world" → ["▁hello", "▁world"]
"नमस्ते" → ["▁न", "मस्", "ते"]  (if Hindi is in the training vocab)
```

### The Vocabulary

The tokenizer has a fixed vocabulary: a lookup table from token string → integer ID.

```
"hello"  → 15339
"world"  → 1917
"▁the"   → 278
" "      → 29871
[BOS]    → 1
[EOS]    → 2
[PAD]    → 0
```

Vocabulary size varies:
- GPT-2: 50,257 tokens
- LLaMA-2: 32,000 tokens
- LLaMA-3: 128,256 tokens (much larger, handles more languages)
- Gemma: 256,128 tokens

### Special Tokens

Every tokenizer has special tokens with fixed IDs:

| Token | Meaning | Example models |
|-------|---------|----------------|
| `[BOS]` / `<s>` | Beginning of sequence | LLaMA, Mistral |
| `[EOS]` / `</s>` | End of sequence | LLaMA, Mistral |
| `[PAD]` | Padding (to make sequences same length) | Most models |
| `[UNK]` | Unknown token (rare in BPE) | BERT-style |
| `[SEP]` | Separator between segments | BERT |
| `[MASK]` | Masked token for MLM training | BERT |
| `<|endoftext|>` | End of text | GPT-2 |
| `<|im_start|>` | Start of a ChatML turn | ChatML format |

For fine-tuning, you often add custom special tokens:
```
[INST], [/INST]   → LLaMA-2 chat format
<|system|>        → Phi-3 chat format
```

### What tokenization produces

When you tokenize a string, you get:
- `input_ids`: list of integer token IDs
- `attention_mask`: 1 for real tokens, 0 for padding tokens
- (sometimes) `token_type_ids`: for models like BERT that distinguish segment A vs B

```python
tokenizer("Hello world")
# {
#   'input_ids': [1, 15043, 3186, 2],
#   'attention_mask': [1, 1, 1, 1]
# }
```

---

## 3. Why This Matters for Fine-Tuning

**Reason 1: Token efficiency affects training cost**

If a sentence in your domain takes 200 tokens instead of 50 (because of poor vocabulary coverage), you are fitting fewer examples in each batch. Your training is 4x less efficient without changing anything else.

**Reason 2: Vocabulary coverage affects model quality**

Words that are rare in the tokenizer's training corpus get split into meaningless pieces. The model has to learn that `["▁fin", "e-", "tun", "ing"]` means fine-tuning, rather than having a single token for it. This makes learning harder.

**Reason 3: Special tokens define the format**

When you fine-tune a chat model, the format of your data (system message, user message, assistant response) is encoded with special tokens. If you format these wrong, the model does not learn what you want it to learn. We will return to this in Chapter 2.

**Reason 4: Max sequence length is in tokens, not characters**

When you set `max_length=2048`, that is 2048 tokens. Code might use 3–4 tokens per word (due to spaces, symbols). Hindi might use 6–8 tokens per word. You need to know your token-to-character ratio for your data.

---

## 4. The Code

Run this yourself. Read every output line. Do not copy-paste and move on.

```python
# Install if needed: pip install transformers

from transformers import AutoTokenizer

# Load a tokenizer (LLaMA-3 style, uses SentencePiece)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

# ── Basic tokenization ──────────────────────────────────────────

text = "Fine-tuning large language models is challenging."
tokens = tokenizer(text)

print("Input text:", text)
print("Input IDs:", tokens["input_ids"])
print("Attention mask:", tokens["attention_mask"])
print("Number of tokens:", len(tokens["input_ids"]))

# ── Convert IDs back to tokens (strings) ────────────────────────

token_strings = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
print("\nToken strings:", token_strings)

# ── See what each word becomes ──────────────────────────────────

words = ["fine-tuning", "transformer", "LoRA", "gradient", "backpropagation",
         "नमस्ते", "中文", "código"]

print("\n── Token count per word ──")
for word in words:
    ids = tokenizer.encode(word, add_special_tokens=False)
    toks = tokenizer.convert_ids_to_tokens(ids)
    print(f"  '{word}' → {len(ids)} tokens → {toks}")

# ── Special tokens ──────────────────────────────────────────────

print("\n── Special tokens ──")
print("BOS token:", tokenizer.bos_token, "| ID:", tokenizer.bos_token_id)
print("EOS token:", tokenizer.eos_token, "| ID:", tokenizer.eos_token_id)
print("PAD token:", tokenizer.pad_token, "| ID:", tokenizer.pad_token_id)

# ── Padding and attention mask ──────────────────────────────────

texts = [
    "Short text.",
    "This is a much longer piece of text that will need more tokens."
]

batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print("\n── Batched tokenization (with padding) ──")
print("Input IDs shape:", batch["input_ids"].shape)
print("Input IDs:\n", batch["input_ids"])
print("Attention mask:\n", batch["attention_mask"])

# ── Decoding: convert IDs back to text ──────────────────────────

ids = [15043, 3186]
decoded = tokenizer.decode(ids)
print("\nDecoded:", decoded)

# ── Vocabulary size ─────────────────────────────────────────────

print("\nVocabulary size:", tokenizer.vocab_size)
```

**Expected things to notice:**
- Multi-lingual words take significantly more tokens than English words
- Padding fills shorter sequences with pad token IDs, and attention mask marks those as 0
- The BOS token is automatically added at the start when `add_special_tokens=True` (default)
- Decoding converts IDs back to readable text

---

## 5. The Experiment

**Experiment 1.1.A — Token Efficiency Audit**

Take any dataset you care about (or use a sample from HuggingFace `datasets`). Pick 100 examples. Compute:
1. Average tokens per example
2. Token-to-character ratio
3. What percentage of examples exceed 512 tokens? 1024 tokens? 2048 tokens?
4. What are the top 20 most frequent tokens in your dataset?

This is called a "token efficiency audit" and it is something you should do before every fine-tuning project.

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")

token_counts = []
all_tokens = []

for example in dataset:
    text = example["instruction"] + " " + example["output"]
    ids = tokenizer.encode(text, add_special_tokens=False)
    token_counts.append(len(ids))
    all_tokens.extend(ids)

print(f"Average tokens per example: {np.mean(token_counts):.1f}")
print(f"Max tokens: {max(token_counts)}")
print(f"Min tokens: {min(token_counts)}")
print(f"% > 512 tokens: {sum(c > 512 for c in token_counts)}%")
print(f"% > 1024 tokens: {sum(c > 1024 for c in token_counts)}%")

top_tokens = Counter(all_tokens).most_common(20)
print("\nTop 20 most frequent tokens:")
for token_id, count in top_tokens:
    print(f"  '{tokenizer.decode([token_id])}' → {count} times")
```

**Write down your observations. What surprised you?**

---

## 6. Interview Checkpoint

**Q: What is a token? Why don't we just use characters or words?**

> A: A token is a unit of text that the model processes. Using individual characters means very long sequences and difficulty learning word-level meaning. Using whole words means a vocabulary of hundreds of thousands of entries (every word form gets its own ID), and new or rare words are completely unseen. Subword tokenization (BPE, SentencePiece) is a middle ground: common words are single tokens, rare words are split into recognizable pieces. This gives a manageable vocabulary while handling any text gracefully.

**Q: Why do some tokens cost more than others? (Common in GPT API questions)**

> A: "Cost" refers to API pricing, but the mechanism is token count. A Hindi sentence might take 3–4x more tokens than an equivalent English sentence because the tokenizer was trained mostly on English text. Hindi words therefore get split into many small subwords. More tokens means more compute, higher cost, and slower generation.

**Q: You used a tokenizer from a pre-trained model. What would happen if you trained a new tokenizer on your domain data and used that?**

> A: The model weights are tied to the original tokenizer's vocabulary. If you change the vocabulary, the embedding table dimensions change, and all the pre-trained weights become invalid — you are essentially starting from scratch. You would need to either retrain from scratch with the new tokenizer, or use techniques like vocabulary extension (adding new tokens while keeping existing ones). In most fine-tuning cases, you use the same tokenizer as the pre-trained model.

---

## 7. Common Mistakes & Misconceptions

❌ **"A token is a word."**
Wrong. "fine-tuning" might be 3–4 tokens. " the" (with a space) is a different token from "the". Numbers are often split digit by digit.

❌ **"The attention mask does not matter for fine-tuning."**
Wrong. The attention mask tells the model which positions to attend to. If you set it incorrectly, the model will attend to padding tokens and learn noise.

❌ **"I can use any tokenizer with any model."**
Wrong. A model's embedding table has exactly `vocab_size` rows, one per token ID. If you use a different tokenizer, the IDs will mean different things, and the model's learned embeddings will be completely wrong.

❌ **"Adding a special token is just about formatting."**
Not quite. When you add a new special token, it gets a new embedding initialized randomly. You need to train that embedding. This is why adding many custom tokens and fine-tuning on a small dataset can hurt — there is not enough data to learn the new embeddings.

