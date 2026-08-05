# Chapter 7 · Lesson 1 — Continued Pretraining / Domain-Adaptive Pretraining (DAPT) and Tokenizer Extension

> **Where this fits:** This lesson exists specifically because of a structural fix made to this curriculum's roadmap — DAPT and tokenizer extension were originally getting collapsed into "fine-tuning," when Chapter 5 (Lessons 2, 8, 11) established they're distinct, larger interventions for foundation-layer (Layer 1) problems. This lesson gives them the standalone treatment they need before the rest of the chapter covers fine-tuning proper.

---

## 1. Why This Comes Before Full Fine-Tuning in This Chapter's Ordering

Directly following Chapter 5, Lesson 11's intervention menu: DAPT and tokenizer extension address **foundation-layer gaps** (Chapter 5, Lesson 2) — missing domain knowledge or vocabulary coverage — which every later lesson in this chapter assumes has already been ruled out or addressed. Fine-tuning (Lesson 2 onward) teaches *behavior* on top of existing knowledge; it's not designed to teach a model facts or vocabulary it fundamentally never encountered in pretraining. Applying fine-tuning to a foundation-layer problem is a common, expensive misdiagnosis this chapter's ordering is built to prevent.

---

## 2. Continued Pretraining / DAPT — What It Actually Is

**Mechanically, DAPT is just... more pretraining** — the exact same next-token-prediction objective from Chapter 2, Lesson 1, continued from an existing checkpoint rather than from random initialization, using a domain-specific corpus instead of (or in addition to) the original general-purpose corpus.

```python
# Structurally near-identical to original pretraining code (Chapter 3),
# the key differences are: starting from a pretrained checkpoint,
# a domain-specific dataset, and a much smaller token budget

model = AutoModelForCausalLM.from_pretrained("base-model-checkpoint")

for batch in domain_corpus_dataloader:  # medical papers, legal documents, etc.
    logits = model(batch["input_ids"])
    loss = causal_lm_loss(logits, batch["input_ids"])  # Chapter 2, Lesson 1 — unchanged
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Why the learning rate needs to be meaningfully lower than original pretraining's peak LR** — this is the single most important hyperparameter difference, worth flagging now since Chapter 8 covers it in full: continuing training from an already-converged checkpoint with too high a learning rate risks catastrophically overwriting the general capabilities the model already has, in favor of overfitting narrowly to the smaller domain corpus. This is the same catastrophic forgetting concern that Lesson 2 covers for full fine-tuning, but the risk is arguably even more pronounced here, since DAPT often uses a domain corpus that's stylistically and topically much narrower than the original diverse pretraining mix.

---

## 3. Tokenizer Extension — A Structurally Different, More Invasive Change

Unlike DAPT (same architecture, same tokenizer, just more training), tokenizer extension actually **changes the vocabulary itself** — adding new tokens for domain-specific terms that were previously fragmented into many subword pieces (Chapter 5, Lesson 2's fertility problem).

**The mechanical steps, worth knowing concretely:**

1. **Identify candidate new tokens** — commonly done by analyzing domain corpus statistics for frequently-occurring multi-subword sequences that would benefit from becoming a single token (similar in spirit to how the original tokenizer was trained via BPE merges, Chapter 1, but now targeted at the domain gap specifically).
2. **Extend the tokenizer's vocabulary** with these new tokens, and correspondingly **resize the model's embedding matrix and output projection layer** to add rows/columns for the new token IDs.
3. **Initialize the new tokens' embeddings** — commonly done by averaging the embeddings of the subword pieces the new token replaces (a reasonable prior, since the new token's meaning is closely related to what those pieces represented), rather than random initialization, which would give the new tokens no useful starting representation at all.
4. **Continue training (typically alongside DAPT)** — the new tokens' embeddings, even with a reasonable initialization, need actual training exposure to become well-calibrated, which is why tokenizer extension is very rarely done in isolation from a DAPT-style continued training phase.

```python
# Conceptual illustration of embedding resize and smart initialization
new_tokens = ["acetylsalicylic", "myocardial", "electrocardiogram"]  # example domain terms
num_added = tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

with torch.no_grad():
    for token in new_tokens:
        old_subword_ids = old_tokenizer.encode(token, add_special_tokens=False)
        old_embeddings = model.get_input_embeddings().weight[old_subword_ids]
        new_token_id = tokenizer.convert_tokens_to_ids(token)
        model.get_input_embeddings().weight[new_token_id] = old_embeddings.mean(dim=0)
        # Reasonable starting point — averaging the fragments' embeddings —
        # not a substitute for the continued training that follows
```

---

## 4. Why These Two Are Usually Paired, and the Cost Profile of Doing So

Directly connecting back to Chapter 5, Lesson 2's finding that vocabulary mismatch and data distribution mismatch frequently co-occur: tokenizer extension alone (without DAPT) leaves the new tokens under-trained — a resized embedding with only an averaged initialization and no further training exposure is a weak starting point, not a finished fix. DAPT alone (without tokenizer extension), when the vocabulary mismatch is severe, still leaves the model working with fragmented, inefficient token sequences for domain terms even after additional training — the underlying representational limitation from fertility (Chapter 5, Lesson 2) isn't addressed by more training alone if the tokenizer itself is the bottleneck.

**The cost, stated plainly, since Chapter 5 Lesson 11 flagged this as the most expensive tier of intervention:** both together require meaningful compute (a real continued-pretraining run, not a quick fine-tune), a genuinely large, high-quality domain corpus (insufficient domain data undermines the whole exercise), and careful hyperparameter choices (Section 2's LR concern) to avoid catastrophic forgetting of general capability. This is why Chapter 5's decision tree treats this pairing as justified only when the domain need is large and permanent enough to amortize that cost — a one-off or moderate need is usually better served by RAG (Chapter 10) instead.

---

## 5. Diagnosis & Mental Models: Signs DAPT/Tokenizer Extension Was the Right or Wrong Call

- **Right call, working as intended:** fertility ratio (Chapter 5, Lesson 2) measurably drops on domain text after tokenizer extension; held-out loss on domain-specific validation text improves substantially; general-capability evals (Chapter 6) on non-domain tasks remain stable, not regressed.
- **Wrong call, or executed poorly:** general-capability evals show meaningful regression after DAPT — a catastrophic forgetting signature (Section 2's LR warning likely wasn't heeded, or the domain corpus was too narrow relative to the training budget spent on it); or fertility on domain text doesn't meaningfully improve despite tokenizer extension — a sign the new-token selection process (Section 3, step 1) didn't actually target the right vocabulary gaps.

---

## Key Takeaways

- DAPT is mechanically identical to original pretraining, just continued from a checkpoint on a domain-specific corpus, with a critically lower learning rate to avoid catastrophic forgetting.
- Tokenizer extension is a structurally different, more invasive change — new vocabulary, resized embeddings, and a smart (not random) initialization strategy for the new tokens.
- These are usually paired, since tokenizer extension without further training leaves new tokens under-calibrated, and DAPT alone doesn't fix a genuine fertility/vocabulary bottleneck.
- Both are the most expensive tier of intervention on Chapter 5's menu — justified specifically when the domain need is large and permanent, with RAG as the usual alternative for smaller or more transient needs.

---

## Self-Check Before Moving to Lesson 2

1. Explain why DAPT typically needs a meaningfully lower learning rate than original pretraining.
2. Walk through the four mechanical steps of tokenizer extension from memory.
3. Why are DAPT and tokenizer extension usually done together rather than independently?
4. What evaluation signature (per Section 5) would indicate that a DAPT run caused catastrophic forgetting rather than successfully adding domain capability?