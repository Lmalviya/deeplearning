# Chapter 5 · Lesson 2 — Data Distribution & Vocabulary Mismatch

> **Where this fits:** This is Layer 1 of Lesson 1's taxonomy — the foundation check that should happen before investigating any specific behavioral capability. It's also the direct diagnostic version of the exact example you raised: "model trained on general web data but used in medical domain."

---

## 1. Two Genuinely Different Problems Hiding Under One Label

"Distribution mismatch" is often used loosely to mean one thing, but it's actually two separable problems with different symptoms and different fixes:

| | Data distribution mismatch | Vocabulary/tokenizer mismatch |
|---|---|---|
| What's wrong | The model was never exposed to this *content* (domain facts, style, terminology usage patterns) during pretraining | The *tokenizer* itself wasn't built with this domain's vocabulary in mind — specialized terms get fragmented into many small subword pieces |
| Symptom signature | Plausible-sounding but factually wrong outputs on domain content | Degraded fluency/coherence specifically around domain-specific terms, even when the model "knows" the concept |
| Typically fixed by | Continued pretraining / DAPT (Chapter 7, Lesson 1), or RAG | Tokenizer extension (also Chapter 7, Lesson 1) — a different, more invasive intervention |

These frequently co-occur (a new domain often has both unfamiliar content *and* unfamiliar vocabulary), but they're diagnosed separately, and worth being able to name as separate problems in an interview rather than one vague "distribution shift" catch-all.

---

## 2. Diagnosing Vocabulary Mismatch — Tokenizer Fertility as a Measurable Signal

**Fertility** = average number of tokens produced per word (or per some standard unit of text) for a given domain's text. This is a concrete, measurable number, not a vague impression.

**Worked example.** Compare fertility for the same tokenizer across two domains:

```
General English text:        "the patient was administered"
                              → ["the", " patient", " was", " administered"]  → 4 tokens, 4 words → fertility ≈ 1.0

Medical terminology text:    "the patient was administered acetylsalicylic acid"
                              → ["the", " patient", " was", " administered", " acet", "yl", "sal", "icyl", "ic", " acid"]
                              → 10 tokens, 6 words → fertility ≈ 1.67
```

A fertility spike on domain-specific terms (here, an uncommon pharmaceutical name being split into 5 subword fragments instead of 1-2) is a direct, measurable symptom of vocabulary mismatch — not a guess. Production tokenizer analysis compares average fertility on a domain corpus against fertility on the tokenizer's original general-purpose training corpus; a large gap is the diagnostic signal.

```python
def measure_fertility(tokenizer, text):
    words = text.split()
    tokens = tokenizer.encode(text)
    return len(tokens) / len(words)

general_fertility = measure_fertility(tokenizer, general_english_sample)
domain_fertility = measure_fertility(tokenizer, medical_domain_sample)
fertility_ratio = domain_fertility / general_fertility
# A ratio meaningfully above 1.0 (e.g. > 1.3-1.5) is a real signal of vocabulary mismatch,
# not just an artifact of domain text being naturally longer
```

**Why high fertility is actually harmful, not just an efficiency nuisance:** every subword fragment consumes a position in the model's limited context window and attention computation, and — more importantly for capability — it forces the model to *reconstruct* a familiar concept from unfamiliar fragment sequences it saw rarely (or never, in this exact fragmentation) during pretraining, rather than working with a token that reliably represents the concept as a whole. This is a genuine capability tax, not just a cost/latency one.

---

## 3. Diagnosing Data Distribution Mismatch — Separately From Vocabulary

Even with a well-fitted tokenizer, a model can lack genuine exposure to a domain's *content*. The diagnostic signal here is different from fertility — it's about factual reliability and stylistic fluency, not tokenization efficiency.

**A concrete diagnostic test:** ask the model to generate domain content *without* any retrieval augmentation, across several independent prompts, and check for:
- **Confident fabrication** — specific-sounding but incorrect facts (a strong signal of a genuine content gap, since the model has clearly learned the *style* of confident domain writing without the underlying accurate content)
- **Consistency of errors across rephrasing** — if the same underlying question, asked three different ways, produces three different wrong answers, that's more consistent with "never learned this content" than with a narrower retrieval or formatting issue (connects back to Lesson 1, Section 2's four-candidate example)
- **Comparison against a retrieval-augmented version of the same prompt** — if providing the correct facts directly in context fixes the answer, the problem is a content/knowledge gap specifically (fixable by RAG or DAPT), not an instruction-following gap (which Lesson 3 covers) or reasoning gap (Lesson 5)

---

## 4. Worked Example: Full Diagnostic Walkthrough, Your Original Scenario

Applying this lesson directly to the exact case you raised — "model trained on general web data but used in medical domain":

1. **Measure fertility** (Section 2) on a sample of real medical terminology from the target use case. Suppose it comes back meaningfully elevated (say, 1.6x general-domain fertility) — vocabulary mismatch is present.
2. **Run the content-gap test** (Section 3) — ask the model medical questions without retrieval augmentation. Suppose it produces confident, specific-sounding, but frequently incorrect answers, and RAG-augmented versions of the same prompts are substantially more accurate — data distribution mismatch is also present.
3. **Both are present simultaneously** — this is common, not a sign the diagnosis went wrong. The intervention decision (Lesson 11, the full decision tree) would then need to weigh: is a full tokenizer extension + DAPT justified by the volume and permanence of medical-domain traffic, or is RAG alone (which sidesteps the content gap without touching the vocabulary problem) sufficient for the actual production need? That's a real cost/benefit call, not a purely technical one — and naming that tradeoff explicitly is what turns a diagnosis into a decision.

---

## 5. Diagnosis & Mental Models: When This Layer Is (and Isn't) the Explanation

- **If fertility is normal AND the content-gap test shows accurate, consistent answers when relevant facts are provided in-context** → Layer 1 is very likely *not* the explanation for whatever symptom prompted the investigation; move to Layer 2 (Lessons 3-9).
- **If fertility is elevated but the content-gap test shows accurate answers** → likely a pure tokenization/efficiency issue rather than a capability issue — worth fixing for cost/latency reasons, but not the explanation for a correctness-related symptom.
- **If both signals are clean but the original symptom persists** → strong evidence the problem lives in a specific Layer 2 capability or even in the surrounding system (retrieval pipeline, formatting) rather than the model's foundational training at all — a useful negative result, not a dead end.

---

## Key Takeaways

- "Distribution mismatch" is really two separable problems — vocabulary/tokenizer mismatch and data content mismatch — with different measurable signals and different fixes.
- Tokenizer fertility is a concrete, computable metric for vocabulary mismatch, not a subjective impression.
- Confident-but-wrong answers that improve substantially with in-context retrieval are the diagnostic signature of a content gap specifically, distinguishing it from instruction-following or reasoning gaps covered in later lessons.
- Both problems commonly co-occur, and the intervention decision after diagnosis is a real cost/benefit tradeoff, not automatic.

---

## Self-Check Before Moving to Lesson 3

1. Explain the difference between vocabulary mismatch and data distribution mismatch using a concrete example other than the medical one used here.
2. How would you compute a fertility ratio for a new domain, and what threshold would make you suspect a real problem versus normal variation?
3. A model gives confidently wrong answers about a legal topic, but the errors change every time you rephrase the question. Does this look more like a content gap or something else, per Section 3's reasoning? Why?