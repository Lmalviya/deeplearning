# Chapter 6 · Lesson 2 — Benchmark Evals: MMLU, HellaSwag, GSM8K, and Their Known Failure Modes

> **Where this fits:** Lesson 1 established perplexity's narrow scope. This lesson covers the next rung up — standardized benchmarks that attempt to measure something closer to capability — and, more importantly, the specific, well-documented ways these benchmarks can mislead you if used naively.

---

## 1. What These Benchmarks Actually Test — Briefly, By Design

Not the focus of this lesson (their content is widely documented), but worth a quick orientation since the failure modes below depend on understanding what's being measured:

- **MMLU (Massive Multitask Language Understanding):** multiple-choice questions across dozens of academic and professional subjects — a broad knowledge/reasoning breadth test.
- **HellaSwag:** commonsense sentence-completion — given a scenario, pick the most plausible continuation from several options, testing commonsense inference.
- **GSM8K:** grade-school-level math word problems requiring multi-step arithmetic reasoning — directly relevant to Chapter 5, Lesson 5's reasoning-capability territory.

**The important framing:** these are all narrow, specific proxies. A model's MMLU score is not "how smart is this model" — it's "how well does this model perform on this specific set of multiple-choice academic questions," which is a much narrower and more gameable claim than the informal way benchmark scores often get discussed.

---

## 2. Contamination — The Single Biggest Threat to Benchmark Validity

**The core problem:** benchmark test sets are public text on the internet. If a benchmark's actual test questions (or very close paraphrases) appear anywhere in a model's pretraining corpus, the model can score well by having effectively memorized answers, not by having the underlying capability the benchmark was designed to measure.

**Why this is a bigger problem than it might sound:** pretraining corpora are typically scraped from a huge fraction of the public internet (Chapter 1), and popular benchmarks — precisely because they're popular and widely discussed, cited, and reproduced in blog posts, papers, and forums — are disproportionately likely to appear, sometimes repeatedly, in that scraped data, compared to a random sample of internet text.

**Detection methods, worth knowing concretely:**
- **N-gram overlap analysis:** search the pretraining corpus for exact or near-exact matches of benchmark question text — a direct, blunt-force detection method, though it can miss paraphrased contamination.
- **Performance-on-canary comparison:** some benchmark releases include deliberately inserted "canary" strings specifically to detect whether a corpus scraper has ingested the benchmark file wholesale.
- **Behavioral tests:** compare model performance on the original benchmark versus a freshly-constructed, structurally similar but genuinely novel test set (same format, new questions) — a large gap between the two is strong evidence of contamination on the original.

---

## 3. Gaming and Overfitting to Benchmarks — A Related but Distinct Problem

Even without literal contamination, benchmarks can be gamed in subtler ways:

- **Training-data curation targeting benchmark style:** deliberately or inadvertently weighting training data toward the specific question formats, phrasing conventions, or topic distributions that popular benchmarks use, without genuinely improving the broader capability the benchmark was meant to proxy for.
- **Multiple-choice format artifacts:** models can sometimes exploit statistical patterns in multiple-choice answer distributions (e.g., certain answer-position biases, or surface-level cues in wrong-answer construction) without engaging with the actual content — a well-known issue across several multiple-choice-format benchmarks historically.
- **Benchmark-specific prompt engineering:** heavily tuning the exact prompt template and few-shot examples used to *evaluate* a model on a specific benchmark, in a way that wouldn't generalize to how the model is actually used in production — inflating the reported number without reflecting real-world capability.

---

## 4. Worked Example: A Skeptical Read of a Benchmark Claim

Suppose a vendor claims their model scores 88% on GSM8K, up from a previous version's 75%. A properly skeptical evaluation, applying Sections 2-3:

1. **Check for contamination disclosure** — did the vendor publish a decontamination methodology (n-gram overlap filtering, canary checks) for their training data specifically with respect to GSM8K? Absence of any stated methodology is itself a signal worth noting, not proof of contamination, but a reason to weight the claim more cautiously.
2. **Check whether the improvement generalizes** — does the same model show a comparable improvement on a structurally similar but different math-reasoning benchmark, or a freshly constructed set of similar problems? If the GSM8K jump is a large outlier relative to improvement on other reasoning benchmarks, that's a specific, checkable red flag for either contamination or narrow overfitting (Section 3).
3. **Connect back to Chapter 5's actual diagnostic tools** — a genuinely improved reasoning capability should also show up in Chapter 5 Lesson 5's perturbation test (does the model correctly adapt to a small change in a novel problem, not just perform well on known benchmark-style problems). This is a much stronger, harder-to-game signal than a single aggregate benchmark percentage, precisely because it's constructed fresh and targets the underlying mechanism rather than a fixed, potentially-memorized question set.

---

## 5. The Practical Takeaway for How to Use These Benchmarks

- **Use published benchmark scores as a rough, first-pass filter** when comparing models you haven't tested yourself — not as a final verdict.
- **Always prefer a benchmark score accompanied by a stated decontamination methodology** over one without, all else equal.
- **Never rely on a benchmark score alone for a capability that matters to your specific use case** — Chapter 5's capability-specific diagnostic techniques (perturbation testing, needle-in-a-haystack, fertility measurement) are harder to game because they're constructed fresh against the model in question, not drawn from a fixed, potentially-memorized public set.

---

## Key Takeaways

- MMLU, HellaSwag, and GSM8K are narrow, specific proxies for broader capabilities, not direct measures of general intelligence or reasoning — treat scores accordingly.
- Contamination (benchmark text appearing in pretraining data) is the single biggest threat to benchmark validity, given how much of the internet ends up in pretraining corpora and how disproportionately represented popular benchmarks are within it.
- Gaming/overfitting is a related but distinct problem — training data or prompting can be tuned toward benchmark-specific patterns without improving the underlying capability.
- A skeptical read of any benchmark claim checks for stated decontamination methodology and generalization to fresh, structurally similar tests — exactly the kind of fresh-construction testing Chapter 5 built throughout.

---

## Self-Check Before Moving to Lesson 3

1. Explain why popular benchmarks are disproportionately likely to be contaminated compared to a random sample of internet text.
2. Name two detection methods for contamination and explain what each one actually checks for.
3. A model shows a large GSM8K improvement but only a small improvement on a freshly constructed, similarly-styled math benchmark. What does this discrepancy suggest, and which Chapter 5 technique would you reach for to investigate further?