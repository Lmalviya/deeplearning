# Chapter 6 · Lesson 4 — Fine-Tuning/Alignment Evals: Win-Rate Judging and LLM-as-Judge Pitfalls

> **Where this fits:** Lessons 1-3 covered evals with objectively checkable answers (loss, multiple-choice, execution-based correctness). This lesson covers the harder case — evaluating open-ended generation quality, where "correct" often isn't binary, which is exactly the situation fine-tuning and alignment work (Chapters 7-9) most often needs to evaluate.

---

## 1. Why Open-Ended Generation Needs a Different Evaluation Approach

A fine-tuned chat model's response to "write a professional email declining a meeting" doesn't have one objectively correct answer the way a GSM8K problem does. Evaluating quality here requires either human judgment or a proxy for it — this is the entire reason win-rate judging and LLM-as-judge methods exist.

---

## 2. Win-Rate Judging — The Basic Mechanism

Instead of scoring a single response in isolation, present a judge (human or model) with **two responses to the same prompt** — one from the model being evaluated, one from a reference/baseline model — and ask which is better, or record a tie.

```
Prompt: "Write a professional email declining a meeting."
Response A (model under test): [response]
Response B (reference model): [response]
Judge: A is better / B is better / Tie
```

Aggregate across many prompts to get a **win rate** — the percentage of comparisons where the model under test was preferred over the reference.

**Why pairwise comparison is generally more reliable than absolute scoring (e.g., "rate this response 1-10"):** absolute scores are more prone to inconsistent scales across different judges or even the same judge at different times (what one judge calls a "7" another calls a "5," for reasons unrelated to actual quality) — a relative judgment ("which of these two is better") is a cognitively easier, more consistent task, a well-documented finding in both human-evaluation and LLM-judge literature.

---

## 3. LLM-as-Judge — The Practical Default at Scale, and Its Specific Pitfalls

Human evaluation is expensive and slow; using a strong LLM to perform the pairwise comparison from Section 2 is the common practical substitute at scale. This introduces its own, well-documented biases worth knowing specifically:

| Bias | What it looks like | Mitigation |
|---|---|---|
| Position bias | The judge favors whichever response is presented first (or second), independent of actual quality | Run each comparison twice with response order swapped, and treat inconsistent results as a tie or discard |
| Verbosity bias | The judge systematically favors longer responses, treating length as a proxy for thoroughness even when the longer response isn't actually better | Explicitly instruct the judge to evaluate conciseness/relevance, or normalize for length in analysis |
| Self-preference bias | An LLM judge tends to favor responses that are stylistically similar to its own outputs (e.g., a GPT-family judge slightly favoring GPT-style phrasing over a differently-styled but equally good response) | Use a judge model from a different family/lineage than either model being compared, where feasible, or corroborate with human spot-checks |
| Sycophancy toward the prompt's apparent framing | If the prompt or surrounding context hints at which response "should" be better, the judge can be swayed by that framing rather than judging independently | Keep judge prompts neutral, blind the judge to which response is the "model under test" vs. "reference" |

**Why this table matters for a credible interview answer:** simply saying "I used an LLM as a judge" without acknowledging these specific, named biases is a shallow answer — a strong one names at least position bias and verbosity bias unprompted, and describes a concrete mitigation (e.g., order-swapping) rather than treating LLM-judge output as ground truth.

---

## 4. Validating the Judge Itself — A Step Often Skipped

Before trusting an LLM judge's verdicts at scale, a real validation step: have the LLM judge and human evaluators independently score the same sample of comparisons, and measure agreement (e.g., what fraction of the time do they reach the same verdict). **Low agreement is a signal the judge isn't a reliable proxy for the human evaluation it's meant to substitute for** — and this check should happen before running a large-scale LLM-judge evaluation, not after results are already being used to make decisions.

```python
def judge_agreement_rate(human_verdicts, llm_verdicts):
    """
    human_verdicts, llm_verdicts: lists of 'A' / 'B' / 'tie' for the same
    set of paired comparisons, same order
    """
    agree = sum(h == l for h, l in zip(human_verdicts, llm_verdicts))
    return agree / len(human_verdicts)

# A commonly cited rough bar: agreement rates well above random chance
# (which for a 3-way A/B/tie judgment is 33%) and reasonably close to
# typical human-human inter-annotator agreement rates on the same task
# is the kind of evidence that justifies trusting the LLM judge at scale.
```

---

## 5. Worked Example: Designing a Fine-Tuning Eval With These Tools

Say you've fine-tuned a customer-support model and need to know if it's actually better than the pre-fine-tune baseline.

1. **Build a representative prompt set** — real or realistic customer queries, ideally including some from Chapter 5's capability-specific test categories (tool-use scenarios, structured-output scenarios) so the eval reflects actual production capability needs, not just generic chat quality.
2. **Run pairwise win-rate comparisons** (Section 2) between the fine-tuned model and the baseline, using an LLM judge with order-swapping to control for position bias (Section 3).
3. **Validate the judge** (Section 4) on a small human-scored subset before trusting the full-scale results.
4. **Report win rate with the tie rate shown separately**, not collapsed into the win rate — a model winning 40% and losing 20% with 40% ties is a very different, more nuanced result than a naive "67% win rate" (computed only over non-ties) would suggest, and collapsing ties out of the reported number can be misleading.

---

## Key Takeaways

- Open-ended generation quality generally requires comparative (pairwise) judgment rather than absolute scoring, since relative judgments are more consistent across judges and time.
- LLM-as-judge is the practical default at scale, but carries specific, well-documented biases — position, verbosity, self-preference, framing sycophancy — each with a concrete mitigation.
- The judge itself should be validated against human judgment on a sample before being trusted at scale — skipping this step is a common, avoidable weakness in fine-tuning eval pipelines.
- Reporting win rate alongside tie rate (not collapsed together) gives a more honest picture than a single number.

---

## Self-Check Before Moving to Lesson 5

1. Name three specific LLM-as-judge biases and a concrete mitigation for each, without looking back.
2. Why is pairwise comparison generally more reliable than absolute 1-10 scoring for evaluating open-ended generation?
3. Explain why validating the judge against human judgment matters, and what a low agreement rate would imply about trusting the judge's results.