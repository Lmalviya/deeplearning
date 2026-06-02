# Lesson 8.3 — LLM-as-Judge Evaluation: Win Rates, Scoring Rubrics, and Bias

---

## The Problem With Automated Metrics for Open-Ended Output

Once you have an instruction-tuned model generating free-form text, the evaluation problem changes fundamentally. Closed-ended benchmarks (MCQ, pass@k for code) have a ground truth. But for open-ended generation — responses to questions, summaries, explanations, creative writing — there is no single correct answer.

Traditional NLP metrics like BLEU, ROUGE, and BERTScore compare model output to a reference answer. They work if your reference is the only valid answer. For instruction-tuned models, many different responses can be equally good or better than the reference. A response that scores 0.2 on ROUGE-L can be far superior to one scoring 0.9 if the high-ROUGE response is verbose, evasive, or poorly organized.

Human evaluation is the gold standard — but it is slow, expensive, and difficult to scale. You cannot run human evaluation after every training step.

**LLM-as-judge** is the practical solution: use a capable LLM (GPT-4, Claude) to evaluate model outputs at scale, with reasonable correlation to human preferences.

---

## Two Evaluation Patterns

### Pattern 1: Pairwise Comparison (Win Rate)

Show the judge model two responses — Response A and Response B — to the same prompt. Ask which is better. Measure the **win rate**: the fraction of comparisons where your model wins.

```
Prompt: "Explain how transformers work to a software engineer."

Response A: [Your model's output]
Response B: [Baseline model's output]

Judge prompt: "Given the above question, which response is better? 
              Consider correctness, clarity, and helpfulness. 
              Answer A, B, or Tie."
```

Win rate = (wins + 0.5 × ties) / total comparisons

This is used in AlpacaEval, Chatbot Arena (human judges), and many custom evaluations. Win rate against a fixed reference (e.g., GPT-4 Turbo or text-davinci-003) is a common alignment paper metric.

**Why pairwise is often better than scoring:** It is much easier for a judge (human or LLM) to say "A is better than B" than to assign a precise score from 1–10. Pairwise comparison avoids calibration issues — different judges may use the scale differently.

### Pattern 2: Absolute Scoring with Rubric

Show the judge one response and a scoring rubric. Ask for a score from 1 to 10 (or 1 to 5). This is how MT-Bench works.

```
You are evaluating an AI assistant's response.

Question: [question]
Response: [model output]

Evaluation criteria:
- Accuracy: Is the information correct? (0-3 points)
- Clarity: Is it clearly explained? (0-3 points)  
- Completeness: Does it fully address the question? (0-4 points)

Provide a score from 1 to 10 and a brief justification.
```

Absolute scoring is more informative than win rate — you get a meaningful number you can track over time rather than a relative comparison. The downside: scores are sensitive to rubric wording and judge model choice.

---

## Biases in LLM-as-Judge

This is the most critical part of the lesson. LLM-as-judge is useful but systematically biased. If you do not know the biases, you will mistake bias for quality signal.

### 1. Position Bias

When shown Response A and Response B, LLMs tend to prefer whichever response is shown first — independent of actual quality. This is not a small effect: in controlled experiments, GPT-4 chose the first response at rates significantly above 50% even when responses were identical or the second was actually better.

**Fix:** Run each comparison twice with A and B swapped. Count a win only if the model wins in both orderings. Count as a tie if the judge flips. This doubles your API cost but eliminates position bias.

### 2. Verbosity Bias

LLMs strongly prefer longer, more detailed responses — even when brevity is clearly correct or preferred. A concise, accurate 3-sentence answer will often lose to a verbose, padding-heavy 15-sentence answer.

**Fix:** Include an explicit instruction in your judge prompt: "Do not prefer responses simply because they are longer. A clear, concise response is often better than a verbose one." Also normalize by response length in your analysis — if your model is winning but its responses are 3× longer, the win rate is misleading.

### 3. Self-Preference Bias (Style Matching)

LLMs tend to prefer responses written in their own style. GPT-4-as-judge will prefer GPT-4-like responses. Llama-3-as-judge will prefer Llama-3-like responses. This is particularly dangerous if you are fine-tuning your model on GPT-4 outputs and using GPT-4 as judge — you will get artificially inflated scores.

**Fix:** Use multiple judge models from different families and average. Watch for suspiciously high win rates that do not correlate with human preference.

### 4. Sycophancy

When the evaluation prompt implies a preferred answer (e.g., "Is this a good response?"), LLMs tend to agree with the implied direction. A well-crafted judge prompt should be neutral and avoid leading the judge.

**Fix:** Frame evaluation prompts as comparisons or explicit criteria-based scoring, not as "is this good?"-style questions. Blind the judge to which response came from which model.

---

## Practical Judge Prompt Structure

A well-designed judge prompt has four parts:

```
[ROLE]
You are an expert evaluator of AI assistant responses. Your task is to judge 
response quality objectively based on the criteria below.

[CRITERIA]  
Evaluate based on:
1. Factual accuracy — is the content correct and well-grounded?
2. Completeness — does the response address all aspects of the question?
3. Clarity — is the response easy to understand and well-organized?
4. Conciseness — is the response appropriately brief without padding?

[INSTRUCTION FOR BIAS AVOIDANCE]
Do not prefer a response solely because it is longer. Do not be influenced 
by the order in which responses are shown. Judge based on quality alone.

[QUESTION + RESPONSES]
Question: {question}

Response A:
{response_a}

Response B:
{response_b}

[OUTPUT FORMAT]
Which response is better? Respond with:
Winner: [A/B/Tie]
Reasoning: [One sentence explanation]
```

---

## Calibrating Against Human Labels

LLM-as-judge is only trustworthy if it correlates with human judgment. Before relying on it, calibrate:

1. Collect 100–200 human preference labels on sample comparisons from your eval set.
2. Run the same comparisons through your LLM judge.
3. Measure agreement rate (what fraction of comparisons does the judge agree with human labelers?).

GPT-4 as judge achieves ~80% agreement with human preferences on MT-Bench style evaluations. This is decent but not perfect. It means ~20% of judgments are wrong — enough to be misleading if your win rate differences are small.

**Practical threshold:** If your win rate difference is < 5%, the signal is within the noise of judge error. Only differences > 10% are reliably meaningful without additional human validation.

> **Interview note:** "How would you evaluate your instruction-tuned model?" A complete answer: "I use a combination. During training: validation loss and early stopping. After training: standard benchmarks (MT-Bench for instruction quality, IFEval for format compliance). For comparative evaluation: pairwise LLM-as-judge with GPT-4, randomizing response order to counter position bias and controlling for verbosity. I calibrate the judge against a small set of human labels to know how much to trust the win rate. I treat win rate differences below 5% as noise."

---

## When LLM-as-Judge Breaks Down

- **Factual evaluation in specialized domains:** GPT-4 cannot reliably judge whether a medical diagnosis reasoning chain is clinically correct, or whether a legal analysis cites the right precedent. Use domain-expert human judges or specialized evaluation models.
- **Code evaluation:** Use execution-based evaluation (run the code) rather than LLM judgment. LLMs confidently call wrong code "correct."
- **Safety evaluation:** LLM judges will sometimes fail to catch subtle policy violations or be overly strict on benign responses. Human review is essential for safety-critical evaluation.

---

## Summary

- LLM-as-judge uses a capable model (GPT-4, Claude) to evaluate open-ended responses where automated metrics fail. It is the practical scaling solution for instruction tuning evaluation.
- **Pairwise comparison** (win rate): show judge two responses, ask which is better. Easier to be consistent. Use AlpacaEval-style against a fixed reference.
- **Absolute scoring** (rubric-based 1–10): show judge one response with criteria. More informative over time. Used in MT-Bench.
- Critical biases: **position bias** (mitigate: swap order, require double-win), **verbosity bias** (mitigate: explicit anti-verbosity instruction, length normalization), **self-preference** (mitigate: multiple judges from different families), **sycophancy** (mitigate: neutral prompt framing).
- Always calibrate your LLM judge against human labels before trusting it. GPT-4 achieves ~80% agreement with humans on MT-Bench style tasks.
- Win rate differences below 5% are within judge noise — require human validation before drawing conclusions.

---
