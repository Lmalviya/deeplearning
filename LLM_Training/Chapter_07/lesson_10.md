# Chapter 7 · Lesson 10 — Interview Lab: Choosing and Defending a Specific Fine-Tuning Method

> **Where this fits:** This lesson is distinct from Chapter 5, Lesson 12's interview lab — that one rebuilt the broader "mental model for fine-tuning" (diagnosis-first thinking). This one is narrower and more technical: once fine-tuning is confirmed as the right intervention, can you choose a specific method, state concrete hyperparameter starting points, and defend the choice under follow-up pressure.

---

## 1. The Format This Question Usually Takes

Typically posed as a scenario: *"You need to fine-tune a [size] model to [do something specific], you have [compute constraint]. Walk me through your approach."* This is Lesson 9's flowchart, performed live, under time pressure, with a real interviewer probing the reasoning at each branch point rather than letting you state a conclusion.

---

## 2. A Full Worked Response to a Realistic Prompt

**Prompt:** *"You need to fine-tune a 7B model to reliably follow a structured output schema for an internal tool, and you have access to a single 40GB GPU. Walk me through your approach."*

> "First, I'd want to confirm this is actually a training-level problem before committing to fine-tuning at all — per Chapter 5's structured-output lesson, I'd check whether constrained decoding alone solves it, since that's near-zero cost compared to any training approach. Assuming that's been checked and ruled insufficient — say the schema is complex enough that generation quality genuinely degrades, not just occasional invalid syntax — I'd move to method selection.
>
> This is a narrow, well-defined capability gap, not a broad behavioral overhaul, so full fine-tuning is overkill — I'd reach for LoRA. A 7B model at bf16 is about 14GB, which fits comfortably in 40GB alongside adapter training overhead, so I wouldn't need QLoRA's quantization here — that's purely a memory-feasibility call, not a quality one, since QLoRA's frozen-weight quantization doesn't inherently reduce achievable quality given gradients never flow through those weights anyway.
>
> For rank, I'd start moderate — something like 16 — since this is a narrower task than a full style overhaul would need, targeting the attention projection matrices as a starting point, with alpha set to roughly twice the rank as a conventional starting point, then tune from there.
>
> For data, I'd build a structured-output-specific dataset following the same discipline as any instruction-tuning data — clean, deduplicated, correctly loss-masked so we're only training on the response portion, with schema-valid examples specifically curated rather than scraped broadly, since Chapter 7's data-triage lesson makes clear that healthy loss curves alone don't guarantee the fine-tune actually teaches the target behavior if the underlying data is flawed.
>
> Then I'd validate with the two-axis structured-output eval from Chapter 6 — format validity and content correctness scored separately — plus a regression check against the model's other capabilities, since even LoRA's smaller footprint doesn't eliminate forgetting risk entirely."

---

## 3. Why This Response Structure Works

Notice the response doesn't just answer "which method" — it walks the full Lesson 9 flowchart out loud, explicitly naming each branch point's reasoning (cheap-fix check first, scope-of-change assessment, memory-feasibility check, rank/alpha starting point, data discipline, validation plan). **This is the direct payoff of building the flowchart as a reusable tool in Lesson 9** — a strong live answer to this kind of question is essentially narrating that flowchart with the specific scenario's numbers plugged in, not improvising from scratch under pressure.

---

## 4. Follow-Up Questions to Have Pre-Loaded

**"Why not just use QLoRA anyway, to be safe on memory?"** → Direct callback to Lesson 9, Q4's reasoning: QLoRA is a response to a memory constraint that doesn't exist in this scenario (14GB comfortably fits in 40GB) — choosing it anyway adds quantization/dequantization overhead at inference and training time for no corresponding benefit, since the constraint it solves isn't present.

**"How did you decide rank 16 specifically, not 8 or 32?"** → Honest answer per Chapter 4, Lesson 7's credible-tuning-answer structure: "16 is a reasonable starting point for a narrow task based on published conventions; I'd validate it with a small sweep — comparing 8, 16, and 32 on a short run — rather than assuming 16 is optimal without evidence, especially since Chapter 7 Lesson 8 showed loss curves alone aren't sufficient validation."

**"What if the eval after training shows the format validity is good but content correctness regressed?"** → This is Chapter 5, Lesson 6's two-axis distinction directly — a regression specifically in content correctness (not format) suggests the fine-tuning data may have been too narrowly focused on schema conformance at the expense of substantive accuracy, pointing back to Lesson 7's data-curation discipline rather than a hyperparameter fix.

**"Your first LoRA run underperforms — what's your next move, in order?"** → Directly reuse Chapter 7, Lesson 8's triage flowchart: check training loss itself first (underfit vs. not), then validation divergence (overfit vs. not), then — if curves look healthy but the capability gap persists — revisit the training data itself before assuming a hyperparameter or method change is needed.

---

## 5. The Compressed Version, for a Faster-Paced Interview

> "I'd confirm cheap fixes are ruled out first, then reach for LoRA given this is a narrow capability gap — full fine-tuning would be overkill, and QLoRA isn't needed since the base model fits in memory at bf16. Moderate rank, targeting attention projections, trained on a clean, correctly-masked, schema-focused dataset, validated with format-validity and content-correctness scored separately, plus a regression check on unrelated capabilities."

---

## Key Takeaways

- This question tests live application of Lesson 9's flowchart under pressure, not memorized facts about any single method in isolation.
- A strong answer explicitly narrates the reasoning at each branch point (cheap-fix check, scope assessment, memory feasibility, rank starting point, data discipline, validation plan) rather than jumping straight to a named method.
- Follow-up questions typically probe whether each choice was actually reasoned through or just named — having Lesson 9's reasoning, not just its conclusions, ready to restate is what survives those follow-ups.
- The compressed version should still touch every stage of the flowchart, just faster — cutting stages entirely (not just detail) is the failure mode to avoid under time pressure.

---

## Self-Check — Full Mock Rep

Generate your own scenario (different model size, different capability gap, different compute constraint) and walk through Lesson 9's flowchart out loud, unscripted, in the style of Section 2's worked response. Then have someone (or a future session with me) fire the four follow-ups from Section 4, adapted to your scenario.