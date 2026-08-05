# Chapter 5 · Lesson 12 — Interview Lab: Rebuilding "Your Mental Model for Fine-Tuning"

> **Where this fits:** This is the direct, complete rebuild of the fourth question from your original interview — the one this entire chapter was built to fix. Your original answer had good bones (it correctly started with "check for data distribution issues first") but collapsed everything downstream into a shallow instruction-tuning → alignment-tuning pipeline. This lesson rebuilds it using the full apparatus from Lessons 1-11.

---

## 1. Your Original Answer, Revisited

> "so there can be multiple issue in the model like model may not trained to the required vocab data like model trained on general web data but used in medical domain so in that case there is training data distribution issue let say model does not have this issue so i will check next does model following the instruction or not if not then i will go for instruction tuning... if model is following the instruction as well then i will go for alignment tunning"

**What was actually right about this, worth preserving:** the instinct to check data/vocabulary distribution *first*, before anything else, was correct — this is exactly Lesson 1's Layer 1 discipline, and it's the reason this chapter's flowchart also checks Layer 1 first. Give yourself credit for that instinct; it just needed everything built on top of it.

**What was missing, mapped directly to this chapter's lessons:**

| Gap in original answer | What this chapter added |
|---|---|
| "Instruction tuning" was the only Layer-2 fix mentioned | Six distinguishable capabilities (Lessons 3-9), each with its own diagnostic test and possibly a completely different fix |
| No mention of checking cheap fixes (prompting, schema, constrained decoding) before assuming a training-level fix | Every Layer 2 lesson explicitly checks cheap fixes first — the single most repeated pattern in the whole chapter |
| "If following instructions, go straight to alignment tuning" — treated as the only remaining option | Lesson 10 shows alignment tuning is specifically for calibration issues (over/under-refusal), not a catch-all for "everything else" |
| No mention of DAPT or tokenizer extension as distinct from fine-tuning | Lesson 11's intervention menu explicitly separates these, since collapsing them was flagged as a real structural error even in this curriculum's own draft |
| The "vocab issue" branch didn't specify what the actual fix is | Lesson 2 + Chapter 7 Lesson 1 — tokenizer extension and/or DAPT, not "fine-tuning" generically |

---

## 2. The Rebuilt Answer — Full Version

> "I'd treat this as a layered diagnosis, not a single check. First, foundation: I'd check for data distribution and vocabulary mismatch — measuring tokenizer fertility on domain text as a concrete signal, and testing whether the model gives confident-but-wrong answers on domain content that improve once I provide the correct facts in-context. If there's a real foundation gap, the fix isn't generic fine-tuning — it's either tokenizer extension, continued pretraining, or RAG, depending on how permanent and large the domain need is.
>
> If foundation checks out, I'd move to specific capabilities rather than a single 'does it follow instructions' check — instruction-following, tool use, reasoning, structured output, code generation, multilingual, and long-context faithfulness are distinguishable capabilities with different failure signatures. For each one, before assuming a training-level fix, I'd rule out cheap explanations first — a tool-use failure might be a bad tool schema, not a model gap; a reasoning failure might resolve with explicit chain-of-thought prompting; a structured-output failure might be better solved with constrained decoding than fine-tuning at all.
>
> Only once I've confirmed a genuine capability gap, ruling out the cheap explanations, would I move to instruction tuning or PEFT methods, matched to the specific diagnosed gap. And separately — alignment tuning specifically targets calibration issues, particularly over-refusal and under-refusal, not just 'whatever's left after instruction-following is fine.' Those are genuinely different problems needing different training approaches. So the mental model isn't a linear pipeline — it's diagnose the layer, rule out cheap fixes, and only then choose the specific, appropriately-costed intervention."

**Why this version is structurally stronger, precisely:** it preserves your original correct instinct (check foundation first) but replaces the two remaining collapsed buckets ("instruction tuning" and "alignment tuning" as catch-alls) with the actual layered structure this chapter built — specific capabilities, cheap-fix-first discipline, and calibration as a genuinely separate concern from general capability.

---

## 3. A Compressed Version, for Time Pressure

> "I wouldn't jump straight to a fix — I'd diagnose in layers. First, foundation: is there a data or vocabulary mismatch, checked via tokenizer fertility and a content-gap test. If that's clean, I'd check specific capabilities separately — instruction-following, tool use, reasoning, structured output, code, multilingual, long-context — since each has different failure signatures and often a cheap fix, like better prompting or a schema fix, before I'd assume a training-level intervention is needed. Once a genuine gap is confirmed, I'd match the fix to it specifically — fine-tuning or PEFT for capability gaps, and alignment tuning specifically for calibration issues like over- or under-refusal, which is a different problem from general capability."

---

## 4. Follow-Up Questions to Have Pre-Loaded

**"Give me a concrete example of a cheap fix resolving something that looked like a capability gap."** → Lesson 5's chain-of-thought example: a model that appears to fail multi-step reasoning because it was never prompted to show intermediate steps, resolved entirely by allowing/encouraging CoT, no training involved.

**"How is alignment tuning different from instruction tuning, in your framework specifically?"** → Instruction tuning teaches general behavior patterns and capability execution (Chapter 7); alignment tuning, per Lesson 10, is specifically about calibrating behavior at the boundary of what the model should/shouldn't do, using preference-driven training (Chapter 9) — and specifically needs *both* over-refusal and under-refusal examples in its training signal, or it drifts toward one failure mode while fixing the other.

**"What's the difference between tokenizer extension and continued pretraining, and when would you use one over the other?"** → Tokenizer extension fixes vocabulary fragmentation (Lesson 2's fertility problem) specifically; continued pretraining/DAPT fixes missing domain *content* knowledge; they're frequently needed together for a genuinely new domain, but they're distinguishable via the fertility test versus the content-gap test respectively (Lesson 2, Section 4's worked example).

**"Isn't this a lot of diagnostic overhead before doing anything? How do you decide how much of this to actually do in practice?"** → A good, honest answer: the depth of diagnosis should scale with the cost of the intervention being considered — a quick prompt-level check costs almost nothing and should basically always be tried; a full DAPT/tokenizer-extension decision, given its cost, deserves the full diagnostic pass. This is itself a demonstration of judgment, not just process for its own sake.

---

## 5. Why This Rebuild Matters Beyond This One Question

This isn't just a better answer to one interview question — it's the demonstration that the diagnostic discipline built across this entire chapter is usable under real interview time pressure, compressible (Section 3) without losing its structure, and robust to follow-up pressure-testing (Section 4). That combination — not the length of the answer — is what separates a memorized response from genuine understanding.

---

## Self-Check — Full Mock Rep

Say the full version (Section 2) out loud, targeting 90-120 seconds. Then say the compressed version (Section 3), targeting 30-40 seconds. Then have someone (or a future session with me) fire the four follow-ups from Section 4 at you in random order.