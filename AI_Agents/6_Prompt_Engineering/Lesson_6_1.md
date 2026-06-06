# Lesson 6.1 — Prompt Engineering Fundamentals

---

## The Problem: The LLM Is Only as Good as Its Instructions

An LLM's behavior is almost entirely determined by what you put in the prompt. The same model that gives a brilliant answer with one prompt gives a confused, wrong, or dangerous answer with a poorly written one. Prompt engineering is not "writing nice sentences" — it is the primary control interface for LLM behavior, and in agent systems, it is the difference between a reliable production agent and an unpredictable one.

This lesson covers the fundamental techniques that every Amazon ML engineer must know.

---

## Technique 1: Zero-Shot Prompting

Ask the LLM to perform a task without providing any examples. The LLM relies entirely on its training knowledge.

```
Classify the sentiment of this product review as Positive, Negative, or Neutral:

Review: "The laptop arrived quickly but the keyboard feels cheap."

Sentiment:
```

**When to use:** Large models (GPT-4, Claude 3.5, Nova Pro) that generalize well from instructions alone. Simple, well-defined tasks.

**When it fails:** Specialized tasks the model has limited training on; tasks requiring a specific output format the model doesn't default to; tasks where the boundary conditions are subtle.

---

## Technique 2: Few-Shot Prompting

Provide 2–8 examples of the task before the actual query. The model learns the pattern from examples.

```
Classify the sentiment of product reviews.

Review: "Battery lasts all day. Highly recommend!"
Sentiment: Positive

Review: "Broke after two weeks. Very disappointed."
Sentiment: Negative

Review: "It's okay. Does the job but nothing special."
Sentiment: Neutral

Review: "The laptop arrived quickly but the keyboard feels cheap."
Sentiment:
```

**When to use:** When zero-shot fails or produces inconsistent output. When the task has a specific format or classification scheme that differs from common patterns. When the model needs to learn edge cases.

**Key insight:** Example selection matters more than example count. 3 well-chosen examples that cover edge cases beat 8 generic examples. Include examples that demonstrate where the model is likely to go wrong.

---

## Technique 3: Role Prompting (Persona)

Assign the LLM a role to activate relevant knowledge and set behavioral expectations.

```
You are an expert Amazon seller policy specialist with 10 years of experience.
You provide precise, accurate answers about Amazon's selling policies.
You cite specific policy sections when relevant.
You never speculate — if unsure, you say "I need to verify this."

Question: Can I sell refurbished electronics on Amazon?
```

**Why it works:** Role assignment narrows the distribution of likely responses. An "expert" produces more precise, confident, appropriately formatted answers than a generic assistant. The behavioral instructions embedded in the role ("never speculate") enforce safety constraints naturally.

**Amazon example:** "You are Rufus, Amazon's helpful shopping assistant. You help customers find products, compare options, and make purchase decisions. You only recommend products available on Amazon.com." This single role instruction shapes thousands of behavioral decisions.

---

## Technique 4: Chain of Thought in Prompts

Already covered in Lesson 2.1. Key reminder: append "Think step by step" or provide examples showing reasoning steps. Critical for math, logic, and multi-step tasks.

---

## Technique 5: Output Format Control

Explicitly specify the output format. This is critical for agent tool outputs, pipeline parsing, and structured data extraction.

```
Extract the following information from this product description and return ONLY valid JSON:

{
  "product_name": string,
  "price": number,
  "key_features": list of strings (max 5),
  "in_stock": boolean
}

Product description: "Sony WH-1000XM5 headphones at $299. Premium noise cancellation,
30-hour battery, multipoint connection. Currently available. Ships in 2 days."
```

**Techniques for reliable structured output:**
- Provide a schema or example of the exact JSON you want
- Say "Return ONLY valid JSON. No explanation, no markdown code blocks."
- Use function calling (Lesson 3.1) instead of prompting — it guarantees schema compliance via API validation

---

## Technique 6: Constraints and Negative Instructions

Tell the model what NOT to do. Positive instructions alone often leave the model free to do unwanted things.

```
You are an Amazon customer support agent.

ALWAYS:
- Verify the order number before taking any action
- Use the exact product name from the order system, not the user's description

NEVER:
- Promise refunds you cannot authorize
- Share other customers' information
- Make up order status if the database returns no results — say "I cannot locate this order"
- Exceed your authority (you can only issue credits up to $50; escalate above that)
```

**Why negative instructions matter:** Without them, the model fills gaps with "reasonable" behavior that may be wrong for your specific context. A model will happily promise a refund if it seems helpful — unless you explicitly forbid it.

---

## Technique 7: Contextual Grounding

Provide specific, relevant context so the model doesn't have to rely on (potentially incorrect) training knowledge.

```
[CUSTOMER ACCOUNT DATA]
Name: John Smith
Order #: 123-456-789
Order Date: 2026-05-15
Item: Sony WH-1000XM5 Headphones ($299)
Shipping Status: Delayed — Expected: 2026-06-08 (was 2026-06-01)
Customer Tier: Prime

[POLICY]
Prime members receive $5 credit for delays > 5 days.
Current delay: 7 days. Credit eligible: YES

Customer message: "My order hasn't arrived yet, I'm frustrated."

Respond to the customer. You may offer the $5 credit if appropriate.
```

The model now has exact facts to work with — order number, delay duration, customer tier, credit policy. It cannot hallucinate because the ground truth is in the prompt.

---

## Prompt Structure for Agent System Prompts

A well-structured agent system prompt has clear sections:

```
[Role and Identity]
You are [name], [role description], [primary responsibility].

[Capabilities]
You can: [list of what the agent is allowed to do]

[Constraints]
You must never: [list of absolute prohibitions]
You must always: [list of required behaviors]

[Tools]
[Tool definitions are injected here by the framework]

[Output Format]
When responding, use this format: [format description]

[Safety Rules]
[Critical safety instructions]
```

---

> **Interview note:** *"What is the difference between zero-shot and few-shot prompting? When would you use each?"*
> Zero-shot: give instructions only, no examples. Use for large models on common tasks where the model has strong priors. Faster (no examples in context), simpler.
> Few-shot: provide 2–8 worked examples before the actual query. Use when: (1) zero-shot produces inconsistent or wrong output, (2) the task has a specific format different from common patterns, (3) edge cases need to be demonstrated explicitly. Few-shot examples should be chosen to cover edge cases, not just typical cases. 3 good examples beat 8 generic ones. For agent tool outputs that must be structured (JSON, XML), use function calling instead of few-shot prompting — it guarantees schema compliance.

> **Interview note:** *"How do you make an LLM agent reliable and consistent in production?"*
> Four practices: (1) Clear, structured system prompt with Role, Capabilities, Constraints (ALWAYS/NEVER), and Output Format sections. The model's behavior is 80% determined by the system prompt. (2) Negative constraints — explicitly say what the model must never do; positive instructions alone leave gaps. (3) Contextual grounding — inject real, specific data into the prompt so the model works with facts, not training knowledge. Reduces hallucination dramatically. (4) Output format control — specify exactly what format you expect (JSON schema, bullet list, max length); validate programmatically in the framework and reject/retry if the format is wrong. Consistency comes from constrained, validated outputs, not from hoping the model will be consistent on its own.

---

## Summary

- **Zero-shot**: instructions only — use for large models on well-defined tasks.
- **Few-shot**: 2–8 examples + query — use when zero-shot is inconsistent or the task has a specific format. Choose edge-case examples, not generic ones.
- **Role prompting**: assign a persona — narrows behavioral distribution, embeds constraints naturally, activates domain-specific knowledge.
- **Output format control**: specify the exact schema — use function calling for guaranteed compliance; use explicit schemas in prompt as a fallback.
- **Negative constraints**: tell the model what NOT to do — prevents the model from filling gaps with "reasonable" but wrong behavior.
- **Contextual grounding**: inject real, specific data — eliminates hallucination on facts that are in the context.
- Well-structured system prompt: Role → Capabilities → Constraints (ALWAYS/NEVER) → Tools → Output Format → Safety Rules.
