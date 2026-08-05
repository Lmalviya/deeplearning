# Chapter 6 · Lesson 5 — Building a Custom Eval Set: A Small Eval Harness in Code

> **Where this fits:** Lessons 1-4 covered the concepts and pitfalls. This lesson makes it concrete — a working, if minimal, eval harness that combines Lesson 3's capability-specific design pattern with Lesson 4's LLM-as-judge mechanics, structured the way a real internal eval tool would be.

---

## 1. The Design Goals for a Minimal But Real Harness

Before code: what a usable eval harness actually needs, distilled from every lesson so far in this chapter —

1. **Structured test cases**, not just a flat list of prompts — each case should carry metadata about *what capability* it's testing (Lesson 3) and *how* to score it (exact-match, execution-based, LLM-judge, etc.), since different capabilities need different scoring logic (Section 4 of Lesson 3's reference table).
2. **Support for both objective and judged scoring** — some Chapter 5 capabilities (structured-output validity, code execution) have objectively checkable answers; others (reasoning quality, tone) need LLM-as-judge (Lesson 4).
3. **Position-swapping built in for any pairwise comparisons** (Lesson 4, Section 3) — not bolted on as an afterthought.
4. **Results broken out by capability category**, not collapsed into one aggregate score — directly following Lesson 3's warning that aggregate numbers hide important patterns.

---

## 2. Code: Test Case Structure

```python
from dataclasses import dataclass, field
from typing import Literal, Callable, Optional
import json

@dataclass
class EvalCase:
    id: str
    capability: str          # e.g. "tool_use", "structured_output", "reasoning"
    prompt: str
    scoring_method: Literal["exact_match", "execution", "llm_judge", "schema_valid"]
    expected: Optional[str] = None          # for exact_match
    schema: Optional[dict] = None            # for schema_valid
    test_fn: Optional[Callable] = None       # for execution-based scoring
    metadata: dict = field(default_factory=dict)  # e.g. {"needle_position": 0.5}
```

**Why `metadata` exists as a free-form field:** directly supports Lesson 3's design pattern — systematically varying one dimension (needle position, schema complexity, conversation turn count) requires being able to tag each case with where it sits on that dimension, so results can later be grouped and plotted by it, not just averaged away.

---

## 3. Code: Scoring Functions Per Method

```python
import jsonschema

def score_exact_match(response, case: EvalCase):
    return 1.0 if response.strip() == case.expected.strip() else 0.0

def score_schema_valid(response, case: EvalCase):
    # Directly implements Chapter 5, Lesson 6's format-validity axis,
    # kept SEPARATE from any content-correctness scoring
    try:
        parsed = json.loads(response)
        jsonschema.validate(parsed, case.schema)
        return 1.0
    except (json.JSONDecodeError, jsonschema.ValidationError):
        return 0.0

def score_execution(response, case: EvalCase):
    # Directly implements Chapter 5, Lesson 7's execution-based
    # code-correctness check — never score code via static inspection alone
    try:
        return 1.0 if case.test_fn(response) else 0.0
    except Exception:
        return 0.0  # crashes count as failures, not errors to ignore

def score_llm_judge(response, reference_response, case: EvalCase, judge_fn):
    # Implements Lesson 4's position-swapping mitigation directly in the scorer
    verdict_1 = judge_fn(case.prompt, response, reference_response)
    verdict_2 = judge_fn(case.prompt, reference_response, response)  # swapped order
    if verdict_1 == "A" and verdict_2 == "B":
        return 1.0   # consistently preferred regardless of position
    elif verdict_1 == "B" and verdict_2 == "A":
        return 0.0   # consistently NOT preferred
    else:
        return 0.5   # inconsistent across position swap — treat as a tie,
                      # per Lesson 4 Section 3's position-bias mitigation
```

---

## 4. Code: The Harness Itself, With Capability Breakdown

```python
from collections import defaultdict

SCORERS = {
    "exact_match": score_exact_match,
    "schema_valid": score_schema_valid,
    "execution": score_execution,
    # llm_judge handled separately below since it needs a reference response
}

def run_eval(model_fn, cases: list[EvalCase], reference_model_fn=None, judge_fn=None):
    results_by_capability = defaultdict(list)

    for case in cases:
        response = model_fn(case.prompt)

        if case.scoring_method == "llm_judge":
            reference_response = reference_model_fn(case.prompt)
            score = score_llm_judge(response, reference_response, case, judge_fn)
        else:
            score = SCORERS[case.scoring_method](response, case)

        results_by_capability[case.capability].append({
            "id": case.id,
            "score": score,
            "metadata": case.metadata,
        })

    # Directly implements this chapter's repeated warning:
    # never collapse to one number without also reporting the breakdown
    summary = {}
    for capability, results in results_by_capability.items():
        scores = [r["score"] for r in results]
        summary[capability] = {
            "mean_score": sum(scores) / len(scores),
            "n": len(scores),
        }

    return summary, results_by_capability
```

---

## 5. Worked Example: Using the Harness for a Needle-in-a-Haystack Sweep

Directly operationalizing Chapter 5, Lesson 9's technique using this harness's `metadata` field:

```python
needle_cases = [
    EvalCase(
        id=f"needle_pos_{pos}",
        capability="long_context",
        prompt=build_needle_test(haystack_text, needle_fact, position_fraction=pos),
        scoring_method="exact_match",
        expected=expected_answer,
        metadata={"needle_position": pos},
    )
    for pos in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # systematic variation, per Lesson 3
]

summary, detailed_results = run_eval(model_fn, needle_cases)

# Now group by metadata to reconstruct the position-vs-accuracy curve
# from Chapter 5, Lesson 9 — this is the step a naive harness (that only
# reports one aggregate "long_context: 0.71" number) would have made impossible
position_accuracy = {
    r["metadata"]["needle_position"]: r["score"]
    for r in detailed_results["long_context"]
}
```

**This is the concrete payoff of the `metadata` design decision from Section 2** — without it, the harness could only ever report a single blended long-context score, hiding exactly the "lost in the middle" pattern that Chapter 5, Lesson 9 identified as the actually useful diagnostic signal.

---

## 6. What a Minimal Harness Like This Doesn't Handle — Honest Scope Limits

Worth naming explicitly rather than overselling this code: it doesn't handle judge validation against human scores (Lesson 4, Section 4 — would need a separate human-scoring collection step), doesn't handle statistical significance testing across runs (important when comparing two close scores — a difference of a few percentage points on a small eval set may not be meaningfully distinguishable from noise), and doesn't handle cost/rate-limiting for LLM-judge calls at scale. A production eval system would need all three — this harness is the conceptual skeleton, not a finished production tool.

---

## Key Takeaways

- A usable eval harness needs structured test cases carrying both capability tags and scoring-method metadata, not a flat prompt list — different capabilities need genuinely different scoring logic.
- Position-swapping for LLM-judge scoring should be built into the scorer itself, not treated as an optional extra step.
- Capability-breakdown reporting (never a single collapsed number) and metadata-tagged systematic variation are what make Chapter 5's diagnostic patterns (like the needle-in-a-haystack curve) reconstructable from harness output.
- A real production harness needs judge validation, statistical significance testing, and cost management on top of this skeleton — worth naming as honest scope limits rather than implying this code is complete.

---

## Self-Check Before Moving to Lesson 6

1. Explain why `metadata` is a necessary field in the `EvalCase` structure, using the needle-in-a-haystack example.
2. Walk through what `score_llm_judge` does with the two swapped-order verdicts, and why an inconsistent result is scored as 0.5 rather than discarded or averaged differently.
3. Name the three honest scope limits of this harness (Section 6) and explain briefly why each matters for a real production system.