# Chapter 5 · Lesson 7 — Code Generation Capability

> **Where this fits:** Continuing Layer 2. Code generation is a useful capability to study because its failures decompose unusually cleanly into distinct categories — syntax, logic, and environment/tool-integration — each with a different diagnostic signature and a different fix, closely mirroring the structured-output split from Lesson 6.

---

## 1. Where Code Capability Comes From — the Pretraining Link

Directly connecting back to Chapter 1's data-mixture content: code generation quality is heavily shaped by the **proportion and quality of code in the pretraining corpus**, not primarily by fine-tuning. A base model with little code exposure during pretraining has a hard capability ceiling that instruction-tuning or fine-tuning cannot fully overcome — fine-tuning can teach a model to *apply* existing code knowledge in a more instruction-following, chat-appropriate way, but it can't manufacture familiarity with a programming language or library the model essentially never saw.

**The diagnostic implication:** if a model is weak across code generation broadly (not just on some narrow sub-task), the first question is whether this is a Layer 1 problem (per Lesson 2's framework) — insufficient code pretraining exposure — before assuming a fine-tuning fix will meaningfully help. A quick check: does the model perform noticeably better on very common languages/libraries (Python, JavaScript) than on a less-represented one (a niche or older language)? A large, consistent gap tracking known pretraining-data popularity is evidence for a Layer 1 explanation.

---

## 2. Three Distinguishable Failure Categories

| Category | Symptom | Diagnostic test |
|---|---|---|
| Syntax errors | Code doesn't parse/compile at all — missing brackets, wrong indentation, invalid tokens for the language | Run the output through a linter/parser; a high raw syntax-error rate, especially on simple snippets, points here |
| Logic errors | Code is syntactically valid, runs without crashing, but produces incorrect results for some inputs | Requires actual execution against test cases, not just parsing — syntax validity is not evidence of logical correctness |
| Tool/environment-integration errors | Code is syntactically and logically fine in isolation, but fails in the target environment — wrong library version's API, incorrect import paths, assumes a dependency that isn't actually available | Requires running the code in the *actual* target environment, not just checking it in isolation |

**Why keeping these separate matters for the intervention decision:** syntax errors on simple code are a strong signal of a genuine, low-level capability gap (rare after Lesson 1's baseline foundation check, but possible for unusual languages). Logic errors often point toward Lesson 5's reasoning-capability territory rather than being a "code" problem per se — code generation for anything beyond boilerplate is fundamentally a reasoning task wearing a programming-language costume. Tool/environment errors are frequently *not* a model problem at all — they're a context/tooling gap (the model wasn't told which library version is in use), closely paralleling Lesson 4's tool-schema-quality point.

---

## 3. Worked Example: A Full Diagnostic Pass

Symptom: a coding assistant frequently produces code that fails when run in the target codebase.

**Step 1 — parse/lint the failing outputs.** Suppose syntax is clean across the board — rules out the syntax-error category entirely, and (per Section 1) suggests the base model's general code-generation foundation is likely fine.

**Step 2 — execute the code against basic test cases in isolation.** Suppose it passes — rules out straightforward logic errors for these cases.

**Step 3 — execute in the actual target environment.** Suppose it now fails, with errors tracing back to calls against an older version of a library API than what the code assumed. **Diagnosis: tool/environment-integration failure, not a model capability gap at all.** The correct fix is providing the model with accurate information about the actual dependency versions in use (a context/prompting fix, or a RAG-style lookup against the actual project's dependency manifest) — not fine-tuning the model on more general code examples, which wouldn't address a problem that's fundamentally about missing project-specific context.

**This mirrors Lesson 4's tool-use diagnosis almost exactly** — worth noticing the structural parallel: a model can be "fine" in isolation and still fail because of an environment/context gap that has nothing to do with the model's trained capability.

---

## 4. A Note on Logic Errors — Why This Often Isn't a "Code" Lesson Problem

Worth stating explicitly, since it's a common point of confusion: for anything beyond simple boilerplate, a logic error in generated code is very often better diagnosed using Lesson 5's reasoning-capability framework (including the perturbation test) than treated as a code-specific issue. A model that makes a logic error in a Python function implementing a multi-step algorithm is exhibiting the same underlying failure — inability to correctly track and combine multiple steps — as a model failing a multi-step word problem. Code is a particularly legible domain for spotting reasoning failures precisely because the "correctness" of each step is checkable by execution, which is worth mentioning if asked why code generation is often used as a reasoning benchmark in the first place.

---

## Key Takeaways

- Code generation capability has a real Layer-1 dependency on pretraining data composition — fine-tuning can't fully manufacture familiarity with a language/library the base model essentially never saw.
- Syntax, logic, and tool/environment-integration failures are diagnostically distinct and require different evidence (parsing, isolated execution, target-environment execution respectively) — don't conflate them.
- Tool/environment-integration failures are frequently not model problems at all, closely paralleling Lesson 4's tool-schema-quality lesson.
- Logic errors in non-trivial code are often better diagnosed through Lesson 5's reasoning framework than treated as a separate "code capability" issue.

---

## Self-Check Before Moving to Lesson 8

1. A model's Python code always parses correctly but frequently produces off-by-one errors in loop boundaries. Which category does this fall into, and which earlier lesson's diagnostic technique would you reach for?
2. Why can't fine-tuning alone fully fix a Layer-1 code-capability gap for an underrepresented programming language?
3. Explain, using Section 3's example, why "the model wrote buggy code" doesn't automatically mean the model itself needs a fix.