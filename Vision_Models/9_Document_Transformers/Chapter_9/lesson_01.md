# 9.1 Why Fixed Softmax Heads Break First (Extended to 50-Class Reality)

## Problem, Recapped

A model fine-tuned with a fixed-size softmax output layer hard-codes the class count into its
architecture — adding a new class means growing that layer and retraining, and the frozen
backbone underneath may never have learned to represent whatever distinguishes the new class
from the original set, since it was only ever pushed to separate those original classes (a
form of catastrophic forgetting / representation staleness). This is why Chapter 2.3 chose an
embedding + prototype/KNN architecture instead, back at the 5-class MVP stage — specifically to
avoid this problem before it ever became acute.

## What Changes Specifically at 50-Class Scale

The embedding + prototype/KNN architecture (Chapter 2.3) already solves the *retraining*
problem. Two new pressures appear as the taxonomy actually grows toward 50 classes that weren't
significant at 5:

**1. Class confusability increases as the taxonomy densifies.** At 5 maximally-distinct classes
(Chapter 1.1's deliberate MVP choice), every class sits far from every other in embedding
space, and a nearest-neighbor decision is easy. At 50 classes, it becomes far more likely that
several classes are genuinely similar to each other (e.g., several distinct financial document
subtypes, several distinct HR document subtypes) — the "Scenario B" confusable-class problem
noted back when the class list was first chosen is no longer avoidable once the taxonomy grows
this large; it has to be actively managed, not sidestepped by class selection alone.

**2. Reference-set maintenance overhead grows linearly with class count.** Each class needs its
own curated reference set (Chapter 4.3's onboarding lifecycle). At 5 classes this is a small,
easily-audited set; at 50 classes, keeping every class's reference set accurate, representative,
and free of drift becomes real, ongoing operational work — a governance problem, not just a
technical one.

## Why This Motivates the Rest of This Chapter

Neither pressure is solved by the embedding/prototype architecture alone — they require
additional structure on top of it:

- **Confusability** is addressed by organizing the taxonomy hierarchically rather than as one
  flat 50-way comparison (Lesson 9.2).
- **Reference-set scale** raises the question of whether brute-force similarity search remains
  viable, or whether a dedicated vector index is needed (Lesson 9.3) — addressed with an honest
  capacity check rather than an assumption.
- **Reference-set curation and new-class discovery** need a deliberate human-in-the-loop
  process, not ad-hoc labeling (Lesson 9.4).

## Trade-offs

| Aspect | At 5 classes (Chapter 2.3's original context) | At 50 classes |
|---|---|---|
| Class separation in embedding space | Large margins, easy nearest-neighbor decisions | Margins shrink as confusable classes appear — flat comparison alone degrades |
| Reference-set curation effort | Small, easy to audit by hand | Grows linearly with class count — needs process, not ad-hoc effort |
| Retraining risk (the original problem this architecture solved) | Solved by the embedding/prototype design | Still solved — this isn't a regression, it's an *additional* set of pressures layered on top |

## Summary

The embedding + prototype/KNN architecture chosen in Chapter 2.3 remains correct at 50-class
scale — it still avoids retraining entirely to add a class. What's new at this scale is that
flat, undifferentiated comparison against all 50 classes' reference sets starts to strain under
increased confusability, and reference-set governance becomes real ongoing work rather than a
one-time setup task. Both pressures are addressed by additional structure — hierarchy (Lesson
9.2) and a deliberate onboarding process (Lesson 9.4) — layered on top of, not replacing, the
architecture already in place.