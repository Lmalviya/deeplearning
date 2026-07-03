# 8.1 Why and When to Decompose

## Problem

Everything built through Chapter 7 is still, architecturally, one deployable pipeline (Chapter
2.1's modular monolith) wrapped in increasingly sophisticated infrastructure — an API layer
(Chapter 3), queues and worker pools (Chapter 5), caching (Chapter 6), and scaling policies
(Chapter 7). "Use microservices at scale" is a common reflex, but adopted without a concrete
justification, it just trades one set of problems (a large deploy unit) for another (network
calls where function calls used to work, distributed tracing, many more deployment pipelines to
operate). The decomposition needs specific, observable triggers — not a assumption that scale
alone demands it.

## Solution / Concept: Four Concrete Triggers, Specific to This System

### Trigger 1 — Divergent scaling ratios

Chapter 1.2 established that the classification stage (GPU-bound embedding inference) needs on
the order of ~80+ GPUs at target scale, while the extraction stage (OCR/HTR, partly GPU-bound
but with a large CPU/IO-bound routing and preprocessing component) and the aggregation/
orchestration logic (lightweight, mostly control flow) have very different resource profiles.
A monolithic deploy unit forces all of these to scale together at the instance level, even
though their actual bottleneck resources differ hugely — wasting either GPU or CPU capacity
depending on which stage currently dominates the deploy unit's sizing.

### Trigger 2 — Divergent deploy cadence

The classification backbone (Chapter 2.3's embedding model) is updated on a model-development
cadence — infrequent, carefully validated, potentially requiring a shadow-deployment or
A/B-test rollout. The OCR/HTR tooling (Chapter 2.2) might be swapped or upgraded independently
(e.g., switching OCR engines, per the swap-without-architecture-change design from Chapter 2.2).
The API contract (Chapter 3) evolves on yet another cadence, driven by client needs. Coupling
all of these into one deploy unit means every taxonomy update, every OCR engine tweak, and every
API change all risk redeploying and restarting the entire pipeline — including the parts that
didn't change.

### Trigger 3 — Team/ownership boundaries

As the system and its usage grow, it becomes realistic for different teams to own extraction
tooling, classification modeling, and the API/product surface. A monolith forces these teams to
share a deploy pipeline and a blast radius — one team's bug can block or break another team's
unrelated change from shipping.

### Trigger 4 — Fault isolation

A resource leak or crash in one pipeline stage (e.g., an OCR engine memory leak under sustained
load) currently risks taking down the entire monolithic process, including stages that were
functioning correctly. Splitting into services isolates failure domains — a crash in extraction
shouldn't be able to take classification down with it.

**A fifth, already-present precedent:** Chapter 5.2 already split real-time and batch traffic
into separate worker *pools* — a lane-based split. The natural next step, once the four
triggers above are actually observed, is splitting along pipeline *stages* as well (extraction,
classification, aggregation), not just lanes — the two splits are orthogonal and compose
cleanly (each service can still have lane-specific instances/scaling, per Chapter 5).

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Decomposing once the triggers above are observed | Independent scaling (GPU vs. CPU resources sized correctly per stage), independent deploy cadence, team autonomy, fault isolation | Network calls replace function calls between stages — added latency and a new class of failure mode (partial failures, timeouts) that a monolith doesn't have; requires distributed tracing/observability (Chapter 10.2) to debug effectively |
| Staying a modular monolith until triggers are observed | No premature operational overhead; the internal modular structure (Chapter 2.1) means the eventual split is mechanical, not a rewrite | Risks all four trigger pains (wasted resources, coupled deploys, shared blast radius, no fault isolation) accumulating unaddressed if decomposition is delayed well past when triggers actually appear |

## When to Use / When Not To

- **Decompose** once at least one of the four triggers is genuinely observed in this specific
  system — most likely trigger 1 (divergent GPU/CPU scaling ratios) first, given the ~80+ GPU
  estimate from Chapter 1.2 makes classification's resource profile obviously different from
  the rest of the pipeline well before team-scaling or deploy-cadence pain becomes acute.
- **Do not decompose preemptively** "because 100M documents sounds like it needs
  microservices" — this notes set's own MVP principle (Chapter 2.1) explicitly warns against
  exactly this reflex, and the same caution applies here: decomposition is justified by
  observed pain, not by the top-line scale target alone.

## Summary

Microservices decomposition here is justified by four concrete, observable triggers — divergent
GPU/CPU scaling ratios between pipeline stages, divergent deploy cadence between model updates
and API/tooling changes, team ownership boundaries, and the need for fault isolation — not by
"scale" as an undifferentiated reflex. The modular internal structure established back in
Chapter 2.1 is what makes this decomposition mechanical once triggered, rather than a rewrite
under production pressure.