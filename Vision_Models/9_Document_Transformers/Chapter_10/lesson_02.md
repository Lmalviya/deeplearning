# 10.2 Monitoring, Drift Detection, and Retraining Triggers

## Problem

Two very different kinds of degradation can affect this system, and they need different
detection mechanisms entirely. **Infrastructure degradation** (queue backlogs, GPU saturation,
rising error rates) has been implicitly assumed-monitored throughout earlier chapters (Chapter
2.5's breakpoints, Chapter 7.3's spike handling) but never made concrete. **Model/data drift** —
the classification model's real-world accuracy silently declining over time even with healthy
infrastructure, as document formats evolve, new vendor templates appear, or a class's true
document population shifts — is a distinct failure mode that infrastructure monitoring alone
cannot detect at all, since nothing about it looks like an infrastructure problem.

## Solution / Concept: Two Monitoring Domains

### Infrastructure monitoring — making earlier chapters' signals concrete

Every breakpoint and SLO stated in earlier chapters becomes a dashboard/alert here, not left as
an abstract concept:

| Signal | Source | Alert condition |
|---|---|---|
| Queue depth and oldest-message age | Real-time and batch queues (Ch 5.2) | Batch: projected time-to-clear threatens the completion-window SLO (Ch 7.3). Real-time: depth growing despite standing floor + autoscaling (Ch 7.2) |
| Latency percentiles per lane | API/Orchestration Service | p95/p99 approaching the SLOs stated in Ch 1.1 |
| GPU/worker utilization | Classification Service (Ch 8.2) | Sustained high utilization without corresponding autoscale response — signals a scaling-policy problem, not just load |
| DLQ size | Chapter 10.1's dead-letter queues | Any non-zero, sustained growth — DLQ accumulation left unwatched defeats its own purpose |
| Error rates per service | All six services (Ch 8.2) | Rising error rate in any one service, isolating the failure domain thanks to the microservices split (Ch 8.1's fault-isolation trigger paying off operationally here) |

This is standard operational monitoring — the specific value this system's design adds is that
every signal above traces directly back to a decision made in an earlier chapter, so an alert
firing points immediately at *which* mechanism (autoscaling policy, DLQ, a specific service) is
implicated, rather than requiring investigation to even locate the right area.

### Model/data drift detection — a genuinely different problem

Infrastructure can be perfectly healthy while the model quietly gets worse at its actual job.
Two leading indicators, both derived from data the system already produces:

- **Confidence score distribution per class, over time.** A creeping decline in average
  confidence for a specific class, or a rising rate of documents falling into the Chapter 9.4
  "unknown" bucket, is a leading indicator that either new document variants are appearing
  within a class, or genuine concept drift is occurring — worth investigating well before it
  shows up as visible accuracy loss.
- **Reviewer correction rate per class, over time** (feeding from Lesson 10.3's review
  workflow). A rising correction rate for a specific class is a lagging but very concrete
  signal that its reference set or the embedding backbone's representation of it needs
  attention — this is ground truth, not a proxy signal like confidence scores.

### Retraining/refresh triggers — matched to the scale of the problem

Not every drift signal warrants the same response. Two genuinely different scales of action:

- **Reference-set refresh** (cheap, frequent, a pure data operation) — triggered by a rising
  correction rate for a *specific class*: pull recent, reviewer-confirmed examples for that
  class and refresh its reference set, using the exact same zero-downtime mechanism from
  Chapter 4.3. This is the default, low-cost response and should handle the large majority of
  observed drift.
- **Backbone retraining or replacement** (rare, expensive) — triggered only by *broad*, not
  class-specific, drift: e.g., confidence distributions shifting downward across many classes
  simultaneously, suggesting the embedding backbone itself (Chapter 2.3) is no longer
  representing the document population well, rather than any single class's reference set being
  stale. This revisits the full domain-adaptation spectrum from the earlier ML-design phase
  (frozen vs. linear-probing vs. fine-tuning vs. a different backbone entirely) — a real,
  costly decision that should be reserved for genuinely broad drift, not triggered by one
  class's correction rate creeping up.
- **Rising unknown-bucket rate specifically** signals a *taxonomy* gap (Chapter 9.4 — a
  candidate new class), not backbone drift — an important distinction, since conflating the two
  would lead to expensive backbone retraining attempting to solve what's actually a missing
  class problem.

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Confidence-distribution and correction-rate monitoring, per class | Detects drift specific to individual classes early, enabling the cheap reference-set-refresh response before it becomes a broader problem | Requires enough review volume (Lesson 10.3) to produce a meaningful correction-rate signal — sparse review coverage for a low-traffic class delays detection |
| Distinguishing reference-set refresh from backbone retraining as separate response tiers | Avoids reaching for an expensive, rare intervention (backbone retraining) when a cheap, frequent one (reference-set refresh) would fully address the observed drift | Requires discipline to correctly diagnose *which* kind of drift is occurring (class-specific vs. broad) before choosing a response — a wrong diagnosis wastes either engineering effort (over-reacting) or lets a real problem persist (under-reacting) |

## When to Use / When Not To

- **Infrastructure monitoring** should be live from the point any of Chapters 5–8's
  infrastructure exists — not something added after an incident reveals its absence.
- **Reference-set refresh** should be the default, low-friction response to any class-specific
  drift signal — cheap enough that there's little reason to delay it once a correction-rate
  threshold is crossed.
- **Backbone retraining** should be reserved for confirmed broad drift, evaluated deliberately
  against the domain-adaptation spectrum trade-offs, not triggered reactively by a single
  class's noisy signal.

## Summary

Infrastructure monitoring makes every SLO and breakpoint from earlier chapters concrete and
alertable, with each signal traceable directly back to the mechanism responsible for it,
thanks to the microservices split from Chapter 8. Model/data drift is a separate, quieter
failure mode, detected via confidence-distribution trends and reviewer correction rates, and
answered with two deliberately different-scale responses: cheap, frequent reference-set
refreshes for class-specific drift (the common case), and rare, expensive backbone
retraining reserved for genuinely broad drift — with a rising unknown-bucket rate correctly
routed to Chapter 9.4's taxonomy-growth process rather than either of these.