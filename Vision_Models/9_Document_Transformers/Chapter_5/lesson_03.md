# 5.3 Worker Pool Design, Dynamic GPU Batching, and Inference Serving

## Problem

GPU inference (running a document/page through the embedding backbone from Chapter 2.3) is
expensive to run one request at a time — a GPU processing a single small input rarely uses its
full parallel compute capacity, so throughput per dollar suffers badly under naive
one-request-per-inference-call handling. The standard fix is **dynamic batching**: grouping
several pending requests into one GPU forward pass. But batching requires *waiting* to
accumulate a batch, which directly conflicts with the real-time lane's latency SLO (Chapter
1.1) — the same technique that maximizes batch-lane throughput would actively hurt real-time
latency if applied uniformly.

## Solution / Concept: Different Batching Strategy Per Lane

Because the two lanes already have fully separate worker pools (Lesson 5.2), each pool can run
its own batching policy, tuned to its own SLO:

### Batch worker pool — aggressive dynamic batching

Accumulate pending page-inference requests until either **N requests** are collected or **T
milliseconds** elapse, whichever comes first, then run one batched GPU forward pass across all
of them. Since the batch SLO (Chapter 1.1) is a completion window, not a per-document latency
target, the wait to accumulate a batch is essentially free — it doesn't threaten any stated
requirement, and it directly maximizes GPU utilization and therefore throughput per dollar
(connects to Chapter 11's cost breakdown).

### Real-time worker pool — minimal or no batching

Process requests with batch size 1, or a very short micro-batching window (e.g., 10–20ms) —
just enough to catch requests that happen to arrive within a few milliseconds of each other,
without meaningfully delaying any individual request. This **deliberately sacrifices GPU
utilization efficiency** to protect the latency SLO — a real, explicit cost, not a free choice:
the real-time pool needs proportionally more GPU capacity per unit of throughput than the batch
pool does, purely because it can't batch as aggressively. This is worth stating plainly, since
it directly affects the cost breakdown in Chapter 11.1.

## Sizing the Two Pools Against Chapter 1.2's GPU Estimate

Chapter 1.2 estimated ≈80–85 GPUs needed at steady state, computed from an aggregate page-
inference rate. Splitting this by lane, using the batch/real-time split from Chapter 1.2
(≈30.9 vs. ≈7.7 docs/sec average, roughly 80/20 by volume):

- **Batch pool:** sized primarily for **average throughput**, since aggressive batching
  achieves high GPU utilization and the completion-window SLO tolerates queue depth absorbing
  variance (Chapter 1.3) — this pool autoscales up gradually in response to growing queue depth,
  not in anticipation of it.
- **Real-time pool:** sized for **peak-percentile load plus the standing capacity buffer**
  established in Chapter 1.3 (the ~62 docs/sec peak estimate), *and* inflated further to
  account for its lower batching efficiency relative to the batch pool doing the same
  aggregate work — this pool cannot rely on gradual autoscaling alone, since the standing
  buffer exists specifically to cover the autoscaling-lag gap for latency-sensitive traffic.

## Model Serving Approach

Rather than hand-rolling request queuing and batching logic inside application code, use a
dedicated model-serving layer (e.g., NVIDIA Triton Inference Server, TorchServe, or an
equivalent) that implements request batching, GPU scheduling, and model versioning as built-in
capabilities. This is treated here as a role, not a deep trade-off debate (similar to the
standard-infra components) — the meaningful design decision in this lesson is the **batching
policy per lane**, not which specific serving framework implements it; any reasonable modern
serving layer supports configurable dynamic batching windows per model/endpoint.

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Aggressive batching for batch-lane workers | Maximizes GPU throughput and cost-efficiency for the 80% majority of traffic | Adds latency per individual request within the batch — irrelevant given the completion-window SLO, but would be unacceptable if applied to real-time |
| Minimal/no batching for real-time-lane workers | Protects the tight latency SLO | Lower GPU utilization efficiency — more GPUs needed per unit of real-time throughput than the batch pool needs for equivalent volume, a real and quantifiable cost |
| Separate batching policy per lane (enabled by Lesson 5.2's pool separation) | Each lane gets the throughput/latency trade-off appropriate to its own SLO | Requires operating and tuning two distinct serving configurations rather than one uniform policy |

## When to Use / When Not To

- **Aggressive batching** is correct wherever the completion-window SLO model applies (batch
  lane) — there's no reason to leave GPU throughput on the table when nothing is waiting on an
  individual result.
- **Minimal batching** is correct wherever a tight per-request latency SLO applies (real-time
  lane) — batching efficiency should never be allowed to trade away a stated latency
  requirement.
- **Revisit batch-window tuning (N and T)** periodically against real observed traffic
  patterns and GPU utilization metrics (Chapter 10.2) — these are hyperparameters, not
  one-time decisions, and the right values shift as traffic volume and shape change.

## Summary

Dynamic GPU batching is a real throughput lever, but its cost — added per-request latency — is
only acceptable on the batch lane's completion-window SLO, not the real-time lane's tight
per-request SLO. Because the two lanes already run on fully separate worker pools (Lesson 5.2),
each can run its own batching policy: aggressive accumulation for batch (maximizing
cost-efficiency), minimal or no batching for real-time (protecting latency at the explicit cost
of needing proportionally more GPU capacity per unit of throughput) — a direct, quantifiable
consequence of the differentiated-SLO requirement stated back in Chapter 1.1.