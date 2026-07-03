# 7.1 Bottleneck Analysis at Each Order of Magnitude

## Problem

"Scale to 100M documents/month" is a single target number, but the system doesn't fail
uniformly as traffic grows toward it — different components become the bottleneck at different
orders of magnitude, and provisioning for the end-state bottleneck (GPU fleet size) while
ignoring an earlier one (e.g., single points of failure at moderate traffic) wastes effort in
the wrong place. Walking through concrete traffic levels, using the same methodology from
Chapter 1.2, shows which component breaks first at each stage — and therefore which chapter's
fix actually needs to be applied, and when.

## Solution / Concept: Walking Traffic Up by Orders of Magnitude

Using the same per-page and per-GPU-second assumptions established in Chapter 1.2:

| Volume | Docs/sec (avg) | Page-inferences/sec (post-early-exit) | GPUs needed (steady state, ~70% util) | Postgres writes/sec (~2 rows/doc) | First real bottleneck |
|---|---|---|---|---|---|
| 1,000 docs/day | ≈0.012 | ≈0.02 | <1 | ≈0.02 | None — MVP (Ch 2) handles this trivially |
| 10,000 docs/day | ≈0.12 | ≈0.17 | <1 | ≈0.23 | Still no throughput bottleneck — but the async batch API contract (Ch 3.1) already requires the queue (Ch 5) to exist as a real backend, not just a design on paper |
| 100,000 docs/day | ≈1.16 | ≈1.7 | ≈2–3 | ≈2.3 | Not throughput — **availability/redundancy**: a single un-load-balanced instance becomes a real single point of failure at this level of real usage, even though raw capacity is nowhere near exhausted |
| 1,000,000 docs/day | ≈11.6 | ≈17.4 | ≈25–29 | ≈23 | **GPU worker pool sizing and autoscaling policy** (Lesson 7.2) becomes the first genuine throughput bottleneck — a fixed small worker pool sized for the previous order of magnitude will visibly queue up |
| 3,333,333 docs/day (100M/month, target) | ≈38.6 | ≈58 | ≈80–85 | ≈77 | Multiple components under real, sustained pressure simultaneously: GPU fleet size (Ch 5.3), database read/write load without caching (Ch 6) or partitioning (Ch 4.4), and queue throughput needing to sustain the full peak numbers from Chapter 1.3 (≈62/sec real-time peak, ≈155/sec batch peak) |

## The General Pattern, as a Heuristic

Reading down the table, the bottlenecks arrive in a fairly consistent order, and this order is
worth internalizing as a general pattern for systems shaped like this one (asynchronous,
GPU-inference-bound, multi-tenant read/write database):

1. **Concurrency/connection handling** breaks first, at surprisingly low volume, if the system
   is still purely synchronous (Chapter 5.1) — this is why the queue is introduced early,
   ahead of when raw throughput alone would demand it.
2. **Availability/redundancy** (single points of failure) becomes a real operational risk once
   the system has genuine production usage, well before it becomes a throughput problem —
   this is a reliability concern (Chapter 10), not a capacity one, and is easy to
   under-prioritize because it doesn't show up in a pure throughput calculation.
3. **GPU inference throughput** becomes the first real capacity bottleneck, because it's the
   most expensive and least elastic resource in the pipeline (Chapter 1.2's ~80+ GPU estimate
   at target scale is the single largest infrastructure line item).
4. **Database read/write contention** becomes pressing next, mitigated first by caching
   (Chapter 6) and partitioning (Chapter 4.4) — both of which should already be in place well
   before this stage, per those chapters' own recommendations, rather than reactively bolted on.
5. **Cross-region latency and true multi-primary scaling needs** (Lesson 7.4) arrive last, and
   are the least likely to actually be needed at this system's stated 100M/month target — they
   become relevant mainly beyond that target, or if geographic/compliance requirements force
   the issue earlier than pure throughput would.

## Trade-offs in Using This Kind of Staged Analysis

| Approach | Gain | Cost |
|---|---|---|
| Identifying the specific first bottleneck at each order of magnitude, rather than provisioning everything for the 100M/month end-state from the start | Effort is spent on the actual current constraint, not a hypothetical future one — directly consistent with the "start simple, respond to observed breakpoints" principle from Chapter 2.5 | Requires genuinely monitoring for each stage's bottleneck signal (Chapter 10.2) — without that visibility, teams find out about the current bottleneck from an incident, the same risk flagged in Chapter 2.5 |

## Summary

Scaling toward 100M documents/month doesn't stress every system component equally at every
traffic level — connection handling and the async contract break first at very low volume,
availability/redundancy becomes a concern at moderate real usage well before raw capacity is
threatened, GPU throughput becomes the first genuine capacity bottleneck at meaningful volume,
and database load and cross-region concerns arrive last, largely already mitigated by
decisions (caching, partitioning) made proactively in earlier chapters rather than reactively
under pressure.