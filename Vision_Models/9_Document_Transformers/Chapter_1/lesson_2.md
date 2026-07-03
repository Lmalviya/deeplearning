# 1.2 Back-of-Envelope Capacity Estimation

## Problem

"100M documents/month" is not, by itself, an actionable number. Every infrastructure decision
later in these notes — how many GPU workers, how big the database, how much storage, how the
queue is sized — needs concrete per-second and per-day figures derived from that top-line
number. Skipping this translation step is how teams either wildly over-provision (burning
budget on capacity nobody needs yet) or under-provision (discovering the real bottleneck only
after it's live and failing). The goal of this lesson is the *method*, not just the final
numbers — the assumptions are explicitly labeled so they can be swapped for real product data
as soon as it exists.

## Method: Working Backward From the Top-Line Number

**Base assumption (from Lesson 1.1):** 100,000,000 documents/month at target scale.

### Step 1 — Convert to a rate

```
100,000,000 docs / 30 days           ≈ 3,333,333 docs/day
3,333,333 docs / 86,400 seconds/day  ≈ 38.6 docs/sec  (average, sustained)
```

This is the **average** rate — real traffic is not flat across the day (see Lesson 1.3 for
peak/spike sizing on top of this baseline).

### Step 2 — Split by traffic mix (80% batch / 20% real-time, from Lesson 1.1)

```
Batch:      80,000,000 docs/month  → 2,666,667 docs/day  → ≈ 30.9 docs/sec average
Real-time:  20,000,000 docs/month  →   666,667 docs/day  → ≈  7.7 docs/sec average
```

This split matters because these two numbers get provisioned very differently — batch can be
smoothed by a queue over its whole SLO window (Ch 1.1: "completed within a few hours"), while
real-time capacity has to be available close to the moment of arrival to meet a tight latency
SLO. Treating 38.6 docs/sec as one undifferentiated number would hide this.

### Step 3 — Estimate pages per document

**Assumption:** average 3 pages/document, reflecting a mix of 1-page documents (ID documents,
receipts) and multi-page documents (contracts).

```
3,333,333 docs/day × 3 pages/doc ≈ 10,000,000 pages/day
10,000,000 pages/day / 86,400s   ≈ 115.7 pages/sec (average)
```

**Adjustment for hierarchical early-exit aggregation** (see the earlier ML-design notes,
Chapter 4.1): not every page of every document actually gets processed — early-exit stops once
confidence clears a threshold. Assumption: on average, only **1.5 of the 3 pages** are actually
run through extraction/classification per document, due to early exit on the (likely majority)
of confidently-classified documents.

```
3,333,333 docs/day × 1.5 pages actually processed ≈ 5,000,000 page-inferences/day
5,000,000 / 86,400s ≈ 57.9 page-inferences/sec (average, post-early-exit)
```

This adjusted number, not the raw 115.7 pages/sec, is the one that should drive GPU
provisioning — using the unadjusted number would over-provision compute that early-exit was
specifically designed to save (see prior ML-design notes, Ch 4.2).

### Step 4 — Estimate compute time per page-inference

**Assumption:** based on the earlier hands-on OCR/classification work, a single page's
extraction + classification (OCR or direct text extraction, plus a forward pass through the
classifier) takes roughly **1 second of GPU time** on a modern single GPU, averaged across
digital-text pages (cheap, no OCR) and scanned/photo pages (OCR-bound, more expensive).

```
5,000,000 page-inferences/day × 1 GPU-second/page-inference ≈ 5,000,000 GPU-seconds/day
5,000,000 GPU-seconds/day / 86,400s/day ≈ 57.9 continuous-GPU-equivalents

Accounting for realistic GPU utilization (~70%, due to batching gaps, warm-up, non-uniform
arrival): 57.9 / 0.70 ≈ 82.7 GPUs needed, continuously running, at steady-state average load.
```

**This number — roughly 80+ GPUs running continuously at steady state — is the single most
important output of this lesson.** It's the figure that later chapters (Ch 5.4 worker pool
design, Ch 7.1 bottleneck analysis, Ch 11.1 cost breakdown) treat as the baseline to scale from.

### Step 5 — Estimate storage

**Assumption:** average raw file size 1MB (mix of small ID-document photos and larger
multi-page scanned PDFs).

```
Raw document storage: 100,000,000 docs/month × 1MB ≈ 100TB/month of new raw storage
```

**Extracted artifacts** (OCR text, layout regions, confidence scores) are much smaller — a few
KB per document — and are negligible next to raw file storage, but still need a place to live
(Ch 4.1).

**Embeddings** (relevant once the open-set/embedding classification architecture from the
prior ML-design notes, Ch 5.2, is in play): a typical embedding vector (e.g., 768-dimensional
float32) is `768 × 4 bytes ≈ 3KB`. Even storing one embedding per document for every document
processed:

```
100,000,000 docs/month × 3KB ≈ 300GB/month
```

— modest compared to raw file storage, but still a real, growing number worth tracking,
especially once vector search over embeddings becomes part of the classification path (Ch 9.3).

## Summary Table (Steady-State Averages at 100M docs/month)

| Metric | Value |
|---|---|
| Documents/day | ≈ 3.33M |
| Documents/sec (average) | ≈ 38.6 |
| Batch documents/sec (average) | ≈ 30.9 |
| Real-time documents/sec (average) | ≈ 7.7 |
| Page-inferences/sec (post-early-exit, average) | ≈ 58 |
| Continuous GPUs needed (steady state, ~70% utilization) | ≈ 80–85 |
| New raw storage/month | ≈ 100TB |
| New embedding storage/month | ≈ 300GB |

## Trade-offs in the Estimation Approach Itself

| Choice | Gain | Cost |
|---|---|---|
| Using explicitly labeled assumptions (3 pages/doc, 1s/page, 70% utilization) rather than a single "trust me" final number | Every number is auditable and replaceable the moment real data exists; disagreements can be resolved by challenging a specific assumption, not the whole estimate | The final numbers are only as good as the assumptions — presenting them without the labels invites false confidence |
| Accounting for early-exit's effect on actual pages processed (Step 3 adjustment) | Avoids over-provisioning GPU capacity for pages that won't actually be run | Requires the aggregation strategy's real-world early-exit rate to be validated in production — if early-exit triggers less often than assumed (e.g., due to a poorly tuned confidence threshold), the true page-inference rate will be higher than planned |

## When to Revisit These Numbers

- As soon as any real production traffic exists, replace every assumption in this lesson
  (pages/doc, early-exit rate, GPU-seconds/page, average file size) with measured values.
- Re-run this estimation whenever the class taxonomy changes significantly (more classes can
  mean lower average confidence, which can reduce the early-exit rate and raise the effective
  page-inference rate — a direct link between Chapter 9's taxonomy scaling and this chapter's
  capacity numbers).

## Summary

Translating "100M documents/month" into actionable numbers requires working through traffic
split, pages-per-document, early-exit's effect on actual pages processed, per-page compute
cost, and storage — each step an explicit, labeled assumption rather than a black-box guess.
The headline output — roughly 80+ GPUs running continuously at steady state, split across an
80/30 batch-heavy and 20/8-real-time-heavy traffic pattern — is the number that grounds every
subsequent infrastructure sizing decision in this notes set.