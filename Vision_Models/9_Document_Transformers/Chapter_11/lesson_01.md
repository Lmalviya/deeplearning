# 11.1 Cost Drivers Breakdown

## Problem

At 100M documents/month, cost stops being a rounding error and becomes a real design
constraint (Chapter 1.1 explicitly states cost-per-document should not scale linearly with
volume). But "reduce cost" is meaningless without knowing *where* the cost actually
concentrates — optimizing database query performance, for instance, is wasted effort if GPU
inference dominates the bill by an order of magnitude. This lesson ranks the actual cost
drivers by relative magnitude, using the numbers already established in earlier chapters, so
Lesson 11.2's optimization effort targets the right place.

## Solution / Concept: Ranking the Cost Drivers

### 1. GPU inference — the dominant cost, by a wide margin

Chapter 1.2 estimated **~80–85 GPUs running continuously** at target steady-state volume — the
single largest infrastructure line item in the entire system, by a large margin over every
other driver below. Within this cost, one asymmetry is worth flagging explicitly: Chapter 5.3
established that the real-time worker pool uses minimal batching to protect latency, at the
cost of lower GPU utilization efficiency than the batch pool's aggressive batching. Concretely:
if real-time traffic is ~20% of volume (Chapter 1.2) but needs, say, meaningfully more GPU-
seconds per document than batch traffic due to this batching-efficiency gap, real-time can
consume a disproportionate share of the GPU budget relative to its share of document volume —
a real, quantifiable cost of the latency SLO stated in Chapter 1.1, not a hidden inefficiency.

### 2. OCR/HTR compute — a meaningful, controllable secondary cost

Within Extraction Service (Chapter 2.2), **direct text extraction from digital PDFs costs
nothing** (no model inference at all) — only pages without a usable text layer incur OCR/HTR
compute cost. This makes the **fraction of traffic that is genuinely digital-text vs.
scanned/photographed** a real, controllable cost lever: a system serving mostly born-digital
PDFs has a meaningfully lower extraction cost profile than one serving mostly scanned/photo
input, independent of document volume. Investing in an accurate, well-tuned text-layer
detection threshold (Chapter 2.1) directly avoids unnecessary OCR spend on pages that didn't
need it.

### 3. Storage — cheap per unit, but compounds over time

Chapter 1.2 estimated ~100TB/month of *new* raw storage. Critically, this is **cumulative, not
merely recurring** — by month 12 at steady state, accumulated raw storage approaches ~1.2PB,
even though per-GB cost is low. This makes **retention and archival policy** a real, ongoing
cost lever rather than a one-time sizing decision: moving older raw files to a cheaper
cold/archival storage tier (enabled cleanly by the time-based partitioning already designed in
Chapter 4.4, which makes identifying "old" data straightforward) directly controls this
growing cost line, and should be revisited periodically as accumulated volume grows, not set
once.

### 4. VLM fallback calls — small in volume, but real per-call cost

For any escalation path to a general VLM (referenced in the earlier ML-design phase's
CLIP+VLM-fallback pattern), each call carries a real, non-trivial per-call API cost, unlike the
embedding-based primary classification path's near-zero marginal cost. The entire point of the
tiered design in Lesson 11.2 is keeping this fallback rate small — a cited industry reference
point from that earlier design phase showed a **~4% fallback rate cutting total classification
cost roughly 10x** versus a VLM-only approach. This driver's actual cost impact is therefore a
direct function of how well the escalation threshold is tuned, not a fixed cost like the three
above.

### 5. Database I/O — the smallest driver, by design

Write load at target scale is modest (~77 writes/sec, Chapter 7.1), and read load — the part
that could otherwise grow into a real cost driver — is specifically kept low by two earlier
decisions: caching (Chapter 6) intercepting the hottest read paths, and partitioning (Chapter
4.4) keeping query performance from degrading as tables grow. This driver ranks last precisely
*because* Chapters 4 and 6 already addressed it proactively — a useful confirmation that those
earlier design decisions are paying for themselves at scale.

## Relative Magnitude Summary

```
GPU inference (Classification + OCR/HTR)  >>  Storage growth  >  VLM fallback (if kept rare)  >  Database I/O
```

## Trade-offs

| Cost driver | Primary lever | Where it's addressed |
|---|---|---|
| GPU inference | Batching policy per lane, worker pool sizing | Chapter 5.3, Chapter 7.2 |
| OCR/HTR compute | Accurate text-layer detection, encouraging digital-native submissions where possible | Chapter 2.1 |
| Storage growth | Retention/archival policy on top of time-based partitioning | Chapter 4.4 |
| VLM fallback calls | Escalation threshold tuning | Lesson 11.2 |
| Database I/O | Already addressed proactively | Chapter 4.4, Chapter 6 |

## When to Use / When Not To

- **Prioritize cost-optimization effort in the order ranked above** — GPU inference and
  storage archival policy first, since they dominate; database tuning last, since Chapters 4
  and 6 already did the proactive work that keeps it a minor driver.
- **Revisit this ranking periodically**, not just once — if, for example, the VLM escalation
  rate creeps upward over time (a drift-related risk connected to Chapter 10.2's monitoring),
  its relative cost ranking could shift meaningfully, and the ranking should be re-derived from
  current data rather than assumed to be permanently fixed.

## Summary

Cost at 100M-document scale is dominated by GPU inference by a wide margin, with a real,
quantifiable asymmetry between the real-time and batch lanes driven directly by the
batching-efficiency trade-off established in Chapter 5.3. Storage is the second-largest driver,
made manageable primarily through retention/archival policy rather than raw per-GB cost. VLM
fallback and database I/O are comparatively minor — specifically *because* earlier chapters
(tiered design, caching, partitioning) were designed with cost in mind from the start, not as
an afterthought bolted on at this chapter.