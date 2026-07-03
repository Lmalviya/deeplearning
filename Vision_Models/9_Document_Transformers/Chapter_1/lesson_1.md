# 1.1 Functional and Non-Functional Requirements

## Problem

Before any architecture decision makes sense, the system needs an explicit, written-down
statement of what it must do and how well it must do it. Skipping this step is how systems end
up over-engineered in the wrong dimension (e.g., heavy real-time infra for a workload that's
actually 80% tolerant batch) or under-engineered in a dimension that quietly becomes the
bottleneck later (e.g., no plan for adding classes, so class #51 requires a redesign). Every
later chapter's decisions trace back to the numbers and constraints stated here.

## Functional Requirements

1. **Accept heterogeneous document submissions** — digital PDF, scanned PDF, handwritten scan,
   photo — as established in the earlier ML-design notes (content extraction, Chapter 2 of the
   prior notes set still applies unchanged as the extraction layer).
2. **Return exactly one document-level class label** (with a confidence score) per submitted
   document, for documents that may span multiple pages.
3. **Support both submission modes:**
   - **Batch mode** — no caller waiting synchronously; results retrieved later via polling,
     webhook, or a downstream export.
   - **Real-time mode** — a caller (often an end user in an upload flow) is waiting on the
     response within the same interaction.
4. **Support a growing, versioned class taxonomy** — starts at 5 classes, must be able to grow
   toward ~50 without requiring a full system redesign or downtime for existing traffic.
5. **Support human review and correction** — some fraction of predictions (low-confidence,
   randomly sampled, or explicitly flagged) must be reviewable by a human, and corrections must
   be capturable as feedback for future improvement (this is a first-class requirement, not an
   afterthought — see Chapter 10.3).
6. **Idempotent resubmission** — submitting the same document twice (e.g., due to a client
   retry after a timeout) must not produce duplicate processing or duplicate billing/records.

## Non-Functional Requirements

| Dimension | Requirement | Why it's stated this way |
|---|---|---|
| **Scale target** | ~100M documents processed at steady state (working assumption: **100M documents/month** at full target scale — this assumption is used for all capacity math in Lesson 1.2, and should be replaced with real product numbers as soon as they exist) | Everything downstream — DB partitioning, queue throughput, GPU fleet size — is sized against this number. Getting the assumption wrong by 10x is a real risk; it's stated explicitly so it can be corrected. |
| **Traffic mix** | **80% batch / 20% real-time**, by document count | This is not a minor detail — it directly shapes queue design (Ch 5.2), worker pool design (Ch 5.3), and cost architecture (Ch 11), since batch and real-time traffic have fundamentally different latency tolerances and should not share one undifferentiated processing path. |
| **Latency SLO — real-time** | Target: p95 end-to-end response under a few seconds (exact number depends on product UX; used as ~5s for capacity math in this notes set) | A user watching an upload spinner has a much tighter tolerance than a background job; this SLO is what forces the parallel-orchestration and worker-headroom decisions in later chapters. |
| **Latency SLO — batch** | Target: completed within a bounded window (e.g., a few hours), not instant | Batch has no human waiting, so the SLO is about a completion deadline, not per-document latency — this is what allows queue depth to absorb spikes instead of requiring instant capacity (Ch 1.3, Ch 7.3). |
| **Class taxonomy growth** | Must support 5 → 50+ classes without full retraining or downtime | Directly rules out a plain fixed-softmax-head architecture as the long-term design (see Ch 9) and requires schema support for adding classes live (Ch 4.3). |
| **Spike tolerance** | Must survive traffic bursts well above steady-state average without violating real-time SLOs or losing batch documents | Real systems don't grow smoothly — sized explicitly in Lesson 1.3. |
| **Durability** | Submitted documents and their predictions must not be lost once accepted, even across a processing failure | Drives the choice to persist and acknowledge receipt *before* processing begins (Ch 3.1, Ch 5.1), not after. |
| **Cost sensitivity** | Cost per document processed should decrease (or at least not scale linearly) as volume grows toward 100M/month | Rules out "run every document through the most expensive model" as a viable end-state; motivates tiered/cascaded design (Ch 11.2). |
| **Auditability** | It must be possible to trace why a given document received a given label | Motivates storing intermediate artifacts (OCR text, confidence scores, model version used) rather than only the final label (Ch 4.2). |

## Trade-offs in How Requirements Are Stated

| Choice | Gain | Cost |
|---|---|---|
| Stating the 100M/50-class target now, but not designing for it yet | Every later decision can be sanity-checked against the real end-state number, avoiding a design that "works today" but has a known dead end | Requires discipline to actually revisit and validate the assumption as real data arrives, or the whole capacity plan is built on a guess |
| Explicit 80/20 batch/real-time split as a stated requirement, not an emergent property | Forces queue/worker/API design to treat the two lanes differently from the start, avoiding a later painful split of an undifferentiated system | If the real split turns out to be different (e.g., 50/50), several downstream sizing decisions need to be revisited — worth re-measuring early in production rather than trusting the assumption indefinitely |

## Summary

The system must ingest heterogeneous documents, return one document-level label per document
across two very different traffic patterns (80% tolerant batch, 20% latency-sensitive
real-time), support a class taxonomy that will grow roughly 10x, and survive both steady growth
and sudden spikes — all while keeping cost per document from scaling linearly with volume. These
requirements, especially the explicit 100M-doc/month and 80/20 traffic-split assumptions, are
the numbers every later capacity and architecture decision in this notes set is checked against.