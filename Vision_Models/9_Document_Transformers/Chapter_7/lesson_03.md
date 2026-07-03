# 7.3 Handling the Spike Case in Practice

## Problem

Chapter 1.3 calculated *how much* capacity to provision for a spike (standing buffer, peak
multipliers) and Lesson 7.2 established *how* autoscaling responds to load. This lesson connects
the two operationally: walking through what actually happens, moment by moment, when a real
spike hits the system, and where each previously-designed mechanism (queue buffering,
autoscaling, standing floor, backpressure) actually engages.

## Solution / Concept: A Spike, Timeline-by-Timeline

### Real-time lane

```
T+0s     Spike begins — real-time submission rate climbs sharply toward or past the
         provisioned peak estimate (Ch 1.3: ~62 docs/sec).
T+0s     The standing capacity floor (Lesson 7.2) is already running and absorbs the
         initial surge — this is the entire point of paying for idle headroom ahead of time.
T+~30s   Reactive autoscaling metrics (utilization, requests-in-flight) cross their
         threshold; the autoscaler begins provisioning additional capacity.
T+2-5min New GPU-backed worker instances become healthy and start serving traffic.
         Until this point, the standing floor is the *only* thing preventing an SLO
         violation — if the spike's magnitude or duration exceeds what the floor covers,
         real-time latency SLO (Ch 1.1) begins degrading during this window.
T+?      If demand keeps climbing even after new capacity is online, and the autoscaling
         ceiling (a defined maximum, cost- or account-limited) is reached: backpressure
         engages (Ch 1.3) — new requests receive an explicit 429/Retry-After response
         rather than being silently queued past the latency SLO.
```

**The critical design commitment here:** the standing floor's size (Chapter 1.3) must be large
enough to cover the T+0 to T+2-5min gap for the *assumed* spike magnitude. If real observed
spikes turn out larger or faster-onset than the Chapter 1.3 assumption, the floor is
under-sized and needs to be revisited — this is exactly the kind of assumption flagged in
Chapter 1.2/1.3 as needing replacement with real data as soon as it exists.

### Batch lane

```
T+0s     Spike begins — batch submission rate climbs sharply (Ch 1.3 example: up to
         ~155 docs/sec).
T+0s     The queue absorbs the burst directly — depth grows, but no submission is
         rejected or delayed at the API layer; producers (Ch 5.1) keep accepting and
         enqueueing at whatever rate they can sustain.
T+~1-2min Queue-depth-driven autoscaling (Lesson 7.2) detects the growing backlog and
         begins provisioning additional batch workers, on a more relaxed timeline than
         the real-time lane needs, since nothing is waiting on an individual result.
T+ongoing Workers scale out gradually and clear the backlog. As long as total time-to-clear
         stays within the completion-window SLO (Ch 1.1), this entire sequence is
         invisible to the batch client beyond a normal completion time.
T+?      If the backlog's projected time-to-clear threatens to exceed the SLO (visible via
         queue-depth-trend monitoring, Ch 10.2), this should trigger an alert well before
         the SLO is actually breached — giving time to add temporary extra capacity
         manually, or communicate proactively with affected customers, rather than
         discovering the breach after the fact.
```

## Trade-offs

| Design element | Gain | Cost |
|---|---|---|
| Standing floor absorbing the real-time spike's first minutes | Protects the latency SLO during the unavoidable autoscaling-lag window | Pure idle-capacity cost during all the time there is no spike — accepted explicitly in Ch 1.3 |
| Queue absorbing the batch spike instead of instant capacity matching | Avoids paying for capacity that would sit idle outside of spikes; batch's SLO tolerates the delay | Requires active queue-depth-trend monitoring — an under-provisioned batch fleet with a growing, unwatched backlog can silently breach its completion-window SLO |
| Backpressure (429/Retry-After) as the real-time ceiling behavior | A predictable, explicit failure mode instead of an invisible SLO violation under extreme load | Requires client-side handling of the 429 response — a real commitment that has to be documented and tested with API consumers (Chapter 3.2), not just implemented silently on the backend |

## When to Use / When Not To

- **This exact sequencing (floor → reactive autoscale → backpressure for real-time; queue →
  gradual autoscale → alerting for batch)** is the correct operational model given the
  differentiated SLOs already established in Chapter 1.1 — it is not a hypothetical, it is the
  direct operational consequence of every design decision made in Chapters 1, 5, and 7.2.
- **Revisit the standing floor size and predictive schedule** the first time an actual observed
  spike either gets fully absorbed with room to spare (floor may be oversized, a cost
  opportunity) or comes close to exhausting it (floor may be undersized, a real risk) — this is
  exactly the kind of assumption that should be tuned against real data, not left at its
  initial estimate indefinitely.

## Summary

A spike is not a single event but a timeline: the real-time lane leans on its standing capacity
floor to survive the unavoidable gap before reactive autoscaling and, in the worst case,
backpressure can respond; the batch lane leans on queue depth to absorb the same relative burst
without needing instant capacity, provided the resulting backlog is actively monitored against
the completion-window SLO rather than left to grow silently. Both mechanisms were sized and
designed in Chapter 1.3 and Lesson 7.2 — this lesson is where those designs are traced through
to what actually happens when a spike hits the running system.