# Capstone — Applied Debugging Scenarios

Real interviews (and real on-call pages) don't hand you a labeled issue — they hand
you a symptom, sometimes with a red herring built in, and expect you to reason your
way to a small set of hypotheses and a plan to test them.

**How to use this chapter:** read only the "Symptom" for each scenario. Write down
your own hypotheses and the first one or two tests you'd run *before* reading the
Reasoning Walkthrough. The value here is entirely in the struggle — reading the
answer first defeats the purpose.

---

## Scenario 1: Retrieval Quality Degrades at Scale

**Symptom:** "With 1,000 documents, retrieval quality is good. When documents
increase to 1M, retrieval quality degrades. Nothing else changed — same embedding
model, same query set style, same chunking. Debug it."

<details>
<summary>Reasoning Walkthrough (click to expand after you've thought it through)</summary>

This is deliberately underspecified — the phrase "nothing else changed" is doing a
lot of work and should be treated with suspicion, not taken at face value.
Competing hypotheses, roughly in order of how cheap they are to check:

1. **ANN index approximation error** — most vector databases switch from exact
   nearest-neighbor search to an approximate index (HNSW, IVF, etc.) once the
   corpus crosses a size threshold, trading some recall for speed. *Cheap test:*
   run the same queries with exact (brute-force) search at 1M scale. If quality
   recovers, the index's approximation parameters (not the model) are the cause —
   fixed by tuning `ef_search`/`nprobe`-style parameters, or accepting the
   recall/speed tradeoff explicitly.
2. **Increased near-duplicate/confusable documents** — at 1K documents, there may
   simply be no close competitors to the correct answer; at 1M, many more
   documents are topically similar, exposing the embedding model's fine-grained
   discrimination weakness (Chapter on embeddings — insufficient hard-negative
   training) that was invisible at small scale because it was never tested.
3. **Embedding dimensionality / capacity ceiling** — a fixed-size embedding vector
   has to represent far more distinct concepts distinctly at 1M documents than at
   1K; if capacity is marginal, this is exactly the scale where it starts to show
   (see the embedding-dimensionality diagnostic: PCA explained variance, similarity
   histogram compression).
4. **Stale/inconsistent index** — a subtler infra check: was the full 1M corpus
   embedded and indexed with the *same* model version and consistently, or did the
   index accumulate documents embedded at different times/model versions
   (embedding drift)?

The right first move: run #1 (exact vs. approximate search) since it's a five-minute
config check and rules out the entire retrieval-infra branch before touching the
model at all. Only if quality is still bad under exact search does this become a
genuine model-capacity or discrimination question (#2/#3).

</details>

---

## Scenario 2: Loss Spikes to NaN Late in a Large Pretraining Run

**Symptom:** "We're pretraining a large transformer. Loss decreases smoothly for
50,000 steps, then suddenly spikes to NaN at step 50,241 and never recovers. The
run had been stable before this. What do you check?"

<details>
<summary>Reasoning Walkthrough</summary>

Competing hypotheses: a data anomaly in that specific batch, a numerical precision
issue (especially if training in fp16), or a gradient explosion that happened to
cross a threshold at that point.

First move: pull the exact batch fed at step 50,241 and inspect it directly — look
for an unusually long sequence, degenerate repeated tokens, or corrupted/encoding-
broken text. This is cheap and often conclusive on its own.

If the batch looks unremarkable, check whether training is in fp16 vs bf16/fp32 —
rerun a short window around that step in fp32 and see if the spike still occurs. If
it disappears in fp32, it's a precision/dynamic-range issue (likely in an
attention softmax or normalization operation overflowing), and the fix is
switching to bf16 or keeping sensitive ops in fp32, not just clipping harder.

If it reproduces even in fp32, look at logged gradient norms per layer in the steps
immediately before 50,241 — a sharp climb right before the spike (rather than an
instant jump) points to a genuine gradient explosion, fixed with clipping and/or a
brief LR reduction and restart from the last good checkpoint.

The key discipline: don't just add gradient clipping and move on without first
checking the batch — if it's a recurring data quality issue, clipping will paper
over it temporarily but the underlying corrupted-data problem will keep resurfacing.

</details>

---

## Scenario 3: Great Offline Metrics, Poor Production Performance

**Symptom:** "Our classifier gets 96% accuracy on our validation set. In production,
user complaints suggest it's wrong constantly — feels closer to 60-70% in practice.
Nothing was changed in the model or code between validation and deployment. Where
do you look?"

<details>
<summary>Reasoning Walkthrough</summary>

A large, unexplained gap between offline and production metrics is one of the
strongest signals to suspect the *evaluation*, not the model itself. Three
competing hypotheses:

1. **Data leakage** — check whether the validation set has any overlap or
   near-duplication with the training set (this alone can inflate offline accuracy
   dramatically without the model being wrong at all in production).
2. **Validation set not representative of production traffic** — pull a fresh
   sample of real production inputs and compare their distribution (topic, length,
   style, class balance) against the validation set. If they look meaningfully
   different, the validation set was measuring the wrong thing all along.
3. **A pipeline mismatch between training/eval preprocessing and the production
   serving path** — e.g., production applies different tokenization, truncation, or
   feature preprocessing than what was used to generate the validation set's
   inputs. Check this by manually tracing a single production request through the
   exact serving code and comparing its final model input against how the
   equivalent validation example is prepared.

Start with #3 if a serving pipeline exists as a separate codepath from the
training/eval pipeline — mismatched preprocessing between "how the model was
tested" and "how the model actually receives input in production" is an extremely
common, and easy to overlook, cause of exactly this kind of gap.

</details>

---

## Scenario 4: Fine-Tuning Improves the New Task but Breaks an Old One

**Symptom:** "We fine-tuned our model on customer support conversations and it got
noticeably better at that. But now it's much worse at writing code, which it used to
do well before this fine-tuning. We didn't touch anything related to code in the
fine-tuning data. Why?"

<details>
<summary>Reasoning Walkthrough</summary>

This is a fairly clean catastrophic forgetting signature (Chapter D5): fine-tuning
on a narrow new domain shifted weights in a way that degraded an unrelated
previously-strong capability, without anyone intending to touch that capability at
all.

Confirming check: directly evaluate the fine-tuned checkpoint against the
pre-fine-tuning checkpoint on the original code-eval suite — if there's a clear,
measurable regression there (not just an impression), that's the direct
confirmation, rather than relying on anecdotal "it feels worse."

Worth also checking, since the fine-tuning data was narrow (a single domain,
customer support): was the fine-tuning run for many epochs / at a fairly high
learning rate relative to how far you want the weights to move? Aggressive
fine-tuning on a narrow dataset is exactly the recipe that produces forgetting most
severely — a much lighter touch (fewer steps, lower LR, or mixing in a small amount
of general/code data during fine-tuning) is the standard fix, along with adopting a
standing regression-eval suite so this is caught before shipping next time, not
after user reports come in.

</details>

---

## Scenario 5: Long-Form Generation Gets Incoherent Toward the End

**Symptom:** "Our model produces coherent, high-quality text for the first
paragraph or two of long-form generation, but as the output gets longer, it
becomes increasingly repetitive and loses the thread of what it was writing about.
Short generations are consistently fine."

<details>
<summary>Reasoning Walkthrough</summary>

Several candidate mechanisms can produce this exact symptom, and they call for
different fixes, so distinguishing them matters:

1. **Exposure bias** — an early small error compounds as generation continues,
   since the model is now conditioning on its own imperfect output rather than
   ground truth. *Test:* compare teacher-forced loss on a long reference text
   against actual autoregressive generation quality on the same prompt — a big gap
   points here.
2. **Positional encoding / length extrapolation limit** — if the point where
   quality drops corresponds closely to the model's trained maximum context length,
   this is a positional-encoding-specific issue, not a general "gets worse over
   length" one. *Test:* check whether the degradation is a sharp cliff at a
   specific known length or a gradual decline across all lengths.
3. **Repetition tendency** — try varying decoding strategy (temperature, nucleus
   sampling, repetition penalty) with the model held fixed. If repetition
   specifically (as opposed to general incoherence) resolves substantially with
   decoding changes alone, that's primarily decoding-side, not a training issue.

Run all three cheap checks before concluding anything, since the fixes are
different: exposure bias points to scheduled sampling during training; a hard
length cliff points to positional encoding scheme or training data length coverage;
pure repetition responsive to decoding parameters points to a much cheaper,
non-retraining fix.

</details>

---

## Scenario 6: Bigger Batch Size Made Training Faster but Worse

**Symptom:** "We increased batch size from 256 to 8,192 to use our hardware more
efficiently and speed up training. Training got noticeably faster wall-clock-wise,
but final validation accuracy dropped compared to the batch-256 run. We didn't
change anything else."

<details>
<summary>Reasoning Walkthrough</summary>

"We didn't change anything else" is the detail to interrogate first: batch size and
learning rate are coupled — an 32x increase in batch size (256 → 8,192) with the
*same* learning rate is very likely under-scaled relative to the larger batch,
which on its own can hurt convergence quality independent of any deeper sharp-minima
story.

First move: check whether LR was scaled up (e.g., linearly, or with warmup adjusted)
to match the new batch size. If not, that's very likely most or all of the
explanation, and the fix is simply LR scaling + warmup — re-run with a properly
scaled LR before concluding anything more exotic.

If LR *was* scaled appropriately and the validation gap persists, then it's worth
investigating the sharp-minima hypothesis specifically: measure the sharpness of
the found minimum (small weight perturbation, check resulting loss increase) for
both batch-size runs. Confirming a demonstrably sharper minimum at the larger
batch size — after ruling out the much more common and mundane LR-scaling
explanation — is what justifies reaching for fixes like SAM or added regularization,
rather than jumping there first.

</details>

---

## Scenario 7: Attention Heads Look the Same for Every Input

**Symptom:** "We're inspecting attention weights in our transformer during error
analysis, and we notice most heads in the later layers seem to attend to the same
first token regardless of what we feed in. Is this a problem, and if so, what's the
fix?"

<details>
<summary>Reasoning Walkthrough</summary>

The instinct to check is whether this pattern is genuinely input-independent or
whether it's a legitimate learned behavior that happens to look similar across the
examples reviewed so far — some heads legitimately learn to consistently track a
structural token (e.g., a sentence-start marker) as part of useful behavior, and
that's not a bug.

The distinguishing test: feed several genuinely different inputs (different topics,
different lengths, different structures) through the model and compare the same
heads' attention patterns across these different inputs. If the pattern is
identical (or nearly so) regardless of how different the inputs are, that's
attention collapse — the head has stopped conditioning on content at all. If the
pattern shifts meaningfully with different input content even though it happens to
often land on early tokens, it may be legitimate.

If collapse is confirmed: check whether it correlates with an aggressive learning
rate early in training (a common cause — heads can collapse into a degenerate
shortcut early on and never recover) before assuming a fundamental architectural
flaw requiring an auxiliary loss or regularization change. The cheaper LR check
should come first.

</details>

---

## Scenario 8: Two Engineers Get Different Results From the "Same" Run

**Symptom:** "Two team members ran what should be an identical training job — same
script, same config file, same random seed — on our multi-GPU cluster, and got
noticeably different final validation accuracy. Neither made any code changes.
What's going on?"

<details>
<summary>Reasoning Walkthrough</summary>

The core clue is "multi-GPU cluster" — this points strongly toward distributed-
training nondeterminism rather than a genuine code or config difference (which
would likely be found immediately by diffing the two configs, a first cheap step
either way).

Confirming test: run the identical configuration on a *single* device (no
distributed training) multiple times. If single-device runs match closely but
multi-device runs don't, that isolates the nondeterminism specifically to the
distributed mechanism — likely non-deterministic gradient reduction order,
asynchronous data loading across workers, or a seed that isn't being propagated
correctly to every worker (e.g., only the main process's seed is set, not each data
loader worker's).

Before concluding "that's just expected variance," it's worth quantifying it: run
several single-device baseline repeats to establish what a normal noise floor looks
like, and compare the observed multi-GPU discrepancy against that floor. If the
discrepancy is much larger than the established baseline noise, that points to a
specific fixable bug (e.g., an unseeded data loader worker) rather than inherent,
acceptable distributed-training variance.

</details>

---

## A closing note on using this chapter

Notice a pattern across all eight: in nearly every case, **the cheapest, most
mundane hypothesis should be checked before the more interesting one.** Learning
rate scaling before sharp minima. The exact batch before assuming a numerical
instability theory. A serving pipeline mismatch before assuming the model itself
regressed. This ordering — cheap and mundane first, exotic and expensive last — is
itself the skill being tested, more than any single piece of textbook knowledge.