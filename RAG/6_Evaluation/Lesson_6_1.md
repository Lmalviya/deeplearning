# Lesson 6.1 — Evaluation Philosophy: Offline vs. Online, Component-Level vs. End-to-End

---

## Why Evaluation Is Hard to Get Right

Building a RAG system is relatively straightforward. Knowing whether it is actually working well is surprisingly difficult.

The naive approach is to use the system and see if it feels right. This is worse than useless — human intuition is systematically biased toward recent, salient, memorable failures while missing consistent subtle failures. A system that spectacularly fails on 1% of queries feels broken even if it correctly handles 99%. A system that consistently returns answers that are 20% wrong feels fine in daily use until a consequential error surfaces.

Rigorous evaluation requires:
- Defining precisely what "good" means for your system.
- Measuring it quantitatively, not qualitatively.
- Measuring it on a representative sample of real queries, not cherry-picked examples.
- Measuring it at multiple levels (not just the final answer).
- Distinguishing between improvements that are real and those that are statistical noise.

This lesson covers the evaluation philosophy and framework. Lessons 6.2 through 6.7 cover the specific metrics for retrieval, generation, and RAG-specific evaluation.

---

## The Two Axes of Evaluation

RAG evaluation spans two independent axes:

**Axis 1 — Timing: Offline vs. Online**

Offline evaluation runs before deployment, on a fixed evaluation dataset, with known ground truth. It is the development and pre-deployment evaluation loop.

Online evaluation runs in production, on real user queries, using behavioral signals (clicks, ratings, reformulations, conversions). It is the production quality monitoring loop.

These are not alternatives — they are complementary. Offline evaluation catches regressions before deployment. Online evaluation catches real-world problems that evaluation datasets miss.

**Axis 2 — Scope: Component-Level vs. End-to-End**

Component-level evaluation measures individual pipeline stages in isolation: how good is retrieval? How good is re-ranking? How faithful is generation?

End-to-end evaluation measures the complete system output: does the final answer correctly satisfy the user's information need?

Again, not alternatives. Component evaluation tells you where problems live. End-to-end evaluation tells you whether those problems actually affect the user experience.

The combination of all four quadrants gives a complete picture:

| | Offline | Online |
|---|---|---|
| **Component** | Retrieval recall on eval set, re-ranker NDCG | Retrieval cache hit rate, latency per stage |
| **End-to-End** | Answer correctness on eval set | User satisfaction, reformulation rate |

---

## Offline Evaluation

### What It Is

You create an evaluation dataset of (query, expected_answer, optional: relevant_chunk_ids) triples. You run the system on all queries in this dataset and compare the system's outputs to the expected answers using defined metrics.

### Building an Evaluation Dataset

This is the hardest part. A good evaluation dataset has three properties:

**Representative:** It covers the distribution of real queries your users ask — not just easy factual lookups, but also ambiguous questions, multi-hop questions, out-of-scope questions, and edge cases.

**Diverse:** It covers different document types, question types, difficulty levels, and domains within your corpus.

**Challenging:** It should include cases where naive RAG would fail — questions requiring multi-hop reasoning, questions where the answer is in an obscure section, questions outside the corpus scope (to test IDK behavior).

**Sources for evaluation data:**

1. **LLM-generated synthetic QA pairs:** For each chunk in a sample of your corpus, ask an LLM to generate 3-5 diverse questions that the chunk answers. Fast, scales to thousands of pairs.

```python
async def generate_eval_pairs_from_chunk(
    chunk_text: str,
    chunk_id: str,
    doc_metadata: dict,
    llm_client,
    n_questions: int = 4
) -> list[dict]:
    """
    Generate evaluation QA pairs from a chunk.
    """
    prompt = f"""Generate {n_questions} diverse questions that can be answered 
from the following text. Include a mix of:
- Simple factual questions
- Questions requiring inference
- Questions about specific numbers or dates
- Questions a real user would ask

For each question, provide the answer based on the text.

Text:
{chunk_text}

Return JSON array:
[
    {{
        "question": "question text",
        "answer": "answer from the text",
        "question_type": "factual | inferential | numerical | temporal",
        "difficulty": "easy | medium | hard"
    }}
]"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=800,
        temperature=0.5  # Some variety in question generation
    )
    
    import json
    pairs = json.loads(response.choices[0].message.content)
    
    # Attach metadata
    for pair in pairs:
        pair["relevant_chunk_id"] = chunk_id
        pair["doc_id"] = doc_metadata.get("doc_id")
        pair["doc_title"] = doc_metadata.get("doc_title")
    
    return pairs
```

2. **User query logs:** Real queries from your deployed system (if available). The most representative source. Requires manual annotation of ground truth answers.

3. **Expert-created questions:** Domain experts write questions that require deep knowledge of the corpus. Highest quality but expensive.

4. **Adversarial questions:** Questions designed to expose failure modes — ambiguous queries, questions outside scope, questions that look answerable but are not, questions with misleading surface similarity to wrong answers.

### Evaluation Dataset Split

Like any machine learning evaluation, split your evaluation data:

- **Development set:** Used during development to guide improvement decisions. You look at this regularly.
- **Test set:** Held out. Used only for final evaluation before deployment. Prevents overfitting the system to the development evaluation set.
- **Regression set:** A small, curated set of previously failing cases you have fixed. Run on every change to ensure you have not reintroduced known failures.

Typical sizes: development (200-500 pairs), test (100-200 pairs), regression (50 critical cases).

### The Overfitting Problem in RAG Evaluation

If you develop the system while looking at development set metrics, you will unconsciously optimize for the evaluation set rather than general performance. This is evaluation overfitting — the system scores well on the evaluation set but not on real queries.

Mitigations:
- Reserve the test set strictly. Only run it when you believe the system is ready for deployment.
- Rotate evaluation questions periodically. A static set eventually gets memorized into development decisions.
- Use online evaluation as the ultimate ground truth. If offline metrics improve but online metrics do not, the offline evaluation is not representative.

---

## Online Evaluation

### What It Is

Online evaluation measures system performance using signals from real users in production. The ground truth is not predefined — it is inferred from user behavior.

### Explicit Feedback Signals

Users directly rate the quality of responses:
- Thumbs up / thumbs down on each response.
- Star ratings.
- "Was this helpful?" binary.
- Written feedback.

```python
async def record_user_feedback(
    query_id: str,
    response_id: str,
    feedback_type: str,  # "thumbs_up", "thumbs_down", "rating", "text"
    feedback_value,  # bool for thumbs, int for rating, str for text
    user_id: str = None
):
    """Record user feedback for a query-response pair."""
    await feedback_store.insert({
        "query_id": query_id,
        "response_id": response_id,
        "feedback_type": feedback_type,
        "feedback_value": feedback_value,
        "user_id": user_id,
        "timestamp": datetime.utcnow()
    })
    
    # Update running metrics
    if feedback_type == "thumbs_up" or feedback_type == "thumbs_down":
        await metrics_store.increment(
            key=f"feedback.{feedback_type}",
            window="daily"
        )
```

Explicit feedback is high quality but sparse — most users do not rate responses. Typical feedback rates are 2-5% of queries.

### Implicit Behavioral Signals

Infer quality from user behavior without asking for explicit ratings:

**Query reformulation rate:** If a user immediately asks the same question again in slightly different words, the first answer did not satisfy them.

```python
def detect_reformulation(
    query: str,
    next_query: str,
    embedding_model,
    similarity_threshold: float = 0.75,
    max_time_gap_seconds: int = 60
) -> bool:
    """
    Detect if the next query is a reformulation of the current query.
    High similarity + short time gap = reformulation (user was unsatisfied).
    """
    from datetime import timedelta
    
    query_emb = embedding_model.embed(query)
    next_emb = embedding_model.embed(next_query)
    
    similarity = cosine_similarity(query_emb, next_emb)
    
    return similarity >= similarity_threshold
```

**Conversation continuation:** Does the user engage further with the answer (asking follow-up questions) or abandon the conversation? Continuation suggests satisfaction.

**Citation clicks:** If the UI shows source citations, do users click to verify them? High click-through may indicate either high trust (want to read more) or low trust (want to verify). Context matters.

**Session abandonment:** Did the user leave immediately after receiving an answer, or continue using the system? Quick abandonment can signal dissatisfaction.

**Answer copy rate:** If the user copies the answer to their clipboard, it was probably useful.

### Online Evaluation Infrastructure

```python
class OnlineEvaluationTracker:
    def __init__(self, metrics_backend):
        self.metrics = metrics_backend
    
    def track_query(self, query_id: str, query: str, response: str, 
                    latency_ms: float, metadata: dict):
        """Track a new query-response pair."""
        self.metrics.record({
            "event": "query",
            "query_id": query_id,
            "query_length": len(query.split()),
            "response_length": len(response.split()),
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow(),
            **metadata
        })
    
    def track_reformulation(self, original_query_id: str, new_query_id: str):
        """Record that a query was reformulated."""
        self.metrics.increment("reformulation_count")
        self.metrics.tag(original_query_id, "reformulated")
    
    def compute_daily_metrics(self) -> dict:
        """Compute daily aggregated quality metrics."""
        queries = self.metrics.get_daily_queries()
        
        return {
            "total_queries": len(queries),
            "thumbs_up_rate": self._compute_thumbs_rate(queries, "up"),
            "thumbs_down_rate": self._compute_thumbs_rate(queries, "down"),
            "reformulation_rate": self._compute_reformulation_rate(queries),
            "avg_latency_p50_ms": self._percentile(queries, "latency_ms", 50),
            "avg_latency_p95_ms": self._percentile(queries, "latency_ms", 95),
            "avg_latency_p99_ms": self._percentile(queries, "latency_ms", 99),
            "idk_rate": self._compute_idk_rate(queries)
        }
```

---

## The Measurement Hierarchy

Not all metrics are equally important. Build a measurement hierarchy so you know which metrics to act on and which are leading indicators vs. lagging indicators.

**North Star metric:** The single metric that most directly captures whether the system is delivering value. For an internal knowledge assistant, this might be "daily active users" or "questions resolved without escalation to human support." This is your ultimate measure of success.

**Primary quality metrics:** The metrics that directly predict the North Star. For RAG: user satisfaction rate (thumbs up), answer accuracy on eval set, IDK rate (are users getting helpful redirects when the answer is not available?).

**Secondary quality metrics:** Metrics that explain the primary metrics. Retrieval recall@K, faithfulness score, context relevance. These help diagnose why primary metrics are good or bad.

**Diagnostic metrics:** Metrics used during debugging and development but not tracked in production dashboards. Per-query latency breakdown, chunk-level relevance scores, individual query failure analysis.

```
North Star
    │
    ├── Primary quality metrics (weekly review)
    │       ├── User satisfaction rate
    │       ├── Eval set accuracy
    │       └── IDK rate
    │
    ├── Secondary quality metrics (on degradation alerts)
    │       ├── Retrieval recall@K
    │       ├── Faithfulness score
    │       └── Context relevance
    │
    └── Diagnostic metrics (ad hoc debugging)
            ├── Per-stage latency
            ├── Per-query failure analysis
            └── Chunk-level scores
```

---

## Evaluation-Driven Development

The right way to develop a RAG system is to let evaluation drive every decision:

**Baseline measurement first.** Before making any change, measure your current system's performance on the evaluation set. This is your baseline. Every subsequent change is measured relative to this baseline.

**One change at a time.** Change one component (chunking strategy, embedding model, re-ranker, prompt) and measure its impact before making the next change. Changing multiple things simultaneously makes it impossible to attribute improvements or regressions.

**Statistical significance.** With a small evaluation set (200 questions), a 2% improvement might be noise. Use statistical tests (McNemar's test for binary outcomes, paired t-test for continuous scores) to determine whether observed differences are significant.

```python
from scipy import stats
import numpy as np

def is_improvement_significant(
    baseline_scores: list[float],
    new_scores: list[float],
    alpha: float = 0.05
) -> dict:
    """
    Test whether the new system significantly outperforms the baseline.
    Uses paired t-test (assumes scores are paired by query).
    """
    assert len(baseline_scores) == len(new_scores), "Must have scores for same queries"
    
    t_stat, p_value = stats.ttest_rel(new_scores, baseline_scores)
    
    mean_improvement = np.mean(new_scores) - np.mean(baseline_scores)
    
    return {
        "mean_improvement": mean_improvement,
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "direction": "improvement" if mean_improvement > 0 else "regression",
        "conclusion": (
            f"Statistically significant {'improvement' if mean_improvement > 0 else 'regression'} "
            f"(p={p_value:.3f}, mean diff={mean_improvement:.3f})"
            if p_value < alpha
            else f"No significant difference (p={p_value:.3f})"
        )
    }
```

**Never deploy without measuring.** Every change to the RAG system — new embedding model, different chunk size, modified prompt, updated re-ranker — must be evaluated on the evaluation set before deployment. No exceptions.

---

## Connecting Offline and Online Evaluation

Offline evaluation is only useful if it predicts online performance. Validate this connection periodically:

```python
def correlate_offline_online_metrics(
    offline_metrics: dict,
    online_metrics: dict
) -> dict:
    """
    Check correlation between offline evaluation metrics and online metrics.
    High correlation = offline eval is a good proxy for real-world quality.
    """
    
    # Track over time: when offline metrics improved, did online metrics also improve?
    # When offline metrics degraded, did online metrics also degrade?
    
    offline_trend = offline_metrics.get("accuracy_trend")  # List of daily accuracy values
    online_trend = online_metrics.get("satisfaction_trend")  # List of daily satisfaction rates
    
    if not offline_trend or not online_trend:
        return {"correlation": None, "insufficient_data": True}
    
    # Align by date and compute correlation
    min_len = min(len(offline_trend), len(online_trend))
    
    correlation, p_value = stats.pearsonr(
        offline_trend[-min_len:],
        online_trend[-min_len:]
    )
    
    return {
        "pearson_correlation": correlation,
        "p_value": p_value,
        "interpretation": (
            "Strong correlation — offline eval is a good proxy" if abs(correlation) > 0.7
            else "Weak correlation — offline eval may not represent real-world quality"
        )
    }
```

If offline and online metrics diverge consistently (offline improves but online does not), your evaluation dataset is not representative of real queries. Refresh it with recent real user queries.

---

## The Evaluation Cadence

How often to evaluate at each level:

| Evaluation type | Frequency | Trigger |
|---|---|---|
| Regression set | Every code change | CI/CD pipeline |
| Development eval set | Every experiment | Manual or experiment tracking |
| Full offline eval | Weekly | Scheduled |
| Online metrics dashboard | Daily | Scheduled |
| Online deep dive | Weekly | Or when metrics alert |
| Offline/online correlation | Monthly | Scheduled |
| Eval dataset refresh | Quarterly | Or when correlation weakens |

---

## Summary

- Evaluation requires defining precisely what "good" means, measuring it quantitatively, and measuring it on representative data — not cherry-picked examples.
- Two axes of evaluation: offline (pre-deployment, fixed dataset, known ground truth) vs. online (production, real queries, behavioral signals); component-level vs. end-to-end.
- Offline evaluation dataset quality is the foundation. Build representative, diverse, challenging datasets with multiple question types. Split into development, test, and regression sets.
- Online evaluation uses explicit feedback (thumbs, ratings) and implicit behavioral signals (reformulation rate, abandonment, copy). Explicit is high quality, low volume. Implicit is high volume, noisier.
- Build a metric hierarchy: North Star → primary quality → secondary quality → diagnostic. Know which tier each metric belongs to.
- Evaluation-driven development: baseline first, one change at a time, statistical significance testing, never deploy without measuring.
- Validate that offline metrics predict online metrics. When they diverge, refresh the evaluation dataset.

---

## What's Next

Lesson 6.2 covers retrieval metrics in depth — Precision@K, Recall@K, MRR, MAP, NDCG, Hit Rate, and Coverage — with the mathematical definitions, intuitions, and how to interpret each in the context of RAG systems.