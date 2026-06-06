# Lesson 6.6 — Online Evaluation: A/B Testing, User Feedback Signals, and Implicit Signals

---

## Why Offline Evaluation Is Not Enough

Offline evaluation on a fixed dataset tells you how the system performs on the questions you thought to include. It does not tell you:

- How real users with real problems experience the system.
- Whether an improvement on the eval set translates to improvement in actual use.
- Which user segments benefit from a change and which are hurt.
- Whether the system is getting better or worse over time on production traffic.

Online evaluation closes this gap. It measures quality using the ultimate ground truth: whether real users are actually satisfied with the answers they get.

The challenge is that you rarely know directly whether an answer was correct. Users rarely rate every response. You need indirect, behavioral signals to infer quality at scale.

---

## The Online Evaluation Stack

A production RAG system needs three layers of online evaluation:

**Layer 1 — Direct feedback:** Explicit user ratings on individual responses.
**Layer 2 — Behavioral signals:** Implicit quality indicators derived from user behavior.
**Layer 3 — A/B testing:** Controlled experiments to measure causal impact of changes.

These layers give you different things. Direct feedback is high quality but sparse. Behavioral signals are high volume but noisy. A/B testing gives you causality, not just correlation.

---

## Layer 1 — Direct Feedback Collection

### Feedback UI Design

The design of the feedback interface significantly affects response rate and quality.

**Thumbs up/down:** Binary. Simple. Gets high response rates (2-5% of queries). Coarse signal.

**Star rating (1-5):** More nuanced. Lower response rates. Requires users to think more.

**Specific attribute rating:** "Was this accurate? Was this helpful? Was it too long?" — very low response rates but highly actionable when received.

**Inline correction:** "The correct answer is..." — extremely valuable but rarely submitted.

**Recommendation:** Use thumbs up/down as the primary signal (high response rate). Add a text box for thumbs-down that asks "What was wrong?" This gets you the nuanced feedback when something actually went wrong, which is exactly when you need it most.

```python
class FeedbackCollector:
    def __init__(self, feedback_store):
        self.store = feedback_store
    
    async def record_feedback(
        self,
        session_id: str,
        query_id: str,
        query: str,
        response: str,
        retrieved_chunk_ids: list[str],
        feedback_type: str,       # "thumbs_up" | "thumbs_down" | "star" | "text"
        feedback_value,           # bool for thumbs, int for star, str for text
        user_context: dict = None # department, role, etc. (if available)
    ) -> str:
        """Record user feedback. Returns feedback_id."""
        
        feedback_id = generate_id()
        
        record = {
            "feedback_id": feedback_id,
            "session_id": session_id,
            "query_id": query_id,
            "query": query,
            "response_preview": response[:200],
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "feedback_type": feedback_type,
            "feedback_value": feedback_value,
            "timestamp": datetime.utcnow(),
            "user_context": user_context or {}
        }
        
        await self.store.insert(record)
        
        # Alert on consecutive negative feedback (may indicate systematic issue)
        await self._check_feedback_streak(session_id)
        
        return feedback_id
    
    async def _check_feedback_streak(
        self,
        session_id: str,
        streak_threshold: int = 3
    ):
        """Alert if a session has multiple consecutive thumbs-down."""
        recent = await self.store.get_recent_by_session(
            session_id=session_id,
            limit=streak_threshold,
            feedback_type="thumbs_down"
        )
        
        if len(recent) >= streak_threshold:
            await alert(
                f"Session {session_id} has {streak_threshold} consecutive negative ratings — "
                "possible systematic failure"
            )
```

### Feedback Rate and Coverage

In most RAG systems, explicit feedback covers only 2-5% of queries. This means your online quality signal has a serious sampling bias: users who rate tend to be either very satisfied or very dissatisfied. The silent majority — users who found the answer acceptable but not worth rating — are underrepresented.

Mitigation strategies:
- Actively solicit feedback on a random 10% sample of queries (not just show a passive thumbs-up button).
- Route high-stakes queries (keywords suggesting important decisions) to mandatory feedback.
- Use LLM-judged quality on a random sample to supplement the sparse explicit feedback.

---

## Layer 2 — Behavioral Signals

Behavioral signals are proxy measures of user satisfaction derived from how users interact with the system — not what they say, but what they do.

### Signal 1 — Query Reformulation Rate

A user reformulates a query when the first answer was unsatisfactory. They rephrase the same question (or a closely related one) shortly after receiving an answer.

```python
class ReformulationDetector:
    def __init__(self, embedding_model, time_window_seconds: int = 120):
        self.embedder = embedding_model
        self.time_window = time_window_seconds
    
    async def detect_reformulation(
        self,
        session_queries: list[dict]  # Ordered by timestamp
    ) -> list[dict]:
        """
        Identify pairs of queries where the second is a reformulation of the first.
        """
        reformulations = []
        
        for i in range(1, len(session_queries)):
            current = session_queries[i]
            previous = session_queries[i - 1]
            
            # Time gap check
            time_gap = (current["timestamp"] - previous["timestamp"]).total_seconds()
            if time_gap > self.time_window:
                continue  # Too long a gap — probably a new topic
            
            # Semantic similarity check
            embeddings = self.embedder.encode(
                [previous["query"], current["query"]],
                normalize_embeddings=True
            )
            import numpy as np
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            
            if similarity >= 0.75:
                reformulations.append({
                    "original_query_id": previous["query_id"],
                    "original_query": previous["query"],
                    "reformulated_query_id": current["query_id"],
                    "reformulated_query": current["query"],
                    "similarity": similarity,
                    "time_gap_seconds": time_gap
                })
        
        return reformulations


async def compute_reformulation_rate(
    sessions: list[list[dict]],  # List of sessions, each a list of queries
    detector: ReformulationDetector
) -> dict:
    """Compute reformulation rate across sessions."""
    total_queries = sum(len(s) for s in sessions)
    total_reformulations = 0
    
    for session in sessions:
        reformulations = await detector.detect_reformulation(session)
        total_reformulations += len(reformulations)
    
    # Queries that were reformulated at least once
    reformulation_rate = total_reformulations / total_queries if total_queries > 0 else 0
    
    return {
        "reformulation_rate": reformulation_rate,
        "total_queries": total_queries,
        "total_reformulations": total_reformulations,
        "interpretation": (
            "Good" if reformulation_rate < 0.05
            else "Acceptable" if reformulation_rate < 0.10
            else "High — users frequently dissatisfied with answers"
        )
    }
```

**Target:** Reformulation rate below 5-8%. Above 10% is a sign of systematic answer quality issues.

### Signal 2 — Session Abandonment After Response

If a user submits a query, receives a response, and immediately ends the session (or has no further activity for > 5 minutes), there are two interpretations:
1. The answer was so complete that the user's need was fully met (positive signal).
2. The answer was so bad that the user gave up (negative signal).

Context disambiguates which is which: if the query was simple and the session abandonment follows a clear, complete-looking answer, it is likely positive. If the query was complex and the answer was short or unclear, abandonment is likely negative.

```python
def classify_abandonment(
    query: str,
    response: str,
    session_duration_seconds: float,
    was_first_query: bool
) -> str:
    """
    Classify whether session abandonment is likely positive or negative.
    """
    response_words = len(response.split())
    query_words = len(query.split())
    
    # Heuristic signals of negative abandonment
    negative_signals = 0
    
    if response_words < 30:
        negative_signals += 1  # Very short response to a real question
    
    if query_words > 10 and response_words < 50:
        negative_signals += 2  # Long query, very short answer
    
    if "I don't" in response or "cannot find" in response:
        negative_signals += 1  # IDK response leading to abandonment
    
    if session_duration_seconds < 10:
        negative_signals += 2  # Abandoned very quickly
    
    if negative_signals >= 3:
        return "negative_abandonment"
    elif negative_signals == 0 and response_words > 100:
        return "positive_completion"
    else:
        return "ambiguous"
```

### Signal 3 — Citation Click-Through

If your RAG system shows source citations with clickable links, track whether users click them:

**High click rate on citations:** Could mean the system is generating good answers that make users want to read more (positive), OR that users do not trust the answer and want to verify (negative). Query-level context disambiguates.

**Zero citation clicks on a question that should be verifiable:** May indicate users trusted the answer implicitly — or never noticed the citations.

This signal is most useful as a relative measure: changes in citation click rate after a system update are more informative than absolute rates.

### Signal 4 — Response Copy Rate

If you can detect clipboard copy events, responses that users copy are almost certainly useful. This is a high-precision positive signal (few false positives — copying a useless answer is rare).

### Signal 5 — Time-to-Next-Query

After receiving an answer, how long before the user asks the next question?

- Very short gap (< 10s) + similar topic → likely reformulation (negative).
- Medium gap (10s-60s) + different topic → likely read the answer, moved on (positive).
- Very long gap (> 5 minutes) → session may effectively be over.

---

## Layer 3 — A/B Testing

A/B testing is the only way to establish that a pipeline change actually causes an improvement in user outcomes, rather than just correlating with them.

### Design Principles for RAG A/B Tests

**What you are testing:** A specific change to one pipeline component — not the whole system. "Adding cross-encoder re-ranking" is a good A/B test scope. "Redesigning the entire pipeline" is not.

**Traffic split:** Randomly assign incoming sessions (not queries) to treatment (new system) or control (current system). Session-level assignment prevents the same user from getting inconsistent experiences within one conversation.

**Metrics to track:** Define primary metric and secondary metrics before running the experiment. Do not go fishing for significant metrics after the fact.

**Minimum detectable effect:** Calculate the minimum improvement you care about and the sample size needed to detect it at statistical significance before starting.

```python
import scipy.stats as stats
import numpy as np

def compute_required_sample_size(
    baseline_rate: float,       # e.g., 0.78 (78% thumbs-up rate)
    minimum_detectable_effect: float,  # e.g., 0.03 (want to detect 3% improvement)
    alpha: float = 0.05,       # Significance level (5%)
    power: float = 0.80        # Statistical power (80%)
) -> dict:
    """
    Compute the required sample size per arm for a proportions A/B test.
    """
    treatment_rate = baseline_rate + minimum_detectable_effect
    
    # Using normal approximation for proportions
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    p_avg = (baseline_rate + treatment_rate) / 2
    
    n = (
        (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) + 
         z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) + 
                          treatment_rate * (1 - treatment_rate)))**2
    ) / (treatment_rate - baseline_rate)**2
    
    n = int(np.ceil(n))
    
    return {
        "required_n_per_arm": n,
        "total_queries_needed": n * 2,
        "baseline_rate": baseline_rate,
        "treatment_rate": treatment_rate,
        "minimum_detectable_effect": minimum_detectable_effect,
        "alpha": alpha,
        "power": power
    }

# Example: baseline thumbs-up rate is 78%, want to detect 3% improvement
sample_size_info = compute_required_sample_size(
    baseline_rate=0.78,
    minimum_detectable_effect=0.03
)
print(f"Need {sample_size_info['required_n_per_arm']} queries per arm")
# Example output: ~2,000 per arm (~4,000 total)
```

### A/B Test Infrastructure

```python
import hashlib

class ABTestRouter:
    def __init__(self, experiments: dict):
        """
        experiments: {
            "exp_001": {
                "name": "Cross-encoder re-ranking",
                "control": "pipeline_v2",
                "treatment": "pipeline_v2_with_reranker",
                "traffic_split": 0.5,  # 50% to treatment
                "status": "running",   # running | paused | concluded
                "start_date": "2024-06-01"
            }
        }
        """
        self.experiments = experiments
    
    def assign_variant(self, session_id: str, experiment_id: str) -> str:
        """
        Deterministically assign a session to control or treatment.
        Same session always gets same variant (consistent experience).
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment or experiment["status"] != "running":
            return "control"
        
        # Hash session_id + experiment_id for deterministic assignment
        hash_input = f"{session_id}:{experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Map to [0, 1) range
        normalized = (hash_value % 10000) / 10000
        
        if normalized < experiment["traffic_split"]:
            return "treatment"
        else:
            return "control"
    
    def get_pipeline_for_session(
        self,
        session_id: str,
        active_experiments: list[str]
    ) -> dict:
        """
        Get the pipeline configuration for this session given active experiments.
        """
        assignments = {}
        pipeline_overrides = {}
        
        for exp_id in active_experiments:
            variant = self.assign_variant(session_id, exp_id)
            assignments[exp_id] = variant
            
            exp = self.experiments[exp_id]
            if variant == "treatment":
                pipeline_overrides[exp_id] = exp["treatment"]
            else:
                pipeline_overrides[exp_id] = exp["control"]
        
        return {
            "assignments": assignments,
            "pipeline_overrides": pipeline_overrides
        }


class ABTestAnalyzer:
    def __init__(self, feedback_store, query_log_store):
        self.feedback = feedback_store
        self.queries = query_log_store
    
    async def analyze_experiment(
        self,
        experiment_id: str,
        primary_metric: str = "thumbs_up_rate",
        min_queries_per_arm: int = 500
    ) -> dict:
        """
        Analyze A/B test results and determine if the treatment is better.
        """
        
        # Fetch feedback by variant
        control_data = await self.feedback.get_by_experiment_variant(
            experiment_id=experiment_id,
            variant="control"
        )
        treatment_data = await self.feedback.get_by_experiment_variant(
            experiment_id=experiment_id,
            variant="treatment"
        )
        
        n_control = len(control_data)
        n_treatment = len(treatment_data)
        
        if n_control < min_queries_per_arm or n_treatment < min_queries_per_arm:
            return {
                "status": "insufficient_data",
                "n_control": n_control,
                "n_treatment": n_treatment,
                "required": min_queries_per_arm
            }
        
        # Compute primary metric
        if primary_metric == "thumbs_up_rate":
            control_positive = sum(1 for d in control_data if d.get("feedback_value") == True)
            treatment_positive = sum(1 for d in treatment_data if d.get("feedback_value") == True)
            
            control_rate = control_positive / n_control
            treatment_rate = treatment_positive / n_treatment
            
            # Two-proportion z-test
            count = np.array([control_positive, treatment_positive])
            nobs = np.array([n_control, n_treatment])
            
            z_stat, p_value = stats.proportions_ztest(count, nobs)
            
        elif primary_metric == "reformulation_rate":
            control_reform = sum(1 for d in control_data if d.get("was_reformulation"))
            treatment_reform = sum(1 for d in treatment_data if d.get("was_reformulation"))
            
            control_rate = control_reform / n_control
            treatment_rate = treatment_reform / n_treatment
            
            count = np.array([n_control - control_reform, n_treatment - treatment_reform])
            nobs = np.array([n_control, n_treatment])
            z_stat, p_value = stats.proportions_ztest(count, nobs)
        
        else:
            raise ValueError(f"Unknown metric: {primary_metric}")
        
        relative_improvement = (treatment_rate - control_rate) / control_rate
        
        return {
            "experiment_id": experiment_id,
            "status": "analyzed",
            "n_control": n_control,
            "n_treatment": n_treatment,
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "absolute_improvement": treatment_rate - control_rate,
            "relative_improvement": relative_improvement,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "recommendation": (
                "deploy_treatment" if (p_value < 0.05 and relative_improvement > 0)
                else "keep_control" if (p_value < 0.05 and relative_improvement < 0)
                else "inconclusive — collect more data"
            )
        }
```

### Common A/B Testing Mistakes in RAG

**Novelty effect:** Users try the new system because it is new, not because it is better. The new system gets inflated metrics during the first week. Run experiments for at least 2 weeks to allow novelty to wear off.

**Segment effects:** An improvement may help one user segment but hurt another. A change that adds more context may help power users but overwhelm casual users. Always check metrics by user segment alongside overall metrics.

**Session contamination:** If you assign variants at query level instead of session level, a user might get treatment in one query and control in another. This corrupts the experience and the measurement.

**Novelty/regression asymmetry:** Users notice and rate quality degradations more sharply than quality improvements. A 5% improvement may be invisible to users while a 5% degradation generates significant negative feedback. Account for this asymmetry in significance thresholds.

**Testing too many things simultaneously:** Running 5 experiments at the same time makes it impossible to attribute effects. Limit concurrent experiments to 2-3 maximum.

---

## Monitoring for Degradation

Beyond A/B testing (which is proactive), you need passive monitoring that alerts on unexpected quality degradation.

```python
class QualityMonitor:
    def __init__(self, metrics_store, alert_service):
        self.metrics = metrics_store
        self.alerts = alert_service
    
    async def check_daily_health(self) -> dict:
        """
        Run daily health checks and alert on anomalies.
        """
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        last_week_avg = await self.metrics.get_week_average(
            metric_names=["thumbs_up_rate", "reformulation_rate", "idk_rate", "p95_latency_ms"]
        )
        
        today_metrics = await self.metrics.get_daily(date=today)
        
        alerts = []
        
        # Check thumbs-up rate
        if today_metrics["thumbs_up_rate"] < last_week_avg["thumbs_up_rate"] * 0.90:
            alerts.append({
                "severity": "high",
                "metric": "thumbs_up_rate",
                "current": today_metrics["thumbs_up_rate"],
                "baseline": last_week_avg["thumbs_up_rate"],
                "drop_pct": (last_week_avg["thumbs_up_rate"] - today_metrics["thumbs_up_rate"]) / last_week_avg["thumbs_up_rate"]
            })
        
        # Check reformulation rate
        if today_metrics["reformulation_rate"] > last_week_avg["reformulation_rate"] * 1.25:
            alerts.append({
                "severity": "medium",
                "metric": "reformulation_rate",
                "current": today_metrics["reformulation_rate"],
                "baseline": last_week_avg["reformulation_rate"]
            })
        
        # Check IDK rate (sudden spike may indicate index gap)
        if today_metrics["idk_rate"] > last_week_avg["idk_rate"] * 1.50:
            alerts.append({
                "severity": "medium",
                "metric": "idk_rate",
                "current": today_metrics["idk_rate"],
                "baseline": last_week_avg["idk_rate"],
                "note": "May indicate data freshness issue or new query type"
            })
        
        # Check latency degradation
        if today_metrics["p95_latency_ms"] > last_week_avg["p95_latency_ms"] * 1.30:
            alerts.append({
                "severity": "high",
                "metric": "p95_latency_ms",
                "current": today_metrics["p95_latency_ms"],
                "baseline": last_week_avg["p95_latency_ms"]
            })
        
        for alert in alerts:
            await self.alerts.send(alert)
        
        return {
            "date": str(today),
            "metrics": today_metrics,
            "baseline": last_week_avg,
            "alerts": alerts,
            "health": "degraded" if any(a["severity"] == "high" for a in alerts) else
                      "warning" if alerts else "healthy"
        }
```

---

## Connecting Offline and Online Evaluation

The complete quality loop:

```
Offline evaluation ──→ Improvement decisions
        ↓                        ↓
   Deploy to A/B test ──→ Validate with real users
        ↓                        ↓
   Full rollout ──→ Continuous online monitoring
        ↓                        
   Alerts trigger diagnosis ──→ Offline root cause analysis
        ↓
   Fix and re-evaluate
```

The critical validation step: before deploying any offline improvement to 100% of traffic, first validate it with an A/B test. Offline improvements that do not improve online metrics (or hurt them) should not be deployed.

Track the correlation between your offline and online metrics over time. When they diverge, refresh your evaluation dataset — it has become unrepresentative of real usage.

---

## Summary

- Online evaluation measures quality on real users with real queries — the ultimate ground truth that offline evaluation approximates.
- Direct feedback (thumbs up/down) is high quality but sparse (2-5% of queries). Solicit actively on a random sample to avoid selection bias.
- Behavioral signals — reformulation rate, session abandonment, citation clicks, copy rate — proxy user satisfaction at high volume with lower quality. Use as complementary signals.
- Reformulation rate below 5-8% is good; above 10% indicates systematic answer quality issues.
- A/B testing is the only way to establish causality. Always assign at session level, define metrics before running, account for novelty effect (run 2+ weeks), check by user segment.
- Required sample size: ~2,000 queries per arm to detect a 3% improvement at 80% power (5% significance).
- Passive quality monitoring with automated alerts catches unexpected degradation between experiments.
- The full loop: offline evaluation → A/B test validation → full rollout → continuous monitoring → alerts → diagnosis → fix → back to offline evaluation.

---

## What's Next

Lesson 6.7 covers data drift and distribution shift — how to detect when your RAG system's inputs and outputs are shifting over time, and how to respond before quality degrades.