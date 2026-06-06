# Lesson 7.3 — Context Window Management and Observability

---

## Part 1: Context Window Management

### The Problem: Agents Fill Their Context and Then Fail

An LLM's context window is its entire working memory. For a long-running agent task — a multi-hour research session, a complex debugging workflow, a customer support conversation spanning 50+ turns — the context fills up. When the context is full:
- Old information gets truncated (silently dropped by the framework).
- The agent loses track of earlier tool results.
- The original task goal may no longer be visible.
- Quality degrades as the agent "forgets" what it was doing.

Context management is not optional for production agents — it is required infrastructure.

---

### Strategy 1: Sliding Window

Keep only the last N turns of conversation. Drop the oldest turns as new ones arrive.

```python
def apply_sliding_window(
    messages: list[dict],
    max_messages: int = 20,
    always_keep: list[str] = ["system"]
) -> list[dict]:
    """
    Keep system messages always. Keep only last max_messages user/assistant turns.
    """
    system_messages = [m for m in messages if m["role"] in always_keep]
    non_system = [m for m in messages if m["role"] not in always_keep]
    
    # Keep only the most recent max_messages
    if len(non_system) > max_messages:
        non_system = non_system[-max_messages:]
    
    return system_messages + non_system
```

**Pros:** Simple, zero latency overhead.
**Cons:** Hard cutoff — important context from early in the conversation is lost permanently. The agent has no knowledge of what it dropped.

---

### Strategy 2: Summarization

When the context approaches a threshold (e.g., 75% of maximum tokens), compress the older portion into a summary:

```python
async def compress_context_with_summary(
    messages: list[dict],
    token_count: int,
    token_threshold: int,
    llm_client,
    keep_recent_n: int = 10
) -> list[dict]:
    """
    When context exceeds threshold:
    1. Summarize the older portion of the conversation.
    2. Replace it with the summary.
    3. Keep the most recent N turns verbatim for continuity.
    """
    
    if token_count < token_threshold:
        return messages
    
    system_messages = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    if len(conversation) <= keep_recent_n:
        return messages  # Nothing to compress
    
    # Split: older portion to summarize, recent portion to keep verbatim
    to_summarize = conversation[:-keep_recent_n]
    to_keep = conversation[-keep_recent_n:]
    
    # Summarize the older portion
    summary_prompt = f"""The following is the earlier portion of an agent conversation. 
Summarize the key information that would be needed to continue this task:
- What was the original goal?
- What has been accomplished so far?
- What tool calls were made and what were the key results?
- What decisions were made and why?
- What is still unresolved?

Keep the summary concise but complete. This will replace the detailed history.

CONVERSATION TO SUMMARIZE:
{format_messages_for_summary(to_summarize)}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=800,
        temperature=0.0
    )
    
    summary_text = response.choices[0].message.content
    
    # Replace old messages with a summary message
    summary_message = {
        "role": "system",
        "content": f"[CONVERSATION SUMMARY — EARLIER CONTEXT]\n{summary_text}\n[END SUMMARY — RECENT CONVERSATION FOLLOWS]"
    }
    
    return system_messages + [summary_message] + to_keep
```

**Pros:** Preserves key information in compressed form. Agent retains awareness of past.
**Cons:** Summary quality depends on summarization LLM. Fine details (exact tool call parameters, precise values) may be lost in compression.

---

### Strategy 3: Hierarchical Memory (Retrieve + Inject)

Don't store everything in context. Store full history externally (vector DB or structured store). At each turn, retrieve only the most relevant past context.

```python
async def build_context_with_retrieval(
    current_query: str,
    recent_turns: list[dict],
    history_store,
    embedding_model,
    max_retrieved: int = 3
) -> list[dict]:
    """
    Build context by combining:
    - Last N turns (always included)
    - Top-K semantically relevant past turns (retrieved from history)
    """
    
    # Always include recent turns
    base_context = recent_turns[-10:]  # Last 10 turns
    
    # Retrieve relevant past context by semantic similarity
    query_embedding = await embedding_model.embed(current_query)
    
    relevant_history = await history_store.search(
        query_embedding=query_embedding,
        k=max_retrieved,
        exclude_ids=[m["id"] for m in recent_turns]  # Don't duplicate recent turns
    )
    
    # Format retrieved history as a "relevant past context" block
    if relevant_history:
        retrieved_block = {
            "role": "system",
            "content": "[RELEVANT PAST CONTEXT]\n" + "\n".join([
                f"[Turn {h['turn_index']}] {h['role']}: {h['content'][:300]}"
                for h in relevant_history
            ]) + "\n[END PAST CONTEXT]"
        }
        return [retrieved_block] + base_context
    
    return base_context
```

**Pros:** Scales to unlimited history. Provides the most relevant past context, not just the most recent.
**Cons:** Retrieval adds latency. May miss important context that is not semantically related to the current query.

---

### Choosing a Strategy

| Session Length | Best Strategy |
|---|---|
| < 20 turns | None needed — full context fits |
| 20–100 turns | Sliding window + keep system prompt |
| 100–500 turns | Summarization when threshold hit |
| > 500 turns or multi-day | Hierarchical retrieval from external store |

In practice: combine strategies — keep recent 10 turns verbatim (sliding window), retrieve relevant past turns (retrieval), summarize medium-history (summarization). The system prompt is never compressed.

---

## Part 2: Observability

### Why Agent Observability Is Hard

In a traditional ML model, observability means monitoring inputs and outputs. In an agent, the interesting things happen *between* inputs and outputs: the reasoning steps, the tool calls, the observations. These intermediate steps are invisible without deliberate instrumentation.

"My agent gave a wrong answer" is much harder to diagnose without knowing: which tool did it call? With what parameters? What did the tool return? Which step did the reasoning go wrong?

Observability makes the agent's internal process visible, traceable, and debuggable.

---

### What to Log: The Complete Agent Trace

Every agent execution should produce a structured trace:

```python
class AgentTrace:
    """
    A complete, structured record of one agent execution.
    Captured automatically by the framework, not the agent.
    """
    
    def __init__(self, trace_id: str, session_id: str, user_id: str):
        self.trace_id = trace_id
        self.session_id = session_id
        self.user_id = user_id
        self.start_time = datetime.utcnow()
        self.steps = []
        self.metadata = {}
    
    def record_step(self, step_type: str, content: dict):
        """Record one step in the agent's execution."""
        step = {
            "step_index": len(self.steps),
            "step_type": step_type,  # "thought", "tool_call", "observation", "response"
            "timestamp": datetime.utcnow().isoformat(),
            "content": content,
            "latency_ms": None  # Filled in by the framework
        }
        self.steps.append(step)
    
    def record_tool_call(
        self,
        tool_name: str,
        parameters: dict,
        result: dict,
        latency_ms: int,
        success: bool
    ):
        self.record_step("tool_call", {
            "tool_name": tool_name,
            "parameters": parameters,  # What was sent to the tool
            "result": result,          # What the tool returned
            "latency_ms": latency_ms,
            "success": success,
            "error": result.get("error") if not success else None
        })
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "total_steps": len(self.steps),
            "total_duration_ms": (datetime.utcnow() - self.start_time).total_seconds() * 1000,
            "steps": self.steps,
            "metadata": self.metadata
        }
```

---

### Key Metrics to Monitor in Production

| Metric | What it signals | Alert threshold |
|---|---|---|
| Mean steps per task | Efficiency; high = inefficient or looping | > 20 steps for simple tasks |
| Tool call failure rate | Tool reliability; high = external dependency issues | > 5% |
| Context utilization % | How full the context window is | > 80% (approaching limit) |
| Task completion rate | Overall agent effectiveness | < target rate |
| P95 latency per step | Individual step slowness | > 5s per step |
| LLM error rate | API quota/rate limit issues | > 1% |
| Guardrail trigger rate | How often inputs/outputs are blocked | Sudden spike |
| Tool call cost per task | Total spend; high = inefficient tool use | Exceeds budget threshold |

---

### Distributed Tracing for Multi-Agent Systems

In a multi-agent system, a single user request spawns multiple agent executions. You need to link them:

```python
# Every agent execution gets a trace_id and a parent_trace_id
# The orchestrator's trace_id becomes the parent for all worker traces

orchestrator_trace = AgentTrace(
    trace_id="orch-001",
    parent_trace_id=None,  # Root
    ...
)

worker_trace = AgentTrace(
    trace_id="worker-001",
    parent_trace_id="orch-001",  # Child of orchestrator
    ...
)
```

With a parent-child trace relationship, you can reconstruct the full execution graph:
```
User request
  └── Orchestrator [orch-001] — 800ms
        ├── Research Worker [worker-001] — 500ms (parallel)
        │     ├── search_web call — 200ms
        │     └── search_web call — 180ms
        └── Analysis Worker [worker-002] — 490ms (parallel)
              └── compute_metrics call — 350ms
```

This is the same pattern used in distributed systems (OpenTelemetry, Jaeger). Agent frameworks like LangSmith, LangFuse, and Weights & Biases provide agent-specific trace visualization.

---

> **Interview note:** *"How do you debug an agent that produces wrong outputs?"*
> Structured observability: every step of the agent's reasoning (thoughts, tool calls, observations, final response) is logged with timestamps, parameters, and results. When debugging, you replay the trace step by step: which step produced the wrong intermediate result? Was it the tool call parameters? The tool's actual response? The LLM's interpretation of the observation? Common root causes: wrong tool selected (bad tool description), wrong parameters (LLM hallucinated a parameter value), tool returned unexpected format (observation was confusing), or context saturation (agent lost track of the original goal). You find the root cause in the trace — without traces, debugging is guesswork.

---

## Summary

- **Context management** is required for long-running agents: sliding window (simple, loses info), summarization (preserves compressed history), hierarchical retrieval (external store, scales to unlimited history).
- Choose strategy by session length: < 20 turns (none needed), 20–100 (sliding window), 100–500 (summarization), > 500 (retrieval-based).
- **Observability**: log every step (thought, tool call, observation, response) with timestamps, parameters, results, and latency. This makes agent behavior debuggable.
- Key production metrics: steps per task, tool failure rate, context utilization, task completion rate, cost per task.
- **Distributed tracing** for multi-agent: parent_trace_id links worker executions to their orchestrator. Reconstruct the full execution tree for debugging.
