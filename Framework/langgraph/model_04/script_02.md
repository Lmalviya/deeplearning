# Video 18: A simple multi-agent pattern — the supervisor

**Target length:** 14–16 minutes
**Goal:** Learner builds a basic supervisor pattern — one node routes to specialized nodes based on task, and those nodes hand control back. Keep this conceptual and lightweight; deep multi-agent orchestration is production-course territory.
**Golden rule:** This is a *pattern*, not a new syntax lesson — every piece used here (conditional edges, routing functions) is already known. Frame it that way explicitly.

---

## Slide 1 — Recap + hook

**On slide:** "You've built graphs with a handful of nodes. What about coordinating whole specialized agents?"

**Speech:**
"So far, every node we've built does one focused job. But what if the work genuinely needs different kinds of expertise — research, writing, reviewing? One option is a single do-everything node. A better option, and today's topic, is a supervisor that delegates to specialists."

---

## Slide 2 — Diagram: the supervisor pattern

**On slide:** *(recreate the diagram shown above — Supervisor node routing to Research, Writer, and Reviewer nodes, with a return path back to Supervisor, and a separate route to End)*

**Speech:**
"Here's the shape we're building. A supervisor node looks at the current task and decides which specialist should handle it next. That specialist does its work, then hands control back to the supervisor — who decides what happens next: maybe another specialist, maybe we're done."

---

## Slide 3 — Why this beats one giant node

**On slide:**
- "Each specialist has a focused, simple job — easier to prompt well"
- "Easier to test and improve one specialist without touching others"
- "The supervisor's only job is deciding, not doing"

**Speech:**
"You could try to build one enormous prompt that does research, writing, and reviewing all at once — but in practice, that tends to produce mediocre results at everything. Splitting responsibility means each specialist can have a tightly focused prompt, and you can improve or debug one without touching the others."

---

## Slide 4 — Live code: the specialist nodes

**Switch to code editor.**

```python
def research_node(state: TaskState) -> dict:
    result = llm.invoke(f"Research this topic: {state['task']}")
    return {"result": result.content, "stage": "researched"}

def writer_node(state: TaskState) -> dict:
    result = llm.invoke(f"Write based on this research: {state['result']}")
    return {"result": result.content, "stage": "written"}

def reviewer_node(state: TaskState) -> dict:
    result = llm.invoke(f"Review and improve this draft: {state['result']}")
    return {"result": result.content, "stage": "reviewed"}
```

**Speech:**
"Three ordinary nodes — the exact same pattern from video 3, nothing new. Each one reads state, does a focused job, returns an update. Notice the `stage` field — that's how the supervisor will know what's already been done."

---

## Slide 5 — Live code: the supervisor's routing logic

**Switch to code editor.**

```python
def supervisor_node(state: TaskState) -> str:
    if state["stage"] == "start":
        return "research"
    if state["stage"] == "researched":
        return "write"
    if state["stage"] == "written":
        return "review"
    return "end"
```

**Speech:**
"And here's the supervisor's decision logic — genuinely just a routing function, exactly like video 5's `add_conditional_edges` pattern, checking a `stage` field to decide what comes next. For a real system, this could instead be an LLM call deciding dynamically — but a simple stage check is a great way to learn the pattern before adding that complexity."

---

## Slide 6 — Wiring it up

**Switch to code editor.**

```python
builder.add_node("research", research_node)
builder.add_node("write", writer_node)
builder.add_node("review", reviewer_node)

builder.add_conditional_edges(START, supervisor_node, {
    "research": "research", "write": "write", "review": "review", "end": END
})
builder.add_conditional_edges("research", supervisor_node, {"write": "write"})
builder.add_conditional_edges("write", supervisor_node, {"review": "review"})
builder.add_conditional_edges("review", supervisor_node, {"end": END})
```

**Speech:**
"Notice the supervisor logic is reused across every conditional edge — it's genuinely one function doing all the deciding, wired in at each handoff point. This is the essence of the supervisor pattern: one central place holding the coordination logic, while each specialist stays simple and focused."

---

## Slide 7 — Running it end to end

**Switch to code editor.** Run with a task, print the `stage` and `result` after each step (or stream it, from video 11).

**Speech:**
"Let's run this and watch it move through research, writing, and review in sequence." *(run live)* "Same underlying mechanics as everything else this course — conditional edges and routing functions — just applied to coordinate whole specialized steps instead of single decisions."

---

## Slide 8 — Where this goes from here (brief, don't dive in)

**On slide:** "Real multi-agent systems add: dynamic LLM-based routing, parallel specialists, error recovery — production-course territory."

**Speech:**
"What we built today is a deliberately simple version of this pattern — a fixed sequence of specialists. Real production multi-agent systems often add dynamic routing where an LLM decides the order itself, specialists running in parallel, and recovery logic when a specialist fails. All genuinely interesting, and out of scope for a beginner course — that's exactly the kind of thing we'll dig into in the production course."

---

## Slide 9 — Recap + what's next

**On slide:**
- "Supervisor pattern = one routing function coordinating several specialist nodes"
- "Every mechanism used: conditional edges and routing functions you already know"
- "Next: the capstone project — everything from this entire course, combined"

**Speech:**
"That wraps up module four, and honestly, the whole foundational part of this course. Every mechanism you've learned — state, nodes, edges, conditional routing, loops, tools, memory, human approval, and now multi-step coordination — comes together next in the capstone project. Let's go build something real."

---

## Production notes

- **Keep the routing logic simple and rule-based (stage checks), not LLM-based, for this video.** Dynamic LLM routing is a natural next question from curious viewers — acknowledge it exists (slide 8) but don't build it here, it adds a layer of complexity that would blur the core pattern this video teaches.
- **This is a good video to genuinely tie the whole course together verbally** — the recap (slide 9) should feel like a real milestone, since it is one.