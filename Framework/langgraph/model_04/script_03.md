# Capstone Part 1: Project setup and core agent loop

**Target length:** 16–20 minutes
**Goal:** Stand up the skeleton of the capstone project — a support/research assistant with a working ReAct-style core loop — with nothing else added yet. Parts 2 and 3 will layer on tools+memory+self-correction, then human approval.
**Golden rule:** Zero new concepts across all three capstone videos. This is the "everything you've learned, one real build" moment — treat it that way explicitly and often.

---

## Slide 1 — Framing: the capstone

**On slide:**
- "One project, built across 3 videos, using every concept from this entire course"
- "A support assistant that: uses tools, remembers conversation, self-corrects, and asks before risky actions"

**Speech:**
"This is it — the project we've been building toward since video 1. Across the next three videos, we're building a real support assistant: it can use tools, remembers your conversation, checks and improves its own answers, and asks permission before doing anything risky. Nothing new to learn — just everything you already know, combined into one real system."

---

## Slide 2 — The spec

**On slide:**
- "Answers user questions, using tools when needed"
- "Remembers the conversation across turns"
- "Critiques and revises its own answer before responding"
- "Pauses for approval before taking any 'action' tool (e.g. filing a ticket)"

**Speech:**
"Let's define the full spec before we write a single line, same discipline as our mini-projects. A user chats with it — it can look things up with tools, it remembers earlier in the conversation, it double-checks its own answers before showing them, and if it ever needs to take a real action, like filing a support ticket, it stops and asks first."

---

## Slide 3 — Designing the state

**Switch to code editor.**

```python
class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    draft_answer: str
    critique: str
    iterations: int
```

**Speech:**
"State design, from video 2's core skill — what does every node actually need? A growing message history, using the reducer pattern from video 9. A `draft_answer` and `critique`, from our self-correction loop back in module one. And an `iterations` counter, from video 6's loop guard. We'll add more fields in part 2 as we layer in tools and the approval step — starting minimal, exactly as video 2 taught."

---

## Slide 4 — This part's scope: just the core loop

**On slide:** "Today: a working assistant node with tools, no memory, no self-correction, no approval yet — those come in parts 2 and 3."

**Speech:**
"For this first video, we're deliberately keeping scope narrow — just get a working tool-using assistant loop running, the ReAct pattern from module two. No memory, no self-correction, no approval gate yet. Build the skeleton first, add the rest on top of something that already works."

---

## Slide 5 — Live code: the assistant node and tools

**Switch to code editor.**

```python
@tool
def check_order_status(order_id: str) -> str:
    """Look up the status of an order by ID."""
    ...

@tool
def file_support_ticket(issue: str) -> str:
    """File a support ticket describing an issue."""
    ...

llm_with_tools = llm.bind_tools([check_order_status, file_support_ticket])

def assistant_node(state: SupportState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

**Speech:**
"Two tools for our support assistant — one to look something up, one that represents a real action, filing a ticket. That second one is deliberately the 'risky' tool we'll gate behind human approval in part 3 — flag it mentally now, we'll come back to it. The assistant node itself is exactly video 9's pattern, unchanged."

---

## Slide 6 — Wiring the core loop

**Switch to code editor.** Build the standard agent → tools → conditional edge → loop pattern from video 9, compile, and run a basic multi-turn tool-using exchange.

**Speech:**
"This wiring is identical to video 9 — agent node, tool node, a routing function checking for tool calls, and an edge looping back. Let's run it and confirm the skeleton actually works before we build anything on top of it." *(run live)*

---

## Slide 7 — Recap + what's next

**On slide:**
- "Working core loop: assistant + tools, no memory or approval yet"
- "Next: add memory and a self-correction pass"

**Speech:**
"That's our skeleton — a working tool-using assistant. In part 2, we add memory so it doesn't forget the conversation, and a self-correction pass so it checks its own answers before showing them, using the loop-guard pattern from way back in module one."

---

## Production notes

- **Resist the urge to add memory or approval in this video, even briefly.** Building strictly incrementally, one working layer at a time, is itself a lesson worth modeling for viewers on how to approach real projects.
- **Make sure this skeleton genuinely runs cleanly before moving to part 2** — a shaky foundation makes every subsequent layer harder to film and harder to follow.