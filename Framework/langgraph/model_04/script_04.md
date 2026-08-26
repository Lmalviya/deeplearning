# Capstone Part 2: Adding memory and self-correction

**Target length:** 16–18 minutes
**Goal:** Layer persistent memory and a self-correction (critique/revise) pass onto part 1's skeleton — still no new concepts, pure combination.
**Golden rule:** Start by running part 1's skeleton live and pointing out its two remaining gaps (no memory, no self-check) before fixing either — keep the "show the gap, then fix it" rhythm from earlier videos.

---

## Slide 1 — Recap + hook

**On slide:** "Part 1 built the skeleton. Today: memory and a self-check pass."

**Speech:**
"Quick recap — last video we got a working tool-using assistant loop running. But it's got two real gaps: it forgets everything between calls, and it never double-checks its own answers. Let's fix both today."

---

## Slide 2 — Adding memory

**Switch to code editor.**

```python
graph = builder.compile(checkpointer=MemorySaver())
```

**Speech:**
"Straight from video 13 — a checkpointer at compile time. Combined with a consistent `thread_id` per conversation, exactly like before, this assistant now remembers context across the whole support conversation, not just one message."

---

## Slide 3 — Adding the critique step to state

**Switch to code editor.**

```python
def critique_node(state: SupportState) -> dict:
    last = state["messages"][-1].content
    review = llm.invoke(f"Critique this support answer for accuracy and helpfulness: {last}\nSay PASS if it's good, otherwise explain what to fix.")
    passed = "PASS" in review.content
    return {"critique": review.content, "iterations": state["iterations"] + 1}
```

**Speech:**
"This is genuinely the exact pattern from mini-project 1 — critique the latest answer, note whether it passed. The only difference from that project is *what* we're critiquing — a support answer instead of a piece of content — the mechanism is identical."

---

## Slide 4 — Wiring the self-correction loop around the existing agent loop

**Switch to code editor.**

```python
def should_revise(state: SupportState) -> str:
    if "PASS" in state["critique"] or state["iterations"] >= 3:
        return "done"
    return "revise"

builder.add_node("critique", critique_node)
builder.add_conditional_edges("agent", tool_or_critique, {"tools": "tools", "critique": "critique"})
builder.add_conditional_edges("critique", should_revise, {"revise": "agent", "done": END})
```

**Speech:**
"Here's the interesting part — we now have *two* different loops layered on the same graph: the tool-use loop from part 1, and this new critique-revise loop wrapped around it, using video 6's exact iteration-guard pattern. `tool_or_critique` decides: does the last message need a tool call, or is it a final answer ready for critique? Same routing-function skill, just checking a slightly different condition than before."

---

## Slide 5 — Running the combined system

**Switch to code editor.** Run a multi-turn conversation, with a question likely to trigger a revision, and show the critique loop engaging before a final answer is returned.

**Speech:**
"Let's run a real conversation." *(run live across a couple of turns, same `thread_id`)* "Watch — it remembers earlier context, uses a tool when it needs to, and this time, watch the critique step — if the first answer isn't quite good enough, it revises before ever showing us the final response."

---

## Slide 6 — Recap + what's next

**On slide:**
- "Now has: tools, memory, self-correction"
- "Still missing: a human approval gate before risky actions"
- "Next: adding that gate, and final polish"

**Speech:**
"So now our assistant remembers conversations and checks its own work before answering. One thing left, and it's an important one — remember that `file_support_ticket` tool from part 1? Right now, nothing stops it from filing a ticket completely on its own. Part 3: we fix that, and do a final polish pass with streaming."

---

## Production notes

- **Two loops in one graph (tool-use loop + critique loop) is genuinely the most complex wiring in the whole course.** Take it slower than usual here — walk through the routing functions on screen carefully rather than rushing past them.
- **Reuse the exact terminology from earlier videos** ("this is video 6's guard pattern," "this is mini-project 1's critique loop") — the repetition is doing real pedagogical work, not padding.