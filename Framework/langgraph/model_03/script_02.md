# Video 14: Inspecting and manipulating graph state

**Target length:** 10–12 minutes
**Goal:** Learner can use `get_state` to see exactly what's saved for a thread, and `update_state` to manually edit it — a genuinely useful debugging skill, and groundwork for human-in-the-loop next video.
**Golden rule:** Frame this explicitly as a debugging superpower, not just an API tour — the "why would I use this" needs to be obvious throughout.

---

## Slide 1 — Recap + hook

**On slide:** "Your agent remembers now. Today: you can actually look inside that memory."

**Speech:**
"Last video, we made state persist across calls using a checkpointer and a thread ID. Today, we open that up — you'll see exactly what's stored, and even learn to edit it directly. This is one of the most underrated debugging tools in LangGraph."

---

## Slide 2 — Live code: get_state

**Switch to code editor.**

```python
snapshot = graph.get_state(config)
print(snapshot.values)
```

**Speech:**
"`get_state`, given the same config with our thread ID, hands back a full snapshot of everything saved for that conversation — every field in our state, exactly as it currently stands. Let's look at one from our memory example last video." *(run live, inspect the messages list)* "This is genuinely useful any time your agent does something unexpected — instead of guessing what's in state, you can just look."

---

## Slide 3 — What else is in a snapshot (briefly)

**On slide:**
- "`.values` — the actual state data"
- "`.next` — which node would run next"
- "`.config` — the thread this snapshot belongs to"

**Speech:**
"A couple other useful fields on that snapshot object worth knowing: `.next` tells you which node is queued to run next — handy for confirming exactly where execution paused, which matters a lot once we cover human-in-the-loop next video. And `.config` just confirms which thread this snapshot belongs to."

---

## Slide 4 — Live code: update_state

**Switch to code editor.**

```python
graph.update_state(config, {"messages": [("user", "Actually, call me Alexandra.")]})
```

**Speech:**
"Now the more powerful move — `update_state` lets you directly inject a change into a thread's saved state, without running the graph at all. Here, I'm manually adding a message correcting the name. Let's run our earlier follow-up question again and see the effect." *(run live, show the updated behavior)* "This is powerful for debugging — you can simulate 'what if state had looked like this' without replaying an entire conversation."

---

## Slide 5 — A quick word on time travel (bonus, keep brief)

**On slide:** "You can also rewind to an earlier point and branch from there — worth knowing exists."

**Speech:**
"One more thing worth knowing exists, even though we won't build it today: LangGraph supports something like 'time travel' — rewinding a thread to an earlier checkpoint and continuing from there instead of the most recent state. It's genuinely powerful for debugging complex runs, but it's a deeper topic than this course needs right now — I'll link the docs if you want to explore it yourself."

*Keep this to one slide, don't demo it live — it's flagged as a pointer for the curious, not a required skill for this course.*

---

## Slide 6 — Recap + what's next

**On slide:**
- "get_state = see exactly what's saved for a thread"
- "update_state = manually edit saved state, no graph run required"
- "Next video: pausing a graph for a human to approve"

**Speech:**
"So: `get_state` to look, `update_state` to edit — both incredibly useful once you're debugging a real agent instead of a toy example. Next video, we use exactly this mechanism — a paused, inspectable state — to build something important: a human approval step before your agent takes a risky action."

---

## Production notes

- **Time travel (slide 5) is the easiest thing to cut if you're running long.** It's flagged as bonus for a reason — a single mention is enough, no live demo required.
- **Try to demo `update_state` on a scenario that visibly changes behavior afterward**, like the name-correction example — an update that doesn't change anything observable undersells why this matters.