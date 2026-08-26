# Video 15: Human-in-the-loop basics

**Target length:** 12–14 minutes
**Goal:** Learner can pause a graph before a risky node using `interrupt_before`, inspect the paused state, and resume — either approving or editing state first.
**Golden rule:** Motivate with a real risk scenario (sending an email, making a purchase, deleting something) — this concept needs a concrete stake to feel meaningful, not an abstract demo.

---

## Slide 1 — Recap + hook

**On slide:** "Some actions shouldn't happen without a human saying yes first."

**Speech:**
"Imagine an agent that can send emails on your behalf. Fully autonomous tool use is great — until the tool is something you really don't want it getting wrong unsupervised. Today: pausing your graph for a human to approve before a risky step happens."

---

## Slide 2 — The scenario

**On slide:**
- "Agent drafts an email"
- "Pauses before actually sending it"
- "Human approves — or edits — then it continues"

**Speech:**
"Here's what we're building toward: a graph with a `draft_email` node, and a `send_email` node. We want execution to pause *right before* `send_email` runs, so a human can look at the draft first."

---

## Slide 3 — Live code: interrupt_before

**Switch to code editor.**

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["send_email"],
)
```

**Speech:**
"One new argument at compile time — `interrupt_before`, a list of node names. Any node in this list, the graph will pause right *before* running it, and hand control back to you. Notice this needs a checkpointer too — from video 13 — because pausing and resuming later only works if state is actually being saved."

---

## Slide 4 — Live code: running until the pause

**Switch to code editor.**

```python
result = graph.invoke({"messages": [("user", "Email the team that the meeting moved to 3pm.")]}, config)
```

**Speech:**
"Let's run this." *(run live)* "Notice — it stops. It ran `draft_email`, and then paused, exactly where we told it to, right before `send_email`. Let's check what's queued next..."

```python
snapshot = graph.get_state(config)
print(snapshot.next)
```

"...and there it is — `send_email`, waiting. This is exactly the `get_state` skill from last video, now put to real use."

---

## Slide 5 — Approving and resuming

**Switch to code editor.**

```python
graph.invoke(None, config)
```

**Speech:**
"To approve and let it continue, we call `.invoke()` again on the same thread — but this time, passing `None` instead of new input. That tells LangGraph: don't add anything new, just continue from where you paused. Let's run it." *(run live, show the email 'sends')* "And now it completes the paused step."

---

## Slide 6 — Rejecting or editing before resuming

**Switch to code editor.**

```python
graph.update_state(config, {"draft": "Corrected draft text here..."})
graph.invoke(None, config)
```

**Speech:**
"And if a human wants to *change* something before approving — say, fix a typo in the draft — that's exactly last video's `update_state` skill. Edit the paused state directly, then resume with `None`. This is why we taught state inspection before this video — it's the exact mechanism a real approval flow depends on."

---

## Slide 7 — A quick note on the newer interrupt() function

**On slide:** "You may also see `interrupt()` used inline inside a node — a newer, more flexible pattern. Worth knowing it exists; check current docs for the latest recommended approach."

**Speech:**
"Quick heads up — LangGraph has been evolving fast in this area, and you may come across a newer pattern using an `interrupt()` function called directly inside a node, which offers more flexibility than `interrupt_before`. I'm teaching you the compile-time version today because it's the clearest way to understand the *concept* — pausing, inspecting, resuming — but if you're building something real, it's worth checking the current docs for whichever pattern is recommended at the time you're building."

*Being upfront about this is exactly the kind of honesty that builds trust — LangGraph's human-in-the-loop API is one of the faster-moving parts of the framework.*

---

## Slide 8 — Recap + what's next

**On slide:**
- "interrupt_before pauses a graph before a named node"
- "get_state + update_state = inspect and edit the pause"
- "invoke(None, config) resumes"
- "Next: mini-project 3, combining memory, inspection, and approval"

**Speech:**
"So: `interrupt_before` pauses, `get_state` and `update_state` let a human look and adjust, and `invoke(None, config)` resumes. This is genuinely one of the most important patterns for building agents you can actually trust with real-world actions. Next video, we combine everything from this module — memory, state inspection, and this approval gate — into one project."

---

## Production notes

- **Use a scenario with real stakes**, even if simulated — "sending an email" or "making a purchase" lands far better than an abstract example, because the *reason* for pausing needs to feel obvious.
- **The honesty slide about `interrupt()` (slide 7) matters more than it might seem.** This corner of LangGraph's API changes faster than most — flagging that openly protects your video's shelf life and your credibility if a viewer notices docs have moved on.