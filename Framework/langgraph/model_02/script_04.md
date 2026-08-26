# Video 11: Streaming outputs

**Target length:** 10–12 minutes
**Goal:** Learner understands why streaming matters for real UX, and can consume a `.stream()` call to see an agent's steps as they happen instead of waiting for one final result.
**Golden rule:** Motivate this with UX, not just syntax — show the "waiting in silence" problem live before showing the fix, same danger-then-fix structure as video 6.

---

## Slide 1 — Recap + hook

**On slide:** "Your agent works. But right now, it's a black box until it's done."

**Speech:**
"Everything we've built so far uses `.invoke()` — you send a request, and you wait, in total silence, until the entire thing finishes. For a multi-step tool-using agent, that silence can be several seconds, sometimes longer. Today we fix that."

---

## Slide 2 — Show the problem live

**Switch to code editor.** Run a multi-tool-call agent with `.invoke()`, and just... wait, visibly, on screen, with nothing happening.

**Speech:**
"Watch this — I'm running our agent from a couple videos ago, something that needs a few tool calls to answer. Nothing appears until it's completely finished." *(let it sit in silence for a few real seconds)* "In a real product, that silence is exactly where users start wondering if something's broken. Let's fix it."

---

## Slide 3 — Live code: .stream() instead of .invoke()

**Switch to code editor.**

```python
for chunk in agent.stream({"messages": [("user", "your question here")]}):
    print(chunk)
```

**Speech:**
- "Same agent, same input — the only change is `.stream()` instead of `.invoke()`, and looping over what it gives back."
- "Instead of one final result, you get a sequence of chunks as the graph actually executes — each one showing you what just happened at that step."

"Let's run it..." *(run live)* "...and now you can see each step land as it happens — the agent's decision, then the tool executing, then the agent again — instead of dead silence followed by one big result."

---

## Slide 4 — What's actually in a chunk

**Switch to code editor.** Print one chunk cleanly and walk through its shape.

**Speech:**
"Each chunk is keyed by the node that just ran. So you'll see something like a `'agent'` key with that step's output, then a `'tools'` key with the tool result, and so on. This is genuinely useful beyond just UX — it's also a great debugging view, since you're watching your graph execute step by step in real time."

---

## Slide 5 — Where this matters most: a simple UI mental model

**On slide:**
- "Show 'thinking...' the moment the agent starts reasoning"
- "Show tool calls as they happen ('searching the web...', 'calculating...')"
- "Stream the final answer's text as it's generated, not all at once"

**Speech:**
"In a real app, you'd use this to drive a UI — the moment a tool call chunk comes through, you might show 'searching...' in your interface. When the final answer starts streaming, you'd render it token by token instead of dumping the whole paragraph at once. We're not building a UI in this course, but understanding this is what makes that possible later."

---

## Slide 6 — Recap + what's next

**On slide:**
- "`.stream()` instead of `.invoke()` — see steps as they happen, not just the end result"
- "Also a genuinely useful debugging tool, not just a UX feature"
- "Next: Mini-project 2 — a tool-using research assistant that streams its reasoning"

**Speech:**
"So: swap `.invoke()` for `.stream()` when you want visibility into what's happening as it happens — for UX, and honestly, for your own debugging sanity too. Next video, we put everything from this module together — tools, the ReAct loop, and streaming — into a real project: a research assistant you can actually watch think."

---

## Production notes

- **The "silence" demo (slide 2) is worth actually sitting through on camera**, even a few uncomfortable seconds of dead air — that discomfort is the entire point, and cutting to make it feel snappier undermines why streaming matters.
- **Don't go deep into different stream modes (values vs. updates vs. messages) here.** One clear mode is enough for a beginner course — mention that other modes exist and point to docs for anyone who wants to go deeper, but don't turn this into a reference video.