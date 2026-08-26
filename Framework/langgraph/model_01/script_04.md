# Video 4: Edges and your first real graph

**Target length:** 12–15 minutes
**Goal:** Learner builds and runs their first actual `StateGraph`, understands `add_node`, `add_edge`, `START`/`END`, `compile()`, and `invoke()` — and sees the manual chaining from video 3 replaced by the real thing.
**Golden rule:** This video pays off videos 2 and 3 directly — keep referencing back to the manual `.update()` chaining from last video ("this is what that was building toward"). Don't introduce conditional edges or loops yet — pure linear flow only.

---

## Slide 1 — Recap + hook

**On slide:** "You've got state. You've got nodes. Today: the thing that connects them."

**Speech:**
"Quick recap — we've got `AgentState` defining our shared notebook, and we've written nodes that read and update it. Last video, I had you manually chain node calls with `.update()`. Today, we replace that manual chaining with LangGraph's actual graph — and I promise, once you see it, it'll feel like a very small step from what you already did by hand."

---

## Slide 2 — StateGraph: the builder

**On slide:**
```
from langgraph.graph import StateGraph
builder = StateGraph(AgentState)
```

**Speech, narrating line by line:**
- "`from langgraph.graph import StateGraph` — this is the core class you'll import in nearly every LangGraph file you ever write."
- "`StateGraph(AgentState)` — we pass in the state schema we built in video 2. This is the moment those two videos connect: the graph builder needs to know the *shape* of the notebook before it can manage it."
- "`builder` is not a runnable graph yet — think of it as a blueprint you're still drawing on. We'll turn it into something runnable in a moment."

---

## Slide 3 — add_node: registering your functions

**Switch to code editor.**

```python
builder.add_node("generate", generate_answer)
builder.add_node("shout", uppercase_answer)
```

**Speech:**
"`add_node` takes two things: a string name for this step, and the actual function — the exact node functions we wrote last video, unchanged. That string name, `"generate"`, is how we'll refer to this step when we wire up edges next. It doesn't have to match the function name, but keeping them similar makes your graph much easier to read later."

---

## Slide 4 — add_edge, START, and END

**Switch to code editor.**

```python
from langgraph.graph import START, END

builder.add_edge(START, "generate")
builder.add_edge("generate", "shout")
builder.add_edge("shout", END)
```

**Speech, narrating line by line:**
- "`START` and `END` are special built-in markers — not nodes you write yourself. Every graph needs to know where execution begins and where it's allowed to stop."
- "`add_edge(START, "generate")` — this says: when the graph runs, begin at the `generate` node."
- "`add_edge("generate", "shout")` — after `generate` finishes, always go to `shout` next. This is an *unconditional* edge — no decision involved, just: this, then that."
- "`add_edge("shout", END)` — and after `shout`, we're done."

"If you forget that last edge to `END`, LangGraph will actually throw an error when you try to run this — it needs to know every path eventually terminates. Keep that in mind, it's a very common first-timer error."

*Flag this error explicitly — it's exactly the kind of real error message beginners will hit within their first ten minutes of coding, and knowing it's expected (not something they broke) saves real frustration.*

---

## Slide 5 — Diagram: what we just built

**On slide:** *(simple 3-box linear flow: START → generate → shout → END, straight arrows, gray boxes — matches the "linear chain" style from video 1's diagram)*

**Speech:**
"Here's the shape of what we just described in code. Nothing fancy yet — a straight line. This is intentionally the simplest possible graph, so the mechanics are crystal clear before we add any branching."

---

## Slide 6 — compile() and invoke()

**Switch to code editor.**

```python
graph = builder.compile()

result = graph.invoke({"user_question": "what is langgraph?", "answer": ""})
print(result)
```

**Speech:**
- "`builder.compile()` — this turns your blueprint into an actual runnable graph. You do this once, after you've finished adding all your nodes and edges."
- "`graph.invoke(...)` — and this is how you run it. You pass in a starting state — notice it matches the shape of `AgentState` — and LangGraph runs it through every node in order, exactly like our manual `.update()` chaining did last video, except now it's automatic."

"Let's run it..." *(run live, show the printed result)* "...and there it is — the same result we got by manually chaining functions last video, except this time, LangGraph did the chaining for us."

*This is the payoff moment of the whole video — make sure the connection back to video 3's manual version is explicit and lands clearly.*

---

## Slide 7 — Visualizing your graph (quick, optional)

**On slide:** "Bonus: you can actually see your graph as a diagram."

**Speech:**
"One quick bonus — LangGraph can generate an actual visual diagram of the graph you just built, which is genuinely useful once graphs get bigger than three nodes. I'll drop a link in the description on how to render this, since the exact setup can change — but know that this exists for when you're debugging something more complex later in the course."

*Keep this brief and treat it as a pointer, not a full walkthrough — exact visualization tooling/setup details are worth double-checking against current docs before you film, since this is the kind of thing that changes between versions.*

---

## Slide 8 — Recap + what's next

**On slide:**
- "StateGraph + add_node + add_edge + compile + invoke = your first real graph"
- "Still fully linear — no decisions yet"
- "Next video: conditional edges — giving your graph a brain"

**Speech:**
"So to recap: you build a `StateGraph`, register your nodes, connect them with edges starting from `START` and ending at `END`, compile it, and invoke it. What we built today only goes one direction — no decisions. Next video, we fix that: conditional edges, which let your graph actually choose what happens next based on state. That's where this starts feeling like a real agent."

---

## Production notes

- **Slide 6 is the emotional payoff of the module so far** — the moment manual chaining becomes automatic. Don't rush past it; let the "and there it is" moment breathe.
- **Don't demo graph visualization tooling in depth.** Mention it exists, move on — a deep dive here would shift focus away from the core mechanics this video is about, and tooling specifics are the most likely thing to go stale between when you record and when someone watches.
- **The "forgot the edge to END" error is worth deliberately showing**, not just mentioning. Comment out the last `add_edge` line, run it, show the real error on screen, then add it back. Seeing an error you were warned about land exactly as described builds real trust.