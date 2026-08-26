# Video 17: Subgraphs and composition

**Target length:** 10–12 minutes
**Goal:** Learner understands why and how to nest a compiled graph as a node inside a larger graph, for organization and reuse.
**Golden rule:** Motivate with a real pain point — a graph that's grown too large to reason about in one file — before showing the fix. Keep the example itself simple; the point is the pattern, not a complex system.

---

## Slide 1 — Recap + hook

**On slide:** "Your graphs are getting bigger. Time to organize them."

**Speech:**
"Everything we've built so far fits comfortably in one file, with a handful of nodes. But real systems grow — imagine a graph with twenty nodes, handling several different concerns. At some point, one flat graph stops being readable. Today: how to break a graph into pieces."

---

## Slide 2 — The idea: a compiled graph can be a node

**On slide:**
> "A subgraph is just a compiled graph — used as a node inside a bigger graph."

**Speech:**
"Here's the core idea, and it's genuinely simple once it clicks: a graph you've already built and compiled can itself be added as a single node inside a *different*, larger graph. From the outside, it looks like just another step. Internally, it's a whole graph doing real work."

---

## Slide 3 — Live code: building a small subgraph

**Switch to code editor.**

```python
sub_builder = StateGraph(AgentState)
sub_builder.add_node("step_a", step_a_fn)
sub_builder.add_node("step_b", step_b_fn)
sub_builder.add_edge(START, "step_a")
sub_builder.add_edge("step_a", "step_b")
sub_builder.add_edge("step_b", END)

subgraph = sub_builder.compile()
```

**Speech:**
"Nothing new in this part at all — this is a graph built exactly the way we've built every graph since video 4. The only thing that makes it a 'subgraph' is what we do with it next."

---

## Slide 4 — Live code: nesting it in a parent graph

**Switch to code editor.**

```python
parent_builder = StateGraph(AgentState)
parent_builder.add_node("intro", intro_fn)
parent_builder.add_node("sub_process", subgraph)
parent_builder.add_edge(START, "intro")
parent_builder.add_edge("intro", "sub_process")
parent_builder.add_edge("sub_process", END)

parent_graph = parent_builder.compile()
```

**Speech:**
"And here's the whole trick — `add_node("sub_process", subgraph)`. Instead of passing a function, we pass our already-compiled graph. As far as the parent graph is concerned, it's just another node. Run this, and internally, it runs the entire two-step subgraph before continuing."

---

## Slide 5 — Why this matters: the state contract

**On slide:**
> "A subgraph needs to share compatible state with its parent — matching keys it reads and writes."

**Speech:**
"One thing to keep in mind: for this to work cleanly, your subgraph's state needs to be compatible with the parent's — generally, sharing the keys it actually reads and writes. For our simple example today, we used the exact same `AgentState` in both, which is the easiest case. More advanced setups can map between different state shapes, but that's beyond what a beginner course needs — know that the concept exists for when you need it."

---

## Slide 6 — When subgraphs actually earn their complexity

**On slide:**
- "A logical unit of work that could stand alone (e.g. a whole 'research' sub-process)"
- "Something reused in more than one place"
- "A team boundary — someone else owns this piece"

**Speech:**
"Don't reach for subgraphs just because you can — for small graphs like the ones we've built this whole course, one flat graph is genuinely easier to follow. Subgraphs earn their complexity when a chunk of your graph is a real logical unit on its own, when you're reusing the same sub-process in multiple places, or when different people are going to own different pieces."

---

## Slide 7 — Recap + what's next

**On slide:**
- "A compiled graph can be used as a node in a bigger graph"
- "Use it for organization and reuse, not by default"
- "Next video: coordinating multiple specialized agents"

**Speech:**
"So: any graph you compile can become a single node somewhere bigger. Use it when it genuinely improves organization, not as a default habit. Next video, we use a related idea to build something new — a supervisor agent that coordinates several specialized agents underneath it."

---

## Production notes

- **Keep the subgraph example intentionally trivial** — two dummy steps is enough. The lesson is the *mechanism* of nesting, not an impressive sub-process.
- **Don't get pulled into state-mapping edge cases.** A single sentence acknowledging the topic exists (slide 5) is the right depth for this course — a deep dive belongs in the production course, if anywhere.