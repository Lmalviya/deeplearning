# Mini-project 1: Self-correcting content generator

**Target length:** 18–22 minutes (this is a build-along, longer than concept videos by design)
**Goal:** Learner builds a complete, working self-correcting content generator end to end, using every concept from videos 2–6 with nothing new introduced — proving to themselves they can build a real thing, not just follow isolated demos.
**Golden rule:** This video introduces **zero new concepts**. Every single piece used here was already taught. If you catch yourself explaining something new, that belongs in an earlier video, not here — this is application, not instruction. Frame it explicitly as "watch everything from this module come together."

---

## Slide 1 — Framing: what we're building and why

**On slide:**
- "A content generator that critiques and revises its own output"
- "Every piece you already know: state, nodes, edges, conditional routing, loops"

**Speech:**
"This is our first real project, and I want to be upfront about something: there is nothing new to learn in this video. Every single piece — state, nodes, edges, conditional edges, loops — you already know how to build. What's new is putting them all together into one working system. That's actually the harder skill, and it's exactly what this video is for."

---

## Slide 2 — The spec, in plain English before any code

**On slide:**
- "Input: a topic"
- "Generate a short piece of content about it"
- "Critique it against a simple quality bar"
- "If it doesn't pass, revise and try again — up to a limit"
- "Output: the final approved (or best-effort) content"

**Speech:**
"Before touching code, let's nail down exactly what we're building. The user gives us a topic. We generate a short piece of writing about it — say, a paragraph. We critique that writing against a simple bar — is it clear, on-topic, reasonably well-written? If it fails, we revise and try again, up to a limit we control. And at the end, we return whatever we've got — either something that passed, or our best attempt after running out of tries."

*Spelling out the spec before code matters here — it mirrors how they should approach their own future projects: define the behavior first, then build.*

---

## Slide 3 — Designing the state (recap from video 2)

**Switch to code editor.**

```python
class ContentState(TypedDict):
    topic: str
    content: str
    critique: str
    passed: bool
    iterations: int
```

**Speech:**
"Let's design our state — this is exactly the skill from video 2. What does every node need to see? A `topic` to write about, the current `content`, the latest `critique`, whether it `passed`, and our `iterations` counter from last video's loop guard. Notice I'm building this fresh for the project rather than reusing `AgentState` directly — a new project often means a genuinely new shape of state, and that's normal. Don't force-fit an old schema onto a new problem."

*This is a useful, honest aside — it prevents learners from thinking they must always reuse the exact same state class shown throughout the module.*

---

## Slide 4 — Writing the generate node

**Switch to code editor.**

```python
def generate_content(state: ContentState) -> dict:
    prompt = f"Write a short paragraph about: {state['topic']}"
    if state.get("critique"):
        prompt += f"\nPrevious attempt was critiqued: {state['critique']}\nPlease address this."
    response = llm.invoke(prompt)
    return {
        "content": response.content,
        "iterations": state.get("iterations", 0) + 1,
    }
```

**Speech:**
"Same node pattern from video 3: receive state, do work, return the update. One new-ish wrinkle worth calling out — not a new concept, just an application of one you know — this node checks if there's a previous critique, and if so, feeds it back into the prompt. That's how the LLM actually improves on retries instead of just generating the same thing again."

---

## Slide 5 — Writing the critique node

**Switch to code editor.**

```python
def critique_content(state: ContentState) -> dict:
    prompt = f"Critique this paragraph for clarity and relevance to '{state['topic']}': {state['content']}\nIf it's good, say PASS. Otherwise, explain what to fix."
    response = llm.invoke(prompt)
    passed = "PASS" in response.content
    return {"critique": response.content, "passed": passed}
```

**Speech:**
"Our critique node — same pattern again. Notice how we're keeping the pass/fail decision simple: we just check whether the LLM's response contains the word 'PASS'. You could get fancier with structured output here, but for our first project, simple and readable wins over clever."

---

## Slide 6 — The routing function with the loop guard (recap from videos 5 and 6)

**Switch to code editor.**

```python
def decide_next(state: ContentState) -> str:
    if state["passed"] or state["iterations"] >= 3:
        return "done"
    return "revise"
```

**Speech:**
"And here's our routing function, straight out of videos 5 and 6 — stop if it passed, or if we've hit our iteration limit, whichever comes first. This exact line is what stands between this project working reliably, and it hanging forever on a topic the LLM struggles with."

---

## Slide 7 — Wiring the graph (recap from video 4)

**Switch to code editor.**

```python
builder = StateGraph(ContentState)
builder.add_node("generate", generate_content)
builder.add_node("critique", critique_content)

builder.add_edge(START, "generate")
builder.add_edge("generate", "critique")
builder.add_conditional_edges(
    "critique",
    decide_next,
    {"revise": "generate", "done": END}
)

graph = builder.compile()
```

**Speech:**
"And now we wire it all up — nothing here is new syntax, it's the exact same `add_node`, `add_edge`, and `add_conditional_edges` calls from videos 4 and 5. This is genuinely the moment where, if you followed the whole module, this should feel almost boring — in a good way. You already know every line here."

---

## Slide 8 — Running it end to end

**Switch to code editor.**

```python
result = graph.invoke({
    "topic": "why octopuses are fascinating",
    "content": "",
    "critique": "",
    "passed": False,
    "iterations": 0,
})
print(result["content"])
print(f"Took {result['iterations']} iteration(s)")
```

**Speech:**
"Let's run it." *(run live)* "And there it is — watch the iteration count. If it passed on the first try, great, that happens. Let's try a trickier topic and see if it needs a revision or two..." *(run again with a topic more likely to need revision, showing the loop actually engage)* "There — you can see it went around the loop before landing on something that passed."

*Try to have at least one run in your recording where the loop genuinely engages more than once — a project video where the loop never visibly triggers undersells the whole point of the module.*

---

## Slide 9 — What you just proved to yourself

**On slide:**
- "You built a working self-correcting agent"
- "Every piece was something you already knew"

**Speech:**
"Take a second on this one — what you just built is a genuinely real pattern used in production agent systems: generate, critique, revise, with a safety guard. And every single piece of it was something you already knew how to do before this video started. That's the whole point of building projects this way — concepts stop being abstract the moment you see them combine into something real."

---

## Slide 10 — Recap + what's next

**On slide:**
- "Module 1 complete: state, nodes, edges, conditional edges, cycles"
- "Next module: making it agentic — tools and real ReAct agents"

**Speech:**
"That wraps up module one — you now have the complete foundation: state, nodes, edges, conditional routing, and safe loops. Next module, we make things properly agentic — giving your graph real tools to use, building a ReAct agent from scratch, and then showing you the prebuilt shortcut once you understand what it's actually doing under the hood. See you there."

---

## Production notes

- **Resist adding anything new.** The temptation in a project video is always to sneak in "oh, and one more thing" — structured output, a fancier prompt technique, error handling. Save all of it. A clean, concept-free project video is far more valuable here than a slightly more impressive one that quietly teaches something ungrounded.
- **Show a run where the loop actually triggers.** If your first test topic passes on iteration one, pick a second topic more likely to need revision (something oddly specific or a bit absurd tends to make a stricter critique more likely) — you want the loop visibly earning its keep on screen at least once.
- **This video doubles as a review.** Treat every "recap from video X" callout as a real signpost, not filler — for anyone who watched the full module, this should feel like assembling pieces they already trust, not learning something new dressed up as a project.