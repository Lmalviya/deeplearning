# Video 5: Conditional edges — giving your graph a brain

**Target length:** 12–15 minutes
**Goal:** Learner can write a routing function and wire it up with `add_conditional_edges`, understands the mapping dict, and sees state genuinely change *which* node runs next — not just what data flows through a fixed path.
**Golden rule:** This is the video where LangGraph stops feeling like "a fancier chain" and starts feeling like its actual purpose. Give this concept real breathing room — don't rush to cycles yet, that's next video.

---

## Slide 1 — Recap + hook

**On slide:** "Last time: a straight line. Today: a fork in the road."

**Speech:**
"Last video, our graph only ever went one direction — `generate`, then `shout`, then done. Every single time, no matter what. Today we change that: we're going to make the graph *decide* what happens next, based on what's actually in state."

---

## Slide 2 — The idea: routing based on state

**On slide:**
> "A conditional edge asks a question about state, and the answer decides which node runs next."

**Speech:**
"Here's the core idea, in plain English: instead of always going from node A to node B, we ask a question — using the current state — and the answer tells us which node to go to. Maybe it's 'was the answer confident enough?' Maybe it's 'did the user ask something we can't handle?' Whatever the question, the answer determines the path."

---

## Slide 3 — Live code: a routing function

**Switch to code editor.**

```python
def check_confidence(state: AgentState) -> str:
    if "I don't know" in state["answer"]:
        return "clarify"
    return "done"
```

**Speech, narrating line by line:**
- "This is what we call a routing function. It takes state — same as a node — but notice the return type is different: it returns a plain string, not a dictionary."
- "That string is a label. It doesn't update state at all — its only job is to say which direction we're going."
- "Here, if the answer contains 'I don't know', we return `"clarify"`. Otherwise, we return `"done"`. Those exact strings are what we'll map to real node names next."

*Explicitly call out that this looks similar to a node but behaves differently — string return, no state mutation — since conflating the two is an easy beginner mix-up.*

---

## Slide 4 — Live code: wiring it up with add_conditional_edges

**Switch to code editor.**

```python
builder.add_conditional_edges(
    "generate",
    check_confidence,
    {
        "clarify": "ask_clarifying_question",
        "done": END,
    }
)
```

**Speech, narrating line by line:**
- "`add_conditional_edges` takes three things. First, the node this decision happens *after* — here, `"generate"`."
- "Second, the routing function itself — `check_confidence`, the one we just wrote."
- "Third, a mapping — a dictionary connecting each possible string the routing function can return, to an actual node name, or `END`."

"So read this as: after `generate` runs, call `check_confidence`. If it returns `"clarify"`, go to the `ask_clarifying_question` node. If it returns `"done"`, we're finished."

---

## Slide 5 — The mistake that causes a real error: incomplete mappings

**On slide:**
> "Every possible return value from your routing function must appear in the mapping — or LangGraph won't know where to go."

**Speech:**
"Here's a mistake worth knowing about before you hit it: if your routing function can return a value that *isn't* in your mapping dictionary, LangGraph will error out the moment that path is actually taken. Let's see that on screen." *(Live: temporarily remove the `"clarify"` entry from the mapping, trigger that path, show the resulting error.)* "That's your sign to go back and check: does my mapping cover every single string my routing function can return? Miss one, and it'll work fine until the day it doesn't."

*Showing this live matters — an error demonstrated on screen sticks far better than a described warning.*

---

## Slide 6 — Diagram: the branching graph

**On slide:** *(reuse the right-hand "graph" panel style from video 1's diagram — Agent/generate node → decision → two branches, one to a new node, one to END — but without the loop-back arrow yet, since we're not covering cycles this video)*

**Speech:**
"Here's the shape of what we just built. One path forks into two, based on state. This is genuinely the shape of most real agent decision points — and next video, we take this one step further by letting one of these branches loop back to an earlier node instead of always moving forward."

---

## Slide 7 — Live run: watching both paths

**Switch to code editor.** Run the graph twice — once with a state that triggers `"done"`, once engineered to trigger `"clarify"` — and show both outputs.

**Speech:**
"Let's actually watch both paths happen. First run — a normal answer, no 'I don't know' in it — routes straight to done. Second run — I'll fake an uncertain answer — and watch, it takes the clarify path instead. Same graph, same code, two completely different paths, because the *state* was different."

---

## Slide 8 — Recap + what's next

**On slide:**
- "`add_conditional_edges` = routing function + mapping dict"
- "Every possible return value needs an entry in the mapping"
- "Next video: what if a branch loops back instead of moving forward?"

**Speech:**
"So: a routing function looks at state and returns a label, and a mapping dict connects each label to a real destination. Make sure every possible output is covered in that mapping. Next video, we take one of these branches and point it *backward* instead of forward — that's where loops come from, and it's the single most powerful thing a graph can do that a chain simply can't."

---

## Production notes

- **This video deserves real pacing room.** It's conceptually the biggest jump in the course so far — don't compress it to hit a shorter runtime.
- **The live error demo (slide 5) is non-negotiable if you're cutting for time — cut something else first.** An incomplete-mapping error is one of the most common real bugs learners will hit on their own; seeing it here means they'll recognize it later instead of panicking.
- **Deliberately don't mention loops this video**, even though it's tempting once branching clicks. Let conditional routing stand fully on its own — cycles get their own dedicated video next for a reason.