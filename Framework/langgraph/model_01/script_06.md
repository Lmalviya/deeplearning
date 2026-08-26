# Video 6: Cycles and loops

**Target length:** 14–16 minutes (this is the module's centerpiece — allow it to run a little longer than the others)
**Goal:** Learner builds a real generate → critique → revise loop with a counter-based guard against infinite loops, and understands *why* the guard is non-negotiable, not optional polish.
**Golden rule:** Introduce the infinite-loop danger *before* showing the fix — let them feel the risk, not just hear a warning about it. This is the single most distinguishing capability of LangGraph vs. a plain chain, so this video should feel like the payoff of the entire module.

---

## Slide 1 — Recap + hook

**On slide:** "You now have branching. Today: branching backward."

**Speech:**
"Last video, a decision could send you forward to one of two nodes. Today, we do something a plain chain literally cannot do: send execution *backward*, to a node that already ran. This is the capability that started this whole course — remember video 1, the chain that couldn't retry? This is the fix."

---

## Slide 2 — What a cycle actually is

**On slide:** *(reuse the generate/critique/check-quality/revise/end diagram shown earlier — Generate → Critique → Check quality → branches to Revise, which loops back to Generate, or to End)*

**Speech:**
"Here's what we're building today. Generate produces an answer. Critique evaluates it. A decision point checks: is this good enough? If not, we go to Revise, which loops back to Generate — and the whole cycle repeats. If it is good enough, we're done. That loop-back arrow is the entire concept — everything else is stuff you already know from the last two videos."

---

## Slide 3 — The danger, shown live before it's fixed

**Switch to code editor.** Deliberately build the loop *without* any guard first.

```python
def check_quality(state: AgentState) -> str:
    if "good" in state["critique"]:
        return "done"
    return "revise"

builder.add_conditional_edges(
    "critique",
    check_quality,
    {"revise": "generate", "done": END}
)
```

**Speech:**
"Let's build this the naive way first, on purpose. Watch what happens if `critique` never actually says the word 'good' — maybe our critique node is a bit too harsh, or the LLM is being inconsistent." *(Run it live with a critique that never contains "good" — let it visibly hang or run for a long time before interrupting it.)* "See that? It's stuck. It'll keep looping generate, critique, generate, critique — forever, or until you run out of API budget, whichever comes first. This is the single most common real bug in LangGraph code, and I wanted you to actually see it happen, not just be warned about it."

*This live "let it hang" moment is the emotional core of the video — don't skip it or just describe it. Feeling the risk is what makes the fix, coming next, feel necessary rather than like extra ceremony.*

---

## Slide 4 — The fix: a counter in state

**Switch to code editor.** First, update the state schema.

```python
class AgentState(TypedDict):
    user_question: str
    answer: str
    critique: str
    iterations: int
```

**Speech:**
"The fix starts back in our state schema — remember, we said in video 2 we'd keep growing this class as we needed to. We add an `iterations` field, an integer, to keep count of how many times we've gone around the loop."

---

## Slide 5 — Incrementing the counter in a node

**Switch to code editor.**

```python
def generate_answer(state: AgentState) -> dict:
    response = llm.invoke(state["user_question"])
    return {
        "answer": response.content,
        "iterations": state["iterations"] + 1,
    }
```

**Speech:**
"Every time `generate` runs, we bump the counter by one — reading the current value out of state, and returning the incremented version as part of our update. Nothing exotic here, just `state["iterations"] + 1`."

---

## Slide 6 — Guarding the routing function

**Switch to code editor.**

```python
def check_quality(state: AgentState) -> str:
    if "good" in state["critique"] or state["iterations"] >= 3:
        return "done"
    return "revise"
```

**Speech:**
"And here's the actual fix — we add a second condition to our routing function. Now we stop either when the critique says 'good', *or* when we've hit three iterations, whichever comes first. That `or state["iterations"] >= 3` is the one line standing between a working graph and an infinite loop. Let's re-run the exact same broken scenario from before..." *(run it live)* "...and this time it stops itself after three attempts, even though the critique never approved it."

*This is the direct payoff to slide 3's danger demo — make the callback explicit: same broken scenario, now safe.*

---

## Slide 7 — Choosing a max iteration count is a design decision, not an afterthought

**On slide:**
- "Too low: gives up before a real fix is found"
- "Too high: wastes time and API cost on a lost cause"

**Speech:**
"Quick but important point: the number you pick here — I used three — isn't arbitrary busywork, it's a real design decision. Too low, and you'll cut off attempts that were about to succeed. Too high, and you're burning API calls and time on something that was never going to converge. There's no universal right answer — it depends on your use case — but always pick a number deliberately, never leave a loop unguarded."

---

## Slide 8 — Recap + what's next

**On slide:**
- "Cycles = an edge pointing back to an earlier node"
- "Always guard loops with a counter or similar stop condition"
- "You now have everything for Mini-project 1"

**Speech:**
"Let's recap: a cycle is just an edge that points backward instead of forward, and it's the one thing a plain chain can never do. But with that power comes a responsibility — always guard your loops, or you risk exactly the kind of infinite loop we saw at the start of this video. And with this, you actually now have every single piece needed to build our first real project — state, nodes, edges, conditional routing, and loops. Next video, we build it end to end."

---

## Production notes

- **The "let it hang" demo (slide 3) is the most important moment in this entire module.** If you're worried about API cost while recording, cap it with a very obviously large number (e.g., let it run 15–20 iterations on screen, sped up in editing) rather than skipping the demonstration — the visceral "oh no, it's not stopping" moment is what makes the guard feel earned rather than arbitrary.
- **Don't introduce any new state fields beyond `iterations` this video.** Resist scope creep — the lesson is the loop-guard pattern, not a bigger example.
- **This is a natural video to run slightly long.** Of all six videos in this module, this is the one where a couple extra minutes for the danger-then-fix arc to breathe is worth it more than staying strictly on a runtime target.