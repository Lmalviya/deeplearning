# Video 10: The prebuilt shortcut — create_react_agent

**Target length:** 8–10 minutes (this should feel like the shortest, easiest video so far — a reward lap after video 9's density)
**Goal:** Learner sees the high-level prebuilt API, understands it's the exact same pattern from video 9 wrapped up, and knows when to reach for it vs. building manually.
**Golden rule:** Don't introduce anything new here — every sentence should be some version of "remember what you built by hand? here it is in one line."

---

## Slide 1 — Recap + hook

**On slide:** "You built this by hand last video. Here's the shortcut."

**Speech:**
"Last video was dense — reducers, loops, routing functions, all wired by hand. Good news: today is the easy one. I'm going to show you a single function that does everything we just built. And because you understand what's underneath it now, it won't feel like a magic trick."

---

## Slide 2 — Live code: the one-liner

**Switch to code editor.**

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools=[get_word_length])
```

**Speech:**
"That's it. `create_react_agent` takes your LLM and your list of tools, and hands you back a fully compiled, runnable graph — the exact agent-tools loop we spent all of last video building by hand."

---

## Slide 3 — Running it — same behavior, familiar interface

**Switch to code editor.**

```python
result = agent.invoke({"messages": [("user", "How many letters are in 'langgraph'?")]})
print(result["messages"][-1].content)
```

**Speech:**
"Let's run it side by side with what we built last video." *(run live, compare outputs)* "Same behavior, same looping tool-call pattern under the hood — `.invoke()`, `.stream()`, all of it works exactly the way you'd expect, because it's still a `StateGraph` underneath, just assembled for you."

---

## Slide 4 — When to use the prebuilt version vs. building manually

**On slide:**
- "Prebuilt: fast, sensible defaults, great for straightforward ReAct agents"
- "Manual: when you need custom routing logic, extra nodes, or non-standard state"

**Speech:**
"So when do you actually reach for each one? Use `create_react_agent` when a standard reason-act loop is genuinely all you need — it's faster to write and has sensible defaults baked in. Go back to building manually, like last video, the moment you need something the prebuilt version doesn't offer — custom routing beyond just 'tool call or not,' extra nodes in the loop, or state that goes beyond a simple message list. Knowing this trade-off, rather than defaulting to one or the other out of habit, is genuinely a skill in itself."

---

## Slide 5 — A peek under the hood (optional, keep brief)

**On slide:** "It's still just a compiled StateGraph — you can inspect it."

**Speech:**
"One quick thing worth knowing: `create_react_agent` isn't some separate system — it returns an actual compiled graph, same as what `builder.compile()` gives you. That means everything else you'll learn in this course — streaming, memory, human-in-the-loop — applies to it exactly the same way."

*Keep this slide short — it's reassurance, not a new lesson.*

---

## Slide 6 — Recap + what's next

**On slide:**
- "create_react_agent = the loop from video 9, prebuilt"
- "Same trade-off skill: know when to build manually instead"
- "Next video: streaming — watching your agent think in real time"

**Speech:**
"So: same agent, one line instead of fifteen, and you now know exactly what's happening underneath because you built it yourself first. Next video, we make this feel a lot more alive — streaming outputs, so you can watch your agent's reasoning happen live instead of waiting for a final answer."

---

## Production notes

- **This video should feel noticeably breezier than video 9.** If it's running long, you're probably over-explaining something already covered — trust the audience remembers last video.
- **Don't add new tools or a new example here.** Reuse the exact same tool and question from video 9 so the comparison is a clean apples-to-apples moment, not a new scenario to parse.