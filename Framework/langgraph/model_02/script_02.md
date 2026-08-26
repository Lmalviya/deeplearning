# Video 9: Building a ReAct agent from scratch

**Target length:** 14–16 minutes
**Goal:** Learner builds a real looping tool-use agent by hand, and learns *reducers* — the mechanism for state fields that accumulate (like a message list) instead of overwrite.
**Golden rule:** Explicitly call back to video 6's loop diagram — this is the same shape, just with a tool call instead of a critique. That reuse is the single best teaching device available here; use it.

---

## Slide 1 — Recap + hook

**On slide:** "Same loop shape you already know — new job for it."

**Speech:**
"Remember the generate-critique-revise loop from video 6? Today we build the exact same *shape* of loop, but instead of critiquing writing, it's an LLM deciding whether to call a tool again or give a final answer. This is what people mean by a 'ReAct agent' — reason, act, observe, repeat."

---

## Slide 2 — The new problem: state that should grow, not overwrite

**On slide:**
> "Every LLM call in this loop needs to see *everything* said so far — not just the last message."

**Speech:**
"Here's a wrinkle we haven't hit yet. In our loop, each time around, we call the LLM again — but this time, it needs to see the *entire conversation so far*, including previous tool calls and results. If we just overwrite a `messages` field each time like we've done with every other field so far, we'd lose history on every loop. We need messages to *accumulate*, not overwrite."

---

## Slide 3 — Live code: the reducer

**Switch to code editor.**

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

**Speech, narrating line by line — this is a genuinely new concept, give it room:**
- "`Annotated[list, add_messages]` — this is new syntax. `Annotated` lets us attach extra information to a type hint. Here, we're saying: this field is a list, *and* whenever a node returns an update to it, don't overwrite it — merge it in using this `add_messages` function."
- "`add_messages` is a prebuilt reducer LangGraph gives you specifically for message lists — it knows how to append new messages, and even update an existing message if you return one with a matching ID."
- "This is the general pattern for any field that should *grow* across your graph's run, not just get replaced. Message lists are the most common case, but the same idea — attach a reducer via `Annotated` — applies to other accumulating fields too."

*This is worth genuinely slowing down for — it's the first time state behavior has been anything other than 'the last write wins,' and it unlocks a huge category of real agents.*

---

## Slide 4 — Live code: the agent node

**Switch to code editor.**

```python
def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

**Speech:**
"Our agent node is almost suspiciously simple now — pass the whole message history to the LLM, get back a response, and return it as a list with one item. The reducer handles appending it to the growing history."

---

## Slide 5 — Live code: the routing function

**Switch to code editor.**

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"
```

**Speech:**
"Our routing function looks at the *last* message. If it has tool calls attached, we're not done — route to the tools node. If not, the LLM gave a final answer, and we're finished. Same routing-function pattern from video 5, just checking a different condition."

---

## Slide 6 — Wiring the loop

**Switch to code editor.**

```python
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
```

**Speech:**
"And here's the loop — `add_edge("tools", "agent")` sends control right back to the agent after any tool executes, exactly like `add_edge("shout", "generate")` did in video 6's revise loop. Same shape, same principle, new use case."

---

## Slide 7 — Running it and watching multiple tool calls happen

**Switch to code editor.** Ask a question requiring 2+ tool calls in sequence, run live, print the message history.

**Speech:**
"Let's ask something that needs more than one tool call to answer fully." *(run live)* "Watch the message list grow — first the agent calls a tool, gets a result, calls the LLM again with that added to history, maybe calls another tool, and eventually gives a final answer with no tool calls attached, which is our stopping condition."

*Note: unlike video 6, this loop doesn't strictly need an iteration counter if your tools are reliable — but flag the risk anyway (see production notes).*

---

## Slide 8 — Recap + what's next

**On slide:**
- "Same loop shape as video 6 — new purpose"
- "Reducers (`Annotated[list, add_messages]`) let state fields accumulate instead of overwrite"
- "Next video: the prebuilt shortcut for all of this"

**Speech:**
"You just built a real ReAct agent, by hand, using nothing but concepts you already had plus one new idea — reducers. Next video, I'll show you the one-line prebuilt version of everything we just built — and because you built it manually first, it won't feel like magic at all."

---

## Production notes

- **Worth a brief callback to video 6's infinite-loop danger.** A tool-calling loop with no bound can also run away if the LLM keeps calling tools indefinitely — mention this in passing and note that `create_react_agent` (next video) has sensible defaults, but a manual version like this one should still consider a max-steps guard for production use.
- **The reducer explanation (slide 3) is the intellectual centerpiece of this video** — don't compress it to fit a shorter runtime. Everything else here is application of prior concepts; this is the one genuinely new idea.