# Video 3: Nodes — functions that transform state

**Target length:** 10–12 minutes
**Goal:** Learner can write a node function, understands it's just a plain function that returns a *partial* update — and can call nodes manually to see them work, without a graph yet.
**Golden rule:** Still no `StateGraph`. We call node functions directly by hand this video. Wiring them into a real graph is video 4 — keep that boundary crisp so each concept lands on its own.

---

## Slide 1 — Recap + hook

**On slide:** "Last time: what state is. Today: the functions that actually change it."

**Speech:**
"Last video we defined `AgentState` — our shared notebook. But a notebook that nobody writes in is useless. Today we write the functions that actually update it: nodes."

---

## Slide 2 — What a node is, conceptually

**On slide:**
> "A node is a function: it receives the current state, and returns the parts it wants to change."

**Speech:**
"A node in LangGraph is nothing exotic — it's just a Python function. It takes the current state as input, does some work, and returns a dictionary of the fields it wants to update. That's the entire contract. No inheritance, no special base class required."

---

## Slide 3 — Live code: your first node

**Switch to code editor.**

```python
def add_greeting(state: AgentState) -> dict:
    return {"answer": f"Hello! You asked: {state['user_question']}"}
```

**Speech, narrating line by line:**
- "`def add_greeting(state: AgentState) -> dict:` — a normal function. The type hints aren't required by Python, but they're genuinely useful here — `state: AgentState` tells you and your editor exactly what shape of data this function expects."
- "Inside, `state['user_question']` — remember, TypedDict makes your state behave like a dictionary, so we access fields with square brackets, not dot notation."
- "`return {"answer": ...}` — and here's the important part: we're *not* returning the whole state back. We only return the key we're changing. LangGraph merges this into the existing state for you."

"Let's actually run this by hand right now, no graph involved — just to see it work."

*Show: create a state dict manually, call `add_greeting(state)`, print the result.*

---

## Slide 4 — The most common beginner mistake: returning the whole state

**On slide:**
> "Return only what changed — not the entire state object."

**Speech:**
"Here's a mistake I want to flag before you make it: don't do `return state` at the end of a modified state dict. Only return the keys you actually changed. If you return the entire state every time, you risk accidentally overwriting fields another node wrote — especially once you have several nodes running. Return the diff, not the whole picture."

*This is worth its own slide because it's genuinely one of the top beginner bugs — silent data loss from returning too much.*

---

## Slide 5 — Live code: a second node, and calling them in sequence

**Switch to code editor.**

```python
def uppercase_answer(state: AgentState) -> dict:
    return {"answer": state["answer"].upper()}
```

**Speech:**
"Let's write a second node, and simulate what a graph will eventually do for us automatically — call them one after another, manually."

```python
state = {"user_question": "what is langgraph?", "answer": ""}
state.update(add_greeting(state))
state.update(uppercase_answer(state))
print(state)
```

"Notice what I'm doing here — I'm manually merging each node's return value back into state with `.update()`. This is *exactly* what LangGraph does for you automatically once we build a real graph next video. Seeing it done by hand first means the graph won't feel like a black box."

*This manual-chaining moment is the single most valuable teaching device in this video — it demystifies the graph before the graph even exists.*

---

## Slide 6 — A real node: calling an LLM

**Switch to code editor.**

```python
def generate_answer(state: AgentState) -> dict:
    response = llm.invoke(state["user_question"])
    return {"answer": response.content}
```

**Speech:**
"Real nodes usually aren't this simple, of course — most of the time a node is calling an LLM, a tool, or doing some real processing. Here's what that looks like: same exact shape, receive state, do work — in this case, call the LLM — and return the update. The *pattern* never changes, no matter how complex the work inside a node gets."

*Keep the LLM setup itself (client init, API key) brief here — assume it was covered in video 1's setup, don't re-teach it.*

---

## Slide 7 — Nodes should be predictable and self-contained

**On slide:**
- "A node shouldn't need to know what happened before or after it"
- "It just needs: what's in state right now"

**Speech:**
"One design habit worth building early: a well-written node doesn't need to know its place in a larger sequence. It just looks at what's currently in state, does its job, and returns an update. This is what makes graphs easy to rearrange and debug later — nodes are interchangeable building blocks, not tightly coupled steps."

---

## Slide 8 — Recap + what's next

**On slide:**
- "Nodes = functions: `(state) -> partial update`"
- "Return only what changed"
- "Next video: wiring nodes into a real graph with edges"

**Speech:**
"So: nodes are just functions, they receive state and return updates, and you only return what changed. Right now we're calling these by hand — next video, we finally build a real `StateGraph`, wire these nodes together with edges, and let LangGraph do this chaining automatically. See you there."

---

## Production notes

- **The manual `.update()` chaining demo (slide 5) is the highlight of this video.** If you're short on time, this is the one segment not to cut — it's what makes video 4's `StateGraph` feel like an upgrade instead of magic.
- **Keep the LLM node (slide 6) light.** One example is enough — don't build multiple LLM-calling nodes here, save that depth for module 2.
- **Reuse `AgentState` from video 2 unchanged.** Don't add new fields yet unless the example genuinely needs them.