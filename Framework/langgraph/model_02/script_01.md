# Video 8: Tool calling inside a graph

**Target length:** 10–12 minutes
**Goal:** Learner can define a tool, bind it to an LLM, and use a prebuilt `ToolNode` to execute it — no loop yet, that's next video.
**Golden rule:** This video is about the *pieces* (tool, bound LLM, ToolNode) in isolation. Don't build the full agent loop yet — one tool call, once, is enough to prove the mechanism works.

---

## Slide 1 — Recap + hook

**On slide:** "Module 1 done: your graph can think. Today: it can act."

**Speech:**
"You've now got a graph that can branch and loop — genuinely powerful. But so far, every node has just called an LLM and returned text. Today we give the LLM something new: the ability to call a tool instead of just talking."

---

## Slide 2 — What a tool actually is

**On slide:** "A tool is just a Python function — the LLM decides when to call it."

**Speech:**
"A tool is nothing magical — it's a regular Python function. What makes it a 'tool' is that we describe it to the LLM, and the LLM can choose to call it instead of generating a plain text answer. The LLM doesn't run the function itself — it just says 'I'd like to call this function with these arguments,' and it's our job to actually run it."

---

## Slide 3 — Live code: defining a tool

**Switch to code editor.**

```python
from langchain_core.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)
```

**Speech:**
- "`@tool` — this decorator, from LangChain, turns a normal function into something an LLM can be told about."
- "The docstring here isn't a comment for humans — the LLM actually reads it to decide *when* to use this tool. A vague docstring gets a tool that never gets called, or gets called at the wrong time. Be specific."
- "The type hints matter too — `word: str` tells the LLM exactly what kind of argument to provide."

---

## Slide 4 — Binding the tool to an LLM

**Switch to code editor.**

```python
llm_with_tools = llm.bind_tools([get_word_length])
```

**Speech:**
"`bind_tools` tells the LLM 'here's what's available to you.' It doesn't change how you call the LLM — you still call `.invoke()` normally — but now the LLM has the *option* to respond with a tool call instead of plain text, if it decides that's the better move."

---

## Slide 5 — Live code: seeing a tool call happen

**Switch to code editor.**

```python
response = llm_with_tools.invoke("How many letters are in 'langgraph'?")
print(response.tool_calls)
```

**Speech:**
"Let's run this and look at what comes back." *(run live)* "Notice — `response.content` is basically empty, but `response.tool_calls` has something in it: the tool name and the arguments the LLM wants to use. This is the key shift: the LLM isn't answering directly anymore, it's *requesting* that a function be run on its behalf."

---

## Slide 6 — ToolNode: the prebuilt executor

**Switch to code editor.**

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([get_word_length])
```

**Speech:**
"Now — something has to actually *run* that tool call. You could write that logic yourself, but LangGraph gives you `ToolNode`, a prebuilt node that looks at the last message in state, finds any tool calls, executes them, and returns the results. One line, and the execution machinery is handled for you."

---

## Slide 7 — Wiring a minimal graph (no loop yet)

**Switch to code editor.** Build a simple `agent -> tools -> END` graph, invoke it once, and show the final state includes the tool's result.

**Speech:**
"Let's wire this into the smallest possible graph — agent node, then tool node, then end. Notice this graph only calls the tool once, then stops — it's not a real agent loop yet, that's deliberate. I want you to see the mechanism cleanly before we add a loop around it."

---

## Slide 8 — Recap + what's next

**On slide:**
- "Tool = decorated function with a clear docstring"
- "bind_tools = tells the LLM what's available"
- "ToolNode = prebuilt executor for tool calls"
- "Next video: looping this into a real ReAct agent"

**Speech:**
"So: define a tool, bind it to your LLM, and let `ToolNode` execute the calls. Right now we only go around once. Next video, we turn this into a real loop — the LLM can call a tool, see the result, and decide to call another tool, or finally answer — that's a proper ReAct agent, and it's going to feel very familiar, because it uses the exact same loop pattern from video 6."

---

## Production notes

- **Keep the tool trivially simple.** A word-length counter or basic calculator is ideal — the point is the mechanism, not an impressive tool.
- **Explicitly flag that this isn't a loop yet.** Beginners who've just learned cycles in module 1 will expect looping immediately — telling them "next video" up front avoids confusion.