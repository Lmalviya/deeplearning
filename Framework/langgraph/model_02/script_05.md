# Mini-project 2: Tool-using research assistant

**Target length:** 16–20 minutes
**Goal:** Build a real tool-using agent — search + calculator tools, streamed reasoning — using nothing but concepts from Module 2. Same "zero new concepts" rule as mini-project 1.
**Golden rule:** No new syntax. This project exists to prove tools + ReAct loop + streaming combine into something genuinely useful, not to teach anything additional.

---

## Slide 1 — Framing: what we're building and why

**On slide:**
- "A research assistant that can search the web, do math, and show its reasoning live"
- "Every piece from this module: tools, the ReAct loop, streaming"

**Speech:**
"Same deal as our first project — nothing new to learn today, just assembly. We're building a research assistant with two tools: a web search tool and a calculator, using the prebuilt agent from video 10, streamed the way we learned in video 11."

---

## Slide 2 — The spec

**On slide:**
- "Input: a research question, possibly requiring lookup and math"
- "Tools: web search, calculator"
- "Output: streamed reasoning, then a final answer"

**Speech:**
"Something like 'what's the population of Japan divided by the population of the UK' — that needs a search tool for the raw numbers, and a calculator for the division. Let's build the tools first."

---

## Slide 3 — Defining the tools

**Switch to code editor.**

```python
@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic."""
    # your search implementation — API of your choice
    ...

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '125000000 / 67000000'."""
    return str(eval(expression))
```

**Speech:**
"Two tools, same `@tool` pattern from video 8 — clear docstrings, since that's what the LLM reads to decide when to use each one. For the search tool, plug in whatever search API you're using — I'll link mine in the description, but the pattern here is identical no matter which provider you pick."

*Flag the `eval()` use honestly — see production notes below.*

---

## Slide 4 — Building the agent

**Switch to code editor.**

```python
agent = create_react_agent(llm, tools=[web_search, calculator])
```

**Speech:**
"Straight from video 10 — one line. This is exactly why we taught the prebuilt version: for a standard tool-use pattern like this, there's no reason to hand-build the loop again."

---

## Slide 5 — Streaming the reasoning

**Switch to code editor.**

```python
for chunk in agent.stream({"messages": [("user", "What's the population of Japan divided by the population of the UK?")]}):
    print(chunk)
```

**Speech:**
"And streamed, from video 11 — let's run this live and actually watch it work through the problem: search for Japan's population, search for the UK's, then calculate the division. Watch the tool calls happen one after another in real time."

*(Run live — narrate each chunk as it appears: "there's the search call... there's the result... now it's calling the calculator...")*

---

## Slide 6 — What you just proved to yourself

**On slide:**
- "A genuinely useful multi-tool research agent"
- "Every piece: something you already knew from this module"

**Speech:**
"What you just built is a real pattern — a lot of production research and support agents are, underneath, exactly this: tools, a reasoning loop, and streamed output. And once again, every piece of it came from concepts you already had."

---

## Slide 7 — Recap + what's next

**On slide:**
- "Module 2 complete: tools, ReAct agents, streaming"
- "Next module: memory — so your agent stops forgetting everything between messages"

**Speech:**
"That's module two wrapped — your agent can now act, not just reason. Next module, we fix something you may have already noticed as a gap: right now, every single `.invoke()` starts from a blank slate. Next up, memory — so your agent can actually remember a conversation."

---

## Production notes

- **Be honest on camera about `eval()`.** It's the simplest way to demo a calculator tool, but mention plainly that `eval()` executing arbitrary strings is not something you'd want in a real production tool without sandboxing — a quick, honest aside, not a rabbit hole.
- **Pick a search API before recording and note the setup cost.** Whatever provider you use will need its own API key — mention that plainly and keep the setup instructions in your video description rather than a long on-camera detour, since exact steps vary by provider and go stale fastest.
- **Try to pick a question that genuinely needs both tools**, not just one — the whole point of this project is showing multi-tool reasoning working together, not a single tool call in isolation.