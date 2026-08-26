# Video 13: Short-term memory — checkpointers and threads

**Target length:** 12–14 minutes
**Goal:** Learner adds a checkpointer to persist state across `.invoke()` calls, and understands `thread_id` as the key that separates one conversation from another.
**Golden rule:** Show the problem (memory loss between calls) live, before the fix — same pattern as videos 6 and 11.

---

## Slide 1 — Recap + hook

**On slide:** "Your agent can act. Today: it remembers."

**Speech:**
"You might have noticed something if you've been experimenting on your own — every time you call `.invoke()`, your agent starts completely fresh. Ask it a question, then ask a follow-up referencing the first one, and it has no idea what you're talking about. Let's see that live."

---

## Slide 2 — Show the problem live

**Switch to code editor.** Two separate `.invoke()` calls: "My name is Alex." then "What's my name?" — show the second one fails to recall.

**Speech:**
"First call: 'My name is Alex.' Second call, right after: 'What's my name?'" *(run both live)* "See that — it has no idea. Every `.invoke()` is a completely blank slate. That's what we're fixing today."

---

## Slide 3 — Live code: adding a checkpointer

**Switch to code editor.**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

**Speech:**
- "`MemorySaver` is the simplest checkpointer LangGraph offers — it keeps your graph's state in memory, in your running Python process."
- "We pass it to `compile()`. This one change is what turns your graph from stateless into stateful."

*Flag plainly: MemorySaver is for local development — it doesn't survive a restart. Production-grade persistence (e.g. Postgres-backed checkpointers) is production-course territory.*

---

## Slide 4 — thread_id: the key that separates conversations

**Switch to code editor.**

```python
config = {"configurable": {"thread_id": "alex-conversation-1"}}

graph.invoke({"messages": [("user", "My name is Alex.")]}, config)
graph.invoke({"messages": [("user", "What's my name?")]}, config)
```

**Speech:**
"Here's the piece that makes this actually usable: `thread_id`. Every call now passes this `config` dictionary with a thread ID — think of it as a conversation ID. As long as you use the *same* thread ID across calls, the checkpointer remembers everything that happened. Let's run this live." *(run both, same thread_id)* "And now — it remembers. Same two questions as before, completely different result."

---

## Slide 5 — Different thread_id, fresh memory

**Switch to code editor.**

```python
new_config = {"configurable": {"thread_id": "someone-else"}}
graph.invoke({"messages": [("user", "What's my name?")]}, new_config)
```

**Speech:**
"And if I switch to a brand new thread ID — watch — completely fresh, no memory of Alex at all." *(run live)* "This is exactly how you'd separate different users, or different conversations from the same user, in a real application: one thread ID per conversation."

---

## Slide 6 — Recap + what's next

**On slide:**
- "checkpointer=MemorySaver() turns state persistent"
- "thread_id separates one conversation from another"
- "Next video: looking inside that saved state"

**Speech:**
"So: add a checkpointer at compile time, and pass a consistent `thread_id` to keep a conversation connected across calls. Next video, we go a level deeper — actually inspecting and even editing that saved state directly, which turns out to be a great debugging tool too."

---

## Production notes

- **The "blank slate" demo (slide 2) needs to actually land as slightly frustrating** — don't rush past it. It's the entire motivation for this video.
- **Be upfront that MemorySaver is dev-only.** A single honest sentence here prevents someone from shipping it to production and being confused later when their state vanishes on a server restart.