# Capstone Part 3: Human approval, polish, and final demo

**Target length:** 18–22 minutes
**Goal:** Add the human-in-the-loop approval gate before the risky tool, add streaming for the final UX polish, and run a complete end-to-end demo of the finished capstone — then close out the course.
**Golden rule:** This is the payoff video for the entire course. Give the final demo real room to breathe — don't rush the ending.

---

## Slide 1 — Recap + hook

**On slide:** "One gap left: nothing stops it from filing a ticket on its own. Let's fix that — and finish."

**Speech:**
"We're on the last piece. Our assistant has tools, memory, and self-correction. But that `file_support_ticket` tool can still fire completely on its own, with no human check. Today, we close that gap, add streaming for a better feel, and run the whole thing end to end."

---

## Slide 2 — Adding the approval gate

**Switch to code editor.**

```python
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],
)
```

**Speech:**
"Straight from video 15 — `interrupt_before`, pausing right before the tools node runs. Now, technically, this pauses before *any* tool call, not just the risky one — for a real system you might want to distinguish which tool is about to run and only pause for the risky ones. Let's actually build that distinction, since it's a realistic refinement."

---

## Slide 3 — Only pausing for the risky tool

**Switch to code editor.**

```python
def check_needs_approval(state: SupportState) -> str:
    last_message = state["messages"][-1]
    for call in last_message.tool_calls:
        if call["name"] == "file_support_ticket":
            return "needs_approval"
    return "safe"

builder.add_conditional_edges("agent", check_needs_approval, {
    "needs_approval": "await_approval",
    "safe": "tools",
})
```

**Speech:**
"This is genuinely just another routing function, video 5's pattern again — check if the pending tool call is our risky one, and route differently if so. We add a tiny `await_approval` node that does nothing itself, and gate *that* specific node with `interrupt_before` instead of gating every tool call. Same mechanism as before, applied more precisely."

---

## Slide 4 — Approving or rejecting the risky action

**Switch to code editor.** Run a conversation that triggers the ticket-filing tool, show it pause, inspect state (video 14), and either approve (`invoke(None, config)`) or reject by editing state to cancel the tool call.

**Speech:**
"Let's trigger this live — I'll ask it to file a ticket about a real issue." *(run live)* "And there — it paused, exactly where we wanted. Let's check what it's about to do..." *(get_state)* "...looks right, let's approve it..." *(invoke None)* "...and it completes. If I didn't like what it was about to file, I could edit the state first, exactly like our human-in-the-loop mini-project."

---

## Slide 5 — Adding streaming for final polish

**Switch to code editor.**

```python
for chunk in graph.stream({"messages": [("user", user_input)]}, config):
    print(chunk)
```

**Speech:**
"Last touch — video 11's streaming, swapped in for our final demo, so we can watch the whole thing happen live instead of waiting on a result. This is genuinely just a drop-in replacement at this point — no new wiring required."

---

## Slide 6 — The full end-to-end demo

**Switch to code editor.** Run a complete, realistic multi-turn conversation covering: normal Q&A with a tool lookup, a self-correction moment if possible, and a ticket-filing request that pauses for approval — all on one `thread_id`, all streamed.

**Speech:**
*(Let this run at a natural, unhurried pace — narrate what's happening, but let the system do the work on screen)* "This is the whole thing — remembering context from earlier in the conversation, using tools, checking its own answers, and stopping to ask before doing anything consequential. Every single piece of this, you built yourself, one concept at a time, since video 1."

---

## Slide 7 — Course wrap-up

**On slide:**
- "You built: state, nodes, edges, conditional routing, loops, tools, ReAct agents, streaming, memory, human-in-the-loop, subgraphs, multi-agent coordination"
- "Three real mini-projects, plus this capstone"

**Speech:**
"Take a moment on this one. Every concept on this list — you didn't just watch it, you built it, by hand, and then used it in something real. That's genuinely the whole philosophy behind how this course was built: nothing abstract, everything eventually put to work."

---

## Slide 8 — What's next

**On slide:**
- "LangGraph for Production — deployment, scaling, observability, advanced multi-agent patterns"
- Thank you + subscribe

**Speech:**
"If you want to take this further — actually deploying something like this, handling real scale, proper observability so you know what's happening in production, more advanced multi-agent orchestration — that's exactly what the next course, LangGraph for Production, covers. Thanks for building this with me — see you there."

---

## Production notes

- **This video deserves the most generous pacing of the entire course.** It's the payoff for everything — resist any urge to rush the final demo to hit a runtime target.
- **Try to genuinely trigger every major behavior in the final demo** (a tool lookup, a self-correction if you can engineer one, and the approval pause) rather than a single happy-path exchange — a demo that shows the *breadth* of what was built lands far better than a short, safe one.
- **The wrap-up (slides 7–8) is a real moment, not boilerplate.** Say it like you mean it — this is the point where a viewer decides whether to trust you with their time on the next course too.