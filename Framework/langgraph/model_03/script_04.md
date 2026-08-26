# Mini-project 3: Persistent assistant with an approval gate

**Target length:** 18–22 minutes
**Goal:** Combine every Module 3 concept — checkpointed memory, thread-based conversations, state inspection, and a human-in-the-loop approval gate — into one working assistant. Zero new concepts, same rule as mini-projects 1 and 2.
**Golden rule:** This project is the reason Module 3 exists as a module — make the throughline explicit: "remember this from video 13... this from video 14... this from video 15" throughout.

---

## Slide 1 — Framing: what we're building and why

**On slide:**
- "A personal assistant that remembers your conversation AND asks before taking a risky action"
- "Every piece: memory, state inspection, human-in-the-loop — all from this module"

**Speech:**
"This project ties together everything from module three. We're building an assistant that remembers what you've told it across a conversation, and pauses to ask permission before doing something risky — in our case, simulating sending a message on your behalf."

---

## Slide 2 — The spec

**On slide:**
- "Remembers user details across turns (memory)"
- "Can draft a message based on conversation context"
- "Pauses for approval before 'sending' it (human-in-the-loop)"
- "A human can approve, or edit the draft before it goes"

**Speech:**
"Here's the flow: you chat with it normally, it remembers context. When you ask it to send something on your behalf, it drafts the message, then stops and waits for you to approve — or edit — before it actually 'sends.'"

---

## Slide 3 — State design (recap from video 2)

**Switch to code editor.**

```python
class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
```

**Speech:**
"Our state — a growing message list, from video 9's reducer pattern, plus a `draft` field to hold the message we're about to send. Nothing new here, just applying what you already know to a new shape of problem."

---

## Slide 4 — The nodes

**Switch to code editor.**

```python
def assistant_node(state: AssistantState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def draft_message_node(state: AssistantState) -> dict:
    draft = llm.invoke(f"Draft a short message based on this conversation: {state['messages']}")
    return {"draft": draft.content}

def send_message_node(state: AssistantState) -> dict:
    print(f"[SIMULATED SEND]: {state['draft']}")
    return {"messages": [("assistant", "Message sent.")]}
```

**Speech:**
"Three straightforward nodes — a general assistant node for normal conversation, a node that drafts a message when asked, and a node that 'sends' it — simulated here with a print statement, since we're not wiring up a real email API for this course."

---

## Slide 5 — Wiring the graph with the approval gate

**Switch to code editor.**

```python
builder = StateGraph(AssistantState)
builder.add_node("assistant", assistant_node)
builder.add_node("draft", draft_message_node)
builder.add_node("send", send_message_node)

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", should_draft, {"draft": "draft", "end": END})
builder.add_edge("draft", "send")
builder.add_edge("send", END)

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["send"],
)
```

**Speech:**
"And here's where module three comes together in one place: a checkpointer for memory, straight from video 13, and `interrupt_before=["send"]`, straight from video 15, pausing right before that risky final step. `should_draft` is just a routing function, video 5's pattern, deciding whether the user's message needs a draft-and-send flow or just a normal reply."

---

## Slide 6 — Running the full flow

**Switch to code editor.** Live demo across several calls on the same `thread_id`:
1. "My name is Jordan, and I'm working on the Q3 report."
2. "Let the team know the report will be a day late."
3. Show it pausing before send.
4. `get_state` to inspect the draft.
5. Either approve directly, or `update_state` to tweak the draft first.
6. `invoke(None, config)` to resume and complete the send.

**Speech:**
*(narrate each step live, explicitly naming which video's concept is in play at each step)* "Watch — it remembers Jordan's name from message one. When I ask it to notify the team, it drafts something, then stops — right where we told it to. Let's look at what it drafted..." *(get_state)* "...and let's say I want to soften the wording a bit before it goes out..." *(update_state)* "...and now I approve it..." *(invoke None)* "...and there's our simulated send, using the edited version."

---

## Slide 7 — What you just proved to yourself

**On slide:**
- "A real assistant pattern: memory + a trustworthy approval gate"
- "Every piece: something you already knew from this module"

**Speech:**
"This pattern — remembering context, then pausing before anything consequential — is genuinely close to how real production assistants that take real-world actions are built. And once again: nothing new today, just module three's three ideas working together."

---

## Slide 8 — Recap + what's next

**On slide:**
- "Module 3 complete: memory, state inspection, human-in-the-loop"
- "Next module: composing bigger systems — subgraphs and multi-agent patterns"

**Speech:**
"That's module three wrapped. Next, we zoom out — how do you organize bigger systems as they grow? Subgraphs for composition, and a simple pattern for coordinating multiple specialized agents. See you there."

---

## Production notes

- **Name the source video for each concept as you use it, out loud, on screen.** This project works best as an explicit "remember this? here it is again" tour — don't be shy about the callbacks, they're the entire pedagogical point.
- **The simulated send (a print statement) is fine and honest — say so on camera.** No need to apologize for not wiring a real email API; briefly explaining the simplification keeps things transparent without derailing the video into third-party API setup.