# Video 2: State — the single idea everything else depends on

**Target length:** 12–15 minutes (this is your first code-heavy video — slightly longer is fine, but watch pacing on the TypedDict/Pydantic section, it's the easiest place to overrun)
**Goal of this video:** Learner understands what state is conceptually, can write a basic TypedDict state schema confidently, and knows *when* to reach for Pydantic instead — without yet touching nodes, edges, or StateGraph.
**Golden rule for this video:** No `StateGraph`, no `add_node`, no compiling or running anything end-to-end. State is taught in complete isolation first — resist the urge to "just show it working," that's video 4's job. If they don't understand state on its own, wiring it into a graph will just look like magic.

---

## Slide 1 — Recap + hook

**On slide:**
- One line: "Last time: why LangGraph exists. Today: the one concept everything else builds on."

**Speech:**
"Last video we talked about *why* LangGraph exists — because plain chains can't loop or branch. Today we start actually building — and we're starting with state, because honestly, almost every LangGraph bug you'll ever hit traces back to a misunderstanding of state. Get this right, and the rest of the course gets a lot easier."

---

## Slide 2 — What is state, conceptually (no code)

**On slide:** *(recreate the diagram shown above — three nodes in a row, each connected down to a shared "State" container below them, labeled "shared data every node reads and writes")*

**Speech:**
"Think of state as a shared notebook that every node in your graph can read from and write to. Node 1 might write down what the user asked. Node 2 reads that, does some work, and adds its own note. Node 3 reads everything written so far and produces the final answer. Nobody has their own private notebook — it's one shared notebook, passed along and updated as the graph runs. That's it. That's the whole concept. Everything we do today is really just: how do we define what that notebook is allowed to contain?"

*Pause after this — it's genuinely the core mental model for the entire course. Don't rush into code yet.*

---

## Slide 3 — Why you need a schema (not just a plain dict)

**On slide:**
- "Why not just use a plain Python dict?"
- Two bullets: "No guarantee of what keys exist" / "Typos become silent bugs"

**Speech:**
"Now, you might think — why not just use a regular Python dictionary for this? You could. But here's the problem: with a plain dict, nothing stops one node from writing `state['respones']` — typo and all — while another node reads `state['response']`. That typo won't throw an error. It'll just silently fail, and you'll spend twenty minutes wondering why your data disappeared. This is exactly why LangGraph wants you to define a *schema* — a strict definition of what your state is allowed to look like."

*Why this slide matters: beginners often skip straight to "here's the syntax" without understanding the problem it solves — and then TypedDict feels like arbitrary ceremony instead of a real fix.*

---

## Slide 4 — Live code: your first state schema (TypedDict)

**Switch to code editor.** Type this live, narrating as you go:

```python
from typing import TypedDict

class AgentState(TypedDict):
    user_question: str
    answer: str
```

**Speech, narrating line by line (first-appearance syntax explanation):**

- "`from typing import TypedDict` — TypedDict comes from Python's built-in `typing` module, nothing to install."
- "`class AgentState(TypedDict):` — we're defining a class, but instead of a regular class, it inherits from TypedDict. This tells Python: this class describes the *shape* of a dictionary, not a normal object."
- "`user_question: str` — this is a type hint. It says: this state will have a key called `user_question`, and its value should be a string. Same for `answer`."

"That's genuinely the whole syntax. Two lines and you've told LangGraph exactly what your state is allowed to contain: two keys, both strings. Any node that tries to read or write a key that isn't defined here — your editor and type checker will flag it before you even run the code."

*Production note: literally type this from scratch on screen, don't paste it in. Watching you type builds more confidence than a pre-written snippet.*

---

## Slide 5 — What TypedDict does *not* do (important honesty moment)

**On slide:**
> "TypedDict is a hint for your editor — not a runtime check."

**Speech:**
"Here's something important that trips people up: TypedDict does *not* stop bad data at runtime. If you pass a number where a string should be, Python won't complain while your code is actually running — your editor and type checker catch it while you're *writing* code, but nothing enforces it live. For a lot of graphs, that's totally fine. But it's worth knowing, because it's exactly why Pydantic exists — which we're covering next."

*This is a deliberately honest slide. Don't gloss over the limitation — being upfront here is what earns trust for the comparison coming up.*

---

## Slide 6 — Live code: the same state in Pydantic

**Switch to code editor.**

```python
from pydantic import BaseModel

class AgentState(BaseModel):
    user_question: str
    answer: str
```

**Speech, narrating line by line:**

- "`from pydantic import BaseModel` — Pydantic is a separate library, it'll already be installed since LangChain depends on it."
- "`class AgentState(BaseModel):` — same idea as before, but now we inherit from `BaseModel` instead of `TypedDict`."
- "The fields look identical — `user_question: str`, `answer: str`. But the behavior underneath is different."

"Watch what happens if I try to create this with a number instead of a string for `user_question`..."

*Show live: instantiate with a bad type, e.g. `AgentState(user_question=123, answer="hi")`, and show the validation error that Pydantic actually raises at runtime.*

"See that? Pydantic actually checked the data *while the program was running* and refused to let bad data through. That's the core difference: TypedDict trusts you, Pydantic checks you."

---

## Slide 7 — TypedDict vs Pydantic: when to use which

**On slide (keep as a clean two-column comparison, not paragraphs):**

| | TypedDict | Pydantic |
|---|---|---|
| Runtime validation | No | Yes |
| Performance overhead | None | Small |
| Best for | Internal state, fast iteration | External input, strict correctness |
| Good default for this course | ✅ Yes, most of the time | Use when input comes from outside your graph |

**Speech:**
"So here's the actual decision, not just the syntax difference. Use TypedDict as your default — it's what most of the official LangGraph examples use, it's zero overhead, and for state that only flows *between your own nodes*, you don't usually need runtime policing. Reach for Pydantic specifically when state is going to hold data coming from *outside* your graph — think: a user submitting a form, or a response from an external API — somewhere bad data could realistically sneak in and you want it caught immediately rather than silently corrupting your graph three nodes downstream."

"One more honest caveat: Pydantic validation in LangGraph only runs on the input to your very first node — it doesn't re-validate on every single node afterward. So it's not a silver bullet throughout your whole graph, it's really about locking down your entry point."

*This caveat is easy to skip but genuinely matters — don't let learners walk away thinking Pydantic protects every step of the graph.*

---

## Slide 8 — A quick word on dataclasses (brief, don't dwell)

**On slide:**
- One line: "A third option exists — Python dataclasses — mainly for when Pydantic's validation cost matters at scale. Not needed yet."

**Speech:**
"Quick mention — you'll sometimes see a third option, Python's built-in `dataclass`, used as a middle ground. I'm not going to cover it today, it's mostly relevant once you're optimizing performance in bigger systems, which is production-course territory. For now, you have everything you need with TypedDict and Pydantic."

*Why include this at all: so learners who see `dataclass` in docs or other tutorials aren't confused about where it fits — without derailing this video teaching a third syntax.*

---

## Slide 9 — Designing your own state (the actual skill)

**On slide:**
- "Ask yourself: what does every node need to see?"
- "Start minimal — add fields only when a node actually needs them"

**Speech:**
"Here's the real skill, beyond syntax: designing what belongs in your state in the first place. A common beginner mistake is cramming everything you can think of into state upfront. Don't. Start minimal — ask, what does my *first* node actually need to receive, and what does it need to produce? Add fields as your graph actually grows and needs them. We're going to keep growing this exact `AgentState` class over the next few videos, so you'll see this evolve naturally instead of guessing all your fields on day one."

*This sets up continuity — a strong tutorial habit: reuse the same evolving example across videos rather than a new toy example every time.*

---

## Slide 10 — Recap + what's next

**On slide:**
- "State = shared notebook every node reads/writes"
- "TypedDict = default choice. Pydantic = when validating outside input"
- "Next video: nodes — functions that actually update state"

**Speech:**
"Let's recap: state is the shared data structure flowing through your graph. Use TypedDict as your default, reach for Pydantic when you need to validate data coming from outside. Next video, we finally make this state *do* something — we'll write our first nodes, which are just functions that read state and return updates to it. See you there."

---

## Production notes (not slide content)

- **Where this video is most likely to overrun:** the TypedDict vs Pydantic comparison (slides 5–8). If you're running long, the dataclass mention (slide 8) is the safest thing to cut entirely — it's a nice-to-have, not essential.
- **Don't run the graph yet.** It'll be tempting to say "let's see this in action" and wire up a quick StateGraph — resist it. State deserves to stand alone conceptually before it's wired into anything, or the two ideas blur together for beginners.
- **Reuse `AgentState` going forward.** From video 3 onward, keep building on this same class rather than introducing a new example — this is what makes the course feel like one continuous build instead of disconnected demos.
- **The live validation-error moment (slide 6)** is worth getting right — an actual Pydantic `ValidationError` traceback on screen does more to teach "runtime validation" than any amount of explanation.