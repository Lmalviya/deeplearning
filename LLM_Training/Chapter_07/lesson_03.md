# Chapter 7 · Lesson 3 — Instruction Tuning: Dataset Formats and Loss Masking

> **Where this fits:** Lesson 2 covered full fine-tuning as a general mechanism. This lesson covers the specific, most common application of it (or of PEFT methods, Lessons 4-6 — the format/masking content here applies regardless of which training method is used) — teaching a model to follow instructions, directly targeting the capability gap diagnosed in Chapter 5, Lesson 3.

---

## 1. Why Instruction-Tuning Data Looks Different From Pretraining Data

Recall Chapter 2, Lesson 1: pretraining data is raw, unstructured text, self-labeling via the next-token shift trick. Instruction-tuning data is explicitly structured as **(instruction, response)** pairs — because the goal is no longer "model the statistics of general text" but specifically "learn the behavior pattern of responding helpfully to a given instruction."

---

## 2. Common Dataset Formats

**Alpaca-style** (a widely-used early format, still commonly seen):
```json
{
  "instruction": "Summarize the following text in one sentence.",
  "input": "The quick brown fox... [longer text]",
  "output": "A fox jumps over a dog."
}
```

**ChatML-style / conversation-turn format** (closer to how modern chat models are actually structured, supporting multi-turn conversations):
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Summarize the following text in one sentence: [text]
<|im_end|>
<|im_start|>assistant
A fox jumps over a dog.
<|im_end|>
```

**Why the shift toward ChatML-style formats over time, worth being able to explain:** Alpaca-style single-turn (instruction, input, output) triples don't naturally represent multi-turn conversations or system-level instructions — real deployed usage (chat interfaces, agents with persistent context) needs a format that natively supports conversation history and role distinction (system/user/assistant), which ChatML-style special-token delimiting handles directly.

---

## 3. Loss Masking — The Critical, Easy-to-Get-Wrong Detail

**The core idea:** during instruction-tuning, you don't want the model to be trained to *predict the instruction/prompt itself* — only to predict the response, given the instruction. Computing loss over the prompt tokens would train the model to become better at generating prompts, which is not the goal.

**Directly reusing the `ignore_index=-100` mechanism from Chapter 2, Lesson 1's padding mask, and Chapter 5, Lesson 2's MLM masking:**

```python
def build_instruction_example(tokenizer, instruction, response, ignore_index=-100):
    prompt_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    full_text = prompt_text + response + "<|im_end|>"

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    labels = full_ids.copy()
    # Mask out the prompt portion — loss is only computed on the RESPONSE tokens
    labels[:len(prompt_ids)] = [ignore_index] * len(prompt_ids)

    return {"input_ids": full_ids, "labels": labels}
```

**Worked example, made concrete:** for the ChatML example in Section 2, everything from `<|im_start|>system` through `<|im_start|>assistant\n` gets `-100` labels — no gradient signal from those tokens at all. Only `"A fox jumps over a dog.<|im_end|>"` contributes to the loss. This is directly analogous to Chapter 2, Lesson 1's causal LM loss formula, just restricted to a subset of positions — exactly the same underlying mechanism as Chapter 5, Lesson 2's MLM masking (which restricted loss to the masked 15%), applied here to restrict loss to the response portion instead.

**What happens if this masking is forgotten — a real, common bug, worth being able to diagnose:** the model gets trained to predict the *instruction* text as well as the response, which can cause it to become worse at genuinely novel instructions (having partially learned to reproduce instruction *patterns* from the training set rather than purely learning response *behavior*), and wastes a meaningful fraction of the gradient signal on a task (predicting the prompt) that has nothing to do with the actual goal.

---

## 4. Multi-Turn Conversations — Masking Gets More Involved

For a multi-turn conversation, the masking needs to cover *every* user/system turn while leaving *every* assistant turn unmasked — not just a single prompt/response split.

```python
def build_multiturn_example(tokenizer, turns, ignore_index=-100):
    """
    turns: list of dicts like {"role": "user"/"assistant"/"system", "content": str}
    """
    full_ids = []
    labels = []

    for turn in turns:
        turn_text = f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
        turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)
        full_ids.extend(turn_ids)

        if turn["role"] == "assistant":
            labels.extend(turn_ids)  # train on assistant turns
        else:
            labels.extend([ignore_index] * len(turn_ids))  # mask user/system turns

    return {"input_ids": full_ids, "labels": labels}
```

**Why this matters beyond just "more code":** a multi-turn conversation gives the model training signal on *how to respond given accumulated context*, not just how to respond to an isolated instruction — this is part of what teaches the long-context/conversation-consistency-adjacent behaviors touched on in Chapter 6, Lesson 3's multi-turn consistency eval example. Getting the masking wrong here (e.g., accidentally training on user turns in a multi-turn setup) is a more consequential bug than the single-turn case, since it compounds across every turn in every conversation in the dataset.

---

## 5. Diagnosis & Mental Models: Symptoms of a Masking Bug

- **Model's outputs start including prompt-like or instruction-like phrasing unprompted** (e.g., repeating back something resembling "Summarize the following:" before its actual answer) → strong signal the prompt wasn't properly masked out of the loss.
- **Model trained on multi-turn data performs oddly worse on later turns than earlier ones** → check whether user/system turn masking was applied consistently across the whole conversation, not just the first turn — a common implementation bug where masking logic is correct for a single-turn case but breaks down for turn 3, 4, 5 of a longer conversation.

---

## Key Takeaways

- Instruction-tuning data is explicitly structured as instruction/response pairs (or multi-turn conversations), unlike pretraining's raw self-labeling text.
- ChatML-style formats with role delimiters have largely superseded simpler Alpaca-style formats because they natively support multi-turn conversations and system-level instructions.
- Loss masking (restricting gradient signal to response/assistant tokens only) is the same `-100`/`ignore_index` mechanism used elsewhere in this curriculum, applied to a new context — and getting it wrong is a real, diagnosable bug with specific symptoms.
- Multi-turn masking needs to correctly handle every turn, not just the first prompt/response pair — a common source of subtle, compounding bugs.

---

## Self-Check Before Moving to Lesson 4

1. Explain why loss must be masked on the instruction/prompt portion during instruction tuning, using the same underlying reasoning as Chapter 2 Lesson 1's shift-by-one trick.
2. What specific model behavior would you expect to observe if the masking was accidentally omitted entirely?
3. Why is multi-turn masking a more consequential place to have a bug than single-turn masking?