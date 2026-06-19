# Lesson 2.5 — Chat Templates and Instruction Formatting
### Chapter 2: What Fine-Tuning Actually Does to a Model

---

## The Problem Story

Karthik fine-tuned Mistral-7B for an internal HR assistant. The loss during training looked perfect — decreasing smoothly, final value around 0.6. He deployed it.

The model was generating incorrect outputs. It would sometimes repeat the user's question as part of its answer. Occasionally it would generate a second "user turn" after answering, as if it were roleplaying both sides of the conversation.

The bug: Karthik did not apply the chat template correctly. He formatted his training data like this:

```
User: What is the leave policy?
Assistant: Employees get 20 days of annual leave.
```

But Mistral's actual expected format is:

```
<s>[INST] What is the leave policy? [/INST] Employees get 20 days of annual leave.</s>
```

And he was computing loss over the entire string — including "User:" and "Assistant:" labels — which taught the model to predict those strings too.

At inference, he was not using the correct prompt format either. The model had learned to expect `[INST]` markers and never saw them.

Two mistakes. Both invisible in the training loss. Both catastrophic in deployment.

This lesson means you will never make either of these mistakes.

---

## The Concept

### Why Chat Templates Exist

A base language model (pre-trained but not instruction-tuned) is trained to continue text — given a sequence of tokens, predict what comes next. It has no concept of "this is a user asking a question" versus "this is an assistant answering."

To fine-tune a model to participate in a conversation with defined roles (system, user, assistant), you need a **convention** that the model learns to recognize. This convention is the chat template.

A chat template does two things:
1. Wraps each conversational turn with special tokens that signal role boundaries
2. Creates a consistent format that the model can learn to parse and produce

The model is not born knowing these conventions. It learns them during instruction fine-tuning by seeing thousands of examples formatted this way, with loss computed only on the assistant turns.

---

### The Major Chat Template Formats

Different model families use different conventions. You must use the exact format that matches your base model — mixing formats causes the model to see a format it was never trained on.

**LLaMA-2 Chat Format:**
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is the capital of France? [/INST] The capital of France is Paris. </s>

<s>[INST] And what is its population? [/INST] Paris has a population of approximately 2.1 million... </s>
```

Key markers: `<s>`, `[INST]`, `[/INST]`, `<<SYS>>`, `<</SYS>>`, `</s>`

**LLaMA-3 / ChatML Format:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The capital of France is Paris.<|eot_id|>
```

Key markers: `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`

**Phi-3 Format:**
```
<|system|>
You are a helpful assistant.<|end|>
<|user|>
What is the capital of France?<|end|>
<|assistant|>
The capital of France is Paris.<|end|>
```

Key markers: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`

**Alpaca Format (older, for base models without chat training):**
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris.
```

No special tokens — just text markers. Used for base models that have no pre-existing chat template. Loss computed only on the response portion.

**ShareGPT Format (data format, not a prompt format):**
```python
{
    "conversations": [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human",  "value": "What is the capital of France?"},
        {"from": "gpt",    "value": "The capital of France is Paris."}
    ]
}
```

This is a data storage format, not the actual tokenized format. You apply the model's chat template to convert this into the tokenized format at training time.

---

### `apply_chat_template()`: The Right Way to Format

HuggingFace tokenizers include the chat template for their model family. You should **always** use `apply_chat_template()` rather than manually constructing the formatted string. Manual formatting is the direct cause of Karthik's bug.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

# Your raw conversation data
messages = [
    {"role": "system",    "content": "You are a helpful HR assistant."},
    {"role": "user",      "content": "What is the annual leave policy?"},
    {"role": "assistant", "content": "Employees receive 20 days of annual leave per year."},
    {"role": "user",      "content": "Can I carry over unused leave?"},
    {"role": "assistant", "content": "Yes, up to 5 days can be carried over to the next year."},
]

# Let the tokenizer apply the correct template
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,       # return string first, to inspect
    add_generation_prompt=False  # True only during inference
)

print(formatted)
```

**`add_generation_prompt=True` vs `False`:**

- `False`: used during training. Produces the full conversation including the final assistant response.
- `True`: used during inference. Adds the assistant turn opening (e.g., `<|assistant|>`) without content, signaling to the model "now you start generating."

```python
# During training (complete conversations):
formatted_train = tokenizer.apply_chat_template(
    messages_with_response,
    tokenize=False,
    add_generation_prompt=False  # include the response
)

# During inference (prompt only):
formatted_inference = tokenizer.apply_chat_template(
    messages_without_response,
    tokenize=False,
    add_generation_prompt=True  # add the "please respond" marker
)
```

---

### Loss Masking on Prompt Tokens: The Critical Detail

This is the most commonly misunderstood aspect of instruction fine-tuning, and it is what separates people who understand fine-tuning from those who just ran a script.

**The question:** In your training data, which tokens should the model be trained to predict?

**The wrong answer:** All tokens.

**Why it is wrong:**

If you compute loss over all tokens in an instruction-response pair, the model is being trained to predict:
1. The system message tokens
2. The special format tokens (`[INST]`, `[/INST]`, etc.)
3. The user instruction tokens
4. The assistant response tokens

Only (4) is what you actually want the model to learn to generate. Items (1), (2), and (3) are part of the input — they are things the model reads, not things it should produce.

Training on (1)-(3) causes several problems:
- The model devotes capacity to predicting things it will never be asked to generate
- The gradient signal is diluted across more positions, slowing learning on what matters
- The model may learn to include system/user markers in its own outputs (Karthik's bug)

**The correct approach — loss masking:**

Set the labels for all non-assistant tokens to -100. PyTorch's cross-entropy with `ignore_index=-100` skips those positions entirely.

```python
# Example: conversation with prompt masking applied

messages = [
    {"role": "system",    "content": "You are helpful."},
    {"role": "user",      "content": "Tell me about Paris."},
    {"role": "assistant", "content": "Paris is the capital of France."},
]

tokenizer.chat_template  # check the model's template

# Full tokenized conversation
full_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]

# Prompt-only tokenized (everything EXCEPT the last assistant response)
prompt_messages = messages[:-1]  # system + user only
prompt_text = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=False,
    add_generation_prompt=True  # add the assistant opening marker
)
prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]

# Create labels: mask everything up to where the response starts
labels = full_ids.clone()
n_prompt = prompt_ids.shape[1]
labels[0, :n_prompt] = -100  # mask the prompt

print(f"Total tokens: {full_ids.shape[1]}")
print(f"Prompt tokens (masked):   {n_prompt}")
print(f"Response tokens (trained): {full_ids.shape[1] - n_prompt}")
print(f"\nLabel values: {labels[0].tolist()}")
```

**Visualizing what is masked:**

```
Full sequence:
<|system|>You are helpful.<|end|><|user|>Tell me about Paris.<|end|><|assistant|>Paris is the capital of France.<|end|>
│                                                                    │                                              │
│◄───────────────── labels = -100 (masked) ──────────────────────►│◄──────── labels = token IDs (train on) ─────►│
```

---

### The TRL `SFTTrainer` Approach (Modern Standard)

Rather than implementing loss masking manually, the TRL library's `SFTTrainer` handles this automatically when you provide data in the messages format.

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # right padding for training

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Dataset in messages format
from datasets import Dataset

data = [
    {
        "messages": [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI where models learn from data."},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "What is deep learning?"},
            {"role": "assistant", "content": "Deep learning uses multi-layer neural networks to learn representations."},
        ]
    },
]
dataset = Dataset.from_list(data)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# SFTTrainer config
sft_config = SFTConfig(
    output_dir="./phi3-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    max_seq_length=512,
    # The key parameter: tell SFTTrainer to use the "messages" column
    # and apply the chat template automatically
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# SFTTrainer automatically:
# 1. Applies the model's chat template to each "messages" list
# 2. Sets labels = -100 for all non-assistant tokens
# 3. Handles padding, truncation, packing

print("SFTTrainer configured. Prompt masking applied automatically.")
print(f"Model trainable params: {model.print_trainable_parameters()}")
```

---

### Multi-Turn Conversations and Loss Masking

Real conversations have multiple turns. Loss masking in multi-turn conversations should mask ALL user/system turns, not just the first one.

```
Turn 1 — User:      "What is Paris?"          ← mask
Turn 1 — Assistant: "Paris is a city in France."  ← TRAIN
Turn 2 — User:      "What is its population?"  ← mask
Turn 2 — Assistant: "About 2.1 million."         ← TRAIN
```

The model should learn to generate both assistant turns. It should not learn to predict the user turns.

Implementing this manually requires finding the token position of every assistant turn and masking everything else. This is why `SFTTrainer` or a well-tested data preprocessing function is strongly preferred over manual implementation.

---

### Format Consistency: Training vs Inference

The single most common deployment bug after format-related bugs is format inconsistency between training and inference.

**Rule: The inference prompt format must be byte-for-byte identical to the training format.**

```python
# Training format (what the model saw during training):
# apply_chat_template(messages, add_generation_prompt=False)
# → "<|system|>\nYou are helpful.<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi!<|end|>"

# Inference format (what you give the model to continue):
# apply_chat_template(messages_without_response, add_generation_prompt=True)
# → "<|system|>\nYou are helpful.<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\n"
#                                                                                    ↑
#                                            model generates from here
```

If your training used `"<|system|>\nYou are helpful.<|end|>\n"` but your inference uses `"<|system|>You are helpful.<|end|>"` (missing the newline), the tokenization is different. The model has never seen this exact sequence of tokens. Output quality degrades.

This is not a theoretical edge case. It is a real, common bug in production deployments.

**The safe pattern:**

Always use `tokenizer.apply_chat_template()` for both training data preparation and inference prompt construction. Never manually construct the formatted string in either case.

---

### Checking Your Loss Masking is Correct

After setting up your data pipeline, always verify that loss masking is applied correctly before running a full training run:

```python
def verify_loss_masking(dataset_example, tokenizer):
    """
    Print which tokens contribute to loss vs which are masked.
    Run this on 3-5 examples before every training run.
    """
    messages = dataset_example["messages"]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],  # everything except last assistant turn
        tokenize=False,
        add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]

    n_prompt = len(prompt_ids)
    n_total = len(full_ids)
    n_response = n_total - n_prompt

    print(f"Total tokens:    {n_total}")
    print(f"Masked (prompt): {n_prompt} ({100*n_prompt/n_total:.1f}%)")
    print(f"Trained (resp):  {n_response} ({100*n_response/n_total:.1f}%)")

    print("\nToken-by-token breakdown (last 10 prompt + all response):")
    print(f"{'Pos':>4} {'Token':<20} {'Trains?'}")
    print("-" * 40)
    for i in range(max(0, n_prompt - 5), n_total):
        token_str = tokenizer.decode([full_ids[i]])
        trains = "YES ←" if i >= n_prompt else "masked"
        print(f"{i:>4} {repr(token_str):<20} {trains}")

    return n_prompt, n_response

# Verify on your first example before training
verify_loss_masking(data[0], tokenizer)
```

Run this function on at least 5 examples from your dataset before starting training. If the masked portion is too large (>90% of tokens) or too small (<10%), something is likely wrong.

---

## The Intuition Bridge

**The template is a language the model has to learn:**

When the base model was pre-trained, it saw millions of text documents but no chat templates. These markers (`[INST]`, `<|assistant|>`, etc.) are a new "micro-language" layered on top of natural language.

During instruction fine-tuning, the model sees thousands of examples in this format with loss computed on the assistant turns. It learns: "when I see `<|assistant|>`, I should generate a helpful response. When I see `<|user|>`, I read and understand."

If you get the template wrong during fine-tuning, the model learns the wrong "micro-language." If you get the template wrong at inference, you are speaking the wrong language to a model that learned the right one.

**Loss masking as teaching by example:**

Imagine you are teaching someone to be a customer service representative. You show them 1000 conversation transcripts — both the customer's words and the agent's responses.

If you test them by asking "what did the customer say in conversation 5?" — you tested the wrong skill. You trained them to understand conversations, but you need them to generate good responses.

Loss masking focuses the model on generating the response, not memorizing the prompt. The model is trained to answer questions, not to repeat them.

---

## Why This Matters for Fine-Tuning

**Every hour you invest in getting the template right saves many hours of debugging later.** Template errors are invisible in training loss and only surface in deployment.

**Loss masking is not optional for instruction fine-tuning.** Computing loss over instruction tokens slows learning, dilutes the gradient signal on what matters, and can teach the model to output format markers in its responses.

**Multi-turn conversations require careful masking.** Single-turn datasets are simpler, but real applications usually have multi-turn conversations. Make sure your masking pipeline handles every assistant turn correctly.

---

## The Code (Full Working Pipeline)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch

# ── Full pipeline: data → template → mask → train ──────────────

model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ── Step 1: Inspect the chat template ──────────────────────────

print("Chat template for this model:")
print(tokenizer.chat_template[:300] if tokenizer.chat_template else "None")

# ── Step 2: Format one example and verify ──────────────────────

example = {
    "messages": [
        {"role": "system",    "content": "You are a concise assistant."},
        {"role": "user",      "content": "What is LoRA?"},
        {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is a PEFT method that adds small trainable matrices to frozen model weights."},
    ]
}

# Full formatted text
full_formatted = tokenizer.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False
)
print("\nFull formatted training example:")
print(repr(full_formatted))

# Inference format (without the response)
inference_formatted = tokenizer.apply_chat_template(
    example["messages"][:-1],
    tokenize=False,
    add_generation_prompt=True
)
print("\nInference format (prompt only):")
print(repr(inference_formatted))

# ── Step 3: Verify loss masking ─────────────────────────────────

full_ids = tokenizer(full_formatted, return_tensors="pt")["input_ids"][0]
inference_ids = tokenizer(inference_formatted, return_tensors="pt")["input_ids"][0]

n_prompt = len(inference_ids)
n_total = len(full_ids)

labels = full_ids.clone()
labels[:n_prompt] = -100

print(f"\nTokens total:    {n_total}")
print(f"Tokens masked:   {n_prompt}")
print(f"Tokens trained:  {n_total - n_prompt}")
print(f"\nFirst 5 labels (should be -100): {labels[:5].tolist()}")
print(f"Last 5 labels (should be token IDs): {labels[-5:].tolist()}")
print(f"Last 5 tokens decoded: {tokenizer.decode(full_ids[-5:])}")

# ── Step 4: Build dataset and train ────────────────────────────

dataset_raw = [
    {
        "messages": [
            {"role": "system",    "content": "You are a helpful ML tutor."},
            {"role": "user",      "content": "What is gradient descent?"},
            {"role": "assistant", "content": "Gradient descent is an optimization algorithm that minimizes a loss function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient."},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": "You are a helpful ML tutor."},
            {"role": "user",      "content": "What is overfitting?"},
            {"role": "assistant", "content": "Overfitting occurs when a model learns the training data too well, including its noise, resulting in poor generalization to new data."},
        ]
    },
]
train_dataset = Dataset.from_list(dataset_raw)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# SFT configuration
sft_config = SFTConfig(
    output_dir="./phi3-chapter2-test",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    bf16=False,  # set True if bf16 supported on your hardware
    fp16=True,
    logging_steps=1,
    max_seq_length=512,
    save_steps=50,
    report_to="none",  # set "wandb" when you have W&B configured
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# ── Step 5: Single training step to verify everything works ────

print("\nRunning 1 training step to verify pipeline...")
trainer.train()
print("Training step completed successfully.")

# ── Step 6: Test inference with correct format ──────────────────

model.eval()
inference_messages = [
    {"role": "system", "content": "You are a helpful ML tutor."},
    {"role": "user",   "content": "What is gradient descent?"},
]

inference_prompt = tokenizer.apply_chat_template(
    inference_messages,
    tokenize=False,
    add_generation_prompt=True  # ← critical for inference
)

inf_inputs = tokenizer(inference_prompt, return_tensors="pt").to(model.device)
tokenizer.padding_side = "left"  # switch to left padding for inference

with torch.no_grad():
    output = model.generate(
        **inf_inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    output[0][inf_inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print(f"\nModel response: '{response}'")
```

---

## The Experiment

**EXP-2.5.A — Template Format Verification**

Goal: Confirm that the format at inference exactly matches what the model was trained on.

```python
# For your specific model, verify this:

tokenizer = AutoTokenizer.from_pretrained("your-model-name")

messages = [
    {"role": "system",    "content": "You are helpful."},
    {"role": "user",      "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?"},
]

# Training format
train_fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

# Inference format
inf_fmt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

print("Training format:")
print(repr(train_fmt))
print("\nInference format:")
print(repr(inf_fmt))
print("\nThe inference format should be a prefix of the training format:")
print(f"Is prefix: {train_fmt.startswith(inf_fmt)}")
```

Fill your experiment log:

```
════════════════════════════════════════════════════════
EXPERIMENT LOG
════════════════════════════════════════════════════════
ID:       EXP-2.5.A
Lesson:   2.5 — Chat Templates and Instruction Formatting
Goal:     Confirm training/inference format consistency
          and verify loss masking is applied correctly

SETUP
Model: [your model]
Template: [paste the template or its name]

RAW OBSERVATIONS
Training format (repr): [paste]
Inference format (repr): [paste]
Is training format a superset of inference format?: ___
Prompt tokens (masked): ___ / ___ total tokens
Response fraction: ___% of total sequence

WHAT SURPRISED ME
[Did the template have any unexpected characters — newlines,
 spaces, specific token orders?]
[Was the response fraction larger or smaller than you expected?]

INTERPRETATION
[Why does the inference format end exactly where the
 model should begin generating?]
[If the formats did not match, what would the model do?]

IMPLICATIONS FOR FINE-TUNING
[For your specific use case, what system message will you use?]
[Will your conversations be single-turn or multi-turn?]
[How will you verify masking for multi-turn conversations?]

OPEN QUESTIONS
[Fill]

NEXT STEP
[Fill — likely: prepare your actual dataset in the correct format]
════════════════════════════════════════════════════════
```

---

## Interview Checkpoint

**Q: What is a chat template and why is it necessary for instruction fine-tuning?**

> A: A chat template is a formatting convention that wraps conversational turns with special tokens indicating role boundaries — system, user, assistant. It is necessary because a base language model has no concept of conversation roles; it was trained to continue text. During instruction fine-tuning, the model learns to associate these special tokens with specific behaviors: generate a response after the assistant marker, read and process the content after the user marker. Without a consistent template, the model has no reliable way to determine when to start or stop generating, or which text is its responsibility to produce.

**Q: What is loss masking in the context of instruction fine-tuning and why is it important?**

> A: Loss masking means setting labels to -100 for all tokens in the prompt (system message, user instruction) so the cross-entropy loss ignores them. The model is only trained to predict the assistant response tokens. This is important for two reasons. First, we want the model to learn to generate good responses, not to predict the user's questions or system instructions — those are inputs, not outputs. Second, computing loss over instruction tokens dilutes the gradient signal with noise from positions the model will never generate in deployment, making learning of the actual task slower and less efficient.

**Q: What is the most common bug in fine-tuning deployment related to chat templates?**

> A: Format inconsistency between training and inference. During training, the data is formatted with the model's chat template including special tokens in a specific order. During inference, if the prompt is formatted differently — even slightly, like a missing newline or different whitespace — the tokenization produces different token IDs, and the model receives an input pattern it never saw during training. The model may generate responses that include format markers, repeat the question, or degrade in quality. The fix is always using `tokenizer.apply_chat_template()` for both training data preparation and inference prompt construction, never constructing the formatted string manually.

**Q: How should you handle loss masking for multi-turn conversations?**

> A: In a multi-turn conversation, you should mask all system and user turns but train on all assistant turns. This means identifying the token positions of every assistant response in the full tokenized sequence and setting everything else to -100. In practice, the TRL `SFTTrainer` handles this automatically when your data is in the standard messages format. If implementing manually, the safest approach is to tokenize the full conversation, then tokenize the prompt up to and including the assistant opening marker for each turn, and use the resulting position indices to build the label mask.

---

## Common Mistakes & Misconceptions

❌ **"I can manually write the chat format string instead of using apply_chat_template()."**
This is the most dangerous mistake in instruction fine-tuning. Chat templates include subtle formatting — specific newlines, spacing around special tokens, header formats — that are easily wrong if written manually. Use `apply_chat_template()` every time, for both training and inference.

❌ **"Loss masking on the prompt makes the model unable to understand the instruction."**
Loss masking only affects what the model is trained to predict. The instruction tokens are still present as inputs — the model still processes them through all the attention layers and builds a representation from them. Masking just means "do not train on predicting these tokens." The model still reads and understands the instruction.

❌ **"All models use the same chat template."**
Every model family has its own template. LLaMA-2, LLaMA-3, Phi-3, Mistral, Qwen, Gemma all have different formats. Using the wrong template for a model produces garbage output because the model never learned that format. Always check `tokenizer.chat_template` for your specific model before preparing training data.

❌ **"add_generation_prompt does not matter during inference."**
`add_generation_prompt=True` adds the assistant turn opening token(s) to signal to the model that it should begin generating a response. Without it, the model sees the conversation end without an assistant marker and may generate user text or the next turn of the conversation rather than a response. This is a common deployment bug that is easy to introduce and hard to debug.

❌ **"I verified the training format, so inference will work."**
Training format and inference format must both be verified, separately. Training format uses `add_generation_prompt=False` and includes the full response. Inference format uses `add_generation_prompt=True` and includes only the prompt. They are different strings. Verify both.

---

## Chapter 2 — Completion Checklist

Before moving to Chapter 3, you should be able to:

- [ ] Explain cross-entropy loss mathematically and say what loss=0.8 means in context
- [ ] Walk through a complete training step (forward → loss → backward → optimizer step) without referring to notes
- [ ] Explain why AdamW uses two running averages and what each one does
- [ ] Implement gradient accumulation and explain why it does not change the math
- [ ] Explain catastrophic forgetting and give three concrete ways to prevent it
- [ ] Name the three types of fine-tuning objectives and when to use each
- [ ] Apply a chat template correctly for both training and inference using apply_chat_template()
- [ ] Implement loss masking and verify it is applied correctly with the verify function
- [ ] Completed and logged EXP-2.1.A, EXP-2.2.A, EXP-2.3.A, EXP-2.4.A, EXP-2.5.A

**The oral test:** Set a timer. Explain out loud, as if to a junior colleague:

> "I am going to fine-tune Phi-3 Mini as an HR assistant for our company. Walk me through every decision I need to make before the first training step even begins — from data format to training objective to loss masking."

A complete answer covers: chat template selection, data format (messages structure), apply_chat_template() for training, loss masking on system/user turns, optimizer choice, learning rate range, gradient accumulation setup, mixed precision choice.

If you can answer this fluently, you are ready for Chapter 3.