# Hands-On Training Pipelines — Kaggle Edition
> SFT → GRPO → DPO in sequence, all runnable on 2x T4 (16GB each)

---

## Kaggle Constraints First

| Resource | Limit | Impact |
|---|---|---|
| GPU | 2x T4, 16GB each | Max ~7B model with QLoRA 4-bit |
| GPU hours | 30 hrs/week | Keep runs short, use small subsets |
| RAM | 29GB CPU | Fine for most datasets |
| Storage | 20GB /kaggle/working | Save checkpoints to /kaggle/working |
| Internet | On by default | HuggingFace downloads work fine |

**Recommended base model for all 3 stages: `Qwen2.5-3B-Instruct`**
- 3B fits 2x T4 easily even without 4-bit
- Strong base reasoning capability
- Good chat template support in TRL
- Fast iteration — full epoch in 15–30 mins on Kaggle

---

## Install Block (same for all notebooks)

```python
%%capture
!pip install -q trl peft bitsandbytes transformers datasets accelerate
!pip install -q liger-kernel  # fused kernels, big speedup on T4

# optional but recommended
!pip install -q wandb          # experiment tracking
```

---

## Pipeline 1 — SFT (Structured Instruction Tuning)

### Dataset Options

| Dataset | Size | Type | HF Path |
|---|---|---|---|
| ✅ **UltraChat 200k** | 200k | Multi-turn chat | `HuggingFaceH4/ultrachat_200k` |
| ✅ **Dolly 15k** | 15k | Single-turn instruct | `databricks/databricks-dolly-15k` |
| ✅ **OpenHermes 2.5** | 1M | GPT-4 instruction | `teknium/OpenHermes-2.5` |
| ✅ **TIGER MathInstruct** | 262k | Math instruction | `TIGER-Lab/MathInstruct` |

**Recommended for first run**: `HuggingFaceH4/ultrachat_200k`
- Already split into train/test
- Already formatted as conversations
- Covers broad skills: summarization, QA, coding, reasoning
- Use 10k subset for Kaggle

---

### Data Pipeline

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# --- 1. Load ---
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
eval_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")

# Use a small subset for Kaggle
dataset = dataset.select(range(10_000))
eval_dataset = eval_dataset.select(range(500))

print(dataset[0])
# {"prompt": "...", "messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

# --- 2. Inspect the schema ---
print(dataset.column_names)  # ['prompt', 'prompt_id', 'messages']
print(dataset[0]['messages'])

# --- 3. Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# --- 4. Verify chat template works ---
sample = tokenizer.apply_chat_template(
    dataset[0]['messages'],
    tokenize=False,
    add_generation_prompt=False
)
print(sample[:500])
```

---

### Model Pipeline

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# --- 4-bit quantization config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # QLoRA double quant
)

# --- Load model ---
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# --- LoRA config ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Apply LoRA ---
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~20M / 3B = ~0.6% ← this is normal
```

---

### Training Pipeline

```python
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import re

# --- Format Monitor Callback (from the paper you read) ---
class FormatMonitorCallback(TrainerCallback):
    def __init__(self, tokenizer, val_samples, check_every=100):
        self.tokenizer = tokenizer
        self.val_samples = val_samples
        self.check_every = check_every
        self.log = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.check_every != 0:
            return
        model.eval()
        scores = []
        for sample in self.val_samples[:3]:   # check 3 samples
            inputs = self.tokenizer(sample, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # check for assistant response presence
            has_response = bool(re.search(r"assistant", text.lower()))
            scores.append(1.0 if has_response else 0.0)
        adherence = sum(scores) / len(scores)
        self.log.append({"step": state.global_step, "adherence": adherence})
        print(f"\n[Step {state.global_step}] Format adherence: {adherence:.1%}")
        model.train()

val_prompts = [
    tokenizer.apply_chat_template(
        eval_dataset[i]['messages'][:1],   # just the user turn
        tokenize=False,
        add_generation_prompt=True
    )
    for i in range(3)
]

# --- SFT Training Config ---
sft_config = SFTConfig(
    output_dir="/kaggle/working/sft_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,     # effective batch = 2*8*2GPU = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    packing=True,                      # pack short sequences → big speedup
    max_seq_length=1024,               # keep short for T4 memory
    logging_steps=25,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=200,
    report_to="none",                  # change to "wandb" if you set it up
)

# --- Trainer ---
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    callbacks=[FormatMonitorCallback(tokenizer, val_prompts, check_every=100)],
)

trainer.train()
trainer.save_model("/kaggle/working/sft_model")
```

---

### Evaluation Pipeline

```python
from peft import PeftModel
import json

# --- Load trained model ---
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model_eval = PeftModel.from_pretrained(base_model, "/kaggle/working/sft_model")
model_eval.eval()

def generate_response(messages, max_new_tokens=256):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model_eval.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# --- Run on eval set ---
results = []
for i in range(50):  # eval on 50 samples
    sample = eval_dataset[i]
    user_turn = [sample['messages'][0]]   # just user message
    predicted = generate_response(user_turn)
    reference = sample['messages'][1]['content']  # assistant ground truth
    results.append({
        "input": user_turn[0]['content'][:100],
        "predicted": predicted[:200],
        "reference": reference[:200],
    })

# --- Basic metrics ---
avg_len_pred = sum(len(r['predicted'].split()) for r in results) / len(results)
avg_len_ref  = sum(len(r['reference'].split()) for r in results) / len(results)
print(f"Avg predicted length: {avg_len_pred:.0f} words")
print(f"Avg reference length: {avg_len_ref:.0f} words")

# Save results for manual inspection
with open("/kaggle/working/sft_eval_results.json", "w") as f:
    json.dump(results[:10], f, indent=2)
```

---

## Pipeline 2 — GRPO (Reasoning with Verifiable Rewards)

> **Prerequisite**: Complete Pipeline 1 first. GRPO fine-tunes the SFT model further.

### Dataset Options

| Dataset | Size | Type | Why good for GRPO |
|---|---|---|---|
| ✅ **GSM8K** | 8.5k | Grade school math | Verifiable answers, clean format |
| ✅ **MATH** | 12.5k | Competition math | Harder, also verifiable |
| ✅ **NuminaMath-CoT** | 860k | Math + CoT traces | Has reasoning traces |
| ✅ **AI2 ARC** | 7.7k | Multiple choice reasoning | Easy reward function |

**Recommended for first run**: `openai/gsm8k`
- Small enough for Kaggle
- Answer is always a number — trivial to verify
- Well-studied benchmark, can compare results

---

### Data Pipeline

```python
from datasets import load_dataset
import re

dataset = load_dataset("openai/gsm8k", "main", split="train")
eval_dataset = load_dataset("openai/gsm8k", "main", split="test")

print(dataset[0])
# {
#   "question": "Natalia sold clips to 48 of her friends...",
#   "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips...\n#### 72"
# }

# GSM8K answer format: reasoning text then #### <number>
def extract_gsm8k_answer(text: str) -> str | None:
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None

# Verify extraction works
print(extract_gsm8k_answer(dataset[0]['answer']))   # → "72"

# Format into prompt/answer pairs
def format_for_grpo(example):
    return {
        "prompt": [
            {"role": "system", "content": "You are a math reasoning assistant. Think step by step inside <think> tags, then give your final answer."},
            {"role": "user", "content": example["question"]},
        ],
        "answer": extract_gsm8k_answer(example["answer"]),
    }

dataset = dataset.map(format_for_grpo)
eval_dataset = eval_dataset.map(format_for_grpo)

# Remove examples where extraction failed
dataset = dataset.filter(lambda x: x['answer'] is not None)
```

---

### Reward Functions

```python
import re

def extract_final_answer(text: str) -> str | None:
    """
    Handles two patterns:
    1. After </think> tags: <think>...</think> Final answer: 72
    2. Last number in response as fallback
    """
    # Try to find answer after think block
    after_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", after_think)
    if numbers:
        return numbers[-1].replace(",", "")
    
    # Fallback: last number in full text
    all_numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return all_numbers[-1].replace(",", "") if all_numbers else None

# --- Reward 1: Correctness (binary) ---
def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]['content'] if isinstance(completion, list) else completion
        predicted = extract_final_answer(text)
        correct = predicted == str(answer).replace(",", "")
        rewards.append(2.0 if correct else 0.0)   # 2.0 for correct, 0 for wrong
    return rewards

# --- Reward 2: Format compliance ---
def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]['content'] if isinstance(completion, list) else completion
        has_think_open  = "<think>" in text
        has_think_close = "</think>" in text
        has_content     = len(text.strip()) > 20
        score = sum([has_think_open, has_think_close, has_content]) / 3.0
        rewards.append(score)
    return rewards

# --- Reward 3: Penalize too-short or too-long responses ---
def length_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]['content'] if isinstance(completion, list) else completion
        words = len(text.split())
        if words < 20:
            rewards.append(-0.5)   # too short, probably no reasoning
        elif words > 500:
            rewards.append(-0.2)   # too long, padding
        else:
            rewards.append(0.0)
    return rewards
```

---

### Training Pipeline

```python
from trl import GRPOTrainer, GRPOConfig

# Continue from SFT checkpoint
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "/kaggle/working/sft_model")
model = model.merge_and_unload()   # merge LoRA weights before GRPO

# Fresh LoRA for GRPO stage
model = get_peft_model(model, lora_config)

grpo_config = GRPOConfig(
    output_dir="/kaggle/working/grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,      # GRPO generates multiple completions, memory heavy
    gradient_accumulation_steps=16,
    learning_rate=5e-6,                 # lower LR than SFT
    num_generations=4,                  # G=4 completions per prompt (G=8 needs more memory)
    max_prompt_length=256,
    max_completion_length=512,
    temperature=0.9,                    # needs diversity in generations
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=[
        correctness_reward,   # weight: highest signal
        format_reward,        # weight: structural compliance
        length_reward,        # weight: anti-padding
    ],
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("/kaggle/working/grpo_model")
```

---

### Evaluation Pipeline

```python
def evaluate_reasoning_model(model, tokenizer, dataset, n=100):
    results = {"correct": 0, "wrong": 0, "no_answer": 0}
    details = []

    for i in range(n):
        sample = dataset[i]
        prompt = tokenizer.apply_chat_template(
            sample['prompt'], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,          # deterministic for eval
                do_sample=False,
            )
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        predicted = extract_final_answer(response)
        ground_truth = sample['answer']
        
        if predicted is None:
            results['no_answer'] += 1
        elif predicted == ground_truth:
            results['correct'] += 1
        else:
            results['wrong'] += 1

        # Track think block presence
        has_think = "<think>" in response and "</think>" in response
        details.append({
            "question": sample['prompt'][-1]['content'][:80],
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": predicted == ground_truth,
            "has_think_block": has_think,
            "response_length": len(response.split()),
        })

    accuracy = results['correct'] / n
    think_rate = sum(d['has_think_block'] for d in details) / n
    
    print(f"Accuracy: {accuracy:.1%}  ({results['correct']}/{n})")
    print(f"No answer rate: {results['no_answer']/n:.1%}")
    print(f"Think block present: {think_rate:.1%}")
    return details

details = evaluate_reasoning_model(model_grpo, tokenizer, eval_dataset, n=100)
```

---

## Pipeline 3 — DPO (Alignment / Preference Tuning)

> **Prerequisite**: Use the SFT model as starting point (standard practice)

### Dataset Options

| Dataset | Size | Type | Notes |
|---|---|---|---|
| ✅ **UltraFeedback Binarized** | 62k | GPT-4 preference pairs | Most popular, broad coverage |
| ✅ **Argilla DPO Mix 7k** | 7k | Curated high quality | Small, good for Kaggle |
| ✅ **Intel Orca DPO Pairs** | 12.9k | Orca-style preference | Clean format |
| ✅ **HH-RLHF** (Anthropic) | 161k | Helpfulness + Harmlessness | Classic, well-studied |

**Recommended for first run**: `argilla/dpo-mix-7k`
- Small enough for Kaggle (7k examples)
- Already binarized (chosen/rejected)
- High quality curation

---

### Data Pipeline

```python
from datasets import load_dataset

dataset = load_dataset("argilla/dpo-mix-7k", split="train")
# Split manually since no default test split
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = splits["train"]
eval_dataset  = splits["test"]

print(train_dataset[0].keys())
# ['system', 'instruction', 'chosen', 'rejected']

# DPO requires this specific schema: prompt / chosen / rejected
def format_for_dpo(example):
    prompt_messages = []
    if example.get("system"):
        prompt_messages.append({"role": "system", "content": example["system"]})
    prompt_messages.append({"role": "user", "content": example["instruction"]})
    
    return {
        "prompt": tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        ),
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

train_dataset = train_dataset.map(format_for_dpo, remove_columns=train_dataset.column_names)
eval_dataset  = eval_dataset.map(format_for_dpo, remove_columns=eval_dataset.column_names)

# Verify
print(train_dataset[0])
# {
#   "prompt":   "<|im_start|>system...<|im_start|>user\n...<|im_start|>assistant\n",
#   "chosen":   "A helpful, detailed response...",
#   "rejected": "A short, unhelpful response.",
# }
```

---

### Training Pipeline

```python
from trl import DPOTrainer, DPOConfig

# Load SFT model as starting point
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "/kaggle/working/sft_model")
model = model.merge_and_unload()

# DPO also needs a frozen reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA on policy model only; ref model stays frozen
model = get_peft_model(model, lora_config)

dpo_config = DPOConfig(
    output_dir="/kaggle/working/dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    beta=0.1,                          # KL penalty — how far to deviate from ref
    max_length=1024,
    max_prompt_length=512,
    bf16=True,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("/kaggle/working/dpo_model")
```

---

### Evaluation Pipeline

```python
# DPO eval is about preference — does the model prefer chosen over rejected?
# Metric: implicit reward (log prob ratio) should be higher for chosen

def compute_implicit_reward(model, ref_model, tokenizer, sample):
    """
    DPO implicit reward: log π_θ(y|x) - log π_ref(y|x)
    Higher = model prefers this completion over reference
    """
    def get_log_prob(m, text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = m(**inputs, labels=inputs['input_ids'])
        return -out.loss.item()   # negative loss = log prob

    chosen_text   = sample['prompt'] + sample['chosen']
    rejected_text = sample['prompt'] + sample['rejected']

    model_chosen_lp   = get_log_prob(model, chosen_text)
    model_rejected_lp = get_log_prob(model, rejected_text)
    ref_chosen_lp     = get_log_prob(ref_model, chosen_text)
    ref_rejected_lp   = get_log_prob(ref_model, rejected_text)

    reward_chosen   = model_chosen_lp - ref_chosen_lp
    reward_rejected = model_rejected_lp - ref_rejected_lp
    return reward_chosen, reward_rejected

# Evaluate on 50 samples
n = 50
correct_preferences = 0
for i in range(n):
    r_chosen, r_rejected = compute_implicit_reward(
        model, ref_model, tokenizer, eval_dataset[i]
    )
    if r_chosen > r_rejected:
        correct_preferences += 1

preference_accuracy = correct_preferences / n
print(f"Preference accuracy: {preference_accuracy:.1%}")
# >50% means model learned to prefer chosen over rejected
# Good models reach 70-85% on held-out DPO data
```

---

## Connecting All Three Pipelines

```
Base Model (Qwen2.5-3B-Instruct)
    │
    ▼
[Pipeline 1: SFT]
Dataset: UltraChat 200k (10k subset)
Goal: Teach format + instruction following
Output: /kaggle/working/sft_model
    │
    ├──────────────────────────────────►  [Pipeline 3: DPO]
    │                                      Start from SFT model
    ▼                                      Dataset: argilla/dpo-mix-7k
[Pipeline 2: GRPO]                         Goal: Preference alignment
Start from SFT model                       Output: /kaggle/working/dpo_model
Dataset: GSM8K
Goal: Reasoning capability
Output: /kaggle/working/grpo_model
    │
    ▼
Best practice: DPO after GRPO
for a reasoning + aligned final model
```

---

## What to Watch During Each Run

| Pipeline | Key metric to watch | Warning sign |
|---|---|---|
| SFT | `eval/loss` going down | Loss stops decreasing after epoch 1 → reduce LR |
| SFT | Format adherence % | <50% at step 500 → check chat template |
| GRPO | `reward/correctness_reward` mean | Stuck at 0 → reward function bug |
| GRPO | `reward/format_reward` mean | Should rise before correctness rises |
| DPO | `rewards/chosen` > `rewards/rejected` | If reversed → beta too high |
| DPO | `logps/chosen` - `logps/rejected` margin | Should grow over training |

---

## Troubleshooting — Kaggle Specific

```python
# --- OOM on T4 ---
# 1. Reduce max_seq_length (1024 → 512)
# 2. Reduce per_device_train_batch_size (2 → 1)
# 3. Increase gradient_accumulation_steps to compensate
# 4. For GRPO: reduce num_generations (4 → 2)

# --- Two GPUs not being used ---
# Check: os.environ["CUDA_VISIBLE_DEVICES"] should not be set
# Use: accelerate with default config, or add this at top
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Slow training ---
# Add to SFTConfig: packing=True (biggest single speedup)
# Add liger kernel:
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
apply_liger_kernel_to_qwen2()  # call before model load

# --- HuggingFace token for gated models ---
from huggingface_hub import login
login(token="your_hf_token")  # add to Kaggle secrets, not hardcoded
```