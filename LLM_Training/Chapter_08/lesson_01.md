# Chapter 8 · Lesson 1 — LoRA-Specific Hyperparameters: Rank, Alpha, Dropout, Target Modules

> **Where this fits:** Chapter 7, Lesson 4 built LoRA's math and code. This lesson goes deep on actually tuning it — concrete ranges, how the hyperparameters interact with each other, and the reasoning behind common conventions rather than just stating them.

---

## 1. Rank — Revisiting With Concrete Ranges

Chapter 7, Lesson 4 established rank controls the capacity/cost tradeoff mathematically. Concrete starting ranges, tied to Chapter 5's diagnosed scope of change (directly reusing Chapter 7, Lesson 9's scope-of-change branch point):

| Task scope | Typical rank range | Reasoning |
|---|---|---|
| Very narrow adjustment (single skill, e.g. a specific output format) | 4-8 | Minimal capacity needed; lower rank also means less catastrophic forgetting risk (Chapter 7, Lesson 2's factor list) |
| Moderate capability gap (e.g. Chapter 5's tool-use or reasoning gaps) | 16-32 | The common default range for most production LoRA fine-tunes |
| Broad behavioral/style shift, approaching full-fine-tuning territory | 64-128+ | Diminishing returns typically appear well before matching full fine-tuning's expressiveness — worth validating empirically rather than assuming higher is always better |

**The diminishing-returns point, worth stating precisely:** doubling rank does not typically double the resulting capability improvement — published ablations commonly show performance plateauing well before very high ranks, meaning the "just increase rank" instinct for a struggling fine-tune often isn't the right lever (Chapter 7, Lesson 8's triage flowchart — checking data quality and task scope first is usually more productive than a blind rank increase).

---

## 2. Alpha — Tuning It as Genuinely Separate From Rank

Chapter 7, Lesson 4, Section 4 established alpha scales the update's effective magnitude independent of rank. **The practical tuning question: does alpha need retuning every time rank changes?** Given the `scaling = alpha / rank` relationship, keeping the alpha-to-rank *ratio* fixed (e.g., always setting `alpha = 2 * rank`) approximately preserves the effective update magnitude as rank varies — this is why the common convention isn't a single fixed alpha value, but a fixed *ratio*, worth understanding as the reason behind the convention rather than memorizing it as an arbitrary rule.

**When to deviate from the fixed-ratio convention:** if a rank sweep shows instability (loss spikes) or a limited effect on outcomes across a wide rank range, adjusting the ratio itself (not just rank) is a legitimate independent axis to explore — a higher alpha-to-rank ratio pushes the adapter to have a stronger effect on the frozen model's behavior per unit of capacity, useful when a stronger nudge is needed without expanding rank (and its parameter/forgetting-risk cost) further.

---

## 3. Dropout — A Smaller but Real Lever

LoRA layers commonly include a dropout applied to the input before the `A` projection (Chapter 7, Lesson 4's `LoRALinear` code can be extended with this), serving the same general regularization purpose as dropout elsewhere in deep learning — randomly zeroing some input elements during training to reduce reliance on any single feature and mitigate overfitting.

**Why this matters more for LoRA fine-tuning than it might in some other contexts:** directly connecting to Chapter 7, Lesson 8's overfitting discussion — small fine-tuning datasets are already at elevated overfitting risk, and LoRA dropout is one of the cheaper, more direct levers available specifically for that risk, alongside epoch count and data diversity (Chapter 7, Lesson 8, Section 4's mitigation list). Typical values are modest (commonly in the 0.0-0.1 range) — LoRA's already-constrained parameter count (versus full fine-tuning) means aggressive dropout is less often necessary than it might be for a much higher-capacity training setup.

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank, alpha, dropout=0.05):
        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        d_out, d_in = base_linear.weight.shape
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * (1 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x):
        base_output = self.base_linear(x)
        lora_output = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_output + self.scaling * lora_output
```

---

## 4. Target Modules — Revisiting the Breadth-vs-Depth Tradeoff

Chapter 7, Lesson 4, Section 6 introduced the tradeoff between concentrating a higher rank on fewer module types versus spreading a lower rank across more module types. Concrete guidance for tuning this choice:

**Start narrow (attention projections only — `q_proj`, `v_proj` at minimum, often `k_proj`, `o_proj` too), then expand if the diagnosed gap (Chapter 5) suggests it's warranted:**
- **Tool-use, structured-output, instruction-following gaps** (Chapter 5, Lessons 3, 4, 6) — these are largely about *how the model attends to and processes* its input/instructions, making attention-matrix targeting a reasonable primary lever.
- **Reasoning, code-generation gaps** (Chapter 5, Lessons 5, 7) — these often benefit from also including the feed-forward layers, since Chapter 3, Lesson 1 established the FFN sub-layer is where a substantial fraction of a transformer's per-token "processing capacity" lives, and reasoning/code correctness plausibly depends more on this processing capacity than pure attention pattern changes.

**Worked example of the breadth/depth interaction:** for a fixed total trainable-parameter budget, targeting only attention projections at rank 32 versus targeting attention + FFN at rank 16 spends the same parameter budget differently — the second spreads a lower per-module rank across more of the network's total computation, and (per Chapter 7, Lesson 4, Section 6's cited empirical finding) this broader-but-shallower allocation sometimes outperforms concentrating capacity narrowly, worth testing rather than assuming either is automatically better for a specific task.

---

## 5. Worked Example: A Full LoRA Hyperparameter Starting Configuration

Applying Sections 1-4 together for a moderate-scope tool-use fine-tune (Chapter 7, Lesson 9's worked example, revisited with actual numbers now):

```python
lora_config = {
    "rank": 16,                                    # Section 1 — moderate scope
    "alpha": 32,                                    # Section 2 — fixed 2x ratio convention
    "dropout": 0.05,                                 # Section 3 — modest, given already-constrained capacity
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Section 4 — attention-focused,
                                                                    # given this is an instruction/tool-use-pattern gap
}
```

**This configuration is explicitly a starting point, not a final answer** — Lesson 4 of this chapter (tuning methods for small vs. large-scale fine-tunes) covers how to actually validate and refine these values rather than treating them as fixed once chosen.

---

## Key Takeaways

- Rank has concrete, scope-dependent starting ranges, with clearly diminishing returns at higher values — a blind rank increase is rarely the right fix for a struggling fine-tune.
- Alpha is conventionally tuned as a fixed ratio to rank, not an independent fixed value, because the `alpha/rank` scaling relationship makes the ratio the more meaningful quantity to preserve.
- LoRA dropout is a real, cheap overfitting lever, particularly relevant given fine-tuning's smaller dataset sizes (Chapter 7, Lesson 8).
- Target module choice should be informed by which part of the network (attention vs. FFN) is most plausibly responsible for the diagnosed Chapter 5 capability gap, not chosen by default.

---

## Self-Check Before Moving to Lesson 2

1. Explain why alpha is typically tuned as a ratio to rank rather than as an independent fixed value.
2. For a diagnosed reasoning-capability gap (Chapter 5, Lesson 5), would you lean toward attention-only or attention+FFN target modules, and why?
3. Why might increasing rank fail to meaningfully improve a struggling fine-tune, and what would you check instead per Chapter 7, Lesson 8?