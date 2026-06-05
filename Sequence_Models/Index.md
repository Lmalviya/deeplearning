# Sequence Models — Interview Preparation Notes

> These notes are built for interview depth, not textbook coverage.  
> Every lesson follows the flow: **Problem → Concept → Diagram → Example → Trade-offs → Interview Questions → Summary.**  
> Read in order. Each lesson builds on the previous one.

---

## Part 1 — Recurrent Neural Networks (RNN)

| Lesson | Topic |
|---|---|
| [Lesson 1.1](1_RNN/Lesson_1_1.md) | RNN Architecture: The Recurrence Equation & Hidden State |
| [Lesson 1.2](1_RNN/Lesson_1_2.md) | Training RNNs: Backpropagation Through Time (BPTT) |
| [Lesson 1.3](1_RNN/Lesson_1_3.md) | The Vanishing & Exploding Gradient Problem |
| [Lesson 1.4](1_RNN/Lesson_1_4.md) | Bidirectional RNN (Bi-RNN) |
| [Lesson 1.5](1_RNN/Lesson_1_5.md) | RNN Pros, Cons, and Real Limitations |

## Part 2 — Long Short-Term Memory (LSTM)

| Lesson | Topic |
|---|---|
| [Lesson 2.1](2_LSTM/Lesson_2_1.md) | The LSTM Insight: Why a Separate Cell State? |
| [Lesson 2.2](2_LSTM/Lesson_2_2.md) | LSTM Gates in Full Detail |
| [Lesson 2.3](2_LSTM/Lesson_2_3.md) | How LSTM Solves the Vanishing Gradient Problem |
| [Lesson 2.4](2_LSTM/Lesson_2_4.md) | Bidirectional LSTM & Stacked LSTM |
| [Lesson 2.5](2_LSTM/Lesson_2_5.md) | LSTM Pros, Cons, and Limitations |

## Part 3 — Gated Recurrent Unit (GRU)

| Lesson | Topic |
|---|---|
| [Lesson 3.1](3_GRU/Lesson_3_1.md) | GRU Architecture: A Streamlined LSTM |
| [Lesson 3.2](3_GRU/Lesson_3_2.md) | GRU Gates: Reset & Update |
| [Lesson 3.3](3_GRU/Lesson_3_3.md) | GRU vs LSTM: The Real Trade-off |

## Part 4 — Attention on RNNs (The Bridge)

| Lesson | Topic |
|---|---|
| [Lesson 4.1](4_Attention_on_RNNs/Lesson_4_1.md) | The Fixed-Size Bottleneck: What LSTM Cannot Do |
| [Lesson 4.2](4_Attention_on_RNNs/Lesson_4_2.md) | Bahdanau Attention: How Attention Fixes the Bottleneck |
| [Lesson 4.3](4_Attention_on_RNNs/Lesson_4_3.md) | From RNN+Attention to Transformers: The Natural Progression |

## Part 5 — Why Transformers Replaced RNN/LSTM/GRU

| Lesson | Topic |
|---|---|
| [Lesson 5.1](5_Why_Transformers_Won/Lesson_5_1.md) | The Sequential Computation Problem |
| [Lesson 5.2](5_Why_Transformers_Won/Lesson_5_2.md) | What Transformers Do Differently (Brief Bridge) |
| [Lesson 5.3](5_Why_Transformers_Won/Lesson_5_3.md) | When Would You Still Choose RNN/LSTM Today? |

---

*Future chapters will cover Transformer internals, BERT, GPT, and beyond.*
