# Pre-LN vs Post-LN — Gradient Flow

---

## The Residual Connection — Why It Matters

Both use:
```
output = x + Sublayer(x)
```

During backprop, gradient splits — one path through the Sublayer, one **identity path directly through x**. The identity path is what prevents vanishing gradients in deep networks.

---

## Post-LN (Original Transformer)

```
x → Sublayer → + → LayerNorm → output
        ↑_______|
```

Gradient path downward:

```
gradient from above
    ↓
LayerNorm        ← must pass through this
    ↓
(residual split)
    ↓
LayerNorm        ← and this at next layer
    ↓
...every layer...
```

**Problem:** Even the identity path merges before LayerNorm — so every gradient, including the shortcut, passes through LayerNorm at each layer. Across 80 layers, this repeated rescaling compounds and can shrink gradients reaching early layers.

---

## Pre-LN (Modern LLMs — GPT2, LLaMA)

```
x → LayerNorm → Sublayer → + → output
        ↑________________________|
```

Gradient path downward:

```
gradient from above
    ┌──────────────┐
    ↓              ↓
through LN+     identity path
Sublayer        (bypasses LN)
    └──────┬───────┘
           ↓
    ┌──────────────┐
    ↓              ↓
through LN+     identity path
Sublayer        (bypasses LN again)
    └──────┬───────┘
           ↓
         ...
```

The residual addition happens **after** the sublayer branch rejoins — so the identity path never touches LayerNorm. Gradient has a clean highway from output all the way to input regardless of depth.

---

## The Core Difference

| | Post-LN | Pre-LN |
|---|---|---|
| Gradient through LN | Every layer, no escape | Only through sublayer branch |
| Identity path clean? | No — merges before LN | Yes — bypasses LN entirely |
| Training stability | Fragile at large depth | More stable |
| Needs LR warmup | Usually yes | Often no |
| Used in | Original Transformer | GPT-2, LLaMA, most modern LLMs |

---

## One Line

Post-LN forces every gradient through LayerNorm at every layer — compounds across depth. Pre-LN keeps the residual path clean — gradient has a direct route that LayerNorm never touches.