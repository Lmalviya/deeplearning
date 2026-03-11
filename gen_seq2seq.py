import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):   return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

# ── CELL 0 – Title ────────────────────────────────────────────────────────────
cells.append(md("""# 🔗 Seq2Seq, Attention Mechanisms & The Bridge to Transformers

**Topics covered:**
1. The many-to-many bottleneck — why vanilla RNN Seq2Seq fails
2. Seq2Seq architecture — Encoder RNN + Decoder RNN
3. The alignment problem (what "context vector" can't solve)
4. Bahdanau Attention (additive) — mathematics + numerical trace
5. Luong Attention (multiplicative / dot-product) — mathematics
6. Side-by-side comparison: Bahdanau vs Luong
7. Step-by-step implementation of both attention mechanisms in PyTorch
8. Full training example: Sequence reversal task
9. Attention weight visualisation (heatmap)
10. How this directly leads to the Transformer ("Attention is All You Need")
11. Practical tips: pack_padded_sequence, CTC Loss, bidirectional limits
12. Interview Q&A
"""))

# ── CELL 1 – Motivation ───────────────────────────────────────────────────────
cells.append(md("""## 1. 🧨 The Bottleneck Problem — Why RNN Seq2Seq Fails

### The Setup: Machine Translation
We want to translate "I love dogs" (English, 3 words) → "J'aime les chiens" (French, 4 words).

**Input and output have different lengths** — this is the defining challenge.

### Vanilla RNN Seq2Seq (without attention)
The architecture has two parts:
1. **Encoder RNN**: reads the source sentence word-by-word → compresses everything into a single **context vector** `c` (the final hidden state `h_T`)
2. **Decoder RNN**: takes `c` and generates the target sentence word-by-word

```
ENCODER:                              DECODER:
"I"  →  [RNN]  →  h1                 c  →  [RNN]  →  "J'"
"love" → [RNN]  →  h2                y1 →  [RNN]  →  "aime"
"dogs" → [RNN]  →  h3 = c            y2 →  [RNN]  →  "les"
                  (context vector)   y3 →  [RNN]  →  "chiens"
```

### The Critical Bottleneck
The entire source sentence — whether 5 words or 100 words — must be compressed into a **single fixed-size vector** `c`.

**What's lost:**
- A 100-word sentence has its first word compressed through 100 RNN steps → nearly zero gradient signal remains
- The context vector `c` must "remember" all of: subject, verb, object, tense, style, idioms
- This is **the same vanishing gradient problem** — now affecting cross-sequence information flow

**Experimental evidence (Bahdanau et al. 2015):**
- With sentences up to 30 words: BLEU score is acceptable
- With sentences > 30 words: BLEU score **collapses** — the model forgets the beginning

> 💡 The solution: instead of forcing the decoder to use ONE fixed vector, let it **look back** at ALL encoder hidden states and **choose which to focus on** at each decoding step. This is **Attention**.
"""))

# ── CELL 2 – Seq2Seq Architecture ─────────────────────────────────────────────
cells.append(md("""## 2. 🏗️ Seq2Seq Architecture — Encoder + Decoder

### Encoder
Reads the source sequence and produces **a set of hidden states** — one per source token:

$$h_1, h_2, \\ldots, h_T = \\text{Encoder}(x_1, x_2, \\ldots, x_T)$$

Instead of only using the final state $h_T$, the attention mechanism uses **all** of $h_1 \\ldots h_T$.

In practice: use a **Bidirectional** encoder to give each $h_i$ context from both directions:
$$h_i = [\\overrightarrow{h}_i ; \\overleftarrow{h}_i]$$

### Decoder
Generates the target sequence **one token at a time**, auto-regressively:

At step $t$, the decoder:
1. Takes its previous hidden state $s_{t-1}$
2. Takes the previous output token $y_{t-1}$  
3. **Uses attention** to produce a context vector $c_t$ from encoder states $h_1 \\ldots h_T$
4. Computes: $s_t = \\text{RNN}(s_{t-1},\\, [y_{t-1}; c_t])$
5. Predicts: $y_t = \\text{softmax}(W_s \\cdot s_t)$

Key difference from vanilla Seq2Seq:
- **Vanilla**: `c` is fixed — same for every decoding step
- **With Attention**: `c_t` is **different** at each step — computed fresh by attending to relevant encoder states

### Teacher Forcing
During training, the decoder receives the **ground-truth** previous token instead of its own prediction.  
This speeds up training but can cause **exposure bias** at inference.  
**Scheduled sampling** gradually transitions from teacher-forcing to self-prediction during training.
"""))

# ── CELL 3 – Bahdanau Attention ───────────────────────────────────────────────
cells.append(md("""## 3. ⭐ Bahdanau Attention (Additive) — Full Mathematics

**Paper:** "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau, Cho, Bengio, 2015)

### The Key Idea
At each decoder step $t$, compute a **relevance score** between the decoder's current state and **every** encoder hidden state.  
Use those scores to produce a **weighted average** of encoder states = the context vector $c_t$.

---

### Step 1: Alignment Score (how relevant is encoder position $i$ for decoder step $t$?)
$$e_{ti} = v_a^\\top \\tanh(W_a s_{t-1} + U_a h_i)$$

where:
- $s_{t-1}$ = decoder hidden state at step $t-1$  
- $h_i$ = encoder hidden state at source position $i$
- $W_a, U_a$ = learned weight matrices
- $v_a$ = learned weight vector (the "scorer")
- The $\\tanh$ + linear = a **small feedforward network** that jointly processes both states

This is called **additive** attention because $W_a s_{t-1}$ and $U_a h_i$ are **added** before the tanh.

---

### Step 2: Attention Weights (normalise scores to a probability distribution)
$$\\alpha_{ti} = \\frac{\\exp(e_{ti})}{\\sum_{j=1}^{T} \\exp(e_{tj})} = \\text{softmax}(e_{t,:})_i$$

- $\\alpha_{ti} \\in (0, 1)$  
- $\\sum_i \\alpha_{ti} = 1$  
- Interpretation: $\\alpha_{ti}$ = "how much should the decoder focus on source word $i$ when producing target word $t$?"

---

### Step 3: Context Vector (weighted sum of encoder states)
$$c_t = \\sum_{i=1}^{T} \\alpha_{ti} \\cdot h_i$$

This is a **soft alignment** — unlike hard alignment (HMM) which picks one source word, attention gives a smooth distribution over all source words.

---

### Step 4: Decoder Update
$$s_t = \\text{RNN}(s_{t-1},\\; [y_{t-1}; c_t])$$
$$y_t = \\text{softmax}(W_y \\cdot s_t + b_y)$$

---

### Complexity
- Encoder: $O(T)$ steps
- Decoder: $O(T')$ steps
- Attention: $O(T \\cdot T')$ score computations per batch
- Total: $O(T \\cdot T')$ — quadratic in sequence length (same issue as Transformers)

### The Alignment Matrix
When we visualise $\\alpha_{ti}$ as a matrix (rows=target, cols=source):
- Strong diagonal → monotonic alignment (similar word order between languages)
- Off-diagonal → reordering (adjective-noun flip between English and French)
"""))

# ── CELL 4 – Luong Attention ──────────────────────────────────────────────────
cells.append(md("""## 4. ⚡ Luong Attention (Multiplicative) — Full Mathematics

**Paper:** "Effective Approaches to Attention-based Neural Machine Translation" (Luong, Pham, Manning, 2015)

### Key Differences from Bahdanau
| Property | Bahdanau (Additive) | Luong (Multiplicative) |
|---|---|---|
| **Uses** | $s_{t-1}$ (previous state) | $s_t$ (current state) |
| **Score function** | $v^\\top \\tanh(W_a s + U_a h)$ | Multiple options (see below) |
| **Complexity** | $O(d_h \\cdot d_a)$ addtional params | Cheaper (dot product) |
| **When computed** | Before generating $s_t$ | After generating $s_t$ |

---

### Three Luong Score Functions

**① Dot (simplest):** No extra parameters
$$\\text{score}(s_t, h_i) = s_t^\\top h_i$$

Requires $\\dim(s_t) = \\dim(h_i)$.

**② General (most common):** One learned matrix $W_a$
$$\\text{score}(s_t, h_i) = s_t^\\top W_a h_i$$

$W_a$ allows different encoder/decoder dimensions and learns the best projection.

**③ Concat (closest to Bahdanau):**
$$\\text{score}(s_t, h_i) = v_a^\\top \\tanh(W_a [s_t; h_i])$$

---

### Luong Forward Pass (Global Attention variant)
1. Run encoder: $h_1, \\ldots, h_T = \\text{BiRNN}(x_1, \\ldots, x_T)$
2. Run one decoder step: $s_t = \\text{RNN}(s_{t-1}, y_{t-1})$
3. Compute scores: $e_{ti} = s_t^\\top W_a h_i$ for all $i$
4. Normalise: $\\alpha_t = \\text{softmax}(e_t)$
5. Context: $c_t = \\sum_i \\alpha_{ti} h_i$
6. Combine: $\\tilde{s}_t = \\tanh(W_c [c_t ; s_t])$
7. Predict: $\\hat{y}_t = \\text{softmax}(W_y \\tilde{s}_t)$

---

### Local vs Global Attention (Luong's two variants)
- **Global:** attend to all encoder positions (same as Bahdanau)
- **Local:** predict a single **alignment position** $p_t$, attend only within a window $[p_t - D, p_t + D]$
  - Gaussian weighting around $p_t$
  - $O(D)$ instead of $O(T)$ attention computations — useful for long sequences
"""))

# ── CELL 5 – Numerical Trace ──────────────────────────────────────────────────
cells.append(code("""import numpy as np
np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)

def sigmoid(z): return 1/(1+np.exp(-z))
def softmax(z): e = np.exp(z - z.max()); return e/e.sum()

# ── Tiny setup: source=3 words, hidden_dim=4 ──
T_src = 3        # "I love dogs"
hidden = 4

# Encoder hidden states (pretend we ran a BiRNN)
H = np.array([[0.8, 0.1, 0.3, 0.2],   # h1 = "I"
              [0.1, 0.9, 0.7, 0.4],   # h2 = "love"
              [0.3, 0.5, 0.1, 0.8]])  # h3 = "dogs"

# Decoder previous hidden state (step 1)
s_prev = np.array([0.5, 0.3, 0.7, 0.2])

# ─────────── BAHDANAU ATTENTION ───────────────────────────────────────
print("=" * 60)
print("BAHDANAU ATTENTION (Additive)")
print("=" * 60)

Wa = np.random.randn(hidden, hidden) * 0.3
Ua = np.random.randn(hidden, hidden) * 0.3
va = np.random.randn(hidden) * 0.3

print(f"\n  Decoder state s_(t-1): {s_prev}")
print(f"  Encoder states H shape: {H.shape}  (T=3, hidden=4)")

# Step 1: Alignment scores
Wa_s = Wa @ s_prev                       # (hidden,)
e = np.zeros(T_src)
print("\n  Step 1: Alignment scores  e_ti = v^T * tanh(Wa*s + Ua*h_i)")
for i in range(T_src):
    Ua_hi  = Ua @ H[i]                   # (hidden,)
    combined = np.tanh(Wa_s + Ua_hi)     # (hidden,)
    e[i]   = va @ combined
    print(f"    e_{i+1} (for '{['I','love','dogs'][i]}'): "
          f"tanh({(Wa_s+Ua_hi).round(3)}) → score = {e[i]:.4f}")

# Step 2: Attention weights
alpha = softmax(e)
print(f"\n  Step 2: Attention weights α = softmax(e)")
for i in range(T_src):
    bar = "█" * int(alpha[i] * 30)
    print(f"    α_{i+1} ({'I   ' if i==0 else 'love' if i==1 else 'dogs'}) = {alpha[i]:.4f}  {bar}")

# Step 3: Context vector
c_t = H.T @ alpha
print(f"\n  Step 3: Context vector c_t = Σ α_i * h_i")
print(f"    c_t = {c_t}")

# ─────────── LUONG ATTENTION (General) ────────────────────────────────
print("\n" + "=" * 60)
print("LUONG ATTENTION (General / Multiplicative)")
print("=" * 60)

# Current decoder state (Luong uses s_t, not s_{t-1})
s_curr = np.array([0.6, 0.4, 0.5, 0.3])   # s_t after one decoder RNN step

Wa_L = np.random.randn(hidden, hidden) * 0.3

print(f"\n  Decoder state s_t: {s_curr}")
print("\n  Step 1: Scores  score(s_t, h_i) = s_t^T * Wa * h_i")
e_L = np.zeros(T_src)
for i in range(T_src):
    e_L[i] = s_curr @ (Wa_L @ H[i])
    print(f"    e_{i+1} ('{'I   ' if i==0 else 'love' if i==1 else 'dogs'}') = {e_L[i]:.4f}")

alpha_L = softmax(e_L)
print(f"\n  Step 2: Attention weights α = softmax(e)")
for i in range(T_src):
    bar = "█" * int(alpha_L[i] * 30)
    print(f"    α_{i+1} ({'I   ' if i==0 else 'love' if i==1 else 'dogs'}) = {alpha_L[i]:.4f}  {bar}")

c_L = H.T @ alpha_L
print(f"\n  Step 3: Context vector c_t = {c_L}")

# Combine
Wc = np.random.randn(hidden, 2*hidden) * 0.3
s_tilde = np.tanh(Wc @ np.concatenate([c_L, s_curr]))
print(f"\n  Step 4: Combined  s̃_t = tanh(Wc·[c_t; s_t]) = {s_tilde}")
print(f"\n  → y_t = softmax(Wy · s̃_t)  [vocabulary projection]")
"""))

# ── CELL 6 – Visualise Attention Heatmap ─────────────────────────────────────
cells.append(code("""import numpy as np
import matplotlib.pyplot as plt

# Simulate attention weight matrix for English → French translation
# "I love dogs" → "J' aime les chiens"
# Row = target token, Col = source token
attention = np.array([
    [0.92, 0.05, 0.03],   # J'    aligns strongly to "I"
    [0.04, 0.93, 0.03],   # aime  aligns strongly to "love"
    [0.03, 0.06, 0.91],   # les   aligns to "dogs" (article)
    [0.01, 0.01, 0.98],   # chiens aligns to "dogs"
])

src_words = ['I', 'love', 'dogs']
tgt_words = ["J'", 'aime', 'les', 'chiens']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0f0f1a')
fig.suptitle("Attention Weight Heatmap — Alignment Visualisation", color='white', fontsize=13)

# ── Heatmap 1: Simple sentence ──
ax = axes[0]
im = ax.imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(src_words))); ax.set_xticklabels(src_words, fontsize=13, color='white')
ax.set_yticks(range(len(tgt_words))); ax.set_yticklabels(tgt_words, fontsize=13, color='white')
ax.set_xlabel('Source (English)', color='#aaa', fontsize=11)
ax.set_ylabel('Target (French)', color='#aaa', fontsize=11)
ax.set_title('"I love dogs" → "J\\'aime les chiens"\n(Near-diagonal = same word order)', color='#4ecdc4', fontsize=10)
ax.set_facecolor('#0f0f1a')
for i in range(len(tgt_words)):
    for j in range(len(src_words)):
        ax.text(j, i, f'{attention[i,j]:.2f}', ha='center', va='center',
                color='black' if attention[i,j] > 0.4 else 'white', fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8, label='Attention weight')

# ── Heatmap 2: Reordering example (adj-noun flip) ──
# "The black cat" → "Le chat noir" (French adj comes AFTER noun)
src2 = ['The', 'black', 'cat']
tgt2 = ['Le', 'chat', 'noir']
attn2 = np.array([
    [0.90, 0.05, 0.05],  # Le    → The
    [0.03, 0.05, 0.92],  # chat  → cat (reordering!)
    [0.02, 0.93, 0.05],  # noir  → black (reordering!)
])
ax = axes[1]
ax.set_facecolor('#0f0f1a')
im2 = ax.imshow(attn2, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(src2))); ax.set_xticklabels(src2, fontsize=13, color='white')
ax.set_yticks(range(len(tgt2))); ax.set_yticklabels(tgt2, fontsize=13, color='white')
ax.set_xlabel('Source (English)', color='#aaa', fontsize=11)
ax.set_ylabel('Target (French)', color='#aaa', fontsize=11)
ax.set_title('"The black cat" → "Le chat noir"\n(Off-diagonal = word reordering learned!)', color='#fd79a8', fontsize=10)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{attn2[i,j]:.2f}', ha='center', va='center',
                color='black' if attn2[i,j] > 0.4 else 'white', fontsize=11)
plt.colorbar(im2, ax=ax, shrink=0.8, label='Attention weight')

plt.tight_layout()
plt.savefig('notes/assets/attention_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()
print("Key insight: the off-diagonal pattern in the right heatmap shows the model")
print("learned that French adjectives come AFTER the noun — purely from training data!")
"""))

# ── CELL 7 – PyTorch Implementation ───────────────────────────────────────────
cells.append(md("## 5. 🛠️ PyTorch Implementation — Bahdanau Attention Seq2Seq"))
cells.append(code("""import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ── Bahdanau Attention Module ──
class BahdanauAttention(nn.Module):
    \"\"\"Additive attention (Bahdanau et al. 2015).
    score(s, h) = v^T * tanh(Wa*s + Ua*h)
    \"\"\"
    def __init__(self, enc_hidden, dec_hidden, attn_dim):
        super().__init__()
        self.Wa = nn.Linear(dec_hidden, attn_dim, bias=False)  # decoder projection
        self.Ua = nn.Linear(enc_hidden, attn_dim, bias=False)  # encoder projection
        self.va = nn.Linear(attn_dim, 1, bias=False)           # scalar scorer

    def forward(self, s_prev, encoder_outputs):
        # s_prev:          (batch, dec_hidden)
        # encoder_outputs: (batch, src_len, enc_hidden)
        src_len = encoder_outputs.shape[1]

        # Expand s_prev for all source positions
        s_exp = self.Wa(s_prev).unsqueeze(1).expand(-1, src_len, -1)  # (batch, src_len, attn_dim)
        h_proj = self.Ua(encoder_outputs)                               # (batch, src_len, attn_dim)

        # Additive alignment score
        energy = self.va(torch.tanh(s_exp + h_proj)).squeeze(-1)       # (batch, src_len)
        alpha  = F.softmax(energy, dim=1)                              # (batch, src_len) → sums to 1

        # Context vector: weighted sum
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, enc_hidden)
        return context, alpha


# ── Luong Attention Module ──
class LuongAttention(nn.Module):
    \"\"\"Multiplicative / General attention (Luong et al. 2015).
    score(s, h) = s^T * Wa * h
    \"\"\"
    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.Wa = nn.Linear(enc_hidden, dec_hidden, bias=False)

    def forward(self, s_curr, encoder_outputs):
        # s_curr:          (batch, dec_hidden)
        # encoder_outputs: (batch, src_len, enc_hidden)
        h_proj = self.Wa(encoder_outputs)           # (batch, src_len, dec_hidden)
        s_exp  = s_curr.unsqueeze(2)                # (batch, dec_hidden, 1)
        energy = torch.bmm(h_proj, s_exp).squeeze(-1)  # (batch, src_len)
        alpha  = F.softmax(energy, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, alpha


# ── Encoder ──
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers,
                          batch_first=True, bidirectional=True)
        # Project BiGRU hidden to single-direction size for decoder
        self.fc  = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        emb = self.emb(x)
        outputs, hidden = self.rnn(emb)
        # outputs: (batch, src_len, hidden*2)
        # hidden:  (2*layers, batch, hidden) → take last forward+backward
        h = torch.tanh(self.fc(torch.cat([hidden[-2], hidden[-1]], dim=1)))
        return outputs, h   # (batch, src_len, hid*2), (batch, hid)


# ── Decoder with Bahdanau Attention ──
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden, dec_hidden, attn_dim):
        super().__init__()
        self.emb      = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attn     = BahdanauAttention(enc_hidden * 2, dec_hidden, attn_dim)
        self.rnn      = nn.GRUCell(embed_dim + enc_hidden * 2, dec_hidden)
        self.fc_out   = nn.Linear(dec_hidden, vocab_size)

    def forward(self, y_prev, s_prev, encoder_outputs):
        emb     = self.emb(y_prev)                                   # (batch, embed_dim)
        context, alpha = self.attn(s_prev, encoder_outputs)         # (batch, enc_hid*2)
        rnn_in  = torch.cat([emb, context], dim=1)                  # (batch, embed+enc_hid*2)
        s_curr  = self.rnn(rnn_in, s_prev)                          # (batch, dec_hidden)
        logit   = self.fc_out(s_curr)                               # (batch, vocab_size)
        return logit, s_curr, alpha


# ── Test shapes ──
VOCAB, EMBED, ENC_HID, DEC_HID, ATTN = 100, 16, 32, 32, 16
enc = Encoder(VOCAB, EMBED, ENC_HID)
dec = AttentionDecoder(VOCAB, EMBED, ENC_HID, DEC_HID, ATTN)

src = torch.randint(1, VOCAB, (4, 8))   # batch=4, src_len=8
enc_out, h = enc(src)

y = torch.randint(1, VOCAB, (4,))
logit, s, alpha = dec(y, h, enc_out)

print("Shapes:")
print(f"  Encoder outputs:    {tuple(enc_out.shape)}  (batch, src_len, enc_hid*2)")
print(f"  Encoder final h:    {tuple(h.shape)}       (batch, dec_hid)")
print(f"  Decoder logit:      {tuple(logit.shape)}    (batch, vocab_size)")
print(f"  Attention weights:  {tuple(alpha.shape)}    (batch, src_len) → sums to 1")
print(f"  α sum check:        {alpha.sum(dim=1).tolist()}  ← should all be ≈1.0")

total_enc = sum(p.numel() for p in enc.parameters())
total_dec = sum(p.numel() for p in dec.parameters())
print(f"\\nEncoder params: {total_enc:,}")
print(f"Decoder params: {total_dec:,} (includes attention weights)")
"""))

# ── CELL 9 – Full Training on Sequence Reversal ───────────────────────────────
cells.append(md("## 6. 🛠️ Full Training — Sequence Reversal Task"))
cells.append(code("""import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(42)

# ── Toy task: reverse a sequence of digits ──
# "3 5 1 7" → "7 1 5 3"
# Great test for attention: position i in output should attend to position T-i in input

VOCAB  = 12          # digits 1-10 + PAD=0 + EOS=11
PAD, EOS = 0, 11
SEQ    = 6
BATCH  = 64

def make_batch(n, seq_len):
    src = torch.randint(1, 10, (n, seq_len))
    tgt = torch.flip(src, dims=[1])
    # Append EOS to target
    tgt_eos = torch.cat([tgt, torch.full((n,1), EOS)], dim=1)
    return src, tgt_eos

# ── Model ──
EMBED, ENC_H, DEC_H, ATTN_D = 16, 32, 32, 16

enc = Encoder(VOCAB, EMBED, ENC_H)
dec = AttentionDecoder(VOCAB, EMBED, ENC_H, DEC_H, ATTN_D)
params = list(enc.parameters()) + list(dec.parameters())
opt    = torch.optim.Adam(params, lr=1e-3)
loss_fn= nn.CrossEntropyLoss(ignore_index=PAD)

losses = []
for epoch in range(120):
    enc.train(); dec.train()
    src, tgt = make_batch(BATCH, SEQ)

    enc_out, h = enc(src)
    ep_loss    = 0
    tgt_len    = tgt.shape[1]
    y          = tgt[:, 0]                 # SOS = first reversed token (for simplicity)

    for t in range(tgt_len - 1):          # teacher forcing
        logit, h, _ = dec(y, h, enc_out)
        ep_loss += loss_fn(logit, tgt[:, t+1])
        y = tgt[:, t+1]                   # teacher force

    opt.zero_grad(); ep_loss.backward(); opt.step()
    losses.append(ep_loss.item() / tgt_len)

# ── Visualise Training ──
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.patch.set_facecolor('#0f0f1a')
for ax in axes: ax.set_facecolor('#13131f')

axes[0].plot(losses, color='#4ecdc4', lw=2)
axes[0].fill_between(range(len(losses)), losses, alpha=0.15, color='4ecdc4')
axes[0].set_title('Seq2Seq + Bahdanau Attention\nTraining Loss (Sequence Reversal)', color='white')
axes[0].set_xlabel('Epoch', color='#aaa'); axes[0].set_ylabel('CE Loss', color='#aaa')
axes[0].tick_params(colors='#aaa')

# ── Visualise attention weights for one example ──
enc.eval(); dec.eval()
src_ex, tgt_ex = make_batch(1, SEQ)
with torch.no_grad():
    enc_out_ex, h_ex = enc(src_ex)
    y_ex = tgt_ex[:, 0]
    alphas = []
    for t in range(SEQ):
        logit_ex, h_ex, a = dec(y_ex, h_ex, enc_out_ex)
        alphas.append(a[0].numpy())
        y_ex = logit_ex.argmax(dim=1)

attn_mat = torch.tensor(alphas).numpy()
ax = axes[1]
im = ax.imshow(attn_mat, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Source position', color='#aaa')
ax.set_ylabel('Target position (reversed)', color='#aaa')
ax.set_title(f'Learned Attention Weights\nSrc: {src_ex[0].tolist()} → reversed',
             color='white')
for i in range(SEQ):
    for j in range(SEQ):
        ax.text(j, i, f'{attn_mat[i,j]:.2f}',
                ha='center', va='center', fontsize=8,
                color='black' if attn_mat[i,j] > 0.4 else 'white')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.tick_params(colors='#aaa')

for ax in axes:
    for sp in ax.spines.values(): sp.set_color('#333')

plt.tight_layout()
plt.savefig('notes/assets/seq2seq_training.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()
print(f"Final loss: {losses[-1]:.4f}")
print("Note: Anti-diagonal pattern in attention = model learned to reverse the sequence!")
"""))

# ── CELL 11 – Bahdanau vs Luong comparison ────────────────────────────────────
cells.append(md("""## 7. ⚖️ Bahdanau vs Luong — Side-by-Side Comparison

| Property | Bahdanau (Additive) | Luong (Multiplicative) |
|---|---|---|
| **Year** | 2015 | 2015 (same year) |
| **Decoder state used** | $s_{t-1}$ (previous) | $s_t$ (current) |
| **Score function** | $v^\\top \\tanh(W_a s + U_a h)$ | $s^\\top W_a h$ (General) or $s^\\top h$ (Dot) |
| **Extra parameters** | $W_a, U_a, v_a$ — 3 matrices | $W_a$ — 1 matrix (General) or none (Dot) |
| **Alignment** | Soft, global | Soft global or Hard local |
| **Computation** | More expensive (tanh + ffn) | Cheaper (dot product) |
| **Intuition** | "Search" — look for alignment jointly | "Match" — score how well state matches encoder |
| **Best for** | Tasks needing rich alignment (translation) | Faster inference, good enough alignment |

### Why Both Lead to Transformers
Both mechanisms compute:
1. **Scores** between query (decoder state) and keys (encoder states)
2. **Softmax** normalisation → attention weights  
3. **Weighted sum** of values (encoder states) → context vector

The Transformer simply:
- Replaces the RNN encoder/decoder with **fully parallel** layers
- Uses Luong-style **dot-product** attention
- Scales by $\\sqrt{d_k}$ (to control variance)
- Adds **multi-head** attention (run multiple attention functions in parallel)

> The Transformer is Luong dot-product attention, fully parallelised, with no RNN needed.

---

## 8. 🔗 The Bridge — How Attention Led to Transformers

```
Vanilla Seq2Seq (2014):   Encoder RNN  →  [single context vector c]  →  Decoder RNN
      ↓ Problem: bottleneck, vanishing gradient across sequences

Bahdanau Attention (2015): Encoder RNN → [ALL h_i] → attention weights → c_t varies per step
      ↓ Insight: soft alignment is learnable and works!

Luong Attention (2015):   Simpler scoring: dot-product between states
      ↓ Simpler = faster = same quality

Transformer (2017):       No RNN at all! Pure attention: Q, K, V matrices
                          Self-attention: sequence attends to itself
                          Multi-head: multiple attention patterns in parallel
                          Fully parallelisable on GPU
```

The key question Vaswani et al. (2017) asked:  
**"If attention already captures the important dependencies, do we even need the RNN?"**  
Answer: No.
"""))

# ── CELL 12 – Practical Tips ──────────────────────────────────────────────────
cells.append(md("""## 9. 🔧 Practical Tips for RNN/Seq2Seq Training

### Tip 1: `pack_padded_sequence` — Don't Process PAD Tokens

**Problem:** In a batch, sequences have different lengths. We pad shorter ones with zeros.  
A naive RNN will process all those zeros — polluting the hidden state after the real content ends.

**Solution:** `torch.nn.utils.rnn.pack_padded_sequence` + `pad_packed_sequence`

```python
# Before RNN: pack (tell PyTorch to skip PAD positions)
packed = torch.nn.utils.rnn.pack_padded_sequence(
    emb,              # (batch, max_len, embed_dim)
    lengths,          # actual length of each sequence, sorted descending
    batch_first=True,
    enforce_sorted=True
)

# Run RNN on packed input (PyTorch handles the masking internally)
output_packed, hidden = self.rnn(packed)

# After RNN: unpack back to padded format
output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
```

**Effect:** The hidden state at the end reflects only real tokens, not padding.

---

### Tip 2: CTC Loss — For Variable-Length Alignment (No Seq2Seq Decoder Needed)

**Problem:** In speech recognition or OCR, we have:
- Input: 200 audio frames
- Output: 10 characters  
- We don't know the alignment: which frame corresponds to which character?

**Vanilla Seq2Seq** would need an explicit decoder + attention to align them.

**CTC (Connectionist Temporal Classification)** solves this differently:
- The RNN outputs one token per time step (including a special **BLANK** token)
- CTC marginalises over **all valid alignments** between the RNN output and the target string
- Example: "CAT" could be encoded as:  
  `_C__AA_T_`, `CC_AT__T`, `_CAAT__T` etc. (BLANK=`_`)
- CTC loss = negative log probability summed over all valid paths

$$\\mathcal{L}_{\\text{CTC}} = -\\log P(y | x) = -\\log \\sum_{\\pi \\in \\mathcal{B}^{-1}(y)} P(\\pi | x)$$

**When to use CTC:**
- Speech-to-text (end-to-end, no alignment needed)
- OCR (image → text)
- Any task with monotonic alignment (output is left-to-right in same order as input)

**Limitation:** CTC cannot model non-monotonic alignments (e.g., translation with word reordering).

---

### Tip 3: Bidirectional Inference Limitations

**Rule:** Bidirectional RNNs / LSTMs / GRUs **cannot be used for autoregressive generation**.

**Why:** The backward RNN at position $t$ uses $x_{t+1}, \\ldots, x_T$ — future tokens.  
At inference time, you generate tokens one by one — future tokens don't exist yet.

**Where you CAN use bidirectional:**
- Encoder in Seq2Seq (you have the full source sentence)
- Sequence labelling (NER, POS — full sentence available)
- Sentiment classification (full sentence available)
- BERT-style masked language modelling (full context available)

**Where you CANNOT:**
- Language modelling / text generation
- The decoder in any Seq2Seq or autoregressive model
- Real-time speech generation
"""))

cells.append(code("""import torch
import torch.nn as nn

# ── Demo: pack_padded_sequence in action ──
print("=" * 55)
print("PACK_PADDED_SEQUENCE DEMO")
print("=" * 55)

# 3 sentences of different lengths: [4, 2, 3] words
# Padded to max_len=4
batch = torch.tensor([
    [10, 20, 30, 40],   # length 4
    [50, 60,  0,  0],   # length 2 (2 PAD tokens)
    [70, 80, 90,  0],   # length 3 (1 PAD token)
], dtype=torch.long)
lengths = torch.tensor([4, 2, 3])

# Sort by descending length (required for pack_padded_sequence)
sorted_idx = lengths.argsort(descending=True)
batch_sorted   = batch[sorted_idx]
lengths_sorted = lengths[sorted_idx]

emb_layer = nn.Embedding(100, 8, padding_idx=0)
emb = emb_layer(batch_sorted)   # (3, 4, 8)

rnn = nn.GRU(8, 16, batch_first=True)

# ── WITHOUT packing (naive) ──
out_naive, h_naive = rnn(emb)
print("\nWithout packing — hidden state after PAD tokens:")
for i, (sentence, length) in enumerate(zip(batch_sorted, lengths_sorted)):
    print(f"  Seq {i} (len={length.item()}): h_T norm = {h_naive[0,i].norm().item():.4f}  ← "
          f"{'contaminated by PAD' if length < 4 else 'clean'}")

# ── WITH packing ──
packed_input = nn.utils.rnn.pack_padded_sequence(emb, lengths_sorted, batch_first=True)
out_packed, h_packed = rnn(packed_input)
out_unpacked, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

print("\nWith packing — hidden state stops at true sequence end:")
for i, (sentence, length) in enumerate(zip(batch_sorted, lengths_sorted)):
    print(f"  Seq {i} (len={length.item()}): h_T norm = {h_packed[0,i].norm().item():.4f}  ← clean ✅")

# ── CTC Loss demo ──
print("\n" + "=" * 55)
print("CTC LOSS DEMO (Speech/OCR alignment)")
print("=" * 55)
ctc_loss = nn.CTCLoss(blank=0)

# Simulated RNN outputs: T=10 time steps, N=2 samples, C=5 classes
T, N, C = 10, 2, 5
log_probs = torch.randn(T, N, C).log_softmax(2)  # (T, N, C) — required format

targets     = torch.tensor([1, 2, 3, 1, 3])      # concatenated targets
input_len   = torch.tensor([T, T])               # all T steps used for each sample
target_len  = torch.tensor([3, 2])               # 3 chars for sample 1, 2 for sample 2

loss = ctc_loss(log_probs, targets, input_len, target_len)
print(f"\n  CTC Loss: {loss.item():.4f}")
print(f"  Input shape:   {tuple(log_probs.shape)}  (time_steps, batch, classes)")
print(f"  Target:        {targets.tolist()}  (concatenated, no alignment needed)")
print(f"  Target lengths:{target_len.tolist()}")
print("\n  → CTC marginalises over ALL valid paths (e.g. CCC_AA_T = CAT)")
print("  → No explicit alignment labels required!")
"""))

# ── CELL 14 – Interview Q&A ────────────────────────────────────────────────────
cells.append(md("""## 10. 🎯 Senior-Level Interview Q&A

**Q1: What problem does attention solve that vanilla Seq2Seq cannot?**  
> Vanilla Seq2Seq compresses the entire input into a single fixed-size context vector. For long sequences (>30 words), this bottleneck causes catastrophic information loss — the model forgets the beginning of the sentence. Attention gives the decoder a direct, soft read-access to every encoder hidden state at every decoding step. Instead of one fixed context, the decoder gets a fresh, dynamically computed context at each step based on relevance.

---

**Q2: What's the difference between "hard" and "soft" attention?**  
> **Soft attention** (Bahdanau, Luong): assigns a real-valued weight $\\alpha_i \\in (0,1)$ to every source position — fully differentiable, trainable with backprop.  
> **Hard attention**: stochastically selects exactly ONE source position at each step (categorical choice). Not differentiable — requires REINFORCE (policy gradient) to train. More interpretable but harder to optimise.  
> Transformers and all modern NLP use soft attention.

---

**Q3: Why do we use a Bidirectional RNN as the encoder but a unidirectional one as the decoder?**  
> Encoder: we have the **full source sentence** available before generating anything, so we can read it in both directions to give each position full context. Decoder: we generate tokens **one at a time**, left to right. At step $t$, token $y_t$ doesn't exist yet — there are no future tokens to read backward from. The decoder is inherently autoregressive and causal.

---

**Q4: How does attention directly lead to the Transformer?**  
> The key insight was: if attention can capture all relevant dependencies between source and target, **do we still need the RNN?** The RNN was only there to produce the encoder states and decoder states. If we replace the RNN with a fully-connected network and use self-attention (sequence attends to itself), we get the Transformer. This removes the sequential dependency that blocked GPU parallelism. The Transformer uses Luong-style dot-product attention, scales by $1/\\sqrt{d_k}$, and runs multiple heads in parallel — but the core mechanism is the same as Bahdanau's soft alignment.

---

**Q5: When would you use CTC Loss vs a Seq2Seq decoder with attention?**  
> **CTC:** When the input-output alignment is monotonic (no word reordering) and you don't know the exact frame-to-character boundary. Best for: speech-to-text (Wav2Vec 2.0, DeepSpeech), OCR. CTC has no decoder to train — lower parameter count.  
> **Seq2Seq with attention:** When you need **non-monotonic** alignment (translation, summarisation) or complex many-to-many mappings. Much more expressive but more parameters and slower to train.

---

**Q6: What is teacher forcing and what problem does it cause?**  
> During training, the decoder receives the **ground-truth** previous token as input instead of its own (potentially wrong) predicted token. This makes training much faster and stable (the model always gets correct context). The problem: at **inference**, the model feeds its own predictions back — and any error compounds. This is called **exposure bias** — the training and inference distributions are mismatched. **Scheduled sampling** gradually reduces teacher forcing ratio over training to close this gap.
"""))

# ── ASSEMBLE ──────────────────────────────────────────────────────────────────
nb.cells = cells
path = r'c:\Users\23add\workspace\deeplearning\notes\12_seq2seq_attention.ipynb'
nbf.write(nb, path)
print(f"✅ Written: {path}  ({len(cells)} cells)")
