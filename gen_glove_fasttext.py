import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

# ── CELL 0 Title ──────────────────────────────────────────────────────────────
cells.append(md("""# 🌐 GloVe & FastText — Global Vectors and Subword Embeddings

**What this notebook covers:**
1. GloVe — Global Vectors from co-occurrence statistics
   - Co-occurrence matrix construction
   - GloVe objective function — full derivation
   - Why GloVe differs from Word2Vec
2. FastText — subword-level embeddings
   - How character n-grams extend embeddings
   - Out-of-vocabulary (OOV) word handling
   - Comparison to Word2Vec

**Pre-requisite:** `02_word2vec.ipynb`
"""))

# ── CELL 1 GloVe Motivation ───────────────────────────────────────────────────
cells.append(md("""## 1. 🌐 GloVe — Global Vectors for Word Representation

**Paper:** Pennington, Socher, Manning — "GloVe: Global Vectors for Word Representation" (2014)

### Word2Vec's Hidden Limitation
Word2Vec learns from **local context windows** — it only sees a few words at a time around each target word. It never explicitly looks at the **global co-occurrence statistics** of the entire corpus.

But these global statistics carry powerful information:

> Consider the ratio: $\\frac{P(\\text{"ice"} | \\text{"solid"})}{P(\\text{"steam"} | \\text{"solid"})}$  
> — How much more likely is "solid" to appear near "ice" than "steam"?

If this ratio is large (say 8.9), it strongly tells us that "ice" IS related to "solid" and "steam" is NOT.  
Consider also: $\\frac{P(\\text{"ice"} | \\text{"water"})}{P(\\text{"steam"} | \\text{"water"})}$ ≈ 1.36 — both ice and steam relate to water similarly.

GloVe's insight: **these co-occurrence ratios already encode meaningful relationships.**  
The model should learn to reconstruct them.

### The Co-occurrence Matrix $X$

Given a corpus, define:
$$X_{ij} = \\text{number of times word } j \\text{ appears in the context of word } i$$

(context = within a window of size $w$ around word $i$)

Example corpus: "the cat sat on the mat the cat sat on the log"

With window $w = 1$:
```
      the  cat  sat  on  mat  log
the   0    2    0    0   0    0
cat   2    0    2    0   0    0
sat   0    2    0    2   0    0
on    0    0    2    0   1    1
mat   0    0    0    1   0    0
log   0    0    0    1   0    0
```

Properties:
- $X$ is **symmetric**: if "cat" appears near "sat", then "sat" also appears near "cat"
- $X$ is **very sparse** for large vocabularies — but it's computed **once** over the entire corpus
- Contains **global** information — unlike Word2Vec's local window sampling

### GloVe Objective

Define:
$$P_{ij} = P(j | i) = \\frac{X_{ij}}{X_i} \\quad \\text{where } X_i = \\sum_k X_{ik}$$

$P_{ij}$ = probability that word $j$ appears in the context of word $i$.

GloVe wants the **dot product of word vectors to equal the log of co-occurrence probability**:
$$w_i^\\top \\tilde{w}_j + b_i + \\tilde{b}_j = \\log X_{ij}$$

Where:
- $w_i$ = embedding of word $i$ (main vector)
- $\\tilde{w}_j$ = embedding of word $j$ (context vector, like W_out in Word2Vec)
- $b_i, \\tilde{b}_j$ = bias scalars

The **loss function** penalises the squared error, but weights it by co-occurrence frequency:
$$J = \\sum_{i,j=1}^{V} f(X_{ij}) \\left( w_i^\\top \\tilde{w}_j + b_i + \\tilde{b}_j - \\log X_{ij} \\right)^2$$

### The Weighting Function $f(X_{ij})$
Not all co-occurrence counts are equally reliable:
- Very rare pairs (e.g., "cat" "quantum"): only appeared once — noisy signal
- Very common pairs (e.g., "the" "cat"): very high count but "the" is meaningless
- Medium frequency pairs: most informative

$$f(X_{ij}) = \\begin{cases} \\left(\\frac{X_{ij}}{X_{\\text{max}}}\\right)^\\alpha & \\text{if } X_{ij} < X_{\\text{max}} \\\\ 1 & \\text{otherwise} \\end{cases}$$

With $X_{\\text{max}} = 100$, $\\alpha = 0.75$:
- Zero co-occurrence → $f = 0$ (ignore — no training signal)
- Low co-occurrence → small weight (don't trust it much)
- High co-occurrence → weight capped at 1 (don't let common words dominate)

### Key Differences: GloVe vs Word2Vec

| Property | Word2Vec | GloVe |
|---|---|---|
| **Training method** | Stochastic (sample pairs) | Batch (over all word pairs) |
| **Uses** | Local context windows | Global co-occurrence matrix |
| **Objective** | Predict context word | Reconstruct log co-occurrence |
| **Result vectors** | $W_{\\text{in}}$ (or average of $W_{\\text{in}}$ and $W_{\\text{out}}$) | $w + \\tilde{w}$ (sum of both vectors) |
| **When to use** | Large dynamic stream data | Pre-computed on fixed corpus |
| **Training speed** | Fast with neg. sampling | Fast — matrix is precomputed |
| **Quality** | Often slightly worse on analogies | Often slightly better on analogies |
| **Pre-trained** | word2vec-google-news-300 | glove-wikipedia-gigaword-100/300 |

> In practice, both produce embeddings of similar quality on most benchmarks. The choice often comes down to available infrastructure and pre-trained model preferences.
"""))

# ── CELL 2 GloVe Code ─────────────────────────────────────────────────────────
cells.append(code("""import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(42)

# ── Tiny corpus ──
corpus_sentences = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
    "the cat ate fish",
    "the dog ate meat",
    "ice is solid water",
    "steam is hot water",
    "water can be solid or liquid",
]
tokens = [w for s in corpus_sentences for w in s.lower().split()]
vocab = sorted(set(tokens))
V = len(vocab)
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}
print(f"Vocab: {V} words: {vocab}")
print()

# ── Step 1: Build co-occurrence matrix ──
window = 2
X = np.zeros((V, V), dtype=float)
idxs = [w2i[w] for w in tokens]
for pos, center in enumerate(idxs):
    lo = max(0, pos - window)
    hi = min(len(idxs), pos + window + 1)
    for ctx_pos in range(lo, hi):
        if ctx_pos != pos:
            context = idxs[ctx_pos]
            # Distance-weighted: words closer to centre count more
            distance = abs(ctx_pos - pos)
            X[center, context] += 1.0 / distance
print("Co-occurrence matrix X  (sample rows):")
focus = ["cat", "dog", "water", "solid", "steam"]
focus_idx = [w2i[w] for w in focus if w in w2i]
focus_words = [i2w[i] for i in focus_idx]
print(f"{'':10}", " ".join(f"{w:>8}" for w in focus_words))
for i in focus_idx:
    print(f"{i2w[i]:10}", " ".join(f"{X[i,j]:>8.2f}" for j in focus_idx))

# ── Step 2: Weighting function f ──
X_max, alpha = 100.0, 0.75
def f_weight(x):
    return min((x/X_max)**alpha, 1.0) if x > 0 else 0.0

# ── Step 3: GloVe Training (simplified SGD) ──
EMB_DIM = 8
w_main = np.random.randn(V, EMB_DIM) * 0.01    # main vectors
w_ctx  = np.random.randn(V, EMB_DIM) * 0.01    # context vectors
b_main = np.zeros(V)
b_ctx  = np.zeros(V)
lr = 0.05
losses = []

for epoch in range(150):
    epoch_loss = 0.0
    for i in range(V):
        for j in range(V):
            if X[i, j] == 0:
                continue
            log_Xij = np.log(X[i, j])
            w = f_weight(X[i, j])
            # Prediction
            pred = np.dot(w_main[i], w_ctx[j]) + b_main[i] + b_ctx[j]
            err  = pred - log_Xij
            loss = w * err ** 2
            epoch_loss += loss
            # Gradients
            grad = 2 * w * err
            w_main[i] -= lr * grad * w_ctx[j]
            w_ctx[j]  -= lr * grad * w_main[i]
            b_main[i] -= lr * grad
            b_ctx[j]  -= lr * grad
    losses.append(epoch_loss)

# Use sum of main and context vectors as final embedding (GloVe convention)
embeddings = w_main + w_ctx

# ── Cosine similarity ──
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)

print("\nGloVe Learned Similarities:")
pairs_to_check = [("cat","dog"), ("cat","fish"), ("water","solid"),
                  ("water","steam"), ("ice","solid"), ("ice","water")]
for w1, w2 in pairs_to_check:
    if w1 in w2i and w2 in w2i:
        sim = cosine(embeddings[w2i[w1]], embeddings[w2i[w2]])
        print(f"  sim({w1:8s}, {w2:8s}) = {sim:+.4f}")

# ── Visualise loss and weighting function ──
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.patch.set_facecolor('#0f0f1a')
for ax in axes: ax.set_facecolor('#13131f')

axes[0].plot(losses, color='#4ecdc4', lw=2)
axes[0].set_title("GloVe Training Loss", color='white')
axes[0].set_xlabel("Epoch", color='#aaa'); axes[0].set_ylabel("Loss", color='#aaa')
axes[0].tick_params(colors='#aaa')
for sp in axes[0].spines.values(): sp.set_color('#333')

x_vals = np.linspace(0, 150, 300)
f_vals = [f_weight(x) for x in x_vals]
axes[1].plot(x_vals, f_vals, color='#fd79a8', lw=2.5)
axes[1].axvline(x=100, color='#fdcb6e', ls='--', lw=1.5, label='X_max=100')
axes[1].set_title("GloVe Weighting Function f(X_ij)", color='white')
axes[1].set_xlabel("Co-occurrence count X_ij", color='#aaa')
axes[1].set_ylabel("Weight f(X_ij)", color='#aaa')
axes[1].tick_params(colors='#aaa')
axes[1].legend(facecolor='#1a1a2e', labelcolor='white')
for sp in axes[1].spines.values(): sp.set_color('#333')

plt.tight_layout()
plt.savefig('llm_basic/assets/03_glove.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()
"""))

# ── CELL 4 FastText ───────────────────────────────────────────────────────────
cells.append(md("""## 2. ⚡ FastText — Subword Character N-gram Embeddings

**Paper:** Bojanowski et al. — "Enriching Word Vectors with Subword Information" (Facebook AI, 2017)

### The Word2Vec/GloVe Problem: Unknown Words
Both Word2Vec and GloVe assign **one vector per word**. If a word was not in the training vocabulary, it has **no representation**.

Real-world problems:
- **New words:** "COVID", "Brexit", "selfie" — not in GloVe trained on 2010 Wikipedia
- **Morphological variants:** "running", "runner", "runs" — each needs a separate embedding  
- **Misspellings:** "recieve" instead of "receive" — no embedding
- **Domain-specific jargon:** Medical, legal, or technical terms unseen in general corpus
- **Names and places:** "Timbermede", "Okonkwo" — rare, possibly unseen

### FastText's Solution: Subword N-grams

FastText decomposes each word into **character n-grams** (substrings of length $n$).

**Example:** For the word "where" with $n \\in \\{3, 4, 5, 6\\}$:
```
Special boundary markers: <where>
  3-grams: <wh, whe, her, ere, re>
  4-grams: <whe, wher, here, ere>
  5-grams: <wher, where, here>
  6-grams: <where, where>
  + the whole word: <where>
```

The `<` and `>` markers distinguish the start/end of words from substrings that appear in the middle.  
For example, `he` as a suffix vs `<he>` as the complete word "he".

**The embedding of a word = sum (or average) of all its subword vectors:**
$$v_{\\text{where}} = \\frac{1}{|G_{\\text{where}}|} \\sum_{g \\in G_{\\text{where}}} z_g$$

where $G_{\\text{where}}$ is the set of all subwords of "where", and $z_g$ is the vector for subword $g$.

### Training
FastText uses the same Skip-gram with Negative Sampling training as Word2Vec, but:
- Instead of looking up one vector per word, it **sums all subword vectors** for the target word
- The gradient is propagated back to **all subword vectors** of the target word

This means common character patterns are shared across words — "ing" in "running", "jumping", "eating" all share the same `ing` subword vector.

### Handling OOV Words
For a word **never seen in training** (e.g., "unhappiness" if only "unhappy" was seen):

1. Decompose into n-grams: `<un`, `unh`, `nha`, `hap`, `app`, `ppi`, `pin`, `ine`, `nes`, `ess`, `ss>`, `<unh`, ...
2. Look up any subword vectors that **do exist** from training
3. Sum/average them → a reasonable embedding is produced!

Even words like "unknownabc123" will get some nonzero embedding from its character patterns.

### FastText vs Word2Vec vs GloVe

| Property | Word2Vec | GloVe | FastText |
|---|---|---|---|
| **OOV words** | ❌ Unknown token | ❌ Unknown token | ✅ Subword n-grams |
| **Morphology** | ❌ Separate vectors | ❌ Separate vectors | ✅ Shared subwords |
| **Vocabulary** | Fixed | Fixed | Can handle any string |
| **Misspellings** | ❌ Fails | ❌ Fails | ✅ Similar to correct spelling |
| **Training speed** | Fast | Medium | Slower (more computations) |
| **Model size** | V × d | V × d | V × d + subword vocab |
| **Best for** | General NLP | General NLP | Morphologically rich languages (Turkish, Finnish), domain text |
| **Languages** | Works for English | Works for English | Essential for agglutinative languages |
"""))

cells.append(code("""import numpy as np
import matplotlib.pyplot as plt

# ── FastText subword decomposition ──
def get_ngrams(word, min_n=3, max_n=6):
    word = f"<{word}>"     # add boundary markers
    ngrams = set()
    ngrams.add(word)       # add the full word token
    for n in range(min_n, max_n+1):
        for i in range(len(word)-n+1):
            ngrams.add(word[i:i+n])
    return ngrams

# Demonstrate subword decomposition
test_words = ["where", "running", "runner", "COVID", "unhappiness"]
print("Subword N-grams (min_n=3, max_n=6):")
print("=" * 60)
for word in test_words:
    ngrams = sorted(get_ngrams(word))
    print(f"\n  '{word}' → {len(ngrams)} subwords:")
    print(f"    {ngrams}")

# ── OOV word embedding via shared subwords ──
print("\n" + "="*60)
print("OOV Handling Demo")
print("="*60)

# Simulate: "runner" was seen in training but "runners" was not
np.random.seed(42)

# Build a tiny subword vocab from seen words
seen_words = ["run", "running", "runner", "fast", "speed", "race"]
subword_vocab = set()
for w in seen_words:
    subword_vocab |= get_ngrams(w)
subword_vocab = sorted(subword_vocab)
sv2i = {s: i for i, s in enumerate(subword_vocab)}

# Random embeddings for each subword (in practice, trained Skip-gram)
DIM = 8
subword_emb = np.random.randn(len(subword_vocab), DIM) * 0.1
print(f"\nSubword vocabulary size: {len(subword_vocab)}")

# Compute embedding for any word by summing its available subwords
def fasttext_embed(word, subword_emb, sv2i, min_n=3, max_n=6):
    ngrams = get_ngrams(word, min_n, max_n)
    vecs   = [subword_emb[sv2i[g]] for g in ngrams if g in sv2i]
    if not vecs:
        return np.zeros(subword_emb.shape[1])
    return np.mean(vecs, axis=0)

# Compare embeddings
words_to_compare = ["runner", "runners", "running", "run", "runnning"]  # last one is misspelled
embeds = {w: fasttext_embed(w, subword_emb, sv2i) for w in words_to_compare}

def cosine(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)

print("\nCosine similarities to 'runner':")
for w in words_to_compare:
    sim = cosine(embeds["runner"], embeds[w])
    oov = "" if any(w_s in seen_words for w_s in [w]) else " [OOV!]"
    print(f"  sim('runner', '{w}'){oov}: {sim:.4f}")

# ── Visualise: subword overlap between words ──
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#13131f')
words_for_viz = ["runner", "running", "runners", "runnning"]
all_ngrams_per_word = {w: get_ngrams(w) for w in words_for_viz}
overlap_mat = np.zeros((len(words_for_viz), len(words_for_viz)))
for i, w1 in enumerate(words_for_viz):
    for j, w2 in enumerate(words_for_viz):
        shared = len(all_ngrams_per_word[w1] & all_ngrams_per_word[w2])
        total  = len(all_ngrams_per_word[w1] | all_ngrams_per_word[w2])
        overlap_mat[i,j] = shared / total if total > 0 else 0

im = ax.imshow(overlap_mat, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(len(words_for_viz))); ax.set_xticklabels(words_for_viz, color='white', rotation=20)
ax.set_yticks(range(len(words_for_viz))); ax.set_yticklabels(words_for_viz, color='white')
ax.set_title("FastText: N-gram Overlap Between Words\n(High overlap = similar subword composition → similar embedding)",
             color='white')
for i in range(len(words_for_viz)):
    for j in range(len(words_for_viz)):
        ax.text(j, i, f'{overlap_mat[i,j]:.2f}', ha='center', va='center',
                color='black' if overlap_mat[i,j] > 0.4 else 'white', fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8, label='Jaccard overlap')
plt.tight_layout()
plt.savefig('llm_basic/assets/03_fasttext.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()
print("\n'runnning' (misspelled) still has high overlap with 'running' → robust to typos!")
"""))

# ── CELL 6 Interview Q&A ─────────────────────────────────────────────────────
cells.append(md("""## 3. 🎯 Interview Q&A

**Q1: What is the key advantage of GloVe over Word2Vec?**
> GloVe explicitly uses **global co-occurrence statistics** from the entire corpus. Word2Vec only sees local context windows during training (a few words at a time). GloVe pre-computes the full co-occurrence matrix $X$ and trains on all word pairs weighted by co-occurrence frequency. GloVe often achieves slightly better word analogy scores because it directly encodes global corpus-level relationships, while Word2Vec must infer them from many local observations.

---

**Q2: What does the weighting function $f(X_{ij})$ do in GloVe?**
> It prevents two types of problems: (1) Very frequent pairs like ("the", "cat") would dominate the loss if unweighted — $f$ caps their weight at 1 via $X_{\\text{max}}$. (2) Very rare pairs ($X_{ij}$ close to 0) are unreliable noise — $f(0) = 0$ ensures they contribute nothing. The middle range, informative co-occurring word pairs, gets weights between 0 and 1 proportional to frequency. The 0.75 exponent provides smooth interpolation.

---

**Q3: Why does FastText handle morphologically rich languages better than Word2Vec?**
> In Turkish or Finnish, a single word root can generate hundreds of valid word forms through agglutination (attaching morphemes). E.g., Turkish "ev" (house) → "evde" (at the house) → "evlerde" (at the houses) → "evlerinizden" (from your houses). Word2Vec would need millions of examples to learn separate vectors for all these forms. FastText shares subword representations across forms: any word containing the root "ev" will have correlated embeddings, even if the exact word form was never seen.

---

**Q4: Can FastText handle emojis or unknown languages?**
> To some extent — yes. If trained on byte-level or Unicode character n-grams, FastText will create subword embeddings for any Unicode sequence, including emojis. However, quality depends on the coverage of similar character patterns in training. For completely foreign scripts (e.g., Arabic in an English model), the subword embeddings will be mostly from the randomly initialised components. True multilingual coverage requires training on multilingual corpora (mFastText, LASER, etc.).

---

**Q5: After training GloVe, how do we get the final word vector?**
> GloVe trains two vectors per word: $w_i$ (main embedding) and $\\tilde{w}_i$ (context embedding). The final recommended embedding is the **sum** $w_i + \\tilde{w}_i$. Intuitively, both vectors encode related but slightly different aspects of the word's usage pattern (as a target vs. as a context). Their sum combines both viewpoints. Using only $w_i$ also works reasonably well and is simpler.
"""))

# ── Assemble ──────────────────────────────────────────────────────────────────
nb.cells = cells
path = 'llm_basic/03_glove_fasttext.ipynb'
nbf.write(nb, path)
print(f"Written: {path}  ({len(cells)} cells)")
