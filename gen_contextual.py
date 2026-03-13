"""
Generate llm_basic/04_contextual_embeddings.ipynb
Uses text blocks written as regular strings (no nested triple-quotes).
"""
import nbformat as nbf, os
os.makedirs('llm_basic/assets', exist_ok=True)
nb = nbf.v4.new_notebook()
cells = []

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

# ─────────────────────────────────────────────────────────────────
#  CELL 0 — Title
# ─────────────────────────────────────────────────────────────────
cells.append(md(
    "# Contextual Embeddings — ELMo, BERT & Sentence Transformers\n\n"
    "**Topics covered:**\n"
    "1. The polysemy problem with static embeddings\n"
    "2. ELMo — context-dependent BiLSTM representations\n"
    "3. BERT — `[CLS]` and token embeddings from Transformers\n"
    "4. Static vs Contextual: when to use each\n"
    "5. Sentence Transformers (SBERT) — semantic similarity at scale\n"
    "6. Mean pooling vs CLS pooling strategies\n\n"
    "**Pre-requisite:** `02_word2vec.ipynb`"
))

# ─────────────────────────────────────────────────────────────────
#  CELL 1 — Polysemy Problem
# ─────────────────────────────────────────────────────────────────
cells.append(md(
    "## 1. The Polysemy Problem — Why Static Embeddings Fall Short\n\n"
    "Every static embedding method (Word2Vec, GloVe, FastText) assigns **one fixed vector per word**. "
    "This vector is the average of all contexts where that word appeared during training.\n\n"
    "Consider the word **\"bank\"**:\n"
    "- \"I deposited money at the **bank**.\" — financial institution\n"
    "- \"The river **bank** was muddy.\" — edge of a river\n"
    "- \"She **bank**ed the aircraft sharply.\" — aircraft manoeuvre\n\n"
    "All three uses collapse to one single vector — a blend of all meanings.\n\n"
    "**More examples of polysemous words:**\n"
    "- **\"crane\"** — the bird OR the construction machine\n"
    "- **\"spring\"** — the season, a coiled object, or to jump\n"
    "- **\"light\"** — not heavy OR electromagnetic radiation OR to ignite\n\n"
    "### What We Really Want\n"
    "For *\"I went to the bank to withdraw cash\"*: "
    "`bank` vector should be close to ATM, money, finance.\n\n"
    "For *\"We sat by the river bank\"*: "
    "`bank` vector should be close to river, shore, mud.\n\n"
    "**Same word, totally different vectors** — that is exactly what contextual embeddings provide.\n\n"
    "---\n\n"
    "## 2. ELMo — Embeddings from Language Models\n\n"
    "**Paper:** Peters et al. (Allen AI, 2018) — *Deep Contextualized Word Representations*\n\n"
    "### Architecture: Deep Bidirectional LSTM Language Model\n\n"
    "ELMo trains a stack of BiLSTMs on raw text using two objectives:\n\n"
    "**Forward LM:** Given $w_1, \\ldots, w_{t-1}$, predict $w_t$\n\n"
    "$$\\log P(w_t \\mid w_1, \\ldots, w_{t-1})$$\n\n"
    "**Backward LM:** Given $w_T, \\ldots, w_{t+1}$, predict $w_t$\n\n"
    "$$\\log P(w_t \\mid w_{t+1}, \\ldots, w_T)$$\n\n"
    "Joint training loss:\n\n"
    "$$\\mathcal{L} = \\sum_{t=1}^{T} \\left[\\log P(w_t \\mid w_{<t}) + \\log P(w_t \\mid w_{>t})\\right]$$\n\n"
    "### Multiple LSTM Layers → Multiple Levels of Representation\n\n"
    "ELMo uses 2 BiLSTM layers + a character CNN at the input:\n\n"
    "```\n"
    "Character CNN input → Layer 0 embedding h0\n"
    "            ↓\n"
    "BiLSTM Layer 1 → h1 = [forward_h1 ; backward_h1]  (syntax-level)\n"
    "            ↓\n"
    "BiLSTM Layer 2 → h2 = [forward_h2 ; backward_h2]  (semantic-level)\n"
    "```\n\n"
    "**Key insight — different layers capture different linguistic levels:**\n"
    "| Layer | What it captures |\n"
    "|---|---|\n"
    "| Layer 0 (char CNN) | Morphology, spelling, POS |\n"
    "| Layer 1 (lower BiLSTM) | Syntactic structure, dependency relations |\n"
    "| Layer 2 (upper BiLSTM) | Word sense, semantic meaning |\n\n"
    "### ELMo Representation — Task-Specific Layer Weighting\n\n"
    "The final ELMo embedding for word $t$ is a **learned weighted sum** of all layer outputs:\n\n"
    "$$\\text{ELMo}_t = \\gamma \\sum_{j=0}^{L} s_j \\cdot h_{t,j}$$\n\n"
    "- $s_j$: softmax-normalised scalars — the model learns which layers matter for THIS task\n"
    "- $\\gamma$: global scale factor\n"
    "- For NER: $s_1$ large (syntax matters). For WSD: $s_2$ large (semantics matters).\n\n"
    "### How ELMo Resolves Polysemy\n\n"
    "For *\"bank\"* in *\"I went to the bank to withdraw cash\"*:\n"
    "- Forward BiLSTM has processed: \"I went to the\" → financial context\n"
    "- Backward BiLSTM has processed: \"to withdraw cash\" → financial reinforcement\n"
    "- ELMo embedding for this \"bank\" → **financial vector**\n\n"
    "For *\"bank\"* in *\"We sat by the river bank at sunset\"*:\n"
    "- Forward BiLSTM: \"We sat by the river\" → natural, geographical\n"
    "- Backward BiLSTM: \"at sunset\" → outdoor\n"
    "- ELMo embedding for this \"bank\" → **river-bank vector**\n\n"
    "Completely different vectors for the same word — polysemy resolved!"
))

# ─────────────────────────────────────────────────────────────────
#  CELL 2 — BERT
# ─────────────────────────────────────────────────────────────────
cells.append(md(
    "## 3. BERT — [CLS] and Contextual Token Embeddings\n\n"
    "**Paper:** Devlin et al. (Google, 2018) — "
    "*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*\n\n"
    "### Key Architectural Difference from ELMo\n\n"
    "| Property | ELMo | BERT |\n"
    "|---|---|---|\n"
    "| **Architecture** | Stacked BiLSTM | Transformer Encoder |\n"
    "| **Bidirectionality** | Approximate (two separate LMs concatenated) | True (all tokens attend to all others jointly) |\n"
    "| **Pre-training task** | Language modelling (forward + backward) | Masked LM + Next Sentence Prediction |\n"
    "| **Parallelism** | Sequential (LSTM) | Fully parallel (attention) |\n\n"
    "### Masked Language Modelling (MLM) — True Bidirectionality\n\n"
    "BERT cannot use next-word prediction (that's causal/left-to-right only). Instead:\n"
    "- Randomly mask 15% of tokens with a `[MASK]` token\n"
    "- Train the model to predict the original masked token using **all other tokens** as context\n\n"
    "$$P(w_t \\mid w_1, \\ldots, w_{t-1}, [MASK], w_{t+1}, \\ldots, w_T)$$\n\n"
    "This forces **every layer** to use context from both left AND right simultaneously.\n\n"
    "### BERT Input Format\n\n"
    "```\n"
    "Original:   \"I went to the bank\"\n\n"
    "Step 1 — Add special tokens:\n"
    "  [CLS] I went to the bank [SEP]\n\n"
    "Step 2 — WordPiece tokenisation (handles OOV):\n"
    "  [CLS] I went to the bank [SEP]\n"
    "  (or: withdraw → with ##draw — subwords split with ## prefix)\n\n"
    "Step 3 — Three embeddings SUMMED:\n"
    "  Token embedding    : lookup for each token ID\n"
    "  Segment embedding  : sentence A (0) or B (1)\n"
    "  Position embedding : learned for position 0, 1, 2, ..., 512\n"
    "```\n\n"
    "### The [CLS] Token as Sentence Representation\n\n"
    "The `[CLS]` token is always placed first. After 12 Transformer layers (BERT-base), "
    "its final hidden state $h_{[CLS]}^{(12)} \\in \\mathbb{R}^{768}$ "
    "is used as a **sentence-level representation** — because through 12 layers of self-attention, "
    "`[CLS]` has attended to every other token.\n\n"
    "For classification: add a linear layer on top and fine-tune end-to-end.\n\n"
    "### Why Raw [CLS] Fails for Semantic Similarity\n\n"
    "> **Critical problem:** BERT was pre-trained with NSP (Next Sentence Prediction), "
    "which uses `[CLS]` to predict if two sentences are consecutive — NOT for semantic similarity.\n\n"
    "Result: cosine similarities between unrelated BERT [CLS] vectors are often above 0.9. "
    "The vectors are **not calibrated for cosine distance**.\n\n"
    "Empirical finding: "
    "cos([CLS](\"A cat sat on the mat\"), [CLS](\"The stock market fell today\")) ≈ **0.92** "
    "— despite having nothing in common semantically.\n\n"
    "**Fix → Sentence Transformers (next section)**"
))

# ─────────────────────────────────────────────────────────────────
#  CELL 3 — Code demo (attention visualisation)
# ─────────────────────────────────────────────────────────────────
code_attention = "\n".join([
    "import numpy as np, matplotlib.pyplot as plt",
    "np.random.seed(42)",
    "",
    "# Simulate attention weights for 'bank' in two sentences",
    "financial_words = 'I went to the bank to withdraw cash'.split()",
    "river_words     = 'We sat by the river bank at sunset'.split()",
    "",
    "# Attention from 'bank' to every other word (simulated for ELMo/BERT)",
    "financial_attn = np.array([0.05, 0.10, 0.05, 0.05, 0.25, 0.05, 0.30, 0.15])",
    "river_attn     = np.array([0.05, 0.05, 0.20, 0.05, 0.20, 0.30, 0.05, 0.10])",
    "financial_attn /= financial_attn.sum()",
    "river_attn     /= river_attn.sum()",
    "",
    "# Simulated word vectors",
    "DIM = 16",
    "all_words = list(set(financial_words + river_words))",
    "word_vecs = {w: np.random.randn(DIM) for w in all_words}",
    "",
    "# Contextualised 'bank' = attention-weighted sum of all word vectors",
    "bank_financial = sum(financial_attn[i]*word_vecs[w] for i, w in enumerate(financial_words))",
    "bank_river     = sum(river_attn[i]*word_vecs[w] for i, w in enumerate(river_words))",
    "bank_static    = word_vecs.get('bank', np.random.randn(DIM))",
    "",
    "def cos(a, b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)",
    "",
    "print('Contextual Embedding Demo for word: bank')",
    "print(f'  cos(bank_financial, bank_river)  = {cos(bank_financial, bank_river):.4f}')",
    "print(f'  cos(bank_financial, bank_static) = {cos(bank_financial, bank_static):.4f}')",
    "print(f'  cos(bank_river,     bank_static) = {cos(bank_river,     bank_static):.4f}')",
    "print()",
    "print('  -> Two contextual vectors differ from each other AND from the static one')",
    "print('  -> This is polysemy resolution in action.')",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))",
    "fig.patch.set_facecolor('#0f0f1a')",
    "for ax in axes: ax.set_facecolor('#13131f')",
    "",
    "for ax, words, attn, title, col in [",
    "    (axes[0], financial_words, financial_attn, 'bank in financial context', '#4ecdc4'),",
    "    (axes[1], river_words,     river_attn,     'bank in river context',     '#fd79a8'),",
    "]:",
    "    bars = ax.barh(range(len(words)), attn, color=col, alpha=0.85)",
    "    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, color='white')",
    "    ax.set_xlabel('Attention weight', color='#aaa')",
    "    ax.set_title(f'Attention weights for: {title}', color='white')",
    "    ax.tick_params(colors='#aaa')",
    "    for sp in ax.spines.values(): sp.set_color('#333')",
    "    if 'bank' in words:",
    "        bi = words.index('bank')",
    "        bars[bi].set_edgecolor('yellow'); bars[bi].set_linewidth(2.5)",
    "",
    "plt.suptitle('Same word, different attention patterns -> different contextual vectors',",
    "             color='white', fontsize=11)",
    "plt.tight_layout()",
    "plt.savefig('llm_basic/assets/04_contextual.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')",
    "plt.show()",
])
cells.append(code(code_attention))

# ─────────────────────────────────────────────────────────────────
#  CELL 4 — SBERT
# ─────────────────────────────────────────────────────────────────
cells.append(md(
    "## 4. Sentence Transformers (SBERT) — Semantic Similarity at Scale\n\n"
    "**Paper:** Reimers & Gurevych (2019) — "
    "*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*\n\n"
    "### The Raw BERT Problem for Semantic Search\n\n"
    "Searching 10,000 sentences for the most similar using raw BERT:\n"
    "- BERT requires BOTH sentences as input (cross-attention between query and each candidate)\n"
    "- For 10K candidates: **10,000 BERT forward passes** → approximately **65 hours** on a GPU\n\n"
    "SBERT fixes this: precompute one embedding per sentence, then use fast cosine similarity.\n\n"
    "### Siamese Network Architecture\n\n"
    "```\n"
    "Sentence A → BERT (shared weights) → Pool → u\n"
    "                                                \\ \n"
    "                                         concat(u, v, |u-v|) → Linear → label\n"
    "                                                /\n"
    "Sentence B → BERT (shared weights) → Pool → v\n"
    "```\n\n"
    "### Pooling Strategies\n\n"
    "**CLS Pooling:** Use the final hidden state of `[CLS]`:\n\n"
    "$$u = h_{[CLS]}^{(L)}$$\n\n"
    "Simple but suboptimal for similarity — BERT [CLS] wasn't calibrated for cosine distance.\n\n"
    "**Mean Pooling (SBERT default):** Average all non-PAD token states:\n\n"
    "$$u = \\frac{\\sum_{t} h_t^{(L)} \\cdot m_t}{\\sum_{t} m_t}$$\n\n"
    "where $m_t \\in \\{0,1\\}$ is the attention mask (0 for PAD tokens).\n\n"
    "Why mean pooling is better:\n"
    "- Uses signal from **every content word** — not just [CLS]\n"
    "- [CLS] can be position-biased (it attends to itself in every layer)\n"
    "- Empirically 2–3% better on STS benchmarks\n\n"
    "**Max Pooling:** Take max across all positions in each dimension:\n\n"
    "$$u_d = \\max_{t} h_{t,d}^{(L)}$$\n\n"
    "Detects whether any token in the sentence activated a particular feature strongly.\n\n"
    "### Training Objectives\n\n"
    "**1. NLI Classification (Entailment / Contradiction / Neutral):**\n\n"
    "$$\\mathcal{L} = \\text{CrossEntropy}(W_c \\cdot (u, v, |u-v|), y)$$\n\n"
    "**2. STS Regression (score in [0,1]):**\n\n"
    "$$\\mathcal{L} = \\text{MSE}(\\cos(u, v), \\text{score})$$\n\n"
    "**3. Triplet Loss (for retrieval):**\n\n"
    "$$\\mathcal{L} = \\max(0, \\|u_a - u_p\\| - \\|u_a - u_n\\| + \\epsilon)$$\n\n"
    "Push anchor closer to positive than to negative by margin $\\epsilon$.\n\n"
    "### Search Speed Comparison\n\n"
    "| Method | Approach | 10K sentence search |\n"
    "|---|---|---|\n"
    "| Raw BERT | Pass every pair through BERT | ~65 hours |\n"
    "| SBERT | Pre-embed + cosine lookup | ~5 seconds |\n"
    "| SBERT + FAISS | Approximate nearest neighbour index | < 1 second |"
))

# ─────────────────────────────────────────────────────────────────
#  CELL 5 — Pooling demo code
# ─────────────────────────────────────────────────────────────────
code_pooling = "\n".join([
    "import numpy as np, matplotlib.pyplot as plt",
    "np.random.seed(42)",
    "",
    "# Simulate BERT final hidden states: 8 tokens, dim=8",
    "tokens = ['[CLS]', 'The', 'cat', 'sat', 'on', 'mat', '.', '[SEP]']",
    "T, D = len(tokens), 8",
    "H = np.random.randn(T, D) * 0.5",
    "H[0] = np.array([0.8, -0.3, 0.5, -0.7, 0.2, 0.9, -0.4, 0.1])  # CLS",
    "H[2, 2] = 1.5   # cat -> high value in dim 2",
    "H[3, 2] = 1.3   # sat -> high value in dim 2",
    "",
    "# Attention mask: exclude [CLS] and [SEP] from mean pooling",
    "mask = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=float)",
    "",
    "cls_pool  = H[0]",
    "mean_pool = (H * mask[:, None]).sum(0) / mask.sum()",
    "max_pool  = np.where(mask[:, None] > 0, H, -np.inf).max(0)",
    "",
    "print('Pooling results for: The cat sat on mat')",
    "print(f'  CLS  pool: {cls_pool.round(3)}')",
    "print(f'  Mean pool: {mean_pool.round(3)}')",
    "print(f'  Max  pool: {max_pool.round(3)}')",
    "",
    "def cos(a, b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)",
    "",
    "print()",
    "print('Pairwise cosine similarity between pooling strategies:')",
    "for na, va in [('CLS', cls_pool), ('Mean', mean_pool), ('Max', max_pool)]:",
    "    for nb, vb in [('CLS', cls_pool), ('Mean', mean_pool), ('Max', max_pool)]:",
    "        print(f'  cos({na}, {nb}) = {cos(va, vb):.4f}')",
    "",
    "# Visualise",
    "fig, ax = plt.subplots(figsize=(10, 4))",
    "fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#13131f')",
    "for label, pool, col in [('CLS pool','#fd79a8',), ('Mean pool','#4ecdc4',), ('Max pool','#fdcb6e',)]:",
    "    pass  # Fix: unpack properly below",
    "for (label, pool, col) in [('CLS pool', cls_pool, '#fd79a8'),",
    "                             ('Mean pool', mean_pool, '#4ecdc4'),",
    "                             ('Max pool', max_pool, '#fdcb6e')]:",
    "    ax.plot(pool, marker='o', label=label, lw=2, color=col)",
    "ax.set_title('CLS vs Mean vs Max Pooling (same sentence)', color='white')",
    "ax.set_xlabel('Embedding dimension', color='#aaa')",
    "ax.set_ylabel('Value', color='#aaa')",
    "ax.legend(facecolor='#1a1a2e', labelcolor='white')",
    "ax.tick_params(colors='#aaa')",
    "for sp in ax.spines.values(): sp.set_color('#333')",
    "plt.tight_layout()",
    "plt.savefig('llm_basic/assets/04_pooling.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')",
    "plt.show()",
])
cells.append(code(code_pooling))

# ─────────────────────────────────────────────────────────────────
#  CELL 6 — Trade-offs + Q&A
# ─────────────────────────────────────────────────────────────────
cells.append(md(
    "## 5. Static vs Contextual — Trade-off Table\n\n"
    "| Property | Static (Word2Vec, GloVe, FastText) | Contextual (ELMo, BERT, SBERT) |\n"
    "|---|---|---|\n"
    "| Vector changes with context? | No — fixed per word | Yes — per sentence |\n"
    "| Handles polysemy? | No — blended vector | Yes — different vector per sense |\n"
    "| Inference speed | Microseconds (lookup) | Milliseconds–seconds (model forward pass) |\n"
    "| Model size | Small (V x d table) | Large (110M–1B+ params) |\n"
    "| OOV words | Word2Vec/GloVe: no. FastText: yes | Yes (subword tokeniser) |\n"
    "| Semantic similarity | OK | SBERT: excellent |\n"
    "| Best for | Real-time, resource-constrained | Quality-critical NLP tasks |\n\n"
    "---\n\n"
    "## 6. Interview Q&A\n\n"
    "**Q1: What is the fundamental difference between static and contextual embeddings?**\n"
    "> Static embeddings assign one fixed vector per word regardless of context. "
    "Contextual embeddings (ELMo, BERT) produce a different vector per word depending on "
    "the full sentence — the same word 'bank' gets a financial vector or river vector "
    "depending on surrounding words. This resolves polysemy.\n\n"
    "---\n\n"
    "**Q2: Why is BERT [CLS] bad for semantic similarity but good for classification?**\n"
    "> BERT's [CLS] was trained with NSP — classifying whether two sentences are "
    "consecutive in a document. This optimises [CLS] for sentence-pair classification, "
    "NOT for cosine-distance-based similarity. For classification you add a linear head "
    "and fine-tune — [CLS] excels there. For similarity/retrieval you need SBERT, "
    "which re-calibrates the embedding space specifically for cosine comparison.\n\n"
    "---\n\n"
    "**Q3: Why is mean pooling generally better than CLS for sentence embeddings?**\n"
    "> Mean pooling aggregates signal from every content token. "
    "[CLS] must represent the whole sentence but may be dominated by self-attention "
    "to its own position. Mean pooling outperforms [CLS] by 2–5% on STS benchmarks.\n\n"
    "---\n\n"
    "**Q4: How does ELMo differ from BERT architecturally?**\n"
    "> ELMo: BiLSTM, forward + backward LMs trained separately, concatenated. "
    "Sequential (not parallel), approximate bidirectionality. "
    "BERT: Transformer encoder, full self-attention in both directions simultaneously "
    "via Masked LM. Fully parallel, true joint bidirectionality. "
    "BERT representations are generally stronger due to true joint bidirectional context.\n\n"
    "---\n\n"
    "**Q5: What training loss does SBERT use — and why not just cosine similarity training?**\n"
    "> SBERT uses NLI (cross-entropy with entailment/neutral/contradiction) and/or "
    "regression on STS scores. Pure cosine similarity training would collapse all embeddings "
    "to a single point (trivial minimum). NLI provides contrastive signal: contradiction pairs "
    "must be far apart, entailment pairs close. Triplet loss directly optimises "
    "the relative ordering that retrieval systems need.\n\n"
    "---\n\n"
    "**Q6: Why is SBERT 65 hours vs 5 seconds for 10K search?**\n"
    "> Raw BERT needs both query AND candidate in the same forward pass for cross-attention. "
    "You cannot pre-cache candidates — every query requires 10K passes. "
    "SBERT produces a fixed vector independently per sentence — cache all 10K once, "
    "then for each query do one BERT pass + 10K dot products (trivially fast on GPU/CPU)."
))

# ─────────────────────────────────────────────────────────────────
#  Write notebook
# ─────────────────────────────────────────────────────────────────
nb.cells = cells
path = 'llm_basic/04_contextual_embeddings.ipynb'
nbf.write(nb, path)
print(f"Written: {path}  ({len(cells)} cells)")
