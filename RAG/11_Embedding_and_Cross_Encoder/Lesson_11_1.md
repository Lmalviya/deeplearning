# Lesson A.1 — From BERT to Sentence Embeddings: Why BERT Fails and How SBERT Fixed It

---

## What BERT Actually Is

Before understanding why BERT fails for sentence similarity, you need to understand what BERT was designed to do.

BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018) is a transformer encoder trained on two tasks:

**1. Masked Language Modeling (MLM):** 15% of tokens in a sentence are masked. The model must predict the masked tokens given all surrounding context. This forces the model to learn deep contextual representations — every token's representation is influenced by every other token in both directions.

**2. Next Sentence Prediction (NSP):** Given two sentences A and B, predict whether B naturally follows A. Trains the model on sentence-level relationships.

BERT's output: for each input token, a 768-dimensional vector (for BERT-base) that encodes that token's meaning in context. The token "bank" in "river bank" produces a different vector than "bank" in "savings bank" — context is baked in.

The [CLS] token (a special classification token prepended to every input) was intended by the original paper to aggregate the whole-sequence meaning and is used for classification tasks. You pass [CLS]'s output vector to a classification head.

This is what BERT was built for: classification, named entity recognition, question answering spans, natural language inference — not sentence-to-sentence similarity retrieval.

---

## Why Raw BERT Fails for Sentence Similarity

Here is the precise problem. Suppose you want to find which of 10,000 sentences is most similar to a query.

**Approach 1 — Cross-encoder style:** Concatenate query + each candidate → feed to BERT → compare. This requires 10,000 BERT forward passes per query. At ~50ms per pass, that is 500 seconds. Completely infeasible.

**Approach 2 — Independent sentence embeddings:** Embed the query once, embed all 10,000 candidates once, store candidate embeddings, compare via cosine similarity. This is what you want. But the question is: what embedding do you use?

The naive answer is: take the [CLS] token output from BERT. This seems reasonable — the NSP task trained [CLS] to represent whole-sentence information.

The problem: BERT's [CLS] representations were not trained to be metrically useful for sentence similarity. "Metrically useful" means: sentences with similar meaning should produce vectors with high cosine similarity, and sentences with different meanings should produce vectors with low cosine similarity.

BERT's [CLS] vectors do not have this property. Two semantically similar sentences can have [CLS] vectors with cosine similarity as low as 0.3. Two completely unrelated sentences can have cosine similarity as high as 0.8. The [CLS] space is not calibrated for retrieval.

**Empirical evidence from the SBERT paper (Reimers & Gurevych, 2019):** They measured average cosine similarity between all [CLS] vectors from BERT-base on the STS (Semantic Textual Similarity) benchmark. The distribution was clustered tightly near 1.0 — nearly all vectors were very similar to each other. This means BERT's CLS space has extremely low anisotropy — all vectors point in roughly the same direction. Cosine similarity between any two vectors is always high, making it useless as a discriminative signal.

Averaging all token embeddings (mean pooling) instead of using [CLS] is slightly better but still not good enough for retrieval.

---

## Why the CLS Space Is Poorly Calibrated

This is a subtle but important point.

BERT was pre-trained on a massive text corpus using MLM and NSP. MLM trains token-level representations to predict surrounding words — not to place semantically similar sentences near each other globally. NSP trains [CLS] to be a binary signal (is B a continuation of A?) — not a graded similarity signal.

After pre-training, if you fine-tune BERT on a classification task, the [CLS] vector becomes useful for that task's decision boundary. But nothing in the training process encourages [CLS] vectors to be metrically arranged in a way where "sentence closeness in embedding space = semantic closeness."

For sentence similarity retrieval to work, you need the embedding space to be calibrated so that:
- Distance in embedding space = semantic distance between sentences.
- Semantically similar sentences cluster together.
- Semantically different sentences are spread apart.

This requires explicit training with sentence-level similarity supervision. BERT never gets that.

---

## What SBERT Does Differently

SBERT (Sentence-BERT) takes BERT and fine-tunes it specifically so that its sentence embeddings are metrically calibrated for similarity retrieval.

The key insight: **use a Siamese network architecture with a similarity objective.**

### The Siamese Architecture

Instead of feeding one sentence to BERT, feed two sentences simultaneously through the same BERT encoder (with shared weights):

```
Sentence A → BERT → Pool → u (vector)
Sentence B → BERT → Pool → v (vector)

Similarity = cosine(u, v)
Loss = f(similarity, ground_truth_similarity)
```

Both sentences are encoded independently by the same model (shared weights — that is what makes it "Siamese"). The pooled outputs are compared using cosine similarity. The loss function trains the model so that similar sentences produce high cosine similarity and dissimilar sentences produce low cosine similarity.

### Mean Pooling: Better Than CLS

SBERT uses mean pooling rather than CLS token:

```
BERT outputs: [token_1_vec, token_2_vec, ..., token_n_vec]
Mean pool: (token_1_vec + token_2_vec + ... + token_n_vec) / n
```

Mean pooling averages all token embeddings. In practice, this produces better sentence representations than [CLS] for most tasks because:
- [CLS] is a single token that has to aggregate all information from one position.
- Mean pooling leverages all token positions — information is distributed across the entire sequence.
- Mean pooling is more robust to the specific way [CLS] was trained (NSP task vs. similarity).

Empirically, mean pooling outperforms [CLS] on almost all sentence similarity benchmarks.

### Training Data

SBERT was originally trained on NLI (Natural Language Inference) data: Stanford NLI and Multi-Genre NLI. NLI datasets contain (premise, hypothesis, label) triples where the label is entailment, contradiction, or neutral.

The training objective:

- (premise, hypothesis, entailment) → train to produce high cosine similarity.
- (premise, hypothesis, contradiction) → train to produce low cosine similarity (negative margin).
- (premise, hypothesis, neutral) → train to produce moderate similarity.

This gives the model explicit supervision on sentence-level similarity. After fine-tuning, the embedding space is calibrated so that semantically similar sentences are near each other.

Later SBERT variants (and the broader sentence-transformers library) train on much richer datasets:
- MS MARCO (query-passage pairs from web search).
- NLI datasets.
- Paraphrase databases.
- Task-specific human annotation.

---

## The Embedding Space After SBERT Training

After SBERT fine-tuning, the embedding space has a fundamentally different structure than raw BERT:

**Before (raw BERT CLS):**
- All vectors cluster in a narrow cone — high cosine similarity between everything.
- Cosine similarity is not informative.
- "heart attack" and "stock market" might have cosine similarity 0.95.

**After (SBERT):**
- Semantically related sentences cluster together.
- Semantically unrelated sentences are spread apart.
- "heart attack" and "myocardial infarction" have cosine similarity ~0.92.
- "heart attack" and "stock market" have cosine similarity ~0.15.

The embedding space now obeys the property you need for retrieval: proximity in embedding space = semantic similarity.

---

## The Speed-Accuracy Trade-off This Creates

The Siamese architecture enables efficient retrieval because sentences are encoded independently:

```
Index time: embed all corpus sentences once → store embeddings
Query time: embed query once → cosine similarity with stored embeddings
```

This is O(n) in precomputation and O(1) per query (or O(log n) with ANN index). Feasible at massive scale.

The cost is accuracy. Because query and document are encoded independently, the model cannot reason about their interaction at encoding time. A bi-encoder SBERT must represent "heart disease prevention strategies" in a vector that will be close to all relevant documents — without knowing which specific document it will be compared against. This is a fundamentally harder task than encoding them together.

This is the core trade-off:
- **Bi-encoder (SBERT):** Fast (precompute all embeddings), less accurate (independent encoding).
- **Cross-encoder:** Slow (must re-encode for every pair), more accurate (joint encoding).

The typical solution in production: use bi-encoder for first-stage retrieval (fast, finds good candidates), use cross-encoder for second-stage re-ranking (slow but accurate, scores only the shortlist).

---

## Pooling Strategies Compared

There are three main pooling strategies, and understanding their differences matters for model selection:

**CLS token pooling:**
```
output = bert_outputs[0]  # The [CLS] token at position 0
```
- Designed for classification (NSP pre-training).
- Single token must represent the entire sequence.
- Works well when the model was fine-tuned with CLS-specific objectives.
- BERT, RoBERTa default to this for classification tasks.

**Mean pooling:**
```
output = bert_outputs.mean(dim=1)  # Average over all token positions
```
- Averages information from all tokens.
- More robust representation.
- Better than CLS for most retrieval/similarity tasks empirically.
- Default in sentence-transformers library.

**Max pooling:**
```
output = bert_outputs.max(dim=1).values  # Max over all token positions
```
- Takes the maximum activation per dimension across all tokens.
- Captures the most prominent feature in each dimension.
- Used less commonly — mean pooling generally outperforms.

**Weighted mean pooling:**
- Weight each token by its attention score before averaging.
- Gives more weight to important tokens (nouns, verbs) and less to function words.
- Marginal improvement over simple mean in most benchmarks.

**In practice:** Use whatever pooling the model was trained with. Do not change pooling from what the model documentation specifies — it will degrade quality significantly.

---

## The Key Insight: Training Objective Determines Embedding Space Quality

The single most important lesson from this lesson:

**The shape and quality of the embedding space is entirely determined by the training objective.**

- Train on MLM → good token-level representations, poor sentence-level.
- Train on NSP → slightly better [CLS] for binary sentence relationship, still poor for graded similarity.
- Train with contrastive similarity objective → metrically calibrated sentence similarity space.
- Train on (query, relevant document) pairs from search logs → retrieval-optimized space.

When you evaluate an embedding model for RAG, you are evaluating: does its training objective produce a space calibrated for your specific retrieval task? A model trained on general NLI data may not be calibrated for medical document retrieval. A model trained on code search (CodeBERT) is calibrated for that task.

This is why domain fine-tuning matters so much: you are re-calibrating the embedding space for your specific (query, document) distribution.

---

## Summary

- BERT was designed for classification and span-level tasks, not sentence-level similarity retrieval.
- BERT's [CLS] space is anisotropic — all vectors cluster near each other, making cosine similarity uninformative.
- SBERT fine-tunes BERT with a Siamese architecture and contrastive similarity objective, creating a metrically calibrated embedding space.
- Mean pooling outperforms CLS pooling for retrieval because it distributes information across all token positions.
- The fundamental trade-off: bi-encoders (SBERT) enable fast independent encoding but sacrifice the accuracy of joint encoding; cross-encoders have joint encoding but cannot scale to large corpora.
- The embedding space quality is entirely determined by the training objective — this is why domain fine-tuning matters.

---

## What's Next

Lesson A.2 goes deep into how embedding models are trained — contrastive learning, triplet loss, InfoNCE, MNRL — and why hard negatives are the single most important ingredient in training a high-quality embedding model.