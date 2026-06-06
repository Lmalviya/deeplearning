# Lesson A.5 — Cross-Encoder Architecture and Training: Why It Is More Accurate

---

## The Fundamental Architectural Difference

To understand cross-encoders deeply, you need to see exactly what changes compared to a bi-encoder at the computational level.

**Bi-encoder (SBERT):**

```
Query: "What is the parental leave policy?"
    ↓ Tokenize: [CLS, what, is, the, parental, leave, policy, ?, SEP]
    ↓ BERT encoder (12 transformer layers)
    ↓ Mean pool all token outputs
    → q_vec: [0.23, -0.11, 0.45, ..., 0.08]  (768-dim vector)

Document: "Employees are entitled to 16 weeks..."
    ↓ Tokenize: [CLS, employees, are, entitled, to, 16, weeks, ..., SEP]
    ↓ BERT encoder (same model, independent pass)
    ↓ Mean pool all token outputs
    → d_vec: [0.19, -0.08, 0.41, ..., 0.11]  (768-dim vector)

Relevance = cosine(q_vec, d_vec)
```

Query and document NEVER see each other during encoding. The model has no information about the document when encoding the query, and no information about the query when encoding the document.

**Cross-encoder:**

```
Input: "[CLS] What is the parental leave policy? [SEP] Employees are entitled to 16 weeks... [SEP]"
    ↓ Tokenize the concatenation
    ↓ BERT encoder (12 transformer layers)
       — every query token attends to every document token
       — every document token attends to every query token
    ↓ [CLS] token output → classification head (linear layer)
    → score: 0.94  (single relevance score)
```

Query and document tokens attend to each other in every transformer layer. The encoding is fundamentally joint.

---

## Why Joint Encoding Produces Better Relevance Scores

Self-attention in a transformer allows every token to attend to every other token. In a cross-encoder, this means:

- Query tokens "look at" document tokens: "Is 'parental leave' mentioned in this document? What does it say about it?"
- Document tokens "look at" query tokens: "Is the information I contain relevant to this specific question about 'parental leave policy'?"

After 12 layers of this mutual attention, the [CLS] token representation captures a highly nuanced understanding of the query-document relationship.

**What bi-encoder misses:**

Consider the query "What is the maximum penalty for late payment?" and two documents:

1. "The late payment penalty is capped at $500 per month."
2. "Late payment fees are calculated at 1.5% monthly, compounding quarterly."

Both documents are about late payment penalties. A bi-encoder encodes both as "late payment penalty content" and the query as "maximum late payment penalty question." The cosine similarity may be similar for both — the bi-encoder cannot easily distinguish whether "capped at $500" answers "maximum penalty" better than "1.5% monthly compounding."

A cross-encoder sees the query "maximum penalty" and the document "capped at $500" together. "Capped" directly addresses "maximum." This interaction is only visible when the two texts are processed jointly.

More broadly, cross-encoders can detect:
- **Term interaction:** Does the query's "maximum" interact with the document's "capped"? (Yes — same concept)
- **Negation:** Does the document use "not" in a way that negates what the query asks about?
- **Specificity matching:** Does the document answer the specific entity the query asks about, or a different one?
- **Pragmatic inference:** Does the document's answer logically imply what the query is looking for even without exact vocabulary match?

None of these can be captured by comparing two independent vectors.

---

## The Mathematical View

**Bi-encoder:** 
```
f(q) · f(d) = relevance
```
Where f is the encoder function. Relevance is a dot product of two independently computed functions.

**Cross-encoder:**
```
g(q, d) = relevance
```
Where g processes both jointly. This is a strictly more expressive function — it can represent any function of (q, d), including ones that are NOT decomposable into f(q) · f(d).

Information theory: any relevance model that can be decomposed as f(q) · f(d) is a strict subset of the models expressible as g(q, d). Cross-encoders are universally approximators of relevance; bi-encoders are constrained to a specific factorized form.

---

## Cross-Encoder Architecture in Detail

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoder:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score (query, document) pairs.
        The model outputs logits; we use sigmoid to get [0,1] scores.
        """
        
        queries = [p[0] for p in pairs]
        documents = [p[1] for p in pairs]
        
        # Tokenize: concatenate query and document with [SEP] between them
        inputs = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Input token structure:
        # [CLS] query_token_1 ... query_token_n [SEP] doc_token_1 ... doc_token_m [SEP]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # outputs.logits shape: (batch_size, 1) for binary relevance
        # Apply sigmoid to get probability
        scores = torch.sigmoid(outputs.logits).squeeze(-1)
        
        return scores.tolist()
```

**What `AutoModelForSequenceClassification` adds to BERT:**

BERT's encoder produces contextualized representations for all tokens. `AutoModelForSequenceClassification` adds:
1. A dropout layer on top of the [CLS] representation.
2. A linear classification head: `nn.Linear(hidden_size, num_labels)`.

For cross-encoder relevance scoring, `num_labels=1` (a single relevance score) or `num_labels=2` (relevant/not-relevant binary classification). Single-label regression is more common in modern practice.

---

## Training a Cross-Encoder: The Three Loss Types

Cross-encoders can be trained with three different loss types, each encoding a different assumption about what relevance data looks like.

### Loss Type 1 — Pointwise (Binary Classification)

Each training example is a single (query, document) pair with a binary label: relevant (1) or not relevant (0).

```python
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

# Training data: (query, document, label) triples
# label = 1 for relevant, 0 for irrelevant
training_examples = [
    InputExample(texts=["query text", "relevant document"], label=1.0),
    InputExample(texts=["query text", "irrelevant document"], label=0.0),
    # ... thousands more
]

model = CrossEncoder("bert-base-uncased", num_labels=1)

# Binary cross-entropy loss
# model predicts probability of relevance
# loss = -label * log(prediction) - (1-label) * log(1-prediction)
```

**Advantage:** Simple. Works with any dataset that has binary relevance labels.

**Disadvantage:** Does not directly optimize ranking — getting 0.6 for a relevant doc and 0.4 for an irrelevant doc gives the same loss as 0.9 vs. 0.1. The model learns to predict labels, not to rank correctly.

### Loss Type 2 — Pairwise (BPR / Margin Loss)

Each training example is a triple: (query, relevant_doc, irrelevant_doc). The loss directly optimizes that the relevant document scores higher than the irrelevant one.

```
L = max(0, score(query, irrelevant) - score(query, relevant) + margin)
```

```python
def pairwise_ranking_loss(
    score_positive: torch.Tensor,   # Score for relevant document
    score_negative: torch.Tensor,   # Score for irrelevant document
    margin: float = 0.0
) -> torch.Tensor:
    """
    Loss = 0 when positive scores higher than negative by at least margin.
    Loss > 0 when negative scores too close or higher than positive.
    """
    loss = torch.clamp(score_negative - score_positive + margin, min=0.0)
    return loss.mean()
```

**Why pairwise is better than pointwise:**

Pairwise directly optimizes the thing you care about: the ranking order. "The relevant document must score higher than the irrelevant document." This is more directly aligned with how you use the cross-encoder at inference time (to rank candidates).

The MS MARCO dataset was used to train most production cross-encoders with pairwise loss. MS MARCO provides (query, positive_passage, hard_negative_passage) triples — directly suited for pairwise training.

### Loss Type 3 — Listwise (LambdaRank / LambdaLoss)

Instead of optimizing pairs, optimize the full ranked list at once. The loss is computed over a set of documents for each query and directly optimizes a ranking metric like NDCG.

```python
def listwise_loss(
    scores: torch.Tensor,       # (n_docs,) predicted scores
    relevance: torch.Tensor,    # (n_docs,) ground truth relevance grades
    metric: str = "ndcg"
) -> torch.Tensor:
    """
    Listwise loss: directly optimize ranking metric.
    """
    # LambdaRank: scale pairwise gradients by NDCG delta
    # When the swap of two documents would improve NDCG a lot,
    # apply a stronger gradient to correct the ranking.
    pass  # Implementation is complex; shown conceptually
```

**Why listwise is theoretically best but rarely used in practice:**

Listwise loss requires knowing the relevance grade of ALL documents for a given query. Most datasets only have sparse labels (we know a few relevant docs, not grades for everything). Pairwise works with sparse labels. Listwise needs dense grades.

For RAG re-ranking, pairwise is the standard. You have (query, positive, hard_negative) triples from mining. This maps directly to pairwise loss.

---

## Training Data for Cross-Encoders

### MS MARCO (The Standard)

MS MARCO is why most cross-encoder benchmarks look the way they do. It contains:
- 500K training queries from Bing search logs.
- Each query has one positive passage (the one the human annotator marked as relevant).
- Hard negatives are mined using BM25 retrieval (top BM25 results that are NOT the positive).

```python
# MS MARCO training triple structure:
{
    "query": "what causes thunder",
    "positive": "Thunder is caused by the rapid expansion of air...",
    "hard_negative": "Thunder is an atmospheric phenomenon that accompanies..."
    # (close to query but incorrect — BM25 retrieved it but it's not the answer)
}
```

Models trained on MS MARCO are good general retrievers for English web-style queries. They are the right starting point but may need fine-tuning for specialized domains.

### Hard Negative Mining for Domain Fine-Tuning

For domain-specific cross-encoder training, you need domain-specific hard negatives:

```python
async def mine_cross_encoder_training_data(
    queries: list[str],
    positive_chunks: list[dict],  # {query_id: chunk_text}
    retriever,                    # Your current retrieval system
    n_hard_negatives: int = 5
) -> list[dict]:
    """
    Mine hard negatives for cross-encoder training using current retrieval.
    """
    
    training_examples = []
    
    for query in queries:
        # Retrieve top-K candidates using current system
        candidates = await retriever.retrieve(query, k=20)
        
        positive = positive_chunks[query]
        positive_ids = {positive["chunk_id"]}
        
        # Hard negatives: retrieved but not the true positive
        hard_negatives = [
            c for c in candidates
            if c["chunk_id"] not in positive_ids
        ][:n_hard_negatives]
        
        for neg in hard_negatives:
            training_examples.append({
                "query": query,
                "positive": positive["text"],
                "negative": neg["text"],
                "label_positive": 1.0,
                "label_negative": 0.0
            })
    
    return training_examples
```

**Why retrieval-mined hard negatives are the best:**

The retrieval system finds chunks that look relevant but are not. These are exactly the cases where the cross-encoder must work hardest — distinguishing near-misses from true positives. Training on these cases directly improves the cross-encoder's performance on the actual task.

---

## Cross-Encoder Score Interpretation

A cross-encoder score is NOT a probability in the traditional sense, even though sigmoid(logit) maps it to [0,1].

```python
# What the score means:
# High score (e.g., 0.95): "This document is very likely relevant to this query"
# Low score (e.g., 0.05): "This document is very likely not relevant"
# Mid score (e.g., 0.50): "Uncertain — could be marginally relevant"

# What you should use it for:
# RANKING: Sort candidates by score (this is the primary use case)
# NOT THRESHOLDING: The scores are not calibrated for a specific threshold
#                    A score of 0.4 on model A may be "relevant" 
#                    while 0.4 on model B is "not relevant"
```

**Important:** Cross-encoder scores from different models are NOT comparable. A MiniLM-L-6 score of 0.7 and a ELECTRA-base score of 0.7 do not mean the same thing. Each model has its own score distribution. Use scores within one model for ranking; do not compare scores across models.

**Calibrating a threshold:** If you want to use the cross-encoder score as a confidence gate (e.g., only pass chunks with score > T to the LLM), calibrate T on your evaluation set:

```python
def calibrate_threshold(
    cross_encoder,
    eval_queries: list[str],
    eval_relevant_chunks: list[list[str]],
    eval_irrelevant_chunks: list[list[str]],
    target_precision: float = 0.90
) -> float:
    """
    Find the minimum score threshold that achieves target_precision.
    """
    import numpy as np
    
    positive_scores = []
    negative_scores = []
    
    for query, positives, negatives in zip(eval_queries, eval_relevant_chunks, eval_irrelevant_chunks):
        for pos in positives[:3]:
            score = cross_encoder.predict([(query, pos)])[0]
            positive_scores.append(score)
        
        for neg in negatives[:3]:
            score = cross_encoder.predict([(query, neg)])[0]
            negative_scores.append(score)
    
    # Find threshold where precision = target_precision
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    for threshold in sorted(thresholds, reverse=True):
        all_above_threshold = (
            [s for s in positive_scores if s >= threshold] +
            [s for s in negative_scores if s >= threshold]
        )
        n_positives_above = sum(1 for s in positive_scores if s >= threshold)
        
        if all_above_threshold:
            precision = n_positives_above / len(all_above_threshold)
            if precision >= target_precision:
                return threshold
    
    return 0.5  # Fallback
```

---

## The Speed-Accuracy Trade-off: Quantified

The reason bi-encoder + cross-encoder is used as a two-stage pipeline comes down to simple arithmetic.

**Setup:** 1M document chunks, k=10 final results needed.

**Bi-encoder only:**
- Index time: embed 1M chunks once (one-time cost).
- Query time: embed query (1 pass), ANN search (O(log N)), return top-10.
- Total query latency: ~30ms.

**Cross-encoder only:**
- Query time: 1M × (query, chunk) forward passes.
- At 1ms per forward pass (fast GPU): 1,000 seconds per query.
- At 10ms per forward pass (CPU): 10,000 seconds per query.
- Completely infeasible.

**Two-stage:**
- Stage 1 (bi-encoder): 30ms, returns top-100 candidates.
- Stage 2 (cross-encoder): 100 forward passes × 1ms/pass = 100ms.
- Total: ~130ms. Feasible.

The cross-encoder's accuracy advantage is captured without paying its linear cost:

```
Retrieval pipeline cost = O(log N) for bi-encoder + O(k_candidates) for cross-encoder
                        ≠ O(N) for cross-encoder alone
```

This is the fundamental insight behind the two-stage retrieval architecture.

---

## Comparing Cross-Encoder Architectures

### MiniLM vs. BERT vs. ELECTRA

**MiniLM cross-encoders** (MiniLM-L-6-v2, MiniLM-L-12-v2):
- Distilled from larger BERT models — smaller, faster, nearly as accurate.
- L-6: 6 transformer layers, ~22M parameters. ~5ms per pair on GPU.
- L-12: 12 layers, ~33M parameters. ~10ms per pair on GPU.
- Good choice for production when latency matters.

**BERT-base cross-encoders:**
- 12 layers, 110M parameters. ~20ms per pair on GPU.
- More accurate than MiniLM but 4× slower.

**ELECTRA-base cross-encoders:**
- Replaced MLM pre-training with Replaced Token Detection — more efficient training signal.
- Similar size to BERT but often better quality.
- Used in `ms-marco-electra-base` — competitive quality with BERT at slightly better efficiency.

**DeBERTa cross-encoders:**
- Disentangled attention (position and content attention computed separately).
- Strongest cross-encoder architecture available.
- 400M+ parameters — high quality but slow. Best for offline or high-latency-tolerance applications.

---

## Summary

- A cross-encoder concatenates query and document into a single input, allowing every query token to attend to every document token in every transformer layer. This joint encoding is what makes cross-encoders more accurate than bi-encoders.
- Bi-encoders cannot capture term interactions, negation, or pragmatic inference between query and document because they encode independently. Cross-encoders can capture all of these.
- Three training loss types: pointwise (binary label per pair), pairwise (triplet with margin loss — the standard), listwise (full ranking metric optimization — rarely used due to data requirements).
- Hard negative mining from retrieval system output is the highest-leverage training data source for domain cross-encoder fine-tuning.
- Cross-encoder scores are for ranking within one model, not for thresholding or comparing across models. Calibrate thresholds on your evaluation set.
- The two-stage architecture (bi-encoder for O(log N) candidate retrieval + cross-encoder for O(k) precise re-ranking) makes cross-encoder accuracy achievable without cross-encoder cost.

---

## What's Next

Lesson A.6 covers the practical considerations for cross-encoders in production: the 512-token truncation problem and its solutions, self-hosted vs. Cohere Rerank trade-offs, fine-tuning workflow, and when to choose cross-encoder vs. ColBERT.