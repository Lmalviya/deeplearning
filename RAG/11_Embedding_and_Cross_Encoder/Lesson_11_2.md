# Lesson A.2 — Contrastive Learning: How Embedding Models Are Trained

---

## The Core Training Problem

You want an embedding model that places semantically similar sentences near each other in vector space. The question is: how do you train a neural network to produce this property?

The answer is contrastive learning — a family of training objectives that directly optimize the geometry of the embedding space by showing the model examples of what "similar" and "different" mean.

The key difference from supervised classification training: you are not training the model to predict a label. You are training it to arrange vectors in space.

---

## The Intuition: Pull and Push

All contrastive learning objectives share the same core intuition:

**Pull similar examples together** in embedding space.
**Push dissimilar examples apart** in embedding space.

You do this by constructing pairs or triplets of examples with known similarity relationships, computing embeddings, and defining a loss that penalizes the model when similar examples are far apart and when dissimilar examples are too close.

The model learns to arrange its embedding space to minimize this loss — and the resulting space has the property you want: proximity = semantic similarity.

---

## Building Block: What Are Positive and Negative Pairs?

Every contrastive training setup requires defining what "similar" and "different" mean for your task.

**Positive pair (anchor, positive):** Two examples that should be embedded close together.
- Query: "What is the parental leave policy?" → Positive: the relevant policy chunk
- Sentence: "The cat sat on the mat" → Positive: "A feline rested on the rug" (paraphrase)
- Question: "What is photosynthesis?" → Positive: "Photosynthesis is the process by which..."

**Negative pair (anchor, negative):** Two examples that should be embedded far apart.
- Query: "What is the parental leave policy?" → Negative: a chunk about expense reimbursement
- Sentence: "The cat sat on the mat" → Negative: "Revenue grew 15% in Q3"

The quality of your positive and negative pairs determines the quality of the trained embedding model more than almost any other factor. This is why hard negatives (discussed later) are so critical.

---

## Training Objective 1 — Triplet Loss

Triplet loss is the most intuitive contrastive objective. Each training example is a triplet:

**(anchor, positive, negative)**

The loss is:

```
L = max(0, distance(anchor, positive) - distance(anchor, negative) + margin)
```

In cosine similarity terms (where higher = more similar):

```
L = max(0, sim(anchor, negative) - sim(anchor, positive) + margin)
```

Where `margin` is a hyperparameter (typically 0.5).

**Interpretation:** The loss is zero when the positive is at least `margin` more similar to the anchor than the negative. When the negative is too close (or closer) to the anchor than the positive, the loss is positive and the model gets a gradient that pushes them apart.

```python
import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.5):
    """
    anchor, positive, negative: normalized embedding vectors (unit norm)
    """
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    
    loss = torch.clamp(neg_sim - pos_sim + margin, min=0.0)
    return loss.mean()
```

**Where training data comes from for triplet loss:**
- NLI datasets: (premise, entailment, contradiction) → natural triplets.
- Paraphrase corpora: (sentence, paraphrase, random sentence).
- Search logs: (query, clicked document, not-clicked document).
- Human-annotated pairs with similarity scores.

**Limitation of triplet loss:** It only uses one negative per anchor-positive pair. This is sample-inefficient. Modern methods use many negatives per example.

---

## Training Objective 2 — InfoNCE / NT-Xent (Multiple Negatives)

InfoNCE (Noise Contrastive Estimation) and its variant NT-Xent (Normalized Temperature Cross-Entropy) are more powerful than triplet loss because they use all other examples in the batch as negatives.

The setup: given a batch of N (anchor, positive) pairs, treat the other N-1 positives in the batch as negatives for each anchor.

```
For anchor_i, the positive is positive_i.
The negatives are: positive_j for all j ≠ i (in-batch negatives).
```

The loss for one anchor:

```
L_i = -log(exp(sim(anchor_i, positive_i) / τ) / Σ_j exp(sim(anchor_i, positive_j) / τ))
```

Where `τ` (temperature) is a hyperparameter (typically 0.05-0.1).

This is a softmax over all positive candidates in the batch. The loss pushes the model to rank the true positive higher than all other positives used as negatives.

```python
import torch
import torch.nn.functional as F

def info_nce_loss(anchors, positives, temperature=0.07):
    """
    anchors: (N, D) normalized anchor embeddings
    positives: (N, D) normalized positive embeddings
    
    Uses all N positives as negatives for each anchor.
    """
    # Compute all pairwise similarities: (N, N) matrix
    similarity_matrix = torch.matmul(anchors, positives.T) / temperature
    
    # Labels: diagonal elements are the true positives
    labels = torch.arange(similarity_matrix.shape[0])
    
    # Cross-entropy loss: maximize diagonal, minimize off-diagonal
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
```

**Why this is more powerful than triplet loss:**

With a batch of N=64 pairs, each anchor gets 63 negatives per training step instead of 1. This gives 63× more signal per forward pass. Large batches create harder negatives on average — some in-batch negatives will be semantically close to the anchor, creating automatically difficult negative examples.

The downside: the quality of negatives depends on what happens to be in the batch. Random in-batch negatives are often easy — completely unrelated to the anchor. You can do better.

---

## Training Objective 3 — MNRL (Multiple Negatives Ranking Loss)

MNRL is the standard training objective in the sentence-transformers library and the most common choice for embedding model fine-tuning in practice.

It is essentially InfoNCE applied specifically to (query, positive_passage) pairs:

```python
from sentence_transformers import losses

# In sentence-transformers:
train_loss = losses.MultipleNegativesRankingLoss(model)
```

The loss treats every other positive in the batch as a negative for each anchor. This is exactly InfoNCE with temperature=1.0 by default, but the library allows customization.

**What makes MNRL practical:**

1. You only need (anchor, positive) pairs — no explicit negatives needed. Negatives come automatically from the batch.
2. Simple to construct training data: (query, relevant document) pairs from user clicks, (sentence, paraphrase) pairs from existing datasets, (question, answer) pairs from QA datasets.
3. Scales naturally: larger batches → more negatives → better training signal.

**The batch size insight:** With MNRL, batch size is a critical hyperparameter. Batch size 16 → 15 negatives per anchor. Batch size 256 → 255 negatives. Larger batch = harder negatives = better trained model. This is why embedding model training typically uses large batch sizes (256-2048) and benefits from multi-GPU training to increase effective batch size.

---

## Hard Negatives: The Most Important Ingredient

Easy negatives (random, completely unrelated examples) teach the model almost nothing useful. The model trivially learns to push these apart — they are far apart in semantic space already.

The examples that actually improve the model are **hard negatives** — examples that are semantically similar to the anchor but are NOT the correct positive. These are the challenging cases where the model currently makes mistakes.

**What hard negatives look like:**

For the query "What is the maternity leave policy?":
- Easy negative: "The weather forecast for tomorrow is sunny."
- Hard negative: "The paternity leave policy allows 2 weeks for fathers."
- Hard negative: "Our family care leave policy covers care for sick relatives."
- Hard negative: "Maternity leave begins no earlier than 4 weeks before the expected birth date." (from a different year's policy that contradicts the current one)

The hard negatives are semantically related to the query but are NOT the correct answer. The model must learn to distinguish them from the true positive — which requires understanding fine-grained semantic differences, not just topic similarity.

**Hard negative mining:**

```python
async def mine_hard_negatives(
    queries: list[str],
    positive_chunks: list[str],
    all_chunks: list[str],
    embedding_model,
    n_hard_negatives: int = 5,
    similarity_lower_bound: float = 0.5,
    similarity_upper_bound: float = 0.9
) -> list[dict]:
    """
    Mine hard negatives: chunks that are similar to the query
    but are NOT the true positive.
    
    Hard negatives have similarity in (lower_bound, upper_bound) to the query.
    Too low = easy negative (not useful).
    Too high = may actually be relevant (risky as negative).
    """
    
    training_examples = []
    
    query_embeddings = embedding_model.encode(queries, normalize_embeddings=True)
    chunk_embeddings = embedding_model.encode(all_chunks, normalize_embeddings=True)
    
    import numpy as np
    
    for i, (query, positive, query_emb) in enumerate(zip(queries, positive_chunks, query_embeddings)):
        # Compute similarity to all chunks
        similarities = chunk_embeddings @ query_emb  # (N,) similarity scores
        
        # Find hard negatives: similar to query but not the true positive
        positive_idx = all_chunks.index(positive)
        
        hard_negative_indices = []
        
        for j, sim in enumerate(similarities):
            if j == positive_idx:
                continue  # Skip the true positive
            
            if similarity_lower_bound <= sim <= similarity_upper_bound:
                hard_negative_indices.append((j, float(sim)))
        
        # Sort by similarity (descending) — hardest negatives first
        hard_negative_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n_hard_negatives
        hard_negatives = [all_chunks[idx] for idx, _ in hard_negative_indices[:n_hard_negatives]]
        
        training_examples.append({
            "query": query,
            "positive": positive,
            "hard_negatives": hard_negatives
        })
    
    return training_examples
```

**Why hard negative quality determines model quality:**

Consider training without hard negatives vs. with:

- **Without hard negatives:** Model learns "maternity leave" is not about "stock prices." Easy. Not very useful.
- **With hard negatives:** Model must learn "maternity leave policy for full-time employees" vs. "maternity leave policy for part-time employees" are different. Hard. Very useful.

The latter forces the model to learn fine-grained distinctions within the same topic area — exactly the kind of discrimination that separates a good retrieval embedding from a mediocre one.

Research consistently shows that models trained with hard negatives outperform those trained with random negatives by 10-20% on retrieval benchmarks, even with the same architecture and training data size.

---

## Training Data Construction: Practical Patterns

### Pattern 1 — From Search Logs (Best Quality, Highest Signal)

```python
# If you have user search logs with click data
training_pairs = []

for log_entry in search_logs:
    query = log_entry["query"]
    clicked_doc = log_entry["clicked_document"]
    not_clicked_docs = log_entry["other_results_not_clicked"]  # Natural hard negatives!
    
    training_pairs.append({
        "query": query,
        "positive": clicked_doc,
        "hard_negatives": not_clicked_docs[:3]  # Retrieved but not clicked = hard negatives
    })
```

Search logs are gold because the "not clicked but retrieved" documents are automatically hard negatives — they were surfaced by the retrieval system as relevant but the user chose something else.

### Pattern 2 — LLM-Generated Synthetic Pairs

```python
async def generate_training_pair(chunk_text: str, llm_client) -> dict:
    """Generate (query, positive) pair from a corpus chunk."""
    
    prompt = f"""Given this document passage, generate one search query that this passage would be the best answer to.
The query should be phrased naturally, as a user would type it.
Also generate 2 plausible but incorrect queries that sound similar but this passage does NOT answer.

Passage: {chunk_text}

Return JSON:
{{
    "correct_query": "...",
    "misleading_queries": ["...", "..."]
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
        temperature=0.5
    )
    
    import json
    result = json.loads(response.choices[0].message.content)
    
    return {
        "query": result["correct_query"],
        "positive": chunk_text,
        # Note: "misleading_queries" → find their true positive chunk for hard negatives
    }
```

### Pattern 3 — Cross-Encoder Teacher for Silver Labels

Use a cross-encoder (more accurate) to generate relevance scores for (query, chunk) pairs. Use these scores to select positives and hard negatives for bi-encoder training. This is called knowledge distillation.

```python
# Cross-encoder generates soft labels
# Bi-encoder is trained to match these labels
# Result: bi-encoder learns from a more accurate teacher
```

---

## The Complete Training Loop

```python
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses
from torch.utils.data import DataLoader

def train_embedding_model(
    base_model_name: str,
    training_examples: list[dict],  # [{query, positive, hard_negatives}]
    output_path: str,
    epochs: int = 3,
    batch_size: int = 64,
    warmup_steps: int = 100
):
    """
    Fine-tune an embedding model using MNRL with hard negatives.
    """
    
    model = SentenceTransformer(base_model_name)
    
    # Convert to InputExample format
    # With hard negatives: [anchor, positive, hard_neg_1, hard_neg_2, ...]
    train_data = []
    for example in training_examples:
        texts = [example["query"], example["positive"]]
        texts.extend(example.get("hard_negatives", []))
        train_data.append(InputExample(texts=texts))
    
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    # MNRL treats everything after the positive as additional negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_save_steps=500,
        checkpoint_path=output_path + "/checkpoints"
    )
    
    return model
```

---

## Temperature: The Critical Hyperparameter

The temperature `τ` in InfoNCE controls how "sharp" or "soft" the similarity distribution is.

**Low temperature (τ = 0.05):** The softmax is very sharp. The model must rank the positive very confidently above all negatives. Creates a harder optimization problem but produces more discriminative embeddings. Risk: training instability if hard negatives are too difficult.

**High temperature (τ = 0.5):** The softmax is softer. The model gets credit for partially ranking the positive above negatives. Easier optimization but produces less discriminative embeddings.

**Typical choice:** τ = 0.05-0.1 for embedding model training. Smaller models may need slightly higher temperature to train stably.

---

## What Good Training Looks Like

After training, you can verify the embedding space has the right structure:

```python
def verify_embedding_quality(model, test_pairs: list[dict]) -> dict:
    """
    Verify that similar pairs have high similarity and
    dissimilar pairs have low similarity.
    """
    import numpy as np
    
    positive_sims = []
    negative_sims = []
    
    for pair in test_pairs:
        anchor_emb = model.encode(pair["anchor"], normalize_embeddings=True)
        pos_emb = model.encode(pair["positive"], normalize_embeddings=True)
        neg_emb = model.encode(pair["negative"], normalize_embeddings=True)
        
        positive_sims.append(float(np.dot(anchor_emb, pos_emb)))
        negative_sims.append(float(np.dot(anchor_emb, neg_emb)))
    
    return {
        "avg_positive_similarity": np.mean(positive_sims),
        "avg_negative_similarity": np.mean(negative_sims),
        "separation": np.mean(positive_sims) - np.mean(negative_sims),
        "good_separation": np.mean(positive_sims) - np.mean(negative_sims) > 0.3
    }
```

A well-trained model should show:
- Average positive similarity: > 0.80
- Average negative similarity: < 0.30
- Separation (positive - negative): > 0.50

Hard negatives will have similarity between 0.40-0.70 even after training — that is expected. The model cannot perfectly separate everything. What matters is that the true positives rank clearly above hard negatives in the similarity ordering.

---

## Summary

- Contrastive learning directly optimizes the geometry of the embedding space by pulling similar examples together and pushing dissimilar ones apart.
- Triplet loss: one positive, one negative per anchor. Simple but sample-inefficient.
- InfoNCE / MNRL: one positive, all other batch items as negatives. More efficient, standard in modern practice.
- Hard negatives are semantically similar to the anchor but not the true positive. Training on hard negatives forces the model to learn fine-grained distinctions, which is what separates good retrieval models from mediocre ones.
- Hard negative mining: use current model to retrieve top-K candidates, use those not marked as relevant as hard negatives.
- Batch size is a critical hyperparameter: larger batches → more negatives → better training signal.
- Temperature controls sharpness: 0.05-0.1 is typical for embedding training.

---

## What's Next

Lesson A.3 covers asymmetric retrieval and instruction-tuned embeddings — why E5 and BGE require different instructions for query vs. document encoding, what mechanically happens without the prefix, and how to ensure you are applying them correctly.