# Lesson 2.2 — Embedding Models: Choosing, Fine-Tuning, Matryoshka Embeddings, and Late Interaction (ColBERT)

---

## Why Embedding Models Are the Foundation of RAG Quality

Everything in your retrieval pipeline depends on the quality of your embeddings. The vector database is just a fast lookup system — it finds vectors that are close to the query vector. If the embedding model maps semantically related text to vectors that are far apart, no amount of clever retrieval logic will fix it.

Think of the embedding model as the translation layer between human language and the mathematical space where retrieval happens. If that translation is lossy or distorted, retrieval fails at the foundation.

This lesson covers:
- How embedding models work internally (enough to reason about their limitations)
- How to choose the right model for your use case
- How to fine-tune when general models are not good enough
- Matryoshka embeddings — a technique to get flexibility in vector size
- ColBERT and late interaction — a fundamentally different approach to retrieval

---

## How Embedding Models Work Internally

An embedding model is a transformer neural network that takes a sequence of tokens as input and produces a fixed-size vector as output.

The core process:

```
Input text: "maternity leave eligibility after 6 months"
    ↓
Tokenizer: [101, 6312, 2994, 17162, 2044, 1020, 2706, 102]
    ↓
Transformer layers: each token attends to all other tokens
    ↓
Pooling: combine all token representations into one vector
    ↓
Output: [0.023, -0.156, 0.891, ..., 0.042]  ← 768-dimensional vector
```

The transformer layers are the key. Each layer runs a self-attention mechanism where every token looks at every other token and updates its representation based on what it sees. After many layers, each token's representation is heavily influenced by its context. "bank" in "river bank" and "bank" in "savings bank" produce different token representations even though they are the same word.

The pooling step collapses all token representations into one vector. The most common pooling approaches:

- **CLS token pooling:** The first special [CLS] token is trained to aggregate the whole sequence meaning. BERT-based models use this.
- **Mean pooling:** Average all token embeddings. Empirically better than CLS for retrieval tasks in most benchmarks. Sentence-transformers models mostly use mean pooling.
- **Weighted mean pooling:** Weight tokens by their attention scores before averaging. Less common but can outperform simple mean pooling.

The resulting vector lives in a high-dimensional space (768, 1024, 1536, or 3072 dimensions depending on the model). The distance between two vectors in this space is a proxy for semantic similarity between the original texts.

### What "Semantic Similarity" Actually Means

This is important to understand precisely. The embedding model is trained to place semantically similar text close together in vector space. But "semantically similar" is defined by the model's training data and objective, not by some universal ground truth.

A model trained on general web text will cluster "heart attack" and "myocardial infarction" closely because they co-occur in medical web content. But it may not cluster a specific legal clause with its definition because that relationship is rare in general web text.

This is why domain matters so much, and why fine-tuning is sometimes necessary.

---

## Bi-Encoder Architecture (Standard Embedding Models)

Standard embedding models used in RAG are **bi-encoders**: they encode the query and the document chunk completely independently.

```
Query: "maternity leave policy"  →  Encoder  →  q_vector
Chunk: "Employees are eligible..."  →  Encoder  →  d_vector

Similarity = cosine(q_vector, d_vector)
```

Because query and chunk are encoded independently, you can pre-compute and store all chunk vectors at index time. At query time, you only need to encode the query (one forward pass) and then do fast vector search against pre-computed chunk vectors.

This is why bi-encoders are fast enough for production retrieval — the expensive encoding of chunks is done offline.

**The limitation:** Because query and chunk never interact during encoding, the model cannot capture fine-grained query-chunk interactions. "What is the maximum penalty for late payment?" and a chunk about "maximum payment limits" may get high cosine similarity even though they are not about the same thing — both contain "maximum" and "payment" without the model being able to reason about whether "penalty" and "limits" are what connects them.

This is exactly what cross-encoders (Lesson 3.6) and ColBERT (later in this lesson) are designed to address.

---

## How to Evaluate and Choose an Embedding Model

### The MTEB Benchmark

The Massive Text Embedding Benchmark (MTEB) is the standard evaluation framework for embedding models. It covers 56 datasets across 8 tasks including retrieval, classification, clustering, and semantic similarity.

The leaderboard (huggingface.co/spaces/mteb/leaderboard) ranks models by average performance across tasks. It is a useful starting point but has critical limitations:

- MTEB is mostly English. If your use case is multilingual, check language-specific benchmarks.
- MTEB retrieval tasks use BEIR datasets — general web/Wikipedia content. If your domain is medical, legal, financial, or highly technical, MTEB rankings may not predict performance on your data.
- MTEB does not account for inference speed or cost. A model that ranks 1st may be 10x slower than rank 3 with only 2% better performance.

**The right approach:** Use MTEB to create a shortlist of 3–5 candidate models, then evaluate them on a sample of your actual data.

### Building a Domain-Specific Evaluation Set

Create 50–100 question-passage pairs from your corpus:
- Write or collect realistic queries your users will ask.
- Manually identify the 1–3 chunks that best answer each query.
- This is your ground truth.

For each candidate model:
1. Embed all chunks and all queries.
2. For each query, retrieve top-10 chunks.
3. Compute Recall@5 and NDCG@10 against your ground truth.

The model with the best numbers on your data is the right choice, regardless of MTEB ranking.

### Key Dimensions to Compare

**Embedding dimension:**
Higher dimension = more expressive representational capacity but more storage and slower search. Common options and their implications:

| Dimension | Storage per 1M chunks | Relative search speed |
|---|---|---|
| 384 | ~1.5 GB | Fastest |
| 768 | ~3 GB | Fast |
| 1536 | ~6 GB | Moderate |
| 3072 | ~12 GB | Slower |

For most use cases, 768 or 1536 dimensions provides a good quality-cost trade-off. 3072 rarely justifies the cost unless you have a very large, diverse corpus.

**Maximum input tokens:**
If your chunks exceed the model's maximum input length, the model truncates silently. This is one of the most common silent bugs in RAG systems.

- Most older models: 512 tokens max.
- Modern models: 8192 tokens (e5-mistral, jina-v2), up to 32768 tokens (some recent models).

Always check the model's max input length and ensure your chunks are smaller. Build an assertion into your indexing pipeline.

**Instruction-tuned models:**
Some models (E5, BGE) require instruction prefixes to differentiate between query and passage encoding. Failing to add these prefixes degrades performance significantly.

E5 example:
```python
# For indexing chunks
chunk_text = "passage: " + chunk_text

# For encoding queries
query_text = "query: " + query_text
```

BGE example:
```python
# For queries only (BGE does not need instruction for passages)
query_text = "Represent this sentence for searching relevant passages: " + query_text
```

**Multilingual support:**
If your corpus includes multiple languages, you need a multilingual model:
- `multilingual-e5-large` — strong multilingual performance
- `paraphrase-multilingual-mpnet-base-v2` — lighter, 50+ languages
- `text-embedding-3-large` (OpenAI) — strong multilingual performance as a managed API

Test specifically on the language pairs you need — general multilingual performance varies widely by language.

### Popular Models Reference

| Model | Dimensions | Max Tokens | Hosting | Notes |
|---|---|---|---|---|
| text-embedding-3-large | 3072 | 8191 | API (OpenAI) | Strong general performance |
| text-embedding-3-small | 1536 | 8191 | API (OpenAI) | Good quality, lower cost |
| BAAI/bge-large-en-v1.5 | 1024 | 512 | Self-hosted | Top open-source general English |
| intfloat/e5-large-v2 | 1024 | 512 | Self-hosted | Strong retrieval, needs instruction prefix |
| intfloat/e5-mistral-7b | 4096 | 32768 | Self-hosted | Very long context, expensive to run |
| jina-embeddings-v2-base | 768 | 8192 | API or self-hosted | Long context, good for long docs |
| Cohere embed-english-v3.0 | 1024 | 512 | API (Cohere) | Strong, supports int8 quantization |
| nomic-embed-text-v1.5 | 768 | 8192 | Self-hosted | Open, long context, Matryoshka support |

---

## Fine-Tuning Embedding Models

### When General Models Are Not Enough

General embedding models are trained on broad web data. They work well for general knowledge retrieval. They underperform when:

- Your domain has specialized vocabulary not common in web text (medical terminology, legal jargon, proprietary product names, internal acronyms).
- Your query style is very different from the text style in your documents (user asks conversational questions; documents are written in formal technical language).
- Your task requires understanding very fine-grained distinctions (two similar-sounding legal clauses that mean different things).

Fine-tuning adapts the model's embedding space to your domain's specific meaning structure. It teaches the model that "MI" means "myocardial infarction" in your medical corpus, or that "SOW" means "Statement of Work" in your procurement documents.

### What Fine-Tuning Means for Embedding Models

Embedding models are fine-tuned using **contrastive learning** — you show the model pairs of (query, relevant chunk) and teach it to produce embeddings where the pair is close in vector space, while pushing apart (query, irrelevant chunk) pairs.

The training objective is typically **Multiple Negatives Ranking Loss** or **InfoNCE loss**:

```
For a batch of (query_i, positive_chunk_i) pairs:
- For each query_i, treat all other positive_chunks in the batch as negatives
- Loss = make cosine(query_i, positive_chunk_i) >> cosine(query_i, negative_chunk_j)
```

This is efficient because you get N² training signal from N pairs — every other positive in the batch serves as a hard negative.

### Building a Fine-Tuning Dataset

This is the most important step and the most work. You need (query, positive_passage) pairs, ideally also with hard negatives.

**Sources for fine-tuning data:**

1. **User query logs:** If your system is already running, actual user queries paired with the chunks they clicked on or rated positively are gold standard training data.

2. **LLM-generated synthetic data:** For each chunk, ask an LLM to generate 3–5 plausible questions that this chunk answers. This is fast and scales easily.

```python
prompt = f"""
Given the following passage from a document, generate 5 diverse questions 
that someone might ask whose answer is contained in this passage.
Return only the questions, one per line.

Passage:
{chunk_text}

Questions:
"""
```

3. **Human annotation:** For high-stakes systems, have domain experts write queries and identify relevant passages. Expensive but highest quality.

4. **Hard negative mining:** Retrieve top-K passages for each query using your current embedding model. Passages that rank highly but are not actually relevant are your hard negatives. Training on hard negatives significantly improves the model's ability to distinguish near-misses.

### Fine-Tuning with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load base model
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Prepare training data
train_examples = [
    InputExample(texts=['query text', 'relevant passage text']),
    InputExample(texts=['another query', 'another relevant passage']),
    # ... thousands more
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# Multiple Negatives Ranking Loss — treats other batch items as negatives
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='./fine-tuned-model'
)
```

### What to Expect from Fine-Tuning

On domain-specific retrieval tasks, fine-tuning typically improves Recall@5 and NDCG@10 by 10–30% over a general base model. The gains are larger when:
- Your domain is highly specialized.
- You have thousands or more training pairs.
- You use hard negatives in training.

The gains are smaller when your domain is close to general web text and your queries follow common patterns.

**Cost consideration:** Fine-tuning a large embedding model (1B+ parameters) requires significant GPU compute. For smaller models (< 300M parameters like bge-base, all-mpnet), fine-tuning on a single A100 for a few hours is practical. Use LoRA (Low-Rank Adaptation) to reduce memory requirements for larger models.

---

## Matryoshka Representation Learning (MRL)

### The Problem It Solves

Standard embedding models produce a fixed-dimension vector. If you train a 1536-dimension model, you always get 1536 dimensions. There is no way to trade quality for speed without switching to a completely different model.

This creates a dilemma: use large dimensions for quality (slow, expensive storage) or small dimensions for speed (lower quality).

Matryoshka Representation Learning (Kusupati et al., 2022) trains a single model to produce embeddings that are useful at multiple dimension levels simultaneously.

### How It Works

A Matryoshka model is trained with a special loss that enforces meaningful representations at nested truncations of the full embedding vector:

```
Full embedding: [d1, d2, d3, ..., d512, d513, ..., d1536]
                 ↑                   ↑                  ↑
          First 64 dims        First 512 dims     Full 1536 dims
          should work           should work        should work
```

The training loss is a weighted sum of losses at multiple dimensions:

```
Total Loss = λ₁ × Loss(first 64 dims) + λ₂ × Loss(first 256 dims) + λ₃ × Loss(full 1536 dims)
```

This forces the model to pack the most important information into the first dimensions. The later dimensions add refinement but are not required for a reasonable representation.

### What This Enables in Production

**Tiered retrieval:** Use small dimensions (64–256) for fast first-stage retrieval to get a large candidate set, then use full dimensions (1536) for precise re-ranking of the candidate set. Both are from the same model — no switching.

**Storage flexibility:** For very large corpora where storage is the bottleneck, store embeddings at 256 or 512 dimensions instead of 1536. Accept a small quality degradation in exchange for 3–6x storage savings and faster search.

**Adaptive quality:** For queries where fast approximate results are acceptable (autocomplete, suggestions), use small dimensions. For queries where precision matters (factual Q&A), use full dimensions.

### Models with Matryoshka Support

- `text-embedding-3-large` and `text-embedding-3-small` (OpenAI) — both support dimension truncation natively via the `dimensions` API parameter.
- `nomic-embed-text-v1.5` — open source, 768 dimensions, Matryoshka-trained.
- `BAAI/bge-m3` — multilingual, multiple retrieval modes including Matryoshka.

### Using Matryoshka Truncation with OpenAI

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(text: str, dimensions: int = 1536) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=dimensions  # Matryoshka truncation
    )
    return response.data[0].embedding

# Fast first-stage retrieval
small_query_embedding = embed(query, dimensions=256)

# Precise re-ranking
full_query_embedding = embed(query, dimensions=1536)
```

---

## Late Interaction Models — ColBERT

### The Fundamental Trade-off ColBERT Addresses

Recall the bi-encoder limitation: query and document are encoded independently. The resulting single vectors cannot capture fine-grained term interactions between query and document.

Cross-encoders fix this by encoding query and document together — but they are too slow for full-corpus search (one forward pass per (query, doc) pair).

ColBERT (Contextualized Late Interaction over BERT, Khattab & Zaharia, 2020) finds a middle ground.

### How ColBERT Works

Instead of producing one vector per document, ColBERT produces one vector **per token** in the document.

```
Standard bi-encoder:
  Document → Encoder → [single 768-dim vector]

ColBERT:
  Document → Encoder → [token₁_vector, token₂_vector, ..., tokenN_vector]
```

At query time, the query is also encoded into per-token vectors. The relevance score between a query and a document is computed using the **MaxSim** operator:

```
For each query token q_i:
    Find the maximum cosine similarity to any document token d_j

Score(query, doc) = Σᵢ max_j cosine(q_i, d_j)
```

In plain English: for every word in the query, find the best matching word in the document, and sum those best matches. A document scores high if every query term has a good match somewhere in the document.

### Why This Is Better Than Standard Bi-Encoders

Standard bi-encoder: "What is the refund policy for enterprise customers?" → one query vector. The vector must simultaneously represent "refund", "policy", "enterprise", and "customers". If the document is about enterprise billing with a paragraph on refunds, the single vectors may not be close enough.

ColBERT: each query token ("refund", "policy", "enterprise", "customers") gets its own vector. MaxSim finds the best match for "refund" in the document, for "policy", for "enterprise", etc. independently. A document that has all these concepts somewhere — even if not together — scores high.

### Why This Is Faster Than Cross-Encoders

In cross-encoders, the query and document interact during the encoding process — you cannot pre-compute document representations. Every query requires a fresh forward pass over (query, document) pairs.

In ColBERT, document token vectors are pre-computed and stored at index time (like a bi-encoder). At query time:
1. Encode the query into per-token vectors (one forward pass).
2. For each candidate document, compute MaxSim using pre-stored token vectors (fast matrix operations).

ColBERT is slower than bi-encoder retrieval (more vectors to compare), but faster than cross-encoder re-ranking (no joint encoding). It sits between the two in the speed-accuracy trade-off.

### ColBERT in Practice

**RAGatouille** is the most practical library for using ColBERT in Python:

```python
from ragatouille import RAGPretrainedModel

# Load ColBERT model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index documents
RAG.index(
    collection=["document text 1", "document text 2", ...],
    index_name="my_index",
    max_document_length=256,
    split_documents=True
)

# Search
results = RAG.search(query="your query here", k=10)
```

**Storage consideration:** ColBERT stores one vector per token per document. A document with 256 tokens generates 256 vectors instead of 1. At 128 dimensions per token vector, this is 256 × 128 × 4 bytes = 131KB per document. For a 100K document corpus, that is ~13GB — significantly more than single-vector bi-encoder storage. Storage compression (product quantization of token vectors) is important at scale.

### When to Use ColBERT

- When bi-encoder retrieval quality is insufficient but cross-encoder latency is too high.
- Domains where term-level matching matters more than holistic semantic similarity (legal, technical documentation, code).
- When you can afford the higher storage cost of per-token vectors.
- As a re-ranker on top of bi-encoder retrieval, rather than full-corpus first-stage retrieval (reduces the storage issue).

---

## Putting It Together: A Decision Framework for Embedding Models

**Step 1: Start with a strong general model.** For English: `bge-large-en-v1.5` (self-hosted) or `text-embedding-3-large` (API). Evaluate on your domain data.

**Step 2: If recall is insufficient in your domain,** fine-tune the general model on domain-specific (query, passage) pairs. Expect 10–30% improvement.

**Step 3: If storage is a bottleneck,** use a Matryoshka model and truncate to smaller dimensions. Measure the quality impact and choose the smallest dimension that meets your accuracy requirement.

**Step 4: If bi-encoder retrieval quality is still insufficient** after fine-tuning, evaluate ColBERT as a re-ranker or as the primary retrieval method.

**Step 5: If your corpus is multilingual,** use `multilingual-e5-large` or `bge-m3` and evaluate across your language pairs.

> **Interview note:** A very common question is "how would you improve retrieval quality?" The structured answer is: (1) evaluate on domain-specific data to find where the current model fails, (2) fine-tune if domain gap is the issue, (3) use Matryoshka for storage/speed flexibility, (4) add ColBERT re-ranking for precision-critical cases. Each step has a cost — justify the investment based on measured improvement.

---

## Summary

- Embedding models are bi-encoders: query and document encoded independently, similarity is cosine distance. Fast but coarse.
- Model choice matters more than retrieval algorithm. Evaluate on your actual domain data, not just MTEB.
- Key dimensions to compare: vector dimension, max input tokens, instruction prefix requirements, multilingual support.
- Fine-tune when your domain has specialized vocabulary or query-document style mismatch. Use contrastive learning on (query, relevant passage) pairs. Expect 10–30% retrieval improvement.
- Matryoshka embeddings pack information into the first dimensions, enabling dimension truncation at query time. Useful for storage and tiered retrieval without switching models.
- ColBERT stores per-token vectors and uses MaxSim for retrieval. More expressive than single-vector bi-encoders, faster than cross-encoders. Higher storage cost.
- There is no universally best embedding model. The right choice depends on domain, scale, latency budget, and storage constraints.

---

## What's Next

Lesson 2.3 covers metadata design and filtering strategies — how to design the metadata schema for your chunks, what fields are worth indexing, and how metadata filtering interacts with vector search to dramatically improve retrieval precision.