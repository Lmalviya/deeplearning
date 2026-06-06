# Lesson A.3 — Asymmetric Retrieval and Instruction-Tuned Embeddings

---

## Symmetric vs. Asymmetric Retrieval

Most introductions to embeddings assume the query and the document are encoded the same way. This is called **symmetric retrieval** — both use identical encoding, and the embedding space is the same for both.

Symmetric retrieval makes sense when the query and document have the same form:
- Finding duplicate questions in a Q&A forum (both are questions).
- Finding paraphrases (both are full sentences of similar length and style).
- Finding similar documents (both are long-form documents).

It breaks down when the query and document have fundamentally different forms — which is almost always the case in RAG:

- Query: "how do employees take parental leave?" (8 words, conversational, question form)
- Document: "Employees are entitled to 16 weeks of fully paid parental leave, commencing no earlier than 4 weeks prior to the expected birth date. Eligibility requires 12 months of continuous employment..." (40+ words, formal prose, declarative statements)

These two texts are semantically identical in intent but lexically and structurally very different. A model trained purely on symmetric pairs may not bridge this gap well.

**Asymmetric retrieval** explicitly acknowledges this difference and trains the model with separate representations for queries and documents.

---

## The Query-Document Distribution Gap

To understand why this matters mechanically, consider what the embedding model must do:

In symmetric retrieval, the model maps both query and document to the same latent space. Since the model has seen many (short question, short question) pairs and (long document, long document) pairs during training, it knows how to compare within each format. But the (short conversational question, long formal document) gap is a different distribution.

The gap has three components:

**1. Length mismatch:** A 5-word query must produce a vector that lands near a 500-token chunk. The model must compress all relevant meaning from 5 words into a vector that has the same orientation as a much richer representation of the same concept.

**2. Style mismatch:** "how do I reset my password" vs. "To reset your account password: 1. Navigate to Settings > Security..." — same intent, completely different vocabulary and structure.

**3. Intent mismatch:** A query expresses an information need. A document expresses information. The model must map an information need to information — which is not the same as mapping information to information.

Without explicit training for asymmetric retrieval, the model's embedding space is not optimally calibrated for the query-document retrieval scenario.

---

## How Asymmetric Training Works

The solution is to train on (query, relevant_document) pairs where the query and document are genuinely different in form. This teaches the model that certain query patterns correspond to certain document patterns.

MS MARCO (Microsoft MAchine Reading COmprehension) is the most important dataset for this. It contains 500,000 genuine web search queries paired with relevant passages from web documents. Real users wrote real queries; the passages are the actual relevant documents. This is the canonical asymmetric retrieval training set.

```
Query: "what causes thunder"
Positive passage: "Thunder is caused by the rapid expansion of air surrounding 
the path of a lightning bolt. The lightning bolt superheats the air to about 
30,000 Kelvin — five times hotter than the sun's surface..."
```

A model trained on MS MARCO learns to map short natural language questions to longer, denser document passages — the core asymmetric retrieval task.

---

## Instruction-Tuned Embeddings: E5 and BGE

Some embedding models take asymmetric retrieval further by using explicit instruction prefixes that tell the model what type of text it is encoding. The two most prominent are E5 (Wang et al., Microsoft, 2022) and BGE (BAAI, 2023).

### E5 (Text Embeddings by Weakly-Supervised Contrastive Pre-training)

E5 was trained on a curated dataset of (text, text) pairs from the web with weak supervision labels. The key architectural decision: all inputs are prefixed with an instruction:

- For queries: `"query: {query_text}"`
- For documents: `"passage: {passage_text}"`

**Why E5 uses these prefixes:**

During E5's training, every input was prepended with either `"query: "` or `"passage: "`. The model learned that:
- Text starting with `"query: "` is an information need (encode for retrieval).
- Text starting with `"passage: "` is a document (encode for being retrieved).

The prefix becomes a routing signal in the model's attention mechanism. The [query] prefix activates a different attentional pattern than the [passage] prefix — essentially switching between two encoding "modes" within the same model.

**What happens without the prefix:**

Without the prefix, E5 treats the input as an ambiguous text of unknown type. Its embedding falls somewhere between "query mode" and "passage mode." The resulting vector is suboptimal for retrieval — it has slightly different characteristics than a properly-prefixed query embedding.

Empirical evidence from the E5 paper: removing the instruction prefix reduces BEIR benchmark recall@100 by 3-8% depending on the dataset. For domain-specific retrieval where the margin is already tight, this is a significant degradation.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")

# CORRECT usage
query = "query: What is the parental leave policy?"
document = "passage: Employees are entitled to 16 weeks of parental leave..."

query_embedding = model.encode(query, normalize_embeddings=True)
doc_embedding = model.encode(document, normalize_embeddings=True)

# WRONG usage (missing prefixes — will degrade retrieval quality)
query_embedding_wrong = model.encode("What is the parental leave policy?", normalize_embeddings=True)
```

### BGE (BAAI General Embedding)

BGE uses a slightly different instruction approach. The instruction is added only to the query, not the document:

- For queries: `"Represent this sentence for searching relevant passages: {query_text}"`
- For documents: `"{document_text}"` (no prefix)

This asymmetry reflects a design choice: the document encoding should be as faithful to the original text as possible, while the query encoding needs a richer signal about what it is.

```python
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# BGE: instruction only on query side
query = "Represent this sentence for searching relevant passages: What is the parental leave policy?"
document = "Employees are entitled to 16 weeks of parental leave..."  # No prefix

query_embedding = model.encode(query, normalize_embeddings=True)
doc_embedding = model.encode(document, normalize_embeddings=True)
```

**Why BGE adds instruction only to queries:**

BGE's training used a large corpus of unlabeled documents for pre-training (they want the document encoder to be as general as possible) and instruction-following pairs for fine-tuning (they want the query encoder to activate a specific "retrieval search" mode). Keeping documents free of prefix allows the same model to encode any document well, while the query prefix specifically activates the retrieval-optimized mode.

---

## What the Prefix Mechanically Does in the Transformer

Understanding the mechanism makes the importance of prefixes concrete rather than just empirical.

A transformer processes input tokens through layers of self-attention. Each token attends to all other tokens and updates its representation based on what it sees.

When you prepend `"query: "` to the input:

1. The tokens `["query", ":"]` become the first two tokens in the sequence.
2. All subsequent tokens (the actual query text) attend to these prefix tokens in every layer.
3. The prefix tokens provide a consistent "context signal" that all other tokens incorporate.
4. After many layers of this, the pooled representation reflects both the query content AND the "this is a query" signal from the prefix.

This is analogous to a task-specific input format that the model learned to condition on during training. It is similar to how instruction-following LLMs use system prompts — the system prompt colors the interpretation of everything that follows.

Without the prefix, the model processes the query without the "this is a retrieval query" signal. The embeddings are still computed, but they lack the consistent conditioning that E5 was trained to expect.

---

## The Practical Implication: Never Forget the Prefix

The most common bug in production RAG systems using E5 or BGE: applying the prefix inconsistently.

**Scenario:** Developer tests the system during development. They add the prefix to queries (because they read the README). Documents are indexed without the prefix (because they forgot or assumed documents do not need it for E5). The system appears to work — recall is 0.82.

Then they upgrade to E5-large to get better quality. Recall stays at 0.82. They investigate and find that documents were indexed without `"passage: "` prefix. They re-index with the prefix. Recall jumps to 0.89.

The bug was always there but was masked by the general quality of the model.

**Prevention:**

```python
class E5Embedder:
    """
    Wrapper that enforces correct prefix usage for E5 models.
    """
    
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with the required 'query: ' prefix."""
        prefixed = f"query: {query}"
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()
    
    def embed_document(self, text: str) -> list[float]:
        """Embed a document passage with the required 'passage: ' prefix."""
        prefixed = f"passage: {text}"
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()
    
    def embed_queries_batch(self, queries: list[str]) -> list[list[float]]:
        prefixed = [f"query: {q}" for q in queries]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()
    
    def embed_documents_batch(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()


class BGEEmbedder:
    """
    Wrapper that enforces correct prefix usage for BGE models.
    BGE uses instruction prefix on queries only, not documents.
    """
    
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, query: str) -> list[float]:
        prefixed = self.QUERY_INSTRUCTION + query
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()
    
    def embed_document(self, text: str) -> list[float]:
        # BGE does NOT use instruction prefix for documents
        return self.model.encode(text, normalize_embeddings=True).tolist()
```

**Add a test that validates prefix usage:**

```python
def test_prefix_consistency(embedder, sample_query: str, sample_doc: str):
    """
    Verify that query and document embeddings have correct prefix behavior.
    Correct: query embedding should be more similar to relevant doc than
             to a randomly-encoded version of itself.
    """
    import numpy as np
    
    query_emb = np.array(embedder.embed_query(sample_query))
    doc_emb = np.array(embedder.embed_document(sample_doc))
    
    # Similarity between correctly-encoded query and relevant document
    correct_sim = float(np.dot(query_emb, doc_emb))
    
    # Encode query as if it were a document (wrong prefix)
    wrong_query_emb = np.array(embedder.embed_document(sample_query))
    wrong_sim = float(np.dot(wrong_query_emb, doc_emb))
    
    print(f"Correct (query prefix) similarity: {correct_sim:.3f}")
    print(f"Wrong (doc prefix on query) similarity: {wrong_sim:.3f}")
    
    assert correct_sim > wrong_sim - 0.02, (
        "Correct prefix should produce higher or equal similarity. "
        "Check that prefix is being applied."
    )
```

---

## Other Instruction-Tuned Models

The instruction-prefix pattern has expanded beyond E5 and BGE. Newer models extend this with much longer, task-specific instructions:

**E5-mistral-7b-instruct (2024):**

```python
# Supports long, task-specific instructions
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"

# Example usage
query = get_detailed_instruct(
    task_description="Given a web search query, retrieve relevant passages that answer the query",
    query="What is the capital of France?"
)
```

**GTE-Qwen2 (Alibaba):**

Uses natural language instructions describing the retrieval task in detail. The model has seen many different instruction types during training and can adapt its encoding mode accordingly.

**The trend:** Instruction following is moving from fixed prefixes ("query:", "passage:") to flexible natural language instructions describing the specific retrieval task. This allows the same model to be used for different retrieval tasks (web search, medical, legal, code) by changing the instruction rather than fine-tuning the model.

---

## When You Do Not Need Instruction Prefixes

Not all models require instruction prefixes. OpenAI's `text-embedding-3-large` and sentence-transformers models like `all-mpnet-base-v2` do not use instruction prefixes.

For these models:
- Query and document are encoded identically.
- The asymmetric query-document gap is handled by training data diversity rather than explicit prefixes.
- These models tend to perform slightly worse on asymmetric retrieval tasks compared to E5/BGE with proper prefixes, but the margin has narrowed with larger training data.

**How to know if a model requires prefixes:**
1. Check the model card on HuggingFace.
2. Look for "query_instruction" or similar fields in the model config.
3. Test with and without prefix on your evaluation set — measure recall difference.

---

## Summary

- Symmetric retrieval treats query and document identically. It works when they have the same form but fails for RAG where a short conversational query must match a long formal document.
- Asymmetric retrieval explicitly trains on (short query, long document) pairs, bridging the form gap.
- E5 adds `"query: "` prefix to queries and `"passage: "` prefix to documents. The prefix is a routing signal that activates different encoding modes within the same model.
- BGE adds a longer instruction only to queries, keeping documents prefix-free for maximum generality.
- The prefix mechanically works through attention: all query tokens attend to the prefix tokens in every layer, conditioning their representation on the "this is a query" signal.
- Forgetting prefixes is one of the most common production bugs. Build prefix enforcement into wrapper classes and add tests.
- Modern models (E5-mistral, GTE-Qwen2) extend fixed prefixes to flexible natural language task instructions.

---

## What's Next

Lesson A.4 covers Matryoshka Representation Learning in depth — how the nested loss works mechanically, why the first dimensions are more important than later ones, and how tiered retrieval exploits this for production speed-accuracy trade-offs.