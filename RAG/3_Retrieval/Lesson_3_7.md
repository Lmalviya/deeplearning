# Lesson 3.7 — Contextual Compression and Context Window Packing

---

## The Problem: Retrieved ≠ Useful

After retrieval and re-ranking, you have your top-K chunks. The naive next step is to concatenate them all into the LLM's context and generate. Most tutorials stop here.

The problem is that retrieved chunks are rarely perfectly matched to what the LLM needs. A retrieved chunk about "parental leave eligibility" may be 600 tokens, but only 80 of those tokens actually answer the user's specific question about part-time employee eligibility. The other 520 tokens cover eligibility for full-time, contractor, and international employees — all irrelevant to this query.

When irrelevant content fills the context window, two things happen:

**Lost in the middle:** LLMs have well-documented difficulty attending to information in the middle of long contexts. Key information buried under noise gets ignored. This was empirically demonstrated by Liu et al. (2023) — LLM performance on multi-document QA degrades significantly when relevant content is not near the beginning or end of the context.

**Token waste:** Every irrelevant token you send to the LLM costs money and increases latency. At $5 per million input tokens (GPT-4o as of 2024), sending 5,000 tokens of noise per query costs $0.025 per query — $25 per 1,000 queries, $25,000 per million queries. At scale this is real money.

This lesson covers how to manage what goes into the LLM's context window — compressing retrieved content to its relevant core and assembling the context to maximize answer quality.

---

## Technique 1 — Contextual Compression

Contextual compression filters or compresses each retrieved chunk to retain only the parts relevant to the specific query.

The key word is "contextual" — the compression is query-specific. The same chunk compressed for query A may retain different sentences than when compressed for query B.

### Approach 1: Extractive Compression (Sentence Filtering)

Split the chunk into sentences, score each sentence for relevance to the query, keep only the top-scoring sentences.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk

class ExtractiveCompressor:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(embedding_model_name)
    
    def compress(
        self,
        query: str,
        chunk_text: str,
        min_sentences: int = 2,
        relevance_threshold: float = 0.5,
        max_sentences: int = 8
    ) -> str:
        """
        Extract relevant sentences from chunk based on query similarity.
        """
        # Split into sentences
        sentences = nltk.sent_tokenize(chunk_text)
        
        if len(sentences) <= min_sentences:
            return chunk_text  # Too short to compress meaningfully
        
        # Embed query and all sentences
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        sentence_embeddings = self.model.encode(sentences, normalize_embeddings=True)
        
        # Score each sentence by cosine similarity to query
        scores = sentence_embeddings @ query_embedding  # dot product = cosine for normalized
        
        # Keep sentences above threshold or top min_sentences if too few qualify
        qualifying_indices = [
            i for i, score in enumerate(scores) 
            if score >= relevance_threshold
        ]
        
        if len(qualifying_indices) < min_sentences:
            # Fall back to top min_sentences by score
            qualifying_indices = np.argsort(scores)[-min_sentences:].tolist()
            qualifying_indices.sort()  # Restore original order
        
        # Cap at max_sentences
        if len(qualifying_indices) > max_sentences:
            # Keep the highest-scoring ones (not just the first ones)
            top_indices = sorted(
                qualifying_indices,
                key=lambda i: scores[i],
                reverse=True
            )[:max_sentences]
            qualifying_indices = sorted(top_indices)  # Restore document order
        
        # Reconstruct text preserving original sentence order
        compressed = " ".join(sentences[i] for i in qualifying_indices)
        
        return compressed
```

**When extractive compression works well:**
- Long chunks where most content is tangential to the query.
- Documents with clear sentence-level information units (FAQs, policy documents).

**When it fails:**
- Chunks where sentences are interdependent — removing context sentences makes retained sentences incomprehensible ("The exception to this rule is..." — "this rule" was in a filtered sentence).
- Tables, lists, and structured content — sentence tokenization destroys their structure.

### Approach 2: LLM-Based Abstractive Compression

Ask an LLM to rewrite the chunk, keeping only what is relevant to the query.

```python
async def abstractive_compress(
    query: str,
    chunk_text: str,
    llm_client,
    max_output_tokens: int = 200
) -> str:
    """
    Use LLM to extract and rewrite only the relevant parts of a chunk.
    """
    
    prompt = f"""Given the following query and passage, extract and rewrite only 
the information from the passage that is directly relevant to answering the query.

If the passage contains no relevant information, respond with: NOT_RELEVANT

Query: {query}

Passage: {chunk_text}

Relevant excerpt (be concise, preserve exact facts and numbers):"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",  # Small model — compression is a simple task
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_output_tokens,
        temperature=0.0
    )
    
    result = response.choices[0].message.content.strip()
    
    if result == "NOT_RELEVANT":
        return ""  # Drop this chunk entirely
    
    return result


async def compress_all_chunks(
    query: str,
    chunks: list[dict],
    llm_client,
    text_key: str = "text"
) -> list[dict]:
    """Compress all retrieved chunks in parallel."""
    
    compressed_texts = await asyncio.gather(*[
        abstractive_compress(query, c[text_key], llm_client)
        for c in chunks
    ])
    
    # Filter out non-relevant chunks and attach compressed text
    result = []
    for chunk, compressed in zip(chunks, compressed_texts):
        if compressed:  # Non-empty = relevant
            result.append({
                **chunk,
                "compressed_text": compressed,
                "original_text": chunk[text_key],
                "compression_ratio": len(compressed) / len(chunk[text_key])
            })
    
    return result
```

**Trade-offs of abstractive compression:**
- Higher quality — the LLM understands context and can sensibly extract the relevant parts.
- One LLM call per chunk — adds latency (run in parallel to mitigate).
- Risk of introducing errors — the compressor may alter specific facts, numbers, or dates.
- For precise domains (legal, financial), abstractive compression is risky because paraphrasing can change meaning. Prefer extractive for precision-critical domains.

### Approach 3: Relevance Filtering (Drop Whole Chunks)

A simpler version: do not compress individual chunks, just drop whole chunks that are below a relevance threshold.

```python
def filter_chunks_by_relevance(
    query: str,
    chunks: list[dict],
    reranker: CrossEncoder,
    min_score: float = 0.3,
    text_key: str = "text"
) -> list[dict]:
    """
    Drop chunks where the cross-encoder score is below threshold.
    Use the re-ranking scores if already computed.
    """
    
    if "rerank_score" in chunks[0]:
        # Re-ranking scores already computed — use them
        return [c for c in chunks if c["rerank_score"] >= min_score]
    
    # Compute scores
    pairs = [(query, c[text_key]) for c in chunks]
    scores = reranker.predict(pairs)
    
    return [
        {**c, "rerank_score": float(s)}
        for c, s in zip(chunks, scores)
        if s >= min_score
    ]
```

This is the cheapest form of compression — no extra LLM calls, no sentence tokenization. Just a threshold cut on re-ranking scores. It does not compress within chunks but prevents clearly irrelevant chunks from polluting the context.

---

## Technique 2 — Context Window Packing

After compression, you need to decide how to arrange the surviving chunks in the LLM's context. This is context window packing — the assembly step.

### The Context Budget

Set a hard token budget for the context block. Everything else in the prompt (system instructions, query, output format instructions) consumes tokens too. Account for them.

```python
def compute_context_budget(
    model_context_window: int,
    system_prompt_tokens: int,
    query_tokens: int,
    expected_output_tokens: int,
    safety_margin: int = 200  # Buffer for tokenization variance
) -> int:
    """How many tokens are available for retrieved context?"""
    
    used = system_prompt_tokens + query_tokens + expected_output_tokens + safety_margin
    available = model_context_window - used
    
    return max(0, available)

# Example:
# GPT-4o: 128K context window
# System prompt: 500 tokens
# Query: 50 tokens  
# Expected output: 1000 tokens
# Safety margin: 200

budget = compute_context_budget(
    model_context_window=128_000,
    system_prompt_tokens=500,
    query_tokens=50,
    expected_output_tokens=1_000,
    safety_margin=200
)
# budget ≈ 126,250 tokens — very generous for most use cases

# For gpt-4o-mini (16K window):
budget = compute_context_budget(16_000, 500, 50, 1_000, 200)
# budget ≈ 14,250 tokens — more constrained
```

### Chunk Selection Under Budget

When you have more content than the budget allows, select which chunks to include:

```python
import tiktoken

def pack_context_within_budget(
    chunks: list[dict],
    token_budget: int,
    text_key: str = "text",
    encoding_name: str = "cl100k_base"  # OpenAI's encoding
) -> list[dict]:
    """
    Greedily pack highest-relevance chunks until budget is exhausted.
    Chunks should be pre-sorted by relevance (most relevant first).
    """
    encoding = tiktoken.get_encoding(encoding_name)
    
    selected = []
    tokens_used = 0
    
    for chunk in chunks:
        chunk_text = chunk.get("compressed_text", chunk[text_key])
        chunk_tokens = len(encoding.encode(chunk_text))
        
        # Add separator tokens (newlines between chunks)
        separator_tokens = 10
        
        if tokens_used + chunk_tokens + separator_tokens <= token_budget:
            selected.append(chunk)
            tokens_used += chunk_tokens + separator_tokens
        else:
            # Try to fit a truncated version
            remaining_tokens = token_budget - tokens_used - separator_tokens
            if remaining_tokens > 100:  # Only include if meaningful content fits
                truncated_tokens = encoding.encode(chunk_text)[:remaining_tokens]
                truncated_text = encoding.decode(truncated_tokens)
                chunk_copy = {**chunk, text_key: truncated_text, "truncated": True}
                selected.append(chunk_copy)
                tokens_used += remaining_tokens + separator_tokens
            break  # No more chunks will fit
    
    return selected
```

### Ordering Strategies

How you order chunks in the context matters because of the "lost in the middle" effect.

**Relevance-first ordering:**
Put the most relevant chunk at the top, least relevant last. Simple. The most important content is in the primacy position.

```python
# Chunks already sorted by rerank_score descending
# Just use that order for context
```

**Sandwich ordering (Lost in the Middle mitigation):**
Put the most relevant chunk first, second most relevant chunk last, others in the middle. Exploits both primacy and recency effects.

```python
def sandwich_order(chunks: list[dict]) -> list[dict]:
    """
    Arrange chunks: most relevant first, second most relevant last,
    rest in between. Maximizes LLM attention on top-2 results.
    """
    if len(chunks) <= 2:
        return chunks
    
    result = [chunks[0]]          # Most relevant: first (primacy)
    result.extend(chunks[2:])     # Middle chunks: middle (ignored less)
    result.append(chunks[1])      # Second most relevant: last (recency)
    
    return result
```

**Chronological ordering:**
For time-sensitive content, order by document date rather than relevance. The LLM can reason about temporal sequences better when content is in chronological order.

**Document-grouped ordering:**
If multiple chunks come from the same document, keep them together. Avoid interleaving chunks from different documents — the LLM handles coherent multi-chunk passages better than fragmented cross-document content.

```python
def group_by_document(chunks: list[dict]) -> list[dict]:
    """
    Group chunks by document, ordering documents by their best chunk's score.
    Within each document, order by chunk position in the document.
    """
    from collections import defaultdict
    
    doc_groups = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk["metadata"]["doc_id"]
        doc_groups[doc_id].append(chunk)
    
    # Order documents by their highest-relevance chunk
    doc_order = sorted(
        doc_groups.keys(),
        key=lambda doc_id: max(c.get("rerank_score", 0) for c in doc_groups[doc_id]),
        reverse=True
    )
    
    # Within each document, sort by chunk position
    result = []
    for doc_id in doc_order:
        doc_chunks = sorted(
            doc_groups[doc_id],
            key=lambda c: c["metadata"].get("chunk_index", 0)
        )
        result.extend(doc_chunks)
    
    return result
```

---

## Technique 3 — Source Tagging and Citation Preparation

Prepare context in a way that enables the LLM to cite sources accurately.

```python
def format_context_with_sources(
    chunks: list[dict],
    text_key: str = "text",
    use_compressed: bool = True
) -> tuple[str, list[dict]]:
    """
    Format chunks as numbered, source-tagged context blocks.
    Returns (formatted_context_string, source_list_for_citation).
    """
    
    context_parts = []
    sources = []
    
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("compressed_text", chunk[text_key]) if use_compressed else chunk[text_key]
        metadata = chunk.get("metadata", {})
        
        # Build source reference
        source = {
            "ref_num": i,
            "doc_title": metadata.get("doc_title", "Unknown Document"),
            "section": metadata.get("heading_path", metadata.get("section", "")),
            "page": metadata.get("page_number"),
            "doc_id": metadata.get("doc_id"),
            "url": metadata.get("source_url")
        }
        sources.append(source)
        
        # Format the context block
        source_header = f"[{i}] {source['doc_title']}"
        if source["section"]:
            source_header += f" — {source['section']}"
        if source["page"]:
            source_header += f" (p.{source['page']})"
        
        context_parts.append(f"{source_header}\n{text}")
    
    formatted_context = "\n\n---\n\n".join(context_parts)
    
    return formatted_context, sources
```

The numbered reference format `[1]`, `[2]`, etc. enables the LLM to cite specific sources in its answer:

```
The maternity leave policy entitles employees to 16 weeks of paid leave [1].
Part-time employees are eligible after completing 6 months of employment [2].
```

---

## Technique 4 — Dynamic Context Assembly Based on Query Type

Not all queries need the same context configuration. A factual lookup needs one precise chunk. A synthesis question needs many chunks from different perspectives. A comparison question needs chunks from different documents about the same topic.

```python
def determine_context_strategy(query: str, query_type: str) -> dict:
    """
    Return context assembly parameters based on query type.
    """
    
    strategies = {
        "factual_lookup": {
            "max_chunks": 3,
            "compress": True,
            "ordering": "relevance_first",
            "context_notes": "Focus on finding the specific fact."
        },
        "synthesis": {
            "max_chunks": 10,
            "compress": False,  # Need full context for synthesis
            "ordering": "document_grouped",
            "context_notes": "Multiple perspectives needed."
        },
        "comparison": {
            "max_chunks": 8,
            "compress": True,
            "ordering": "alternating_source",  # Alternate between compared entities
            "context_notes": "Comparing across multiple sources."
        },
        "procedural": {
            "max_chunks": 5,
            "compress": False,  # Need full steps
            "ordering": "chronological",  # Steps in order
            "context_notes": "Follow the procedure steps."
        },
        "default": {
            "max_chunks": 6,
            "compress": True,
            "ordering": "sandwich",
            "context_notes": ""
        }
    }
    
    return strategies.get(query_type, strategies["default"])
```

---

## The Full Context Assembly Pipeline

Putting all techniques together into a coherent context assembly stage:

```python
async def assemble_context(
    query: str,
    reranked_chunks: list[dict],
    llm_client,
    embedding_model,
    context_budget: int,
    query_type: str = "default",
    text_key: str = "text"
) -> tuple[str, list[dict]]:
    """
    Full context assembly pipeline.
    Returns (formatted_context, sources).
    """
    
    strategy = determine_context_strategy(query, query_type)
    max_chunks = strategy["max_chunks"]
    
    # Step 1: Take top chunks from re-ranked list
    candidates = reranked_chunks[:max_chunks * 2]  # Take extra, will trim after compression
    
    # Step 2: Compress (if strategy says so and chunks are long)
    if strategy["compress"]:
        avg_chunk_length = sum(len(c[text_key].split()) for c in candidates) / len(candidates)
        
        if avg_chunk_length > 150:  # Only compress if chunks are meaningfully long
            candidates = await compress_all_chunks(query, candidates, llm_client, text_key)
    
    # Step 3: Apply max_chunks limit after compression (compression may eliminate some)
    candidates = candidates[:max_chunks]
    
    # Step 4: Apply ordering strategy
    if strategy["ordering"] == "sandwich":
        candidates = sandwich_order(candidates)
    elif strategy["ordering"] == "document_grouped":
        candidates = group_by_document(candidates)
    elif strategy["ordering"] == "relevance_first":
        pass  # Already sorted by relevance from re-ranking
    
    # Step 5: Pack within token budget
    candidates = pack_context_within_budget(candidates, context_budget, text_key)
    
    # Step 6: Format with source tags
    formatted_context, sources = format_context_with_sources(
        candidates,
        text_key=text_key,
        use_compressed=strategy["compress"]
    )
    
    return formatted_context, sources
```

---

## Common Context Assembly Mistakes

**Mistake 1: No deduplication.**
Two chunks from the same parent (in parent-child retrieval) appear as separate context blocks. The LLM sees the same content twice, wasting tokens and potentially confusing its attention.

```python
def deduplicate_by_parent(chunks: list[dict]) -> list[dict]:
    seen_parents = set()
    result = []
    for chunk in chunks:
        parent_id = chunk["metadata"].get("parent_id", chunk["metadata"].get("chunk_id"))
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            result.append(chunk)
    return result
```

**Mistake 2: Ignoring source metadata in compression.**
Abstractive compression strips source information from chunk text. Always preserve metadata separately — never compress it away.

**Mistake 3: Fixed chunk count regardless of query.**
Always sending 10 chunks regardless of whether the query needs 1 or 20 is wasteful in one direction and incomplete in another. Adapt chunk count to query complexity.

**Mistake 4: Compressing before re-ranking.**
Compress after re-ranking, not before. Re-ranking needs full chunk text to accurately score relevance. Compressed text may lack the signals the cross-encoder relies on.

**Mistake 5: Truncating at the context budget without warning.**
If a chunk is truncated because the budget ran out, the LLM may receive a statement that ends mid-sentence. Tag truncated chunks and consider adding "[...continued]" to indicate incompleteness.

---

## Summary

- Retrieved chunks often contain significant irrelevant content. Sending them to the LLM unmodified wastes tokens, increases cost, and hurts answer quality via the lost-in-the-middle effect.
- Extractive compression: sentence-level relevance scoring retains only relevant sentences. Fast, no LLM call, but cannot bridge sentence dependencies.
- Abstractive compression: LLM rewrites each chunk to retain only relevant content. Higher quality, risks paraphrasing errors, adds latency (parallelize).
- Relevance filtering: drop entire chunks below a threshold. Cheapest approach, does not compress within chunks.
- Context budget: compute exactly how many tokens are available for retrieved content given system prompt, query, and expected output. Enforce it hard.
- Ordering: sandwich order (most relevant first and last) mitigates lost-in-the-middle. Document-grouped ordering improves coherence for multi-chunk passages from the same source.
- Source tagging with numbered references enables accurate LLM citation. Always preserve metadata separately from compressed text.
- Compress after re-ranking, deduplicate before assembly, and adapt chunk count to query type.

---

## What's Next

Lesson 3.8 covers retrieval failure modes and how to diagnose them — the systematic debugging framework for when your retrieval pipeline is underperforming and you need to find exactly where it is breaking down.