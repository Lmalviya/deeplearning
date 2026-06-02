# Lesson 3.8 — Retrieval Failure Modes and How to Diagnose Them

---

## Why Retrieval Failures Are Hard to Debug

A broken RAG system does not announce itself. It returns an answer — it just returns a wrong one. Users see a confident, fluent response that is either incorrect, incomplete, or based on outdated information. Without deliberate instrumentation, you have no way of knowing whether the failure was in retrieval, re-ranking, context assembly, or generation.

This makes RAG debugging fundamentally different from debugging traditional software. A null pointer exception tells you exactly where the program broke. A wrong RAG answer gives you no stack trace. You have to build your own diagnostic tooling.

This lesson covers the taxonomy of retrieval failure modes — each one has a distinct cause, a distinct diagnostic signal, and a distinct fix. By the end, you should be able to take any retrieval failure and identify which stage broke and why.

---

## The Two-Stage Diagnostic Framework

Every RAG failure traces back to one of two root causes:

**Retrieval failure:** The right chunks were not in the top-K. The LLM never had access to the information needed to answer correctly.

**Generation failure:** The right chunks were retrieved. The LLM had access to the answer but still produced a wrong or incomplete response.

These require completely different fixes. Improving the LLM prompt does nothing for a retrieval failure. Improving hybrid search does nothing for a generation failure.

The first diagnostic step for any failure is always: **"Was the relevant content retrieved?"**

```python
async def diagnose_failure(
    query: str,
    expected_answer: str,
    expected_source_chunk_ids: list[str],
    retriever,
    k: int = 10
) -> dict:
    """
    Identify whether a failure is retrieval or generation.
    Requires knowing what the correct chunk(s) should have been.
    """
    
    results = await retriever.retrieve(query, k=k)
    retrieved_ids = [r["chunk_id"] for r in results]
    
    # Check if correct chunks were retrieved
    hits = [cid for cid in expected_source_chunk_ids if cid in retrieved_ids]
    recall = len(hits) / len(expected_source_chunk_ids)
    
    if recall == 0:
        failure_type = "retrieval"
        detail = "Correct chunks not retrieved at all"
    elif recall < 1.0:
        failure_type = "partial_retrieval"
        detail = f"Only {len(hits)}/{len(expected_source_chunk_ids)} correct chunks retrieved"
    else:
        failure_type = "generation"
        detail = "All correct chunks retrieved but answer still wrong"
    
    return {
        "failure_type": failure_type,
        "detail": detail,
        "recall_at_k": recall,
        "retrieved_ids": retrieved_ids,
        "expected_ids": expected_source_chunk_ids,
        "hits": hits
    }
```

Once you know it is a retrieval failure, the next step is isolating which stage of the retrieval pipeline broke.

---

## Retrieval Failure Mode 1 — Embedding Space Mismatch

**Symptom:** The correct chunk exists in the index, but dense retrieval does not return it even in the top-50. BM25 also misses it. The query and chunk are about the same topic but use completely different vocabulary.

**Root cause:** The embedding model does not map the query and the relevant chunk to nearby vectors. This happens when:
- The domain vocabulary is specialized and not well-represented in the embedding model's training data.
- The query style is very different from the document style (colloquial vs. formal, question vs. statement).
- The embedding model is too small or general for the corpus.

**Diagnosis:**

```python
def diagnose_embedding_mismatch(
    query: str,
    relevant_chunk_text: str,
    all_chunks: list[str],
    embedding_model
) -> dict:
    """
    Check if the relevant chunk is actually near the query in embedding space.
    """
    query_emb = embedding_model.embed(query)
    chunk_embs = embedding_model.embed_batch(all_chunks)
    relevant_emb = embedding_model.embed(relevant_chunk_text)
    
    import numpy as np
    
    # Similarity between query and relevant chunk
    query_relevant_sim = float(np.dot(query_emb, relevant_emb))
    
    # Where does the relevant chunk rank among all chunks?
    all_sims = np.dot(chunk_embs, query_emb)
    relevant_idx = all_chunks.index(relevant_chunk_text)
    rank = int(np.sum(all_sims > all_sims[relevant_idx])) + 1
    
    return {
        "query_relevant_similarity": query_relevant_sim,
        "relevant_chunk_rank": rank,
        "total_chunks": len(all_chunks),
        "diagnosis": "embedding_mismatch" if rank > 20 else "ok"
    }
```

If the similarity is low (< 0.6) or the rank is very high (> 50), the problem is in the embedding model.

**Fixes:**
- Fine-tune the embedding model on domain (query, relevant chunk) pairs.
- Add HyDE — embed a hypothetical answer instead of the raw query.
- Use a better base model (switch from all-MiniLM to bge-large or text-embedding-3-large).
- Add keyword expansion to the query so BM25 can find the chunk even when dense fails.

---

## Retrieval Failure Mode 2 — Chunking Boundary Problem

**Symptom:** Dense retrieval finds a chunk near the correct one, but the actual answer is split across a chunk boundary. The retrieved chunk has half the answer; the other half is in the adjacent chunk that was not retrieved.

**Root cause:** Fixed-size or recursive chunking split a coherent semantic unit across two chunks. Neither chunk alone is sufficient.

**Diagnosis:**

```python
def check_boundary_split(
    query: str,
    retrieved_chunks: list[dict],
    all_chunks: list[dict],
    expected_answer: str
) -> dict:
    """
    Check if the answer is split across a chunk boundary.
    """
    
    retrieved_texts = [c["text"] for c in retrieved_chunks]
    combined_retrieved = " ".join(retrieved_texts)
    
    # Check if expected answer keywords appear in retrieved content
    answer_words = set(expected_answer.lower().split())
    retrieved_words = set(combined_retrieved.lower().split())
    answer_coverage = len(answer_words & retrieved_words) / len(answer_words)
    
    if answer_coverage < 0.5:
        # Check if neighboring chunks contain the missing content
        for chunk in retrieved_chunks:
            chunk_idx = chunk["metadata"].get("chunk_index")
            doc_id = chunk["metadata"].get("doc_id")
            
            if chunk_idx is None:
                continue
            
            # Check next chunk
            next_chunk = next(
                (c for c in all_chunks 
                 if c["metadata"].get("doc_id") == doc_id 
                 and c["metadata"].get("chunk_index") == chunk_idx + 1),
                None
            )
            
            if next_chunk:
                combined = chunk["text"] + " " + next_chunk["text"]
                combined_words = set(combined.lower().split())
                combined_coverage = len(answer_words & combined_words) / len(answer_words)
                
                if combined_coverage > 0.8:
                    return {
                        "diagnosis": "boundary_split",
                        "split_at_chunk": chunk_idx,
                        "fix": "Use parent-child chunking or increase chunk size"
                    }
    
    return {"diagnosis": "not_boundary_split", "answer_coverage": answer_coverage}
```

**Fixes:**
- Switch to parent-child chunking — retrieve small child chunks but return parent context to LLM.
- Use semantic chunking to align chunk boundaries with semantic breaks.
- Use sentence window retrieval to expand context around retrieved chunks.
- Increase chunk overlap so content near boundaries appears in multiple chunks.

---

## Retrieval Failure Mode 3 — Metadata Filter Over-restriction

**Symptom:** The query works fine without metadata filters. With filters applied, the correct chunk is not returned even though it exists in the index.

**Root cause:** A metadata filter that seemed correct is actually excluding the relevant chunk. Common causes:
- Date filter too restrictive — the relevant document is slightly older than the cutoff.
- Department filter too narrow — the relevant policy applies to "All Employees" but the filter was set to only one department.
- Status filter — the relevant document is marked "draft" or "archived" but is still the correct answer.

**Diagnosis:**

```python
async def diagnose_filter_exclusion(
    query: str,
    metadata_filter: dict,
    vector_db,
    embedding_model,
    expected_chunk_id: str,
    k: int = 50
) -> dict:
    """
    Check if metadata filter is excluding the correct chunk.
    """
    query_emb = await embedding_model.embed(query)
    
    # Search without filter
    unfiltered = await vector_db.search(
        collection="documents",
        query_vector=query_emb,
        limit=k
    )
    unfiltered_ids = [r.id for r in unfiltered]
    
    # Search with filter
    filtered = await vector_db.search(
        collection="documents",
        query_vector=query_emb,
        filter=metadata_filter,
        limit=k
    )
    filtered_ids = [r.id for r in filtered]
    
    in_unfiltered = expected_chunk_id in unfiltered_ids
    in_filtered = expected_chunk_id in filtered_ids
    
    if in_unfiltered and not in_filtered:
        # Fetch the chunk's actual metadata to see which filter excluded it
        chunk_payload = await vector_db.get_chunk(expected_chunk_id)
        return {
            "diagnosis": "filter_exclusion",
            "chunk_exists_in_index": True,
            "excluded_by_filter": True,
            "chunk_metadata": chunk_payload,
            "applied_filter": metadata_filter,
            "fix": "Review filter — chunk exists but does not pass filter conditions"
        }
    elif not in_unfiltered:
        return {
            "diagnosis": "not_in_index",
            "chunk_exists_in_index": False,
            "fix": "Chunk not indexed — check indexing pipeline for this document"
        }
    else:
        return {"diagnosis": "retrieved_correctly"}
```

**Fixes:**
- Widen date range filters.
- Check document status values — ensure "active" and "superseded" statuses are correctly assigned.
- For department filters, include "All Employees" or "Global" as fallback values.
- Add logging to capture which filters are applied and which chunks they exclude.

---

## Retrieval Failure Mode 4 — Re-ranking Demotion

**Symptom:** The correct chunk appears in the top-50 from hybrid retrieval but is ranked below top-10 after re-ranking and never makes it into the LLM context.

**Root cause:** The cross-encoder re-ranker scores the relevant chunk lower than the final ranking suggests it should be. Common causes:
- Cross-encoder's 512-token limit truncates the relevant part of the chunk.
- The query phrasing creates a misleading similarity with irrelevant chunks that the cross-encoder scores higher.
- General cross-encoder was not trained on your domain — domain-specific phrasing confuses it.

**Diagnosis:**

```python
def diagnose_reranking_demotion(
    query: str,
    retrieval_results: list[dict],   # Top-50 from hybrid retrieval
    reranked_results: list[dict],    # Top-10 after re-ranking
    expected_chunk_id: str
) -> dict:
    """
    Check if the correct chunk was demoted during re-ranking.
    """
    
    retrieval_rank = next(
        (i + 1 for i, r in enumerate(retrieval_results) 
         if r["chunk_id"] == expected_chunk_id),
        None
    )
    
    rerank_rank = next(
        (i + 1 for i, r in enumerate(reranked_results) 
         if r["chunk_id"] == expected_chunk_id),
        None
    )
    
    if retrieval_rank and not rerank_rank:
        # In top-50 but not top-10 after re-ranking
        
        # Find what the cross-encoder scored this chunk
        chunk_score = next(
            (r.get("rerank_score") for r in retrieval_results 
             if r["chunk_id"] == expected_chunk_id),
            None
        )
        
        return {
            "diagnosis": "reranking_demotion",
            "retrieval_rank": retrieval_rank,
            "rerank_score": chunk_score,
            "top_reranked_scores": [r.get("rerank_score") for r in reranked_results[:5]],
            "fix": [
                "Increase K passed to re-ranker",
                "Fine-tune cross-encoder on domain data",
                "Check if chunk is being truncated at 512 tokens"
            ]
        }
    
    return {
        "retrieval_rank": retrieval_rank,
        "rerank_rank": rerank_rank,
        "diagnosis": "ok" if rerank_rank and rerank_rank <= 10 else "not_retrieved"
    }
```

**Fixes:**
- Increase the number of candidates passed to the re-ranker (K=50 → K=100).
- Fine-tune the cross-encoder on domain-specific (query, relevant chunk) pairs.
- Use the child chunk for re-ranking (shorter, fits within 512 tokens) then fetch parent for LLM.
- Add a minimum score floor — if the top-ranked chunk has a very low cross-encoder score, something is wrong with this query's retrieval.

---

## Retrieval Failure Mode 5 — Index Coverage Gap

**Symptom:** The document with the correct answer exists in your document store but has not been indexed, or was indexed incorrectly.

**Root cause:**
- Ingestion pipeline failed silently for certain document types or formats.
- A document was added to the source system but the incremental indexing worker did not pick it up.
- The document was indexed but the content was garbled by a bad parser (especially for complex PDFs).
- The chunk containing the answer was filtered out by the quality checker.

**Diagnosis:**

```python
async def diagnose_coverage_gap(
    expected_doc_id: str,
    expected_chunk_content_fragment: str,
    registry,
    vector_db,
    embedding_model
) -> dict:
    """
    Check whether a document and its content are properly indexed.
    """
    
    # Check 1: Is the document in the registry?
    registry_entry = await registry.get(expected_doc_id)
    
    if not registry_entry:
        return {
            "diagnosis": "not_in_registry",
            "fix": "Document never submitted for indexing — check ingestion pipeline"
        }
    
    if registry_entry["status"] == "failed":
        return {
            "diagnosis": "indexing_failed",
            "error": registry_entry.get("error_message"),
            "fix": "Indexing failed — retry with debugging enabled"
        }
    
    # Check 2: Are chunks for this document in the vector DB?
    chunks_in_db = await vector_db.scroll(
        collection="documents",
        scroll_filter={"doc_id": {"$eq": expected_doc_id}},
        limit=10
    )
    
    if not chunks_in_db:
        return {
            "diagnosis": "indexed_but_chunks_missing",
            "registry_status": registry_entry["status"],
            "fix": "Registry shows indexed but no chunks in vector DB — re-index"
        }
    
    # Check 3: Does any chunk contain the expected content?
    fragment_words = set(expected_chunk_content_fragment.lower().split())
    
    for chunk in chunks_in_db:
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(fragment_words & chunk_words) / len(fragment_words)
        if overlap > 0.7:
            return {
                "diagnosis": "content_found",
                "chunk_id": chunk["chunk_id"],
                "coverage": overlap
            }
    
    return {
        "diagnosis": "content_not_in_chunks",
        "chunks_exist": True,
        "fix": "Content may have been lost during parsing or filtered by quality checker"
    }
```

**Fixes:**
- Add content hash verification to the indexing pipeline — compare extracted text length to source document size.
- Log the quality filter decisions — track how many chunks are filtered out per document.
- For critical documents, spot-check indexed content manually by fetching chunks for that doc_id.
- For persistent failures with specific document types, investigate the parser routing logic.

---

## Retrieval Failure Mode 6 — Scale Degradation

**Symptom:** Recall was high (> 90%) at 10K documents. After growing to 500K documents, recall dropped to 70%. Nothing changed in the pipeline — just more documents.

**Root cause:** This is one of the most important failures to understand and is covered in depth in Lesson 7.2. The short version:

- **HNSW graph quality degrades:** As vectors are added incrementally, the graph structure becomes suboptimal for the new distribution. Nodes inserted early have connections that reflect the smaller corpus; nodes inserted late have better local connections but poor long-range connections.
- **BM25 IDF shift:** As the corpus grows, term frequencies change. Terms that were rare (high IDF, high weight) become common (lower IDF, lower weight) as more documents are added. Existing BM25 scores become less accurate relative to the new corpus statistics.
- **Metadata filter selectivity increases:** More documents means metadata filters that once returned 10% of the corpus now return 0.5%, hitting the small-subset ANN degradation problem.
- **Embedding distribution shift:** If new documents have a different topical distribution than original documents, the embedding space becomes more crowded in some regions and sparser in others.

**Diagnosis:**

```python
async def diagnose_scale_degradation(
    test_queries: list[tuple[str, list[str]]],  # (query, relevant_chunk_ids)
    retriever,
    k: int = 10
) -> dict:
    """
    Measure recall@K across a sample of test queries.
    Run this periodically as corpus grows.
    """
    recalls = []
    
    for query, relevant_ids in test_queries:
        results = await retriever.retrieve(query, k=k)
        retrieved_ids = set(r["chunk_id"] for r in results)
        relevant_set = set(relevant_ids)
        
        recall = len(retrieved_ids & relevant_set) / len(relevant_set)
        recalls.append(recall)
    
    mean_recall = sum(recalls) / len(recalls)
    
    return {
        "mean_recall_at_k": mean_recall,
        "k": k,
        "n_test_queries": len(test_queries),
        "low_recall_queries": [
            q for (q, _), r in zip(test_queries, recalls) if r < 0.5
        ]
    }
```

**Fixes:**
- Rebuild the HNSW index periodically (weekly or monthly for fast-growing corpora).
- Increase ef parameter at query time to compensate for graph degradation.
- Recompute BM25 IDF weights across the full corpus periodically.
- Shard the index by topic or time period to keep each shard's internal distribution coherent.

---

## Building a Retrieval Tracing System

The most powerful diagnostic tool is a trace that records every stage of the retrieval pipeline for every query.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class RetrievalTrace:
    trace_id: str
    query: str
    timestamp: datetime
    
    # Query understanding
    resolved_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    sub_questions: list[str] = field(default_factory=list)
    
    # Retrieval
    dense_results: list[dict] = field(default_factory=list)
    sparse_results: list[dict] = field(default_factory=list)
    hyde_results: list[dict] = field(default_factory=list)
    merged_results: list[dict] = field(default_factory=list)
    
    # Re-ranking
    reranked_results: list[dict] = field(default_factory=list)
    
    # Context assembly
    final_chunks: list[dict] = field(default_factory=list)
    context_token_count: int = 0
    
    # Generation
    llm_response: Optional[str] = None
    
    # Outcome
    user_feedback: Optional[str] = None  # thumbs_up, thumbs_down, None
    

class InstrumentedRetriever:
    def __init__(self, retriever, trace_store):
        self.retriever = retriever
        self.trace_store = trace_store
    
    async def retrieve_and_trace(self, query: str) -> tuple[list[dict], RetrievalTrace]:
        trace = RetrievalTrace(
            trace_id=generate_trace_id(),
            query=query,
            timestamp=datetime.utcnow()
        )
        
        # Query understanding
        resolved = await resolve_conversational_query(query, ...)
        trace.resolved_query = resolved
        
        rewritten = await rewrite_query(resolved, ...)
        trace.rewritten_query = rewritten
        
        # Retrieval
        trace.dense_results = await dense_retrieve(resolved)
        trace.sparse_results = await sparse_retrieve(resolved)
        trace.merged_results = reciprocal_rank_fusion([
            trace.dense_results, trace.sparse_results
        ])
        
        # Re-ranking
        trace.reranked_results = reranker.rerank(
            resolved, trace.merged_results
        )
        
        # Context assembly
        trace.final_chunks = assemble_context(trace.reranked_results)
        
        # Persist trace
        await self.trace_store.save(trace)
        
        return trace.final_chunks, trace
```

With traces stored, you can run queries like:
- "Show me all queries where the correct chunk was in dense results but not in sparse results."
- "Show me all queries where re-ranking demoted the top dense result below rank 5."
- "Show me all queries where the context had fewer than 3 chunks despite k=10."

These queries turn debugging from guesswork into a systematic search through evidence.

---

## The Debugging Playbook

When a specific query is failing, work through this sequence:

```
Step 1: Check if it is a retrieval or generation failure.
        → Was the correct chunk retrieved?
        → If yes: the bug is in generation (Lesson 4.5). 
        → If no: continue.

Step 2: Check index coverage.
        → Is the document indexed? Is the specific chunk in the vector DB?
        → If no: fix ingestion/indexing pipeline.

Step 3: Check dense retrieval.
        → What is the cosine similarity between the query and correct chunk?
        → What rank does the correct chunk get in unfiltered dense search?
        → If rank > 50: embedding mismatch (fix: HyDE, fine-tuning, better model).

Step 4: Check sparse retrieval.
        → Does BM25 find the correct chunk?
        → If not: vocabulary mismatch (fix: query expansion, BM25 tokenization).

Step 5: Check metadata filters.
        → Run the query without filters. Does the correct chunk appear?
        → If yes: a filter is excluding it (fix: widen filter conditions).

Step 6: Check re-ranking.
        → Is the correct chunk in top-50 after fusion but not top-10 after re-ranking?
        → If yes: re-ranker demotion (fix: increase K, fine-tune re-ranker).

Step 7: Check context assembly.
        → Is the correct chunk in top-10 after re-ranking but not passed to LLM?
        → If yes: budget exhaustion or deduplication issue (fix: increase budget, check dedup logic).
```

---

## Summary

- Every RAG failure is either a retrieval failure (wrong chunks retrieved) or a generation failure (right chunks retrieved, wrong answer generated). Diagnose which it is before fixing anything.
- The six retrieval failure modes: embedding space mismatch, chunking boundary split, metadata filter over-restriction, re-ranking demotion, index coverage gap, and scale degradation.
- Each failure mode has a distinct diagnostic signal: check similarity scores, ranks at each stage, filter conditions, index registry, and recall trends over time.
- Build a retrieval tracing system that records every pipeline stage for every query. Without traces, debugging is guesswork.
- Follow the debugging playbook in sequence: coverage → dense → sparse → filter → re-rank → assembly. Fix the earliest broken stage first.
- Run periodic recall@K measurements as corpus grows. Scale degradation is invisible until it is measured.

---

## What's Next

Part 3 is complete. Part 4 begins with Lesson 4.1 — prompt design for RAG: how to write system prompts that ground the LLM in retrieved context, enforce citation, and handle the case where the context does not contain the answer.