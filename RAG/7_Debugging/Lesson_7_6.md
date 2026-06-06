# Lesson 7.6 — Common Failure Patterns Catalog and Diagnostic Playbook

---

## How to Use This Lesson

This lesson is a reference catalog, not a linear read. Every failure pattern follows the same structure:

- **Symptom:** What the user or system observes.
- **Root cause:** What actually went wrong.
- **Diagnostic signal:** How to confirm this is the issue.
- **Fix:** The specific remediation.
- **Prevention:** How to stop it from happening again.

Bookmark this lesson. When you encounter a failure in production, come here first, match the symptom, and follow the diagnostic signal before applying the fix.

---

## Category 1 — Retrieval Failures

### Pattern R-01: "The system doesn't know about X" — Document Not Indexed

**Symptom:** User asks about a document or topic that exists in the corpus. System responds with IDK or hallucinates a wrong answer. The document is findable in the source system.

**Root cause:** The document was never indexed, indexing failed silently, or the document was recently added and the incremental indexing pipeline has not picked it up yet.

**Diagnostic signal:**
```python
# Check the registry for this document
doc = await registry.get_by_source_path(document_path)
print(doc)  # None = never submitted for indexing
            # status: "failed" = indexing failed
            # indexed_at < created_at = not yet picked up
```

**Fix:**
- If never submitted: check why the document was not discovered by the indexing pipeline (file location, format, permissions).
- If failed: check error_message in registry, fix root cause, re-trigger indexing.
- If pending: force immediate re-index or check why the worker is not picking it up.

**Prevention:** Add document coverage monitoring (Lesson 7.4). Alert when source document count diverges from indexed document count by more than 2%.

---

### Pattern R-02: "Wrong answer on a topic that's in our corpus" — Embedding Mismatch

**Symptom:** The relevant document exists and is indexed. But retrieval does not return it. System answers from parametric knowledge or a different, less relevant document.

**Root cause:** The query and the relevant chunk embed to vectors that are far apart in the embedding space. This happens with vocabulary mismatch (user says "heart attack", document says "myocardial infarction") or domain-specific terminology.

**Diagnostic signal:**
```python
# Compute direct similarity between query and known-relevant chunk
query_emb = embedder.embed(failing_query)
chunk_emb = embedder.embed(known_relevant_chunk_text)
similarity = cosine_similarity(query_emb, chunk_emb)
print(similarity)  # If below 0.55, embedding mismatch confirmed
```

**Fix:**
- Short-term: add HyDE for this query type, or query rewriting that bridges the vocabulary.
- Medium-term: fine-tune the embedding model on domain-specific (query, chunk) pairs.
- Long-term: add query expansion that introduces domain synonyms.

**Prevention:** Monitor per-document-type recall@K. When recall drops for a specific document type, investigate embedding quality for that domain.

---

### Pattern R-03: "The answer was there last week" — Stale Index

**Symptom:** A query that worked correctly before now returns an outdated answer, or the answer reflects an old policy that has since been updated.

**Root cause:** The source document was updated but the index was not. The incremental indexing pipeline missed the update, or the update propagation was delayed beyond acceptable freshness SLA.

**Diagnostic signal:**
```python
registry_entry = await registry.get(doc_id)
source_modified = get_file_modified_time(source_path)

lag_hours = (source_modified - registry_entry.indexed_at).total_seconds() / 3600
print(f"Indexing lag: {lag_hours:.1f} hours")
# If lag > SLA hours, confirmed stale index
```

**Fix:** Force re-index of the affected document immediately. Investigate why the incremental pipeline missed it (webhook failure, polling interval too long, content hash mismatch bug).

**Prevention:** Set up staleness monitoring alerts (Lesson 2.6). Track time from document modification to index update. Alert when staleness exceeds your SLA.

---

### Pattern R-04: "Searches a very specific filter and gets nothing" — Filter Over-Exclusion

**Symptom:** A query with metadata filters returns zero or very few results. Removing the filter returns good results. The filter seemed correct but is excluding the relevant chunk.

**Root cause:** The metadata field on the relevant chunk does not match the filter value. Common causes: inconsistent enum values ("Active" vs "active"), date format mismatch, missing metadata field, overly specific department filter.

**Diagnostic signal:**
```python
# Run query WITHOUT filter
unfiltered = await vector_db.search(query_embedding, limit=10)
print("Without filter:", [r.payload.get("document_status") for r in unfiltered[:5]])

# Compare with what the relevant chunk actually has
chunk = await vector_db.get_by_id(expected_chunk_id)
print("Relevant chunk metadata:", chunk.payload)
```

**Fix:** Fix the metadata on affected chunks (re-index with corrected metadata). Fix the filter logic to be more tolerant (case-insensitive match, broader enum values).

**Prevention:** Include metadata field values in your evaluation dataset. Validate metadata schema at ingestion time with strict validation that fails loudly on unexpected values.

---

### Pattern R-05: "Works great for popular topics, fails for obscure ones" — Long-Tail Coverage Gap

**Symptom:** Common, frequently asked questions work well. Rare or specialized questions fail. The relevant documents exist but are obscure (few documents on that topic, specialized vocabulary).

**Root cause:** Embedding space is crowded for common topics and sparse for rare ones. ANN search has good coverage of dense regions but poor coverage of sparse regions. BM25 also suffers because rare terms have very high IDF, making any noise in the query have outsized impact.

**Diagnostic signal:**
```python
# Check how many chunks exist for the failing topic
topic_embedding = embedder.embed(failing_query)
results = await vector_db.search(topic_embedding, limit=50)
topic_scores = [r.score for r in results]

# If scores drop sharply (e.g., 0.85, 0.84, 0.60, 0.58...) = sparse coverage
# If scores are uniformly high = crowding in that topic area
print(topic_scores[:10])
```

**Fix:** For sparse topics, ensure all relevant documents are indexed and consider creating additional synthetic chunks (summaries, Q&A pairs) that provide more retrieval surface area for the topic.

**Prevention:** Track hit rate@10 by topic cluster. Alert when specific clusters have hit rate below 0.80.

---

### Pattern R-06: "Re-ranking makes things worse" — Cross-Encoder Miscalibration

**Symptom:** Recall before re-ranking is acceptable but recall after re-ranking is significantly lower. The relevant chunk was in top-50 before re-ranking but got pushed below top-10 after.

**Root cause:** The cross-encoder is scoring the relevant chunk as irrelevant. Causes: (1) the relevant chunk exceeds 512 tokens and is truncated, losing the part that makes it relevant; (2) the cross-encoder was not trained on your domain's query-chunk style; (3) the relevant chunk is long-form context while the query is short and specific.

**Diagnostic signal:**
```python
# Directly score the query-chunk pair with the cross-encoder
pairs = [(failing_query, known_relevant_chunk[:500])]
scores = cross_encoder.predict(pairs)
print(f"Cross-encoder score: {scores[0]:.3f}")
# If below 0.4 while irrelevant chunks score above 0.6, cross-encoder miscalibration confirmed

# Check if truncation is the issue
token_count = len(tokenizer.encode(known_relevant_chunk))
print(f"Chunk tokens: {token_count}")  # > 450 = likely being truncated
```

**Fix:**
- If truncation: use parent-child chunking so the re-ranker scores the child chunk, then fetch the parent for context.
- If domain mismatch: fine-tune the cross-encoder on domain-specific (query, relevant chunk, irrelevant chunk) triples.
- Short-term: increase K passed to re-ranker, or reduce re-ranker's strictness by keeping more candidates.

**Prevention:** Track pre-rerank vs. post-rerank recall on your evaluation set. If re-ranking consistently reduces recall, the cross-encoder is miscalibrated.

---

## Category 2 — Generation Failures

### Pattern G-01: "System answers from memory, ignores the document" — Parametric Override

**Symptom:** The correct chunk was retrieved. It appears in the context. But the LLM's answer does not reflect the retrieved content — it reflects what the LLM learned during training (which may be outdated or domain-incorrect).

**Root cause:** The LLM's training has stronger "confidence" in its parametric knowledge about this topic than in the retrieved context. Strong, frequently-reinforced parametric beliefs override contextual instructions.

**Diagnostic signal:**
```python
# Generate with PERFECT context (give the LLM the exact relevant chunk)
response_with_context = await llm.generate(
    system="Answer ONLY from the provided context.",
    context=known_relevant_chunk,
    query=failing_query
)

response_without_context = await llm.generate(
    system="Answer this question.",
    query=failing_query
)

# If both are the same (or context-less is better), parametric override confirmed
print("With context:", response_with_context)
print("Without context:", response_without_context)
```

**Fix:** Strengthen grounding instructions (Lesson 4.1 escalation strategies). Add explicit conflict instruction ("the provided context supersedes your training knowledge"). For persistent cases, use two-step generation (extract facts, then synthesize only from extracted facts).

**Prevention:** Run faithfulness audits on sampled production queries monthly (Lesson 4.2).

---

### Pattern G-02: "Answer is right but cites the wrong source" — Citation Hallucination

**Symptom:** The answer content is correct and grounded in the context, but the citation references ([1], [2]) point to the wrong source, or a citation is fabricated.

**Root cause:** The LLM confuses which source supports which claim when multiple retrieved chunks discuss related topics. It assigns the closest-sounding citation rather than the correct one.

**Diagnostic signal:**
```python
# Verify each citation against its source
for claim_with_citation in extract_citations_from_answer(answer):
    source_chunk = retrieved_chunks[claim_with_citation.ref_num - 1]
    supported = verify_citation(claim_with_citation.claim, source_chunk.text)
    print(f"[{claim_with_citation.ref_num}] {claim_with_citation.claim[:50]}... → {supported}")
```

**Fix:** Use the citation verification pipeline from Lesson 4.4. In the prompt, instruct the LLM to cite inline (immediately after each claim) rather than at the end. Add explicit instruction: "Do not cite a source unless you can identify the specific sentence that supports this claim."

**Prevention:** Run citation verification on a sample of production responses. Track unsupported citation rate as a quality metric.

---

### Pattern G-03: "Answer is correct for wrong reasons" — Lucky Hallucination

**Symptom:** The answer is factually correct and matches the expected answer, but faithfulness evaluation reveals the answer was not grounded in the retrieved context. The LLM got the right answer from parametric knowledge despite the retrieval failing.

**Root cause:** Retrieval failed, but the LLM happened to know the answer from training. This is most common for well-known facts.

**Why this matters:** This failure is invisible in output quality metrics (the answer is correct!) but invisible problems get worse over time. If you do not detect it, you have no idea your retrieval is failing because the correct metric stays high. Then when retrieval fails on domain-specific knowledge the LLM does not have parametrically, you get wrong answers.

**Diagnostic signal:** Faithfulness score below 0.7 despite correct answer. Retrieval recall@K showing the correct chunk was not retrieved.

**Fix:** Fix retrieval (the answer happened to be correct, but retrieval is broken and will fail on harder domain-specific queries). Do not ignore this just because the answer was correct.

**Prevention:** Always measure faithfulness separately from correctness. They can diverge.

---

### Pattern G-04: "Answer is incomplete — covers only part of the question" — Context Truncation

**Symptom:** For complex questions requiring multiple facts, the answer addresses some parts but silently ignores others. The user has to ask follow-up questions to get the complete picture.

**Root cause:** (1) Context budget ran out and some relevant chunks were truncated; (2) Sub-question decomposition did not identify all required information needs; (3) Retrieved chunks covered some but not all aspects of the question.

**Diagnostic signal:**
```python
# Check context token usage
trace = await trace_store.get_trace(trace_id)
print(f"Context tokens: {trace['ret_context_tokens']}")
print(f"Context budget: {MAX_CONTEXT_TOKENS}")

# Check if all sub-questions were retrievable
for sub_q in identified_sub_questions:
    results = await retriever.retrieve(sub_q)
    print(f"Sub-question: {sub_q[:50]}... → Top score: {results[0].get('rerank_score', 0):.2f}")
```

**Fix:**
- If budget exceeded: increase context budget allocation, or use compression to fit more relevant content.
- If sub-questions not identified: improve decomposition logic.
- If retrieval missing some aspects: query expansion for complex questions.

**Prevention:** For complex questions, use decomposition and verify that each sub-question has a good retrieval result before assembling context.

---

### Pattern G-05: "IDK response when the answer is clearly in the corpus" — False IDK

**Symptom:** System responds "I don't have information about this" but the answer is clearly in the indexed documents.

**Root cause:** (1) Retrieval failed (covered by retrieval patterns above); (2) Retrieval succeeded but context assembly truncated the relevant content; (3) The LLM's IDK instruction is too aggressive; (4) The retrieved chunk is relevant but the LLM judges it as not answering the specific question.

**Diagnostic signal:**
```python
# Check if relevant chunk was retrieved
trace = await trace_store.get_trace(trace_id)
final_chunks = trace["_full_trace"]["retrieval"]["final_context_chunks"]
relevant_chunk_in_context = any(
    c["chunk_id"] == known_relevant_chunk_id 
    for c in final_chunks
)
print(f"Relevant chunk in context: {relevant_chunk_in_context}")

# If yes, the LLM is producing false IDK
# If no, check if it was retrieved at all (retrieval failure)
```

**Fix:**
- If relevant chunk in context: loosen IDK threshold in the prompt ("only say IDK if the specific fact is completely absent, not if the context is only partially relevant").
- If relevant chunk not retrieved: fix retrieval (R-01 through R-06 patterns).
- If context truncation: increase context budget.

**Prevention:** Monitor IDK rate. When IDK rate spikes, run a sample of IDK queries through the debugger to classify how many are true IDK vs. false IDK.

---

## Category 3 — System-Level Failures

### Pattern S-01: "Works for some users but not others" — Multi-Tenant Filter Bug

**Symptom:** Some users get correct answers consistently. Other users get IDK or wrong answers for the same questions. Documents that should be available to the second group appear unavailable.

**Root cause:** Metadata-based access control filters are incorrectly configured. The filter is too restrictive for some users (not including documents they should have access to) or using the wrong field name.

**Diagnostic signal:**
```python
# Simulate the exact query for both users
user_a_filter = build_access_filter(user_a_context)
user_b_filter = build_access_filter(user_b_context)

results_a = await vector_db.search(query_emb, filter=user_a_filter)
results_b = await vector_db.search(query_emb, filter=user_b_filter)

print(f"User A results: {len(results_a)}")
print(f"User B results: {len(results_b)}")
print(f"User B filter: {user_b_filter}")  # Inspect the actual filter being applied
```

**Fix:** Fix the filter construction logic for the affected user group. Verify that all documents they should have access to have the correct metadata values.

**Prevention:** Add multi-tenant filter testing to your evaluation suite. Test with user personas from different groups.

---

### Pattern S-02: "Performance degrades Monday morning" — Cold Cache Effect

**Symptom:** Latency spikes on Monday mornings or after weekends. Resolves after a few hours of traffic. Systems that were fast Friday are slow Monday.

**Root cause:** Vector database and LLM API connection pools, embedding caches, and result caches expire over weekends. The first queries of the week pay full cold-start costs. Additionally, vector database HNSW graph indices may have been evicted from RAM during low-traffic periods.

**Diagnostic signal:** Check P95 latency trends by hour of week. If latency consistently spikes on Monday mornings and declines over the first hour, cold cache confirmed.

**Fix:** Implement a warm-up job that runs before the first users arrive. Send a sample of representative queries to populate caches.

```python
async def warm_up_system(sample_queries: list[str], retriever):
    """Send warm-up queries before peak traffic to populate caches."""
    print("Running system warm-up...")
    for query in sample_queries[:20]:
        try:
            await retriever.retrieve(query, k=10)
        except Exception:
            pass  # Ignore errors during warm-up
    print("Warm-up complete.")
```

**Prevention:** Schedule the warm-up job as a cron that runs before peak hours. Ensure HNSW index is configured with `always_ram=True` for the most frequently accessed collection.

---

### Pattern S-03: "Intermittent wrong answers" — Nondeterminism Without Cause

**Symptom:** The same query sometimes returns correct answers and sometimes wrong answers with no apparent pattern. Temperature is set to 0.1 or lower. There is no change in the system.

**Root cause candidates:** (1) Retrieval is pulling different chunks on different calls due to HNSW approximate nearest neighbor randomness; (2) LLM API is routing to different model versions or servers; (3) Context differs between calls due to caching inconsistency; (4) Metadata filters are timestamp-based and documents are crossing a freshness boundary.

**Diagnostic signal:**
```python
# Run the same query 5 times and inspect retrieved chunk IDs each time
for i in range(5):
    results = await retriever.retrieve(failing_query)
    print(f"Run {i}: {[r['chunk_id'] for r in results[:5]]}")
# If chunk IDs differ significantly between runs, HNSW nondeterminism confirmed
```

**Fix:**
- If HNSW nondeterminism: increase ef parameter. At ef=512, results are more consistent (though still approximate). For critical applications, consider exact search on a filtered subset.
- If LLM model version drift: pin to a specific model version (e.g., "gpt-4o-2024-05-13" instead of "gpt-4o").
- If timestamp-based freshness filters: ensure filter logic is stable (use date() not datetime() for daily granularity).

---

### Pattern S-04: "Costs are 10x higher than expected" — Context Window Bloat

**Symptom:** LLM API costs are much higher than expected. Token usage is far beyond what you estimated.

**Root cause:** Context is larger than anticipated because: (1) retrieved chunks are long and not being compressed; (2) conversation history is accumulating without bounds; (3) system prompt is much longer than intended; (4) multi-query expansion is sending many more tokens than planned.

**Diagnostic signal:**
```python
# Sample trace token breakdown
traces = await trace_store.query_traces(limit=100)
avg_context_tokens = sum(t["ret_context_tokens"] for t in traces) / len(traces)
avg_gen_input_tokens = sum(t["gen_total_input_tokens"] for t in traces) / len(traces)

print(f"Average context tokens: {avg_context_tokens:.0f}")
print(f"Average total input tokens: {avg_gen_input_tokens:.0f}")
print(f"Expected input tokens: {EXPECTED_INPUT_TOKENS}")
```

**Fix:** Apply contextual compression (Lesson 3.7). Set explicit token budgets and enforce them. Consider smaller models for query understanding and re-ranking stages where the full capability of GPT-4o is not needed.

**Prevention:** Set per-query token budget alerts. Alert when any query exceeds 2× the expected token budget.

---

### Pattern S-05: "Accuracy high on eval set, poor in production" — Evaluation Dataset Leakage

**Symptom:** Offline evaluation metrics are excellent (recall@10 = 0.95, faithfulness = 0.92). But user satisfaction is low and negative feedback rate is high.

**Root cause:** The evaluation dataset is not representative of production queries. It was built during development and reflects the developers' mental model of user queries, not actual user behavior. The system has effectively been over-tuned to the evaluation set.

**Diagnostic signal:**
```python
# Compare embedding distributions of eval queries vs production queries
eval_embeddings = embed_queries(eval_dataset["questions"])
prod_embeddings = embed_queries(recent_production_queries)

# PCA and KS test (from Lesson 6.7)
distribution_shift = compute_distribution_shift(eval_embeddings, prod_embeddings)
print(f"KS statistic: {distribution_shift['mean_ks_statistic']:.3f}")
# If above 0.15, eval set is not representative
```

**Fix:** Refresh the evaluation dataset with recent production query samples. Annotate 100-200 real user queries and add them to the eval set. Retire old eval questions that no longer reflect user behavior.

**Prevention:** Check offline-online metric correlation monthly. If they diverge by more than 10 percentage points, refresh the evaluation dataset immediately.

---

## The Quick Reference Diagnostic Card

When a failure is reported, run through this card:

```
1. Is the relevant document indexed?
   NO → Pattern R-01 (Coverage Gap)
   YES → continue

2. Does the query embed near the relevant chunk? (similarity > 0.6)
   NO → Pattern R-02 (Embedding Mismatch)
   YES → continue

3. Is the relevant chunk in top-50 of dense retrieval?
   NO → Pattern R-02 (Embedding Mismatch) or R-05 (Long-tail)
   YES → continue

4. Does a metadata filter exclude it?
   YES → Pattern R-04 (Filter Over-Exclusion)
   NO → continue

5. Is the relevant chunk in top-10 after re-ranking?
   NO → Pattern R-06 (Cross-Encoder Miscalibration)
   YES → continue

6. Is the relevant chunk in the final context passed to the LLM?
   NO → Context truncation (Pattern G-04)
   YES → continue

7. Does the LLM use the context in its answer?
   NO → Pattern G-01 (Parametric Override)
   YES → answer quality issue

8. Is the answer complete?
   NO → Pattern G-04 (Incomplete) or G-02 (Citation Hallucination)
   YES → false negative report — verify the actual failure is real
```

---

## Summary

This lesson is a catalog of 14 failure patterns across three categories:

**Retrieval failures (R-01 to R-06):** Document not indexed, embedding mismatch, stale index, filter over-exclusion, long-tail coverage gaps, and cross-encoder miscalibration. All diagnosed by checking successively later pipeline stages.

**Generation failures (G-01 to G-05):** Parametric override, citation hallucination, lucky hallucination, context truncation, and false IDK. All diagnosed by verifying whether the correct context was available and whether the LLM used it.

**System-level failures (S-01 to S-05):** Multi-tenant filter bugs, cold cache effects, HNSW nondeterminism, context window bloat, and evaluation dataset leakage. Diagnosed with infrastructure and distribution analysis.

The quick reference diagnostic card provides a systematic decision tree for attributing any failure to one of these patterns without trial-and-error.

---

## What's Next

Part 7 is complete. Part 8 begins with Lesson 8.1 — the vector database landscape: Qdrant, Pinecone, Weaviate, pgvector, Milvus — what they are, how they differ, and how to choose the right one for your use case.