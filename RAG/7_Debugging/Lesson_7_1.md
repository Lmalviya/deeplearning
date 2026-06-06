# Lesson 7.1 — Systematic Debugging Framework: Isolating Retrieval vs. Generation Failures

---

## Why RAG Debugging Is Different

Traditional software debugging follows a clear path: error messages, stack traces, log lines pointing to the exact failure. RAG debugging has none of this. The system produces output for every query — the output is just wrong, incomplete, or faithfully incorrect. There is no exception, no null pointer, no obvious signal.

This makes RAG debugging a diagnostic exercise, not a code-reading exercise. You are a physician looking at symptoms and working backward to a root cause, not an engineer reading a stack trace.

The discipline required is: resist the urge to immediately try fixes. First, understand precisely what went wrong and where. A fix applied to the wrong stage costs time and may obscure the real problem.

---

## The Diagnostic Hierarchy

Every RAG failure traces to one of five possible root causes, and they have a natural diagnostic order — start with the earliest stage and work forward:

```
Stage 0 — Index Coverage: Is the answer in the index at all?
Stage 1 — Retrieval: Did retrieval find the right chunks?
Stage 2 — Re-ranking: Did re-ranking surface the right chunks to the top?
Stage 3 — Context Assembly: Did the right chunks make it into the LLM context?
Stage 4 — Generation: Did the LLM use the context correctly?
```

Diagnosing from the earliest stage forward prevents the most common mistake in RAG debugging: spending time tuning re-ranking or prompts when the actual problem is a coverage gap or embedding mismatch in the retrieval stage.

---

## The Debugging Starter Kit

Before you can debug systematically, you need the ability to inspect each stage independently. This requires building tracing into your pipeline from the beginning (as described in Lesson 3.8), and having tools to query each stage in isolation.

```python
class RAGDebugger:
    """
    A diagnostic tool for inspecting RAG pipeline failures.
    """
    
    def __init__(
        self,
        vector_db,
        embedding_model,
        cross_encoder,
        llm_client,
        registry
    ):
        self.vdb = vector_db
        self.embedder = embedding_model
        self.reranker = cross_encoder
        self.llm = llm_client
        self.registry = registry
    
    async def diagnose_query(
        self,
        query: str,
        expected_answer: str = None,
        expected_chunk_ids: list[str] = None
    ) -> dict:
        """
        Run a complete diagnostic for one failing query.
        Returns a structured report of what each stage did.
        """
        
        report = {
            "query": query,
            "expected_answer": expected_answer,
            "expected_chunk_ids": expected_chunk_ids,
            "stages": {}
        }
        
        # Stage 0: Index Coverage
        report["stages"]["index_coverage"] = await self._diagnose_coverage(
            expected_chunk_ids
        )
        
        # Stage 1: Dense Retrieval
        report["stages"]["dense_retrieval"] = await self._diagnose_dense_retrieval(
            query, expected_chunk_ids
        )
        
        # Stage 2: Sparse Retrieval (BM25)
        report["stages"]["sparse_retrieval"] = await self._diagnose_sparse_retrieval(
            query, expected_chunk_ids
        )
        
        # Stage 3: Hybrid Retrieval + RRF
        report["stages"]["hybrid_retrieval"] = await self._diagnose_hybrid(
            query, expected_chunk_ids
        )
        
        # Stage 4: Re-ranking
        report["stages"]["reranking"] = await self._diagnose_reranking(
            query, expected_chunk_ids
        )
        
        # Stage 5: Generation
        if expected_chunk_ids:
            report["stages"]["generation"] = await self._diagnose_generation(
                query, expected_chunk_ids, expected_answer
            )
        
        # Synthesize findings
        report["diagnosis"] = self._synthesize_diagnosis(report["stages"])
        report["recommended_fix"] = self._recommend_fix(report["diagnosis"])
        
        return report
    
    async def _diagnose_coverage(self, expected_chunk_ids: list[str]) -> dict:
        """Stage 0: Are the expected chunks in the index?"""
        if not expected_chunk_ids:
            return {"status": "skipped", "reason": "no expected chunk IDs provided"}
        
        results = {}
        for chunk_id in expected_chunk_ids:
            chunk = await self.vdb.get_by_id(chunk_id)
            results[chunk_id] = {
                "exists": chunk is not None,
                "metadata": chunk.get("payload") if chunk else None
            }
        
        all_present = all(r["exists"] for r in results.values())
        
        return {
            "status": "pass" if all_present else "fail",
            "chunk_presence": results,
            "missing_chunks": [cid for cid, r in results.items() if not r["exists"]]
        }
    
    async def _diagnose_dense_retrieval(
        self,
        query: str,
        expected_chunk_ids: list[str]
    ) -> dict:
        """Stage 1: Where does the relevant chunk rank in dense retrieval?"""
        query_embedding = await self.embedder.embed(query)
        
        results = await self.vdb.search(
            collection="documents",
            query_vector=query_embedding,
            limit=100  # Retrieve many to find true rank
        )
        
        retrieved_ids = [r.id for r in results]
        scores = {r.id: r.score for r in results}
        
        diagnostics = {}
        if expected_chunk_ids:
            for chunk_id in expected_chunk_ids:
                rank = next(
                    (i + 1 for i, cid in enumerate(retrieved_ids) if cid == chunk_id),
                    None
                )
                diagnostics[chunk_id] = {
                    "rank": rank,
                    "score": scores.get(chunk_id),
                    "in_top_10": rank is not None and rank <= 10,
                    "in_top_50": rank is not None and rank <= 50,
                    "found": rank is not None
                }
        
        # Also show what the top-5 results are
        top_5 = [
            {"chunk_id": r.id, "score": r.score, "text_preview": r.payload.get("text", "")[:100]}
            for r in results[:5]
        ]
        
        all_found_top_50 = all(
            d.get("in_top_50", False) for d in diagnostics.values()
        ) if diagnostics else None
        
        return {
            "status": "pass" if all_found_top_50 else "fail" if all_found_top_50 is not None else "unknown",
            "chunk_ranks": diagnostics,
            "top_5_results": top_5,
            "query_embedding_norm": float(sum(x**2 for x in query_embedding)**0.5)
        }
    
    async def _diagnose_sparse_retrieval(
        self,
        query: str,
        expected_chunk_ids: list[str]
    ) -> dict:
        """Stage 2: Does BM25 find the relevant chunk?"""
        from rank_bm25 import BM25Okapi
        
        # This requires access to your BM25 index
        # Simplified: query the text search index
        bm25_results = await self.vdb.text_search(query=query, limit=50)
        bm25_ids = [r.id for r in bm25_results]
        
        diagnostics = {}
        if expected_chunk_ids:
            for chunk_id in expected_chunk_ids:
                rank = next(
                    (i + 1 for i, cid in enumerate(bm25_ids) if cid == chunk_id),
                    None
                )
                diagnostics[chunk_id] = {
                    "rank": rank,
                    "found_in_top_50": rank is not None and rank <= 50
                }
        
        return {
            "status": "pass" if all(d.get("found_in_top_50") for d in diagnostics.values()) else "fail",
            "chunk_ranks": diagnostics,
            "top_5_bm25_results": [
                {"chunk_id": r.id, "text_preview": r.payload.get("text", "")[:100]}
                for r in bm25_results[:5]
            ]
        }
    
    async def _diagnose_hybrid(
        self,
        query: str,
        expected_chunk_ids: list[str]
    ) -> dict:
        """Stage 3: After RRF fusion, where does the relevant chunk rank?"""
        # Run hybrid retrieval (dense + sparse + RRF)
        hybrid_results = await self._run_hybrid_retrieval(query, k=50)
        hybrid_ids = [r["chunk_id"] for r in hybrid_results]
        
        diagnostics = {}
        if expected_chunk_ids:
            for chunk_id in expected_chunk_ids:
                rank = next(
                    (i + 1 for i, cid in enumerate(hybrid_ids) if cid == chunk_id),
                    None
                )
                diagnostics[chunk_id] = {
                    "rank": rank,
                    "rrf_score": next(
                        (r.get("rrf_score") for r in hybrid_results if r["chunk_id"] == chunk_id),
                        None
                    ),
                    "in_top_20": rank is not None and rank <= 20
                }
        
        return {
            "status": "pass" if all(d.get("in_top_20") for d in diagnostics.values()) else "fail",
            "chunk_ranks": diagnostics
        }
    
    async def _diagnose_reranking(
        self,
        query: str,
        expected_chunk_ids: list[str]
    ) -> dict:
        """Stage 4: After cross-encoder re-ranking, is the relevant chunk in top-10?"""
        hybrid_results = await self._run_hybrid_retrieval(query, k=50)
        
        # Re-rank
        pairs = [(query, r["text"]) for r in hybrid_results[:50]]
        rerank_scores = self.reranker.predict(pairs)
        
        reranked = sorted(
            zip(hybrid_results, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        reranked_ids = [r["chunk_id"] for r, _ in reranked]
        
        diagnostics = {}
        if expected_chunk_ids:
            for chunk_id in expected_chunk_ids:
                rank = next(
                    (i + 1 for i, cid in enumerate(reranked_ids) if cid == chunk_id),
                    None
                )
                rerank_score = next(
                    (score for r, score in reranked if r["chunk_id"] == chunk_id),
                    None
                )
                diagnostics[chunk_id] = {
                    "rank": rank,
                    "rerank_score": float(rerank_score) if rerank_score is not None else None,
                    "in_top_10": rank is not None and rank <= 10
                }
        
        return {
            "status": "pass" if all(d.get("in_top_10") for d in diagnostics.values()) else "fail",
            "chunk_ranks": diagnostics,
            "top_3_reranked": [
                {"chunk_id": r["chunk_id"], "score": float(score), "text_preview": r.get("text", "")[:100]}
                for r, score in reranked[:3]
            ]
        }
    
    async def _diagnose_generation(
        self,
        query: str,
        relevant_chunk_ids: list[str],
        expected_answer: str
    ) -> dict:
        """Stage 5: Given the correct context, does the LLM answer correctly?"""
        
        # Fetch the relevant chunks
        context_chunks = []
        for chunk_id in relevant_chunk_ids:
            chunk = await self.vdb.get_by_id(chunk_id)
            if chunk:
                context_chunks.append(chunk.payload.get("text", ""))
        
        if not context_chunks:
            return {"status": "fail", "reason": "could not fetch relevant chunks"}
        
        context = "\n\n".join(context_chunks)
        
        # Generate answer with perfect context
        response = await self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using only the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            max_tokens=500,
            temperature=0.0
        )
        
        generated = response.choices[0].message.content
        
        # Score against expected answer
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        embeddings = model.encode([generated, expected_answer], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        
        return {
            "status": "pass" if similarity > 0.85 else "fail",
            "generated_answer": generated,
            "expected_answer": expected_answer,
            "semantic_similarity": similarity,
            "diagnosis": (
                "Generation works correctly with perfect context"
                if similarity > 0.85
                else "Generation fails even with perfect context — check prompt or LLM"
            )
        }
    
    def _synthesize_diagnosis(self, stages: dict) -> str:
        """Identify the earliest failing stage."""
        
        stage_order = [
            "index_coverage",
            "dense_retrieval",
            "sparse_retrieval",
            "hybrid_retrieval",
            "reranking",
            "generation"
        ]
        
        for stage in stage_order:
            if stages.get(stage, {}).get("status") == "fail":
                return f"Root cause: {stage.replace('_', ' ')} failure"
        
        if all(stages.get(s, {}).get("status") == "pass" for s in stage_order if s in stages):
            return "All stages pass — may be a context assembly or prompt issue"
        
        return "Undetermined — check stages with 'unknown' status"
    
    def _recommend_fix(self, diagnosis: str) -> str:
        fixes = {
            "index coverage": "Add missing document to corpus and re-index. Check ingestion pipeline for gaps.",
            "dense retrieval": "Improve embedding quality (fine-tune model or use HyDE). Check for vocabulary mismatch.",
            "sparse retrieval": "Improve tokenization, add synonyms, or rely more on dense retrieval for this query type.",
            "hybrid retrieval": "Tune RRF k parameter. Check that both dense and sparse are running correctly.",
            "reranking": "Increase K passed to re-ranker. Fine-tune cross-encoder on domain data. Check 512 token limit.",
            "generation": "Strengthen grounding instructions in prompt. Check for parametric knowledge conflict."
        }
        
        for keyword, fix in fixes.items():
            if keyword in diagnosis.lower():
                return fix
        
        return "Run stage-by-stage inspection to identify the specific failure point."
```

---

## Batch Failure Analysis

Individual query debugging is for specific failures. Batch failure analysis is for understanding systematic patterns across many failures.

```python
async def analyze_failure_batch(
    failing_queries: list[dict],  # Queries that produced wrong answers
    debugger: RAGDebugger
) -> dict:
    """
    Diagnose a batch of failures and identify patterns.
    """
    
    stage_failure_counts = {
        "index_coverage": 0,
        "dense_retrieval": 0,
        "sparse_retrieval": 0,
        "hybrid_retrieval": 0,
        "reranking": 0,
        "generation": 0,
        "undetermined": 0
    }
    
    failure_examples = {stage: [] for stage in stage_failure_counts}
    
    for query_item in failing_queries[:50]:  # Sample for efficiency
        report = await debugger.diagnose_query(
            query=query_item["query"],
            expected_answer=query_item.get("expected_answer"),
            expected_chunk_ids=query_item.get("expected_chunk_ids")
        )
        
        diagnosis = report.get("diagnosis", "undetermined")
        
        # Find the failing stage
        failing_stage = "undetermined"
        for stage_name in ["index_coverage", "dense_retrieval", "sparse_retrieval", 
                           "hybrid_retrieval", "reranking", "generation"]:
            if stage_name in diagnosis.lower():
                failing_stage = stage_name
                break
        
        stage_failure_counts[failing_stage] += 1
        
        if len(failure_examples[failing_stage]) < 3:
            failure_examples[failing_stage].append({
                "query": query_item["query"],
                "diagnosis": diagnosis
            })
    
    total = len(failing_queries[:50])
    
    return {
        "total_analyzed": total,
        "failure_distribution": {
            stage: {"count": count, "pct": round(count / total * 100, 1)}
            for stage, count in stage_failure_counts.items()
        },
        "primary_failure_stage": max(stage_failure_counts, key=stage_failure_counts.get),
        "examples_by_stage": failure_examples,
        "recommended_priority": _prioritize_fixes(stage_failure_counts, total)
    }


def _prioritize_fixes(failure_counts: dict, total: int) -> list[dict]:
    """
    Prioritize which failures to fix first based on frequency and impact.
    """
    
    # Impact multiplier: early stage failures affect more queries
    impact_weights = {
        "index_coverage": 1.5,   # Highest impact — no fix possible without it
        "dense_retrieval": 1.3,
        "sparse_retrieval": 1.1,
        "hybrid_retrieval": 1.0,
        "reranking": 0.9,
        "generation": 0.8        # Can sometimes work around with better prompting
    }
    
    priorities = []
    for stage, count in failure_counts.items():
        if count == 0:
            continue
        
        pct = count / total
        weight = impact_weights.get(stage, 1.0)
        priority_score = pct * weight
        
        priorities.append({
            "stage": stage,
            "failure_count": count,
            "failure_pct": round(pct * 100, 1),
            "priority_score": round(priority_score, 3)
        })
    
    return sorted(priorities, key=lambda x: x["priority_score"], reverse=True)
```

---

## Query Difficulty Classification

Not all failures are equal. Some failures are on "hard" queries where even perfect retrieval might struggle. Separating easy from hard failures helps prioritize what to fix.

```python
async def classify_query_difficulty(
    query: str,
    llm_client
) -> dict:
    """
    Classify how hard a query is to answer correctly.
    """
    
    prompt = f"""Classify the difficulty of this question for a RAG system.

Question: {query}

Consider:
- How specific/precise is the answer required to be?
- How many documents might contain relevant information?
- Does answering require combining information from multiple sources?
- Is there potential for vocabulary mismatch between query and documents?
- Is this the kind of question that might expose gaps in a knowledge base?

Return JSON:
{{
    "difficulty": "easy" | "medium" | "hard" | "very_hard",
    "difficulty_score": 1-10,
    "reasons": ["list of factors making it easy or hard"],
    "primary_challenge": "retrieval" | "reasoning" | "knowledge_gap" | "ambiguity"
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
        temperature=0.0
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

When you have a batch of failures, separate them by difficulty. Hard failures may be acceptable — even excellent RAG systems fail on genuinely hard queries. The failures you should prioritize fixing are the easy ones: queries where a good system should succeed but yours does not.

---

## The Error Taxonomy

Build a taxonomy of error types to make debugging consistent across your team:

```python
ERROR_TAXONOMY = {
    "R-COV": {
        "name": "Coverage Gap",
        "stage": "indexing",
        "description": "Relevant content not in the index",
        "fix": "Ingest missing documents or re-index corrupted ones"
    },
    "R-EMB": {
        "name": "Embedding Mismatch",
        "stage": "dense_retrieval",
        "description": "Query and relevant chunk are semantically related but embedding model does not capture this",
        "fix": "HyDE, query expansion, or embedding model fine-tuning"
    },
    "R-KWD": {
        "name": "Keyword Miss",
        "stage": "sparse_retrieval",
        "description": "BM25 cannot find the chunk due to vocabulary mismatch",
        "fix": "Query rewriting, synonym expansion, SPLADE instead of BM25"
    },
    "R-FLT": {
        "name": "Filter Exclusion",
        "stage": "retrieval",
        "description": "Relevant chunk is filtered out by metadata filter",
        "fix": "Widen filter conditions, check metadata accuracy"
    },
    "R-RNK": {
        "name": "Ranking Demotion",
        "stage": "reranking",
        "description": "Relevant chunk found but pushed below top-K by re-ranker",
        "fix": "Increase K, fine-tune re-ranker, check token truncation"
    },
    "G-HAL": {
        "name": "Hallucination",
        "stage": "generation",
        "description": "LLM generates content not in retrieved context",
        "fix": "Stronger grounding instructions, NLI faithfulness check"
    },
    "G-IGN": {
        "name": "Context Ignored",
        "stage": "generation",
        "description": "Correct context was retrieved and provided but LLM answered from memory",
        "fix": "Parametric conflict handling, two-step generation"
    },
    "G-INC": {
        "name": "Incomplete Answer",
        "stage": "generation",
        "description": "Answer is correct but missing important parts of the expected answer",
        "fix": "Increase context (more chunks), check context compression over-aggressiveness"
    },
    "G-IDK": {
        "name": "False IDK",
        "stage": "generation",
        "description": "LLM said it doesn't know when the answer was in the context",
        "fix": "Loosen IDK instruction threshold, check context assembly for truncation"
    }
}


async def classify_failure(
    query: str,
    generated_answer: str,
    expected_answer: str,
    debug_report: dict,
    llm_client
) -> dict:
    """
    Classify a failure into the error taxonomy.
    """
    
    diagnosis = debug_report.get("diagnosis", "")
    stages = debug_report.get("stages", {})
    
    # Check coverage first
    if stages.get("index_coverage", {}).get("status") == "fail":
        return {"error_code": "R-COV", **ERROR_TAXONOMY["R-COV"]}
    
    # Check dense retrieval
    dense = stages.get("dense_retrieval", {})
    if dense.get("status") == "fail":
        return {"error_code": "R-EMB", **ERROR_TAXONOMY["R-EMB"]}
    
    # Check sparse retrieval
    sparse = stages.get("sparse_retrieval", {})
    if sparse.get("status") == "fail" and dense.get("status") == "pass":
        return {"error_code": "R-KWD", **ERROR_TAXONOMY["R-KWD"]}
    
    # Check reranking
    reranking = stages.get("reranking", {})
    hybrid = stages.get("hybrid_retrieval", {})
    if hybrid.get("status") == "pass" and reranking.get("status") == "fail":
        return {"error_code": "R-RNK", **ERROR_TAXONOMY["R-RNK"]}
    
    # Generation failures
    gen = stages.get("generation", {})
    if gen.get("status") == "fail":
        # Was the answer "I don't know" when it should not have been?
        idk_phrases = ["don't have information", "cannot find", "not in the provided"]
        if any(phrase in generated_answer.lower() for phrase in idk_phrases):
            return {"error_code": "G-IDK", **ERROR_TAXONOMY["G-IDK"]}
        
        # Was the answer partially right but incomplete?
        if gen.get("semantic_similarity", 0) > 0.5:
            return {"error_code": "G-INC", **ERROR_TAXONOMY["G-INC"]}
        
        return {"error_code": "G-HAL", **ERROR_TAXONOMY["G-HAL"]}
    
    return {"error_code": "UNKNOWN", "description": "Unclassified failure — manual inspection required"}
```

---

## Building a Debug Dashboard

A practical debug interface for your team:

```python
async def generate_debug_report(
    failing_queries: list[dict],
    debugger: RAGDebugger
) -> dict:
    """
    Generate a comprehensive debug report for a set of failing queries.
    Outputs a structured report suitable for team review.
    """
    
    # Batch analysis
    batch_analysis = await analyze_failure_batch(failing_queries, debugger)
    
    # Detailed diagnosis of top 5 failures from the primary failing stage
    primary_stage = batch_analysis["primary_failure_stage"]
    primary_examples = batch_analysis["examples_by_stage"].get(primary_stage, [])
    
    detailed_diagnoses = []
    for example in primary_examples[:5]:
        query_item = next(
            (q for q in failing_queries if q["query"] == example["query"]),
            None
        )
        if query_item:
            diagnosis = await debugger.diagnose_query(
                query=query_item["query"],
                expected_answer=query_item.get("expected_answer"),
                expected_chunk_ids=query_item.get("expected_chunk_ids")
            )
            detailed_diagnoses.append(diagnosis)
    
    return {
        "summary": {
            "total_failures_analyzed": batch_analysis["total_analyzed"],
            "primary_failure_stage": primary_stage,
            "failure_distribution": batch_analysis["failure_distribution"],
            "fix_priority": batch_analysis["recommended_priority"]
        },
        "detailed_examples": detailed_diagnoses,
        "action_items": _generate_action_items(batch_analysis)
    }


def _generate_action_items(batch_analysis: dict) -> list[dict]:
    """Convert diagnostic findings into concrete action items."""
    
    actions = []
    distribution = batch_analysis["failure_distribution"]
    
    if distribution.get("index_coverage", {}).get("pct", 0) > 10:
        actions.append({
            "priority": "P0",
            "action": "Audit indexing pipeline",
            "detail": f"{distribution['index_coverage']['pct']}% of failures are coverage gaps",
            "owner": "data-engineering"
        })
    
    if distribution.get("dense_retrieval", {}).get("pct", 0) > 20:
        actions.append({
            "priority": "P1",
            "action": "Improve embedding quality",
            "detail": "Consider HyDE for short queries, domain fine-tuning for vocabulary mismatch",
            "owner": "ml-engineering"
        })
    
    if distribution.get("reranking", {}).get("pct", 0) > 15:
        actions.append({
            "priority": "P1",
            "action": "Fine-tune or replace cross-encoder",
            "detail": "Re-ranker is demoting relevant chunks — may be domain vocabulary issue",
            "owner": "ml-engineering"
        })
    
    if distribution.get("generation", {}).get("pct", 0) > 20:
        actions.append({
            "priority": "P1",
            "action": "Strengthen prompt grounding",
            "detail": "LLM is not using retrieved context reliably",
            "owner": "prompt-engineering"
        })
    
    return sorted(actions, key=lambda x: x["priority"])
```

---

## The Debugging Protocol: Step-by-Step

When you encounter a RAG failure in production, follow this exact protocol:

**Step 1 — Reproduce the failure.**
Run the exact query through the current production system and confirm you see the reported failure. Many "failures" reported by users are not reproducible — the context changed, the document was updated, or the user misread the response.

**Step 2 — Check if it is an isolated failure or systematic.**
Query the production logs for similar queries. If it is a one-off, it may be noise. If 30 similar queries are failing, it is systematic and high priority.

**Step 3 — Run the stage-by-stage diagnostic.**
Use the `RAGDebugger.diagnose_query()` method to find the earliest failing stage. Do not skip stages.

**Step 4 — Classify the error.**
Map to the error taxonomy. This forces precision about what specifically failed and what the fix is.

**Step 5 — Propose a fix at the right stage.**
Fix only the failing stage. Do not add prompt complexity to fix a retrieval problem. Do not re-index everything to fix a generation prompt problem.

**Step 6 — Test the fix in isolation.**
Before deploying, verify the fix resolves the specific diagnostic failure. Run the full batch of similar failures to confirm the fix generalizes.

**Step 7 — Add to regression test suite.**
The query that exposed the failure becomes a regression test case. Run it on every future deployment.

---

## Summary

- RAG debugging is diagnostic, not code-reading. Resist the urge to try fixes before understanding the failure stage.
- The diagnostic hierarchy: index coverage → dense retrieval → sparse retrieval → hybrid/RRF → re-ranking → generation. Always start from stage 0.
- The `RAGDebugger` class provides stage-by-stage isolation — run each stage independently to find the first failure.
- Batch failure analysis identifies patterns across many failures and prioritizes which stage to fix first.
- The error taxonomy (R-COV, R-EMB, R-KWD, R-FLT, R-RNK, G-HAL, G-IGN, G-INC, G-IDK) standardizes failure classification for team communication.
- Always fix the earliest failing stage. Tuning downstream stages to compensate for upstream failures is technical debt.
- Every diagnosed failure should become a regression test case.

---

## What's Next

Lesson 7.2 covers the "accuracy drops at 5K documents" problem in depth — root cause analysis, systematic investigation, and the fixes for the specific failure modes that cause retrieval accuracy to degrade at scale.