# Lesson 5.1 — Corrective RAG (CRAG): Self-Assessment and Fallback Retrieval

---

## The Problem CRAG Solves

Standard RAG has a silent failure mode: when retrieval fails, the system does not know it failed.

The pipeline retrieves top-K chunks, passes them to the LLM, and generates an answer — regardless of whether those chunks are actually relevant to the query. If the local knowledge base does not contain the answer, or if retrieval returns the wrong chunks, the LLM either hallucinates a confident wrong answer or produces a response grounded in irrelevant content.

The user sees a fluent, confident response with no indication that retrieval failed. This is worse than an obvious failure — it erodes trust over time as users discover the system is confidently wrong.

CRAG (Corrective Retrieval Augmented Generation, Shi et al., 2024) introduces a self-assessment step: after retrieval, before generation, evaluate the quality of what was retrieved. If retrieval quality is insufficient, take corrective action before generating.

The corrective action can be:
- Discard the retrieved content and fall back to a broader knowledge source (web search).
- Reformulate the query and retrieve again.
- Combine local retrieval with external sources.
- Return IDK rather than generating from bad context.

CRAG transforms RAG from a blind pipeline into a self-aware one.

---

## The CRAG Architecture

```
User Query
    ↓
Local Retrieval (your vector DB + BM25)
    ↓
Retrieval Evaluator
    ↓
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Score HIGH          Score AMBIGUOUS        Score LOW       │
│  (correct)           (partial)              (wrong)         │
│      ↓                    ↓                    ↓            │
│  Use local          Use local + web       Discard local     │
│  context            search results        → Web search      │
│  directly           combined              or IDK            │
└─────────────────────────────────────────────────────────────┘
    ↓
Knowledge Refinement (strip irrelevant content from both sources)
    ↓
Generation
```

The key component is the **Retrieval Evaluator** — a model that assesses whether the retrieved documents are actually relevant to the query.

---

## The Retrieval Evaluator

The original CRAG paper uses a fine-tuned T5 model as the evaluator. In practice, you can use an LLM-based evaluator, a cross-encoder score threshold, or a combination.

### Option 1 — Cross-Encoder Score Threshold

The simplest approach: use the re-ranking scores from your cross-encoder as the evaluation signal. If the highest-scoring retrieved chunk has a cross-encoder score above a threshold, retrieval is considered successful.

```python
def evaluate_retrieval_quality_by_score(
    reranked_results: list[dict],
    high_threshold: float = 0.7,
    low_threshold: float = 0.3
) -> str:
    """
    Classify retrieval quality based on cross-encoder scores.
    Returns: 'correct', 'ambiguous', or 'incorrect'
    """
    if not reranked_results:
        return "incorrect"
    
    top_score = reranked_results[0].get("rerank_score", 0)
    
    # Count chunks above low threshold
    above_low = sum(1 for r in reranked_results if r.get("rerank_score", 0) >= low_threshold)
    
    if top_score >= high_threshold and above_low >= 2:
        return "correct"
    elif top_score >= low_threshold:
        return "ambiguous"
    else:
        return "incorrect"
```

**Limitation:** Cross-encoder scores are calibrated for the MS MARCO dataset by default. A score of 0.7 on a general cross-encoder may mean different things on your specific domain. Calibrate thresholds by sampling queries and manually reviewing what retrieval quality looks like at each score level.

### Option 2 — LLM-Based Retrieval Evaluator

Ask an LLM to assess whether the retrieved documents can answer the query.

```python
async def evaluate_retrieval_quality_llm(
    query: str,
    retrieved_chunks: list[dict],
    llm_client,
    text_key: str = "text"
) -> dict:
    """
    Use an LLM to evaluate whether retrieved chunks can answer the query.
    Uses a small, fast model — this is on the critical path.
    """
    
    # Format top-3 chunks for evaluation (don't need all of them for this check)
    top_chunks = retrieved_chunks[:3]
    chunks_text = "\n\n---\n\n".join([
        f"Document {i+1}: {c[text_key][:500]}"
        for i, c in enumerate(top_chunks)
    ])
    
    prompt = f"""Evaluate whether the provided documents can answer the given question.

Question: {query}

Retrieved documents:
{chunks_text}

Assessment:
- CORRECT: The documents directly contain the answer to this question
- AMBIGUOUS: The documents are partially relevant but may not fully answer the question
- INCORRECT: The documents are not relevant and cannot answer this question

Respond with JSON:
{{
    "quality": "CORRECT" | "AMBIGUOUS" | "INCORRECT",
    "reason": "brief explanation",
    "missing_information": "what information would be needed to answer (if quality != CORRECT)"
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",  # Fast, cheap — on critical path
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
        temperature=0.0
    )
    
    import json
    result = json.loads(response.choices[0].message.content)
    result["quality"] = result["quality"].lower()
    
    return result
```

### Option 3 — Hybrid Evaluation (Recommended for Production)

Combine both approaches: use the fast cross-encoder score threshold as a first pass, and only run the LLM evaluator when the score is in the ambiguous range.

```python
async def evaluate_retrieval_hybrid(
    query: str,
    reranked_results: list[dict],
    llm_client,
    high_threshold: float = 0.7,
    low_threshold: float = 0.3
) -> dict:
    """
    Fast scoring first, LLM evaluation only when ambiguous.
    """
    
    if not reranked_results:
        return {"quality": "incorrect", "method": "no_results"}
    
    top_score = reranked_results[0].get("rerank_score", 0)
    
    # Clear cases — no LLM call needed
    if top_score >= high_threshold:
        return {"quality": "correct", "method": "score_threshold", "top_score": top_score}
    
    if top_score < low_threshold:
        return {"quality": "incorrect", "method": "score_threshold", "top_score": top_score}
    
    # Ambiguous range — use LLM evaluator
    llm_eval = await evaluate_retrieval_quality_llm(
        query, reranked_results, llm_client
    )
    llm_eval["method"] = "llm_evaluator"
    llm_eval["top_score"] = top_score
    
    return llm_eval
```

This hybrid minimizes LLM calls (most queries will be clear correct or incorrect based on score alone) while maintaining accuracy for edge cases.

---

## The Corrective Actions

Based on the evaluation, take one of three corrective actions.

### Action 1 — Use Local Context (Quality: Correct)

No correction needed. Proceed with normal RAG generation using the retrieved chunks.

```python
async def action_use_local(
    query: str,
    retrieved_chunks: list[dict],
    llm_client
) -> dict:
    context = format_context(retrieved_chunks)
    answer = await generate_from_context(query, context, llm_client)
    
    return {
        "answer": answer,
        "source": "local",
        "chunks_used": retrieved_chunks
    }
```

### Action 2 — Combine Local + Web (Quality: Ambiguous)

Local retrieval has partial but potentially incomplete information. Augment with web search for the missing pieces.

```python
async def action_combine_sources(
    query: str,
    local_chunks: list[dict],
    web_search_client,
    llm_client,
    evaluation_result: dict
) -> dict:
    """
    Combine local retrieval with web search to fill gaps.
    """
    
    # Determine what is missing from local retrieval
    missing_info = evaluation_result.get("missing_information", query)
    
    # Search the web for the missing information
    web_results = await web_search_client.search(
        query=missing_info,
        num_results=5
    )
    
    # Fetch and extract relevant content from top web results
    web_chunks = []
    for result in web_results[:3]:
        content = await fetch_and_extract(result["url"])
        if content:
            web_chunks.append({
                "text": content[:2000],
                "metadata": {
                    "source": "web",
                    "url": result["url"],
                    "title": result["title"]
                }
            })
    
    # Knowledge refinement — strip irrelevant content before combining
    refined_local = await refine_knowledge(query, local_chunks, llm_client)
    refined_web = await refine_knowledge(query, web_chunks, llm_client)
    
    # Combine and generate
    all_chunks = refined_local + refined_web
    context = format_context_with_sources(all_chunks)
    
    answer = await generate_from_context(
        query, context, llm_client,
        source_note="This answer combines information from internal documents and web sources."
    )
    
    return {
        "answer": answer,
        "source": "combined",
        "local_chunks_used": refined_local,
        "web_chunks_used": refined_web
    }
```

### Action 3 — Fall Back to Web Search (Quality: Incorrect)

Local retrieval failed. The internal knowledge base does not contain relevant information. Discard the local results and rely on web search or return IDK.

```python
async def action_web_fallback(
    query: str,
    web_search_client,
    llm_client,
    allow_web_fallback: bool = True
) -> dict:
    """
    Discard local retrieval results and fall back to web search.
    """
    
    if not allow_web_fallback:
        return {
            "answer": "I don't have information about this in the available documents.",
            "source": "idk",
            "reason": "Local retrieval failed and web search is not enabled."
        }
    
    # Reformulate query for web search
    web_query = await reformulate_for_web(query, llm_client)
    
    web_results = await web_search_client.search(
        query=web_query,
        num_results=5
    )
    
    if not web_results:
        return {
            "answer": "I couldn't find relevant information for this question.",
            "source": "idk"
        }
    
    # Build context from web results
    web_chunks = [
        {
            "text": result.get("snippet", "") + "\n" + result.get("content", "")[:1000],
            "metadata": {"source": "web", "url": result["url"], "title": result["title"]}
        }
        for result in web_results
    ]
    
    context = format_context_with_sources(web_chunks)
    
    answer = await generate_from_context(
        query, context, llm_client,
        source_note="Note: This answer is based on web search results, not internal documents."
    )
    
    return {
        "answer": answer,
        "source": "web",
        "web_results_used": web_results[:3]
    }


async def reformulate_for_web(query: str, llm_client) -> str:
    """
    Rewrite a query originally designed for internal document retrieval
    into a query better suited for web search.
    """
    prompt = f"""Rewrite the following question as a web search query.
Make it more general and suitable for finding information on the public web.
Remove any company-specific references.

Original question: {query}
Web search query:"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()
```

---

## Knowledge Refinement

A key step in CRAG that is often overlooked: before passing retrieved content (local or web) to the LLM for generation, refine it to remove irrelevant content.

Web search results are particularly noisy — pages contain navigation menus, ads, unrelated content, and boilerplate alongside the relevant passage. Local chunks may also contain surrounding context that is not relevant to this specific query.

```python
async def refine_knowledge(
    query: str,
    chunks: list[dict],
    llm_client,
    text_key: str = "text"
) -> list[dict]:
    """
    Strip irrelevant content from chunks, keeping only what is
    relevant to answering the query.
    """
    
    refined = []
    
    for chunk in chunks:
        original_text = chunk[text_key]
        
        if len(original_text.split()) < 50:
            refined.append(chunk)  # Too short to refine meaningfully
            continue
        
        prompt = f"""Extract only the information from the following text that is
directly relevant to answering this question.

Question: {query}

Text: {original_text[:3000]}

If the text contains relevant information, return only the relevant portions.
If the text contains NO relevant information, respond with: IRRELEVANT"""
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.0
        )
        
        refined_text = response.choices[0].message.content.strip()
        
        if refined_text != "IRRELEVANT":
            refined.append({
                **chunk,
                text_key: refined_text,
                "original_text": original_text,
                "refined": True
            })
    
    return refined
```

Knowledge refinement reduces the noise that enters the LLM's context window, improving generation quality and reducing hallucination risk from irrelevant content.

---

## The Complete CRAG Pipeline

```python
class CRAGPipeline:
    def __init__(
        self,
        local_retriever,
        web_search_client,
        llm_client,
        allow_web_fallback: bool = True,
        high_quality_threshold: float = 0.7,
        low_quality_threshold: float = 0.3
    ):
        self.local_retriever = local_retriever
        self.web_search = web_search_client
        self.llm = llm_client
        self.allow_web = allow_web_fallback
        self.high_threshold = high_quality_threshold
        self.low_threshold = low_quality_threshold
    
    async def answer(self, query: str) -> dict:
        """
        Full CRAG pipeline: retrieve → evaluate → correct → generate.
        """
        
        # Step 1: Local retrieval
        local_results = await self.local_retriever.retrieve(query)
        
        # Step 2: Evaluate retrieval quality
        evaluation = await evaluate_retrieval_hybrid(
            query=query,
            reranked_results=local_results,
            llm_client=self.llm,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold
        )
        
        quality = evaluation["quality"]
        
        # Step 3: Take corrective action based on evaluation
        if quality == "correct":
            # Local retrieval succeeded — use it directly
            refined = await refine_knowledge(query, local_results, self.llm)
            result = await action_use_local(query, refined, self.llm)
        
        elif quality == "ambiguous":
            if self.allow_web:
                # Combine local + web search
                result = await action_combine_sources(
                    query, local_results, self.web_search, self.llm, evaluation
                )
            else:
                # No web access — use local despite ambiguity, flag uncertainty
                refined = await refine_knowledge(query, local_results, self.llm)
                result = await action_use_local(query, refined, self.llm)
                result["uncertainty_flag"] = True
        
        else:  # incorrect
            result = await action_web_fallback(
                query, self.web_search, self.llm, self.allow_web
            )
        
        # Attach evaluation metadata to result
        result["retrieval_evaluation"] = evaluation
        result["query"] = query
        
        return result
```

---

## CRAG Without Web Search

Many enterprise RAG systems cannot use web search — they operate on sensitive internal data and cannot make external calls. CRAG is still useful in this constrained setting; the fallback options change.

When local retrieval is incorrect without web search:

**Option A — Query reformulation and retry:**
```python
async def reformulate_and_retry(query: str, local_retriever, llm_client) -> dict:
    """Reformulate the query and try local retrieval again."""
    
    reformulated = await rewrite_query(
        query, 
        instruction="Rewrite this query using different keywords that might match internal documents",
        llm_client=llm_client
    )
    
    new_results = await local_retriever.retrieve(reformulated)
    new_evaluation = await evaluate_retrieval_hybrid(query, new_results, llm_client)
    
    if new_evaluation["quality"] != "incorrect":
        return {"results": new_results, "query_used": reformulated, "reformulated": True}
    
    return {"results": [], "query_used": query, "reformulated": False}
```

**Option B — Broader corpus search:**
If your primary index is filtered (e.g., current documents only), retry with a wider index (include archived, all departments, all time periods).

**Option C — Graceful IDK:**
Return a helpful IDK response that explains what the system cannot find and suggests where the user might find it.

```python
async def graceful_idk(query: str, llm_client) -> dict:
    """Generate a helpful IDK response that guides the user."""
    
    prompt = f"""A user asked: {query}

Our document retrieval system could not find relevant information.
Generate a helpful response that:
1. Clearly states the information is not in available documents
2. Suggests what type of document or team might have this information
3. Offers an alternative (e.g., "you might ask HR directly" or "check the intranet")

Keep it concise and constructive."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.1
    )
    
    return {
        "answer": response.choices[0].message.content,
        "source": "idk",
        "helpful_redirect": True
    }
```

---

## When CRAG Adds the Most Value

CRAG is not necessary for every RAG system. It adds complexity and latency (the evaluation step). The value is highest when:

**Your corpus is incomplete.** If your knowledge base covers 80% of the queries users ask, CRAG handles the 20% that would otherwise silently fail.

**Query distribution is unpredictable.** Users ask questions beyond your anticipated scope. CRAG catches these gracefully rather than hallucinating.

**The cost of confident wrong answers is high.** In domains where a confidently wrong answer causes harm or erodes user trust significantly, CRAG's evaluation gate is worth the latency cost.

**You have reliable web search access.** CRAG's full value is realized when there is a meaningful fallback. Without it, CRAG still adds value via graceful IDK handling and query reformulation.

**Use standard RAG without CRAG when:**
- Your corpus is comprehensive and well-maintained.
- Query distribution is predictable and well-covered by your index.
- Latency is critical and the evaluation step is too expensive.
- IDK responses are acceptable and you have strong prompt-level IDK instructions.

---

## CRAG vs. Self-RAG

CRAG and Self-RAG both add self-assessment to RAG. They differ in how:

| | CRAG | Self-RAG |
|---|---|---|
| **When assessment happens** | After retrieval, before generation | During generation (token by token) |
| **What is assessed** | Retrieval quality | Whether to retrieve at all + output quality |
| **Assessment mechanism** | Separate evaluator model | Reflection tokens in the LLM itself |
| **Fallback action** | Web search or IDK | Skip retrieval or re-retrieve |
| **Implementation** | Standard LLM + evaluator | Requires specially trained model |
| **Best for** | Incomplete corpus, unpredictable queries | Reducing unnecessary retrieval, latency optimization |

CRAG is more immediately deployable — it does not require a specially trained model. Self-RAG requires a model fine-tuned to produce reflection tokens.

---

## Summary

- CRAG adds a retrieval quality evaluation step between retrieval and generation. It transforms RAG from a blind pipeline into a self-aware one.
- The retrieval evaluator classifies retrieval quality as correct, ambiguous, or incorrect. Implementations range from cross-encoder score thresholds (fast) to LLM-based evaluation (accurate) to hybrid (best of both).
- Corrective actions: use local context (correct), combine local + web (ambiguous), fall back to web search or IDK (incorrect).
- Knowledge refinement strips irrelevant content from both local and web sources before generation, reducing noise in the LLM context.
- Without web search: corrective actions include query reformulation and retry, broader corpus search, or graceful IDK with helpful redirects.
- CRAG adds value when the corpus is incomplete, queries are unpredictable, or wrong answers are costly. It adds latency — weigh the trade-off.

---

## What's Next

Lesson 5.2 covers Self-RAG in depth — the selective retrieval architecture where the LLM decides whether to retrieve, evaluates retrieved content, and critiques its own output using reflection tokens.