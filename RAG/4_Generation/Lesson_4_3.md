# Lesson 4.3 — Long-Context Generation: Stuffing, Iterative, and Map-Reduce Patterns

---

## When Standard RAG Generation Is Not Enough

Standard RAG retrieves a handful of chunks, packs them into the context window, and generates one response. This works when:
- The answer lives in a small number of focused chunks.
- The query is specific enough that 5–10 chunks provide complete context.
- The total context fits comfortably within the LLM's window.

It breaks down for a class of queries that require synthesizing across many documents or large documents:

- "Summarize all 47 customer complaints about Product X from Q3."
- "What are the common themes across our 12 vendor contracts?"
- "Analyze all risk factors mentioned in this 200-page annual report."
- "Compare our marketing strategy across all quarterly plans from 2022–2024."

For these queries, retrieval returns too many or too large chunks to fit in a single context window, and the answer genuinely requires looking at all of them — not just the top-5 most similar ones.

This lesson covers three architectural patterns for handling this class of problem.

---

## Pattern 1 — Context Stuffing

The simplest approach: retrieve everything relevant and stuff it all into a large context window.

### When It Works

With models offering 128K (GPT-4o) or 200K (Claude) token context windows, a surprisingly large amount of content fits. A 200-page document is roughly 100,000–150,000 tokens — within range for a single call.

```python
async def context_stuffing(
    query: str,
    all_relevant_chunks: list[dict],
    llm_client,
    model: str = "gpt-4o",
    max_context_tokens: int = 120_000
) -> str:
    """
    Pack all relevant content into a single large context call.
    Best for medium-length document collections that fit within the window.
    """
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    # Greedily add chunks until we hit the token budget
    context_parts = []
    tokens_used = 0
    
    for chunk in all_relevant_chunks:
        chunk_text = chunk["text"]
        chunk_tokens = len(enc.encode(chunk_text))
        
        if tokens_used + chunk_tokens > max_context_tokens:
            break
        
        context_parts.append(chunk_text)
        tokens_used += chunk_tokens
    
    full_context = "\n\n---\n\n".join(context_parts)
    
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a thorough analyst. Read all provided context carefully before responding."
            },
            {
                "role": "user",
                "content": f"Context:\n{full_context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on all the provided context."
            }
        ],
        max_tokens=2000
    )
    
    return response.choices[0].message.content
```

### Limitations of Context Stuffing

**Lost in the middle is severe at large scales.** Research shows LLM attention quality degrades significantly for context over 32K tokens. At 100K tokens, information in the middle of the context receives substantially less attention than content near the beginning or end. For synthesis tasks, this means the LLM may systematically miss information buried in the middle of the context.

**Cost scales linearly with context size.** At $5 per million input tokens (GPT-4o), 100K tokens costs $0.50 per query. For a system handling 1,000 queries per day of this type, that is $500/day just in input tokens. At 10,000 queries/day, $5,000/day.

**Latency is high.** Processing 100K tokens takes 5–15 seconds for first token generation, depending on the model.

**Beyond 200K tokens, it is impossible.** No current model handles a million-token context. Truly large corpora require one of the other patterns.

**Use context stuffing when:** The total content is under 50K tokens, you can tolerate the cost, and the query requires holistic understanding that retrieval would fragment (e.g., "analyze this entire contract for risk factors").

---

## Pattern 2 — Iterative / Sequential Generation

Instead of processing all content at once, process it in sequential passes. Each pass builds on the result of the previous one.

### Basic Iterative Accumulation

```python
async def iterative_generation(
    query: str,
    chunks: list[dict],
    llm_client,
    batch_size_tokens: int = 8000
) -> str:
    """
    Process chunks in sequential batches, accumulating findings.
    Each batch sees the accumulated findings from previous batches.
    """
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    
    accumulated_findings = ""
    current_batch = []
    current_tokens = 0
    
    async def process_batch(batch: list[dict], prior_findings: str) -> str:
        batch_text = "\n\n".join(c["text"] for c in batch)
        
        prompt = f"""You are analyzing documents to answer: {query}

Previous findings so far:
{prior_findings if prior_findings else "None yet — this is the first batch."}

New documents to analyze:
{batch_text}

Based on these new documents, what additional relevant information do you find?
Add to or refine the previous findings. Be concise — focus on what is new or changes the picture."""
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        return response.choices[0].message.content
    
    # Process chunks in batches
    for chunk in chunks:
        chunk_tokens = len(enc.encode(chunk["text"]))
        
        if current_tokens + chunk_tokens > batch_size_tokens and current_batch:
            # Process current batch
            accumulated_findings = await process_batch(current_batch, accumulated_findings)
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens
    
    # Process final batch
    if current_batch:
        accumulated_findings = await process_batch(current_batch, accumulated_findings)
    
    # Final synthesis pass
    synthesis_prompt = f"""Based on the following accumulated findings from analyzing
all relevant documents, provide a final comprehensive answer to: {query}

Accumulated findings:
{accumulated_findings}

Final answer:"""
    
    final_response = await llm_client.chat.completions.create(
        model="gpt-4o",  # Use better model for final synthesis
        messages=[{"role": "user", "content": synthesis_prompt}],
        max_tokens=1500,
        temperature=0.1
    )
    
    return final_response.choices[0].message.content
```

### The Drift Problem in Iterative Generation

Iterative generation has a subtle problem: accumulated findings from early batches can bias interpretation of later batches. If the first batch of documents establishes a narrative frame ("the primary issue is X"), subsequent batches get filtered through that frame — even if later documents would, in isolation, suggest a different conclusion ("the primary issue is Y").

Mitigation: on every N batches, run a "recalibration" step that asks the LLM to assess whether the accumulated findings still accurately represent what has been seen so far, given the new context.

```python
async def recalibrate_findings(
    query: str,
    accumulated_findings: str,
    recent_batch_summary: str,
    llm_client
) -> str:
    prompt = f"""Review and recalibrate these accumulated findings.

Original question: {query}

Current accumulated findings:
{accumulated_findings}

New batch revealed:
{recent_batch_summary}

Does the new information change, contradict, or add nuance to previous findings?
Update the accumulated findings to reflect the complete picture so far.
Remove any conclusions that are no longer supported or need qualification."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.0
    )
    return response.choices[0].message.content
```

**When to use iterative generation:**
- When content must be processed in logical order (timeline analysis, sequential reasoning).
- When accumulated context from prior documents is needed to interpret later documents.
- When you need to track how findings evolve across a document set.

**Limitation:** Sequential processing — each batch must wait for the previous one to complete. Total latency = N batches × per-batch LLM latency. For 20 batches at 2 seconds each, that is 40 seconds total. Not suitable for real-time user-facing queries.

---

## Pattern 3 — Map-Reduce Generation

Map-reduce is the most powerful pattern for large-scale synthesis. It processes batches in parallel (map), then combines the results (reduce). This makes it much faster than iterative processing.

```
                     [Query]
                        |
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   [Batch 1]        [Batch 2]       [Batch 3]    ← MAP (parallel)
   Map LLM call     Map LLM call    Map LLM call
        ↓               ↓               ↓
   [Summary 1]     [Summary 2]     [Summary 3]
        └───────────────┼───────────────┘
                        ↓
                  [Reduce LLM call]             ← REDUCE (single)
                        ↓
                 [Final Answer]
```

### Map Step

Each batch of chunks is independently summarized or analyzed with respect to the query.

```python
async def map_step(
    query: str,
    chunk_batch: list[dict],
    llm_client
) -> str:
    """
    Process one batch of chunks and extract relevant information.
    Runs in parallel across all batches.
    """
    batch_text = "\n\n---\n\n".join(c["text"] for c in chunk_batch)
    
    prompt = f"""Analyze the following documents to extract information relevant 
to answering this question: {query}

Documents:
{batch_text}

Extract and summarize:
1. All facts, figures, and data points relevant to the question
2. Any positions, arguments, or viewpoints expressed
3. Any contradictions or tensions with the question topic
4. Key quotes that directly address the question

If these documents contain NO relevant information, respond with: NOT_RELEVANT

Be concise. Focus only on what is relevant to the question."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",  # Fast, cheap model for parallel map calls
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.0
    )
    
    result = response.choices[0].message.content
    return result if result.strip() != "NOT_RELEVANT" else ""
```

### Reduce Step

Combine all map summaries into a final answer.

```python
async def reduce_step(
    query: str,
    map_summaries: list[str],
    llm_client,
    source_metadata: list[dict] = None
) -> str:
    """
    Synthesize map summaries into a final comprehensive answer.
    """
    # Filter out empty/not-relevant summaries
    relevant_summaries = [s for s in map_summaries if s.strip()]
    
    if not relevant_summaries:
        return "No relevant information was found in the provided documents."
    
    # Format summaries with source attribution if available
    if source_metadata:
        formatted = []
        for i, (summary, meta) in enumerate(zip(relevant_summaries, source_metadata)):
            if summary:
                source_label = meta.get("doc_title", f"Source {i+1}")
                formatted.append(f"From {source_label}:\n{summary}")
        combined_summaries = "\n\n".join(formatted)
    else:
        combined_summaries = "\n\n---\n\n".join(relevant_summaries)
    
    prompt = f"""You have analyzed multiple documents to answer this question:
{query}

Here are the findings from each document batch:

{combined_summaries}

Now synthesize these findings into a comprehensive, well-organized answer.

Instructions:
- Identify the main themes and patterns across all sources
- Note where sources agree or provide complementary information  
- Note where sources contradict each other (do not resolve contradictions — report them)
- Use specific evidence from the findings to support each point
- Organize by theme/category, not by source
- Be comprehensive but avoid repetition"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",  # Use best model for final synthesis
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.1
    )
    
    return response.choices[0].message.content
```

### Complete Map-Reduce Pipeline

```python
async def map_reduce_generation(
    query: str,
    chunks: list[dict],
    llm_client,
    batch_token_size: int = 6000,
    max_concurrent_maps: int = 10
) -> dict:
    """
    Full map-reduce generation pipeline.
    Returns answer and processing metadata.
    """
    import tiktoken
    from asyncio import Semaphore
    
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    
    # Step 1: Build batches
    batches = []
    current_batch = []
    current_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = len(enc.encode(chunk["text"]))
        
        if current_tokens + chunk_tokens > batch_token_size and current_batch:
            batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    # Step 2: Map — process all batches in parallel (with concurrency limit)
    semaphore = Semaphore(max_concurrent_maps)
    
    async def map_with_semaphore(batch: list[dict]) -> str:
        async with semaphore:
            return await map_step(query, batch, llm_client)
    
    map_tasks = [map_with_semaphore(batch) for batch in batches]
    map_summaries = await asyncio.gather(*map_tasks)
    
    relevant_count = sum(1 for s in map_summaries if s.strip())
    
    # Step 3: Reduce — synthesize all map outputs
    final_answer = await reduce_step(
        query=query,
        map_summaries=list(map_summaries),
        llm_client=llm_client
    )
    
    return {
        "answer": final_answer,
        "total_batches": len(batches),
        "relevant_batches": relevant_count,
        "total_chunks": len(chunks),
        "map_summaries": list(map_summaries)  # For debugging/tracing
    }
```

### Multi-Level Map-Reduce

For very large document sets, a single reduce step may receive too many map summaries to handle well. Use hierarchical reduce — reduce the map outputs in groups, then reduce those group summaries:

```
Map    → [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10]
Reduce1 → [R_A(S1+S2+S3), R_B(S4+S5+S6), R_C(S7+S8+S9+S10)]
Reduce2 → Final answer from (R_A + R_B + R_C)
```

```python
async def hierarchical_reduce(
    query: str,
    map_summaries: list[str],
    llm_client,
    group_size: int = 5
) -> str:
    """
    Multi-level reduction for very large numbers of map outputs.
    """
    relevant = [s for s in map_summaries if s.strip()]
    
    if len(relevant) <= group_size:
        # Small enough for single reduce
        return await reduce_step(query, relevant, llm_client)
    
    # Group into chunks and do intermediate reductions
    intermediate_summaries = []
    
    for i in range(0, len(relevant), group_size):
        group = relevant[i:i + group_size]
        group_summary = await reduce_step(
            query=query,
            map_summaries=group,
            llm_client=llm_client
        )
        intermediate_summaries.append(group_summary)
    
    # Final reduce over intermediate summaries
    return await reduce_step(query, intermediate_summaries, llm_client)
```

---

## Choosing the Right Pattern

The decision depends on three factors: total content volume, whether sequential processing matters, and latency requirements.

```
                        Content volume
                    Small          Large
                   (<50K t)       (>50K t)
                    ┌─────────────┬─────────────┐
Sequential     Low  │  Context    │  Iterative  │
ordering       need │  Stuffing   │  Generation │
required?          ├─────────────┼─────────────┤
               High │  Iterative  │  Iterative  │
                    │  (w/ prior  │  (w/ prior  │
                    │  context)   │  context)   │
                    └─────────────┴─────────────┘

Latency        Low tolerance → Map-Reduce (parallel)
requirement:   High tolerance → Iterative (sequential) or Map-Reduce
```

More specifically:

**Use context stuffing when:**
- Total content is under 50K tokens.
- Holistic reading matters (the LLM needs to see relationships across the entire document).
- Query is about a single document ("summarize this contract", "find all risks in this report").
- Latency and cost are not the primary concern.

**Use iterative generation when:**
- Content must be processed in a meaningful order (timeline analysis, building understanding sequentially).
- Later documents reference or modify earlier ones.
- The accumulated context from prior documents is needed to correctly interpret subsequent ones.
- Latency is not critical (background processing, async jobs).

**Use map-reduce when:**
- Large volumes of independent documents need synthesis ("summarize all 500 customer reviews about Product X").
- Parallel processing is needed to meet latency requirements.
- Documents are largely independent — each can be analyzed without knowledge of the others.
- The query is a synthesis or aggregation question ("what are the common themes?", "list all occurrences of X").

---

## Handling Contradictions in Multi-Document Synthesis

When synthesizing across many documents, contradictions are common and important. The map-reduce reduce step must handle them explicitly.

```python
async def contradiction_aware_reduce(
    query: str,
    map_summaries: list[str],
    llm_client
) -> dict:
    """
    Reduce with explicit contradiction detection and reporting.
    """
    combined = "\n\n---\n\n".join(s for s in map_summaries if s.strip())
    
    prompt = f"""Synthesize the following document summaries to answer: {query}

Document summaries:
{combined}

In your synthesis:
1. CONSENSUS: What do most or all documents agree on?
2. CONTRADICTIONS: Where do documents directly contradict each other? 
   List each contradiction with the conflicting claims.
3. GAPS: What aspects of the question are not addressed by any document?
4. ANSWER: A direct answer to the question, noting where there is and is not consensus.

Format as JSON with keys: consensus, contradictions, gaps, answer"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.0
    )
    
    try:
        import json
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"answer": response.choices[0].message.content, "parse_error": True}
```

Contradiction-aware synthesis is particularly important for:
- Legal document analysis (conflicting clause interpretations).
- Multi-vendor contract comparison (different terms for the same concept).
- Research synthesis (conflicting study findings).
- Historical document analysis (events described differently across sources).

---

## Cost and Latency Comparison

For 100 documents, each 1,000 tokens (100K total tokens):

| Pattern | LLM Calls | Input Tokens | Latency | Cost (GPT-4o) |
|---|---|---|---|---|
| Context stuffing | 1 | 100K | 10–15s | ~$0.50 |
| Iterative (10 batches) | 10 + 1 synthesis | 110K total | 30–60s | ~$0.10 (mini) + $0.05 (synthesis) |
| Map-reduce (10 batches, parallel) | 10 + 1 | 110K total | 5–8s | ~$0.10 (mini maps) + $0.05 (reduce) |

Map-reduce wins on both latency (parallelism) and cost (cheap mini model for maps, single capable model for reduce). Context stuffing has the highest fidelity but highest cost. Iterative is the slowest but handles sequential dependencies.

---

## Practical Tip: Adaptive Pattern Selection

Build a router that selects the appropriate pattern based on query characteristics and content volume:

```python
async def adaptive_long_context_generation(
    query: str,
    chunks: list[dict],
    llm_client,
    embedding_model
) -> str:
    
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    total_tokens = sum(len(enc.encode(c["text"])) for c in chunks)
    
    # Detect query characteristics
    needs_sequential = any(phrase in query.lower() for phrase in [
        "over time", "chronological", "timeline", "history", "evolution",
        "how did", "changed", "progression"
    ])
    
    needs_synthesis = any(phrase in query.lower() for phrase in [
        "summarize all", "compare all", "common themes", "across all",
        "aggregate", "overall", "in total", "throughout"
    ])
    
    # Route to appropriate pattern
    if total_tokens < 50_000 and not needs_synthesis:
        # Small enough to stuff, and not a large-scale synthesis query
        return await context_stuffing(query, chunks, llm_client)
    
    elif needs_sequential:
        # Must process in order
        return await iterative_generation(query, chunks, llm_client)
    
    else:
        # Large scale or synthesis — map-reduce
        result = await map_reduce_generation(query, chunks, llm_client)
        return result["answer"]
```

---

## Summary

- Standard RAG generation (retrieve top-K, single context window) fails for queries requiring synthesis across many documents or very large documents.
- Context stuffing packs all content into a large context window. Works for under 50K tokens and holistic document queries. High cost and lost-in-the-middle quality degradation at large scale.
- Iterative generation processes batches sequentially, each building on accumulated findings. Handles sequential dependencies but is slow (no parallelism) and prone to early-batch framing bias.
- Map-reduce processes batches in parallel (map) then synthesizes the results (reduce). Fastest pattern for large independent document sets. Use cheap fast models for map, best model for reduce.
- Multi-level reduce handles very large numbers of map outputs by reducing in groups hierarchically.
- Contradiction-aware reduce explicitly identifies and reports conflicts across sources rather than silently blending them.
- Choose based on: content volume (stuffing if small), sequential dependency (iterative if yes), synthesis need (map-reduce if large independent set).

---

## What's Next

Lesson 4.4 covers structured output generation from retrieved context — when the answer needs to be JSON, a table, a report, or another structured format, and how to reliably extract and validate structured outputs from RAG pipelines.