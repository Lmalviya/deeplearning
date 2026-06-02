# Lesson 4.2 — Handling Conflicting Context vs. Parametric Knowledge

---

## The Conflict Problem in Precise Terms

Every LLM carries two knowledge sources simultaneously:

**Parametric knowledge:** Information encoded in the model's weights during pretraining and fine-tuning. This is static — it reflects the world as it was in the training data, up to the training cutoff. It cannot be updated without retraining.

**Contextual knowledge:** Information provided in the prompt at inference time — your retrieved chunks. This is dynamic, current, and specific to your domain.

In a perfectly designed RAG system, these two sources never conflict because the LLM defers entirely to contextual knowledge for domain-specific answers. In practice, they conflict constantly:

- Your HR policy was updated last month. The LLM's training data contains the old version. Both are "known" to the model simultaneously.
- Your retrieved context says the CEO is Jane Smith. The model was trained when John Doe held the role.
- Your internal process has 7 steps. A general process the model learned has 5 steps. They overlap but differ in specifics.
- Your legal contract uses a term in a domain-specific way that differs from its general usage.

When conflict occurs, LLMs do not reliably defer to context — they sometimes blend the two sources, producing hybrid answers that are partially wrong in subtle ways, or they ignore the context entirely and answer from parametric memory.

This lesson covers why this happens, how to detect it, and the layered strategies beyond prompt instructions that address it systematically.

---

## Why LLMs Do Not Always Defer to Context

Understanding why the problem occurs is necessary to fix it properly.

### Confidence Asymmetry

LLMs assign implicit confidence levels to information based on how consistently it appeared in training data. If the LLM saw "The standard notice period for employee termination is 2 weeks" across thousands of documents during pretraining, it holds that fact with high confidence. When your retrieved context says "Notice period is 90 days per our enterprise agreement," the LLM faces a conflict between a high-confidence parametric belief and a single contextual statement.

In this situation, LLMs frequently either:
- Ignore the context and state the parametric belief.
- Blend them: "While standard practice is 2 weeks, your agreement specifies 90 days."
- Correctly defer to context but with hedging that undermines the authority of your actual policy.

The second outcome — blending — is particularly dangerous because it sounds plausible and produces a response that appears to use the context while actually diluting it with incorrect information.

### Attention Competition

In a long context window with many retrieved chunks, the LLM's attention is distributed across thousands of tokens. Parametric knowledge does not "compete" for attention — it is always available. Contextual knowledge competes with other contextual content. If the specific conflicting detail is not prominently placed (first or last in context, or bolded, or repeated), it may lose the attention competition to parametric memory.

### Instruction-Following Degradation

Even with explicit grounding instructions ("use ONLY the provided context"), LLMs do not follow instructions perfectly, especially when:
- The context is very long and the instruction is far from the relevant passage.
- The instruction conflicts with the model's training to be "helpful" (which creates a pull toward producing complete-sounding answers even from memory).
- The model was fine-tuned on data where context override was not consistently enforced.

This is why prompt instructions alone are necessary but not sufficient.

---

## Detecting Conflicts at Inference Time

Before resolving conflicts, you need to detect them. There are two detection approaches: pre-generation (detect before sending to LLM) and post-generation (detect in the LLM's output).

### Pre-Generation Conflict Detection

Before assembling the final prompt, check whether retrieved context contradicts the LLM's likely parametric beliefs on the same topic.

This is done by querying the LLM about the topic without context (pure parametric recall), then comparing to the retrieved content.

```python
async def detect_parametric_conflict(
    query: str,
    retrieved_context: str,
    llm_client,
    conflict_threshold: float = 0.3
) -> dict:
    """
    Check if retrieved context likely conflicts with LLM parametric knowledge.
    
    Strategy: Ask the LLM what it knows about the topic without context,
    then compare to the retrieved content.
    """
    
    # Step 1: Get LLM's parametric answer (no context)
    parametric_response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer briefly based on your training knowledge."},
            {"role": "user", "content": f"What do you know about: {query}"}
        ],
        max_tokens=300,
        temperature=0.0
    )
    parametric_answer = parametric_response.choices[0].message.content
    
    # Step 2: Ask LLM to assess whether the retrieved context agrees
    conflict_check_prompt = f"""Compare these two pieces of information about the same topic.

Information A (general knowledge):
{parametric_answer}

Information B (specific document):
{retrieved_context[:2000]}

Do these two pieces of information contradict each other on any specific facts, 
figures, dates, or policies? 

Respond with JSON:
{{
    "conflict_detected": true/false,
    "conflict_description": "description of specific contradiction or 'none'",
    "conflicting_claim_in_A": "the specific claim from A that conflicts",
    "conflicting_claim_in_B": "the specific claim from B that conflicts"
}}"""
    
    detection_response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": conflict_check_prompt}],
        max_tokens=200,
        temperature=0.0
    )
    
    try:
        import json
        result = json.loads(detection_response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        return {"conflict_detected": False, "error": "parse_failure"}
```

**Cost consideration:** Pre-generation detection adds two extra LLM calls. This is expensive on the critical path. Use it selectively:
- Only for queries in domains known to have high conflict risk (regulatory, policy, factual/numerical).
- Only when the query matches patterns that suggest high-confidence parametric knowledge (common facts, well-known statistics, standard practices).
- As a background process for high-stakes queries where latency budget is relaxed.

### Post-Generation Conflict Detection

Detect conflicts after the LLM has generated its response, by checking whether the response is faithful to the retrieved context.

```python
async def check_response_faithfulness(
    query: str,
    retrieved_context: str,
    llm_response: str,
    llm_client
) -> dict:
    """
    Check if LLM response is faithful to retrieved context or introduces
    information not present in or contradicting the context.
    """
    
    prompt = f"""You are evaluating whether an AI response faithfully represents 
the information in provided source documents.

Query: {query}

Source documents:
{retrieved_context}

AI Response:
{llm_response}

Evaluate:
1. Does the response contain any claims not supported by the source documents?
2. Does the response contradict any information in the source documents?
3. Does the response add information from outside the provided context?

Respond with JSON:
{{
    "faithful": true/false,
    "unsupported_claims": ["list of claims in response not found in context"],
    "contradictions": ["list of claims that contradict context"],
    "external_additions": ["list of information added from outside context"],
    "faithfulness_score": 0.0-1.0
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",  # Need capable model for nuanced faithfulness check
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.0
    )
    
    try:
        import json
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"faithful": None, "error": "parse_failure"}
```

Post-generation detection is used in:
- **Evaluation pipelines** (offline, not on the critical path) to measure faithfulness across a sample of queries.
- **Online monitoring** on a sample of live queries to track faithfulness degradation over time.
- **High-stakes query handlers** where you can afford the latency of a faithfulness check before returning the response.

---

## Resolution Strategy 1 — Explicit Conflict Acknowledgment in Prompt

When pre-generation detection identifies a conflict, modify the prompt to explicitly flag it.

```python
def build_conflict_aware_prompt(
    query: str,
    context: str,
    conflict_info: dict
) -> str:
    
    base_prompt = f"""Context documents:
{context}

Question: {query}"""
    
    if conflict_info.get("conflict_detected"):
        conflict_note = f"""
IMPORTANT NOTE: The provided context may differ from general knowledge on this topic.
Specifically: {conflict_info['conflict_description']}

You MUST use the information from the context documents above, NOT your general 
knowledge. The context represents the current, authoritative information for this 
specific situation."""
        
        return conflict_note + "\n\n" + base_prompt
    
    return base_prompt
```

The conflict note is injected between the system prompt's grounding instruction and the context block, making it prominent. LLMs respond better to specific conflict identification ("the context says 90 days, which differs from the common 2-week standard") than to generic authority assertions.

---

## Resolution Strategy 2 — Confidence Calibration Through Framing

Reframe the retrieved context to signal its authority relative to general knowledge. The goal is to increase the LLM's implicit confidence in the contextual information.

```python
def frame_authoritative_context(
    chunks: list[dict],
    authority_signal: str = "official"
) -> str:
    """
    Frame context chunks with authority signals that increase the LLM's
    confidence in contextual over parametric knowledge.
    """
    
    authority_phrases = {
        "official": "Official policy document",
        "current": "Current version (supersedes previous policies)",
        "authoritative": "Authoritative source document",
        "verified": "Verified and approved documentation",
        "legal": "Legally binding agreement"
    }
    
    phrase = authority_phrases.get(authority_signal, "Source document")
    
    formatted_chunks = []
    for i, chunk in enumerate(chunks, 1):
        doc_title = chunk["metadata"].get("doc_title", "Document")
        version = chunk["metadata"].get("version", "")
        date = chunk["metadata"].get("effective_date", "")
        
        header_parts = [f"[{i}] {phrase}: {doc_title}"]
        if version:
            header_parts.append(f"Version {version}")
        if date:
            header_parts.append(f"Effective {date}")
        
        header = " | ".join(header_parts)
        formatted_chunks.append(f"{header}\n{chunk['text']}")
    
    return "\n\n".join(formatted_chunks)
```

The authority framing ("Official policy document | Version 3.2 | Effective January 2024") signals to the LLM that this is a specific, current, authoritative source — not a general reference that might be superseded by more recent training data.

---

## Resolution Strategy 3 — Evidence Extraction Before Generation

Instead of asking the LLM to both find evidence and generate an answer simultaneously, split these into two steps:

**Step 1:** Ask the LLM to extract the specific relevant facts from the context.
**Step 2:** Ask the LLM to construct the final answer from the extracted facts only.

```python
async def two_step_generation(
    query: str,
    context: str,
    llm_client
) -> dict:
    """
    Two-step generation that separates evidence extraction from answer synthesis.
    Reduces parametric contamination by grounding each step explicitly.
    """
    
    # Step 1: Extract relevant facts from context
    extraction_prompt = f"""From the following context documents, extract all 
information directly relevant to answering this question.

Question: {query}

Context:
{context}

Extract the relevant facts as a bulleted list. Include exact figures, dates, 
and quotes. Only include information from the context — do not add any external 
knowledge.

Extracted facts:"""
    
    extraction_response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extraction_prompt}],
        max_tokens=500,
        temperature=0.0
    )
    
    extracted_facts = extraction_response.choices[0].message.content
    
    # Step 2: Generate answer from extracted facts only
    synthesis_prompt = f"""Using ONLY the following extracted facts, answer the question.
Do not add any information that is not in the extracted facts list.

Question: {query}

Extracted facts (these are the ONLY source of information for your answer):
{extracted_facts}

Answer:"""
    
    synthesis_response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": synthesis_prompt}],
        max_tokens=600,
        temperature=0.1
    )
    
    return {
        "answer": synthesis_response.choices[0].message.content,
        "extracted_facts": extracted_facts
    }
```

**Why this works:** The extraction step forces the LLM to explicitly identify what is in the context before generating. This makes parametric contamination in Step 2 harder because the LLM's "working memory" for the synthesis is explicitly the extracted facts list, not the full context plus parametric knowledge.

**Cost:** Two LLM calls instead of one. Acceptable for high-stakes queries where faithfulness is critical; too expensive for high-volume applications.

---

## Resolution Strategy 4 — Retrieval Confidence Scoring

When the retrieval system is uncertain (low similarity scores, few results), the LLM is more likely to fall back to parametric knowledge. Use retrieval confidence as a signal to adjust generation behavior.

```python
def compute_retrieval_confidence(
    reranked_results: list[dict],
    min_score_threshold: float = 0.4
) -> dict:
    """
    Assess how confident the retrieval system is about this query.
    """
    if not reranked_results:
        return {"confidence": "none", "score": 0.0}
    
    top_score = reranked_results[0].get("rerank_score", 0)
    mean_score = sum(r.get("rerank_score", 0) for r in reranked_results[:5]) / min(5, len(reranked_results))
    above_threshold = sum(1 for r in reranked_results if r.get("rerank_score", 0) >= min_score_threshold)
    
    if top_score >= 0.7 and above_threshold >= 3:
        confidence = "high"
    elif top_score >= 0.4 and above_threshold >= 1:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "confidence": confidence,
        "top_score": top_score,
        "mean_score": mean_score,
        "chunks_above_threshold": above_threshold
    }


def build_confidence_adaptive_prompt(
    query: str,
    context: str,
    retrieval_confidence: dict
) -> str:
    """
    Adapt the generation prompt based on retrieval confidence.
    Low confidence = stronger warning about using context only.
    """
    
    if retrieval_confidence["confidence"] == "low":
        confidence_note = """WARNING: The retrieved documents may not contain 
complete information about this query. If the context below does not directly 
address the question, say so explicitly rather than using general knowledge to fill gaps."""
    
    elif retrieval_confidence["confidence"] == "medium":
        confidence_note = """Note: Use the provided context as your primary source.
If information seems incomplete, acknowledge the limitation."""
    
    else:
        confidence_note = ""
    
    return f"""{confidence_note}

Context:
{context}

Question: {query}"""
```

Low retrieval confidence should trigger either:
- A stronger IDK-oriented prompt (as above).
- A fallback to CRAG behavior — discard the low-confidence context and either respond with IDK or trigger a secondary retrieval strategy.

---

## Resolution Strategy 5 — CRAG-Style Fallback

When retrieved context is of insufficient quality for a query, do not pass it to the LLM at all. Instead, either:
- Return IDK and direct the user to the appropriate resource.
- Trigger a secondary retrieval strategy (web search, different index, broader query).

```python
async def crag_generation(
    query: str,
    retrieved_chunks: list[dict],
    retrieval_confidence: dict,
    llm_client,
    fallback_retriever=None
) -> dict:
    """
    CRAG-style generation: evaluate retrieval quality before generating.
    """
    
    if retrieval_confidence["confidence"] == "none":
        return {
            "answer": "I don't have any relevant documents to answer this question.",
            "source": "idk",
            "chunks_used": []
        }
    
    if retrieval_confidence["confidence"] == "low":
        if fallback_retriever:
            # Try a secondary retrieval strategy
            fallback_chunks = await fallback_retriever.retrieve(query)
            
            if fallback_chunks:
                retrieved_chunks = fallback_chunks
                retrieval_confidence = {"confidence": "medium", "source": "fallback"}
            else:
                return {
                    "answer": "I don't have sufficient information in the available documents to answer this confidently.",
                    "source": "idk",
                    "chunks_used": []
                }
        else:
            # No fallback — return IDK
            return {
                "answer": "The available documents don't seem to contain enough information to answer this question accurately.",
                "source": "idk",
                "chunks_used": []
            }
    
    # Confidence is medium or high — proceed with generation
    context = format_context(retrieved_chunks)
    answer = await generate_with_context(query, context, llm_client)
    
    return {
        "answer": answer,
        "source": "retrieval",
        "confidence": retrieval_confidence["confidence"],
        "chunks_used": retrieved_chunks
    }
```

---

## Detecting Conflicts at Scale: Offline Monitoring

For production systems, run faithfulness monitoring on a sample of queries to detect systematic conflict patterns.

```python
async def run_faithfulness_audit(
    query_response_pairs: list[dict],  # [{query, context, response}]
    llm_client,
    sample_size: int = 100
) -> dict:
    """
    Offline audit: measure faithfulness across a sample of production queries.
    Run nightly or weekly.
    """
    import random
    
    sample = random.sample(query_response_pairs, min(sample_size, len(query_response_pairs)))
    
    faithfulness_scores = []
    conflict_patterns = []
    
    for item in sample:
        result = await check_response_faithfulness(
            query=item["query"],
            retrieved_context=item["context"],
            llm_response=item["response"],
            llm_client=llm_client
        )
        
        faithfulness_scores.append(result.get("faithfulness_score", 0))
        
        if not result.get("faithful", True):
            conflict_patterns.append({
                "query": item["query"],
                "unsupported_claims": result.get("unsupported_claims", []),
                "contradictions": result.get("contradictions", [])
            })
    
    mean_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    
    return {
        "mean_faithfulness": mean_faithfulness,
        "sample_size": len(sample),
        "unfaithful_count": len(conflict_patterns),
        "unfaithful_rate": len(conflict_patterns) / len(sample),
        "common_conflict_patterns": conflict_patterns[:10]  # Top 10 for review
    }
```

Run this weekly and alert when faithfulness drops below your threshold (typically 0.85 for general applications, 0.95 for high-stakes domains). Review the conflict patterns to identify systematic issues — if the same type of conflict appears repeatedly, it indicates a retrieval gap (the parametric knowledge the LLM is falling back on should be retrieved from your corpus instead).

---

## The Full Resolution Priority Order

When you identify a conflict, apply resolutions in this order of preference:

```
Priority 1 — Fix the retrieval (preferred).
If the LLM is falling back to parametric knowledge, it is often because the 
correct context was not retrieved. Add the missing document, improve chunking 
on the relevant section, or fix the query understanding to retrieve better.
Parametric fallback is often a symptom of retrieval failure.

Priority 2 — Strengthen the prompt instruction.
If retrieval is correct but the LLM still ignores context, escalate the 
grounding instruction (Lesson 4.1 escalation strategies).

Priority 3 — Use two-step generation.
For consistently problematic query types, add the extraction step before synthesis.

Priority 4 — Add pre-generation conflict detection.
For high-stakes queries where neither of the above is sufficient, add the 
detection+flagging step to explicitly tell the LLM about the conflict.

Priority 5 — CRAG fallback.
If retrieval confidence is too low to trust, prefer IDK over a parametric answer.
```

> **Interview note:** "How do you handle conflicts between what the LLM knows and what your documents say?" — The answer interviewers want: (1) the problem exists because LLMs have strong parametric beliefs that compete with contextual information, (2) prompt grounding instructions help but are not sufficient alone, (3) retrieval confidence scoring lets you detect when the LLM is likely to fall back on memory, (4) two-step extraction-then-synthesis reduces contamination for critical domains, (5) faithfulness monitoring detects systematic conflicts offline. Show you understand this is a multi-layer problem, not a one-prompt fix.

---

## Summary

- Parametric vs. contextual conflict is a fundamental LLM behavior problem — models do not reliably defer to retrieved context when they hold confident parametric beliefs.
- Conflicts occur because of confidence asymmetry (parametric beliefs are stronger than single-context statements), attention competition (contextual information must compete for LLM attention), and imperfect instruction following.
- Detect conflicts pre-generation (check context against parametric recall before prompting) or post-generation (check response faithfulness against context after generating).
- Resolution strategies in priority order: fix retrieval first, then strengthen prompt, then two-step generation, then explicit conflict flagging, then CRAG fallback.
- Two-step generation — extract facts from context, then synthesize from extracted facts only — is the most effective single-query resolution for high-stakes domains.
- Retrieval confidence scoring adapts generation behavior based on how good the retrieval was. Low confidence triggers stronger IDK orientation.
- Run offline faithfulness audits on production query samples weekly. Systematic conflicts in audit results indicate retrieval gaps — the content the LLM falls back on should be in your index.

---

## What's Next

Lesson 4.3 covers long-context generation strategies — how to handle queries that require synthesizing information from many documents or very large documents: context stuffing, iterative generation, map-reduce, and when each pattern applies.