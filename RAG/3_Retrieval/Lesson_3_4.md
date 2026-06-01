# Lesson 3.4 — Query Understanding: Rewriting, Expansion, and Decomposition

---

## Why Raw Queries Are Poor Retrieval Signals

The user's raw query is almost never the optimal retrieval signal. This is not a criticism of users — it is a fundamental property of how people communicate. People write queries the way they think and speak, not the way documents are written.

The vocabulary gap between how users ask questions and how documents are written is one of the oldest problems in information retrieval. It was identified in 1960. Every technique in this lesson is a specific strategy to bridge this gap.

The challenge is that query understanding is a double-edged sword. Done well, it dramatically improves recall. Done poorly — wrong rewrite, overly aggressive expansion, bad decomposition — it actively hurts retrieval by introducing irrelevant signals and diluting the query intent. This lesson covers each technique with equal attention to when it helps and when it hurts.

---

## The Five Problems Query Understanding Solves

Before diving into techniques, be precise about the problems:

**1. Vocabulary mismatch.** User asks "how to get out of a deal" — document says "contract termination clauses." No shared vocabulary despite identical intent.

**2. Ambiguity.** User asks "policy renewal" in a system with both HR policies and insurance policies. The query is underspecified.

**3. Missing context.** In a multi-turn conversation, "what about the deductible?" refers to a topic established three turns ago. The raw query has no retrieval signal.

**4. Query complexity.** "Compare the refund policies of our Pro and Enterprise plans and explain which is better for a small business" contains multiple distinct information needs that cannot all be satisfied by a single retrieval pass.

**5. Query length mismatch.** Very short queries ("maternity leave") give the embedding model too little signal. Very long queries (100+ words) dilute the embedding — the embedding model tries to represent too many concepts at once.

Different techniques address different problems. Choose based on which problem your system actually has.

---

## Technique 1 — Query Rewriting

Query rewriting transforms the raw query into a cleaner, more retrieval-friendly form using an LLM.

### What Good Rewriting Looks Like

```
Raw:      "how do i get out of my contract early"
Rewritten: "early termination clause contract exit conditions penalty fees"

Raw:      "what happens if someone steals my stuff at work"
Rewritten: "workplace theft policy employee property loss compensation liability"

Raw:      "can i work from home"
Rewritten: "remote work policy work from home eligibility requirements approval process"
```

The rewriter converts colloquial, intent-based queries into formal, keyword-rich queries that match document vocabulary better.

### Implementation

```python
async def rewrite_query(
    query: str,
    domain_context: str = "",
    llm_client = None
) -> str:
    """
    Rewrite a user query for better retrieval.
    Returns the rewritten query string.
    """
    
    system_prompt = """You are a search query optimizer. Your task is to rewrite 
user questions into better search queries for a document retrieval system.

Rules:
- Use formal, technical language that matches how documents are written
- Include synonyms and related terms that might appear in relevant documents
- Expand abbreviations and informal terms
- Preserve all specific terms, codes, or identifiers exactly as given
- Keep the output concise (under 30 words)
- Do NOT add information that was not implied by the original query
- Return ONLY the rewritten query, nothing else"""

    user_prompt = f"""Domain: {domain_context}
Original query: {query}
Rewritten query:"""

    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",  # Use a small, fast model — not your most capable one
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=100,
        temperature=0.1  # Low temperature — you want deterministic rewrites
    )
    
    return response.choices[0].message.content.strip()
```

### Design Decisions

**Keep the original query.** A rewriter can lose important nuance. Run retrieval on both the original and the rewritten query, then merge results with RRF. Never discard the original entirely.

```python
async def retrieve_with_rewrite(query: str, ...) -> list[dict]:
    rewritten = await rewrite_query(query)
    
    # Retrieve with both original and rewritten
    original_results = await retrieve(query)
    rewritten_results = await retrieve(rewritten)
    
    # RRF merge — original and rewritten get equal weight
    return reciprocal_rank_fusion([original_results, rewritten_results])
```

**Use a small, fast model.** Query rewriting is on the critical path — every query goes through it. GPT-4o-mini, Claude Haiku, or a fine-tuned small model adds 100–200ms. GPT-4o or Claude Sonnet adds 500–1500ms. The quality improvement from a larger model is rarely worth the latency for a rewriting task.

**Domain context helps.** Providing the system domain ("HR policy system", "financial compliance", "software engineering knowledge base") helps the rewriter use appropriate formal vocabulary. Without context, the rewriter may guess the wrong domain.

**Low temperature.** Rewriting should be deterministic and precise, not creative. temperature=0.0 or 0.1. Higher temperatures introduce randomness that may occasionally produce better rewrites but more frequently produces irrelevant ones.

### When Rewriting Hurts

**Specific identifiers.** "Form I-94 expiration date" rewritten to "immigration document expiration arrival date" loses the specific form number that BM25 would have matched exactly. The rewriter should preserve codes, numbers, and proper nouns unchanged.

**Simple, precise queries.** "What is the employee referral bonus?" does not need rewriting — the vocabulary already matches typical document language. Rewriting may change it to something worse.

**Add a bypass check:** if the query is already well-formed (uses formal vocabulary, contains specific identifiers, is a simple factual question), skip rewriting.

```python
def should_rewrite(query: str) -> bool:
    """Heuristic: only rewrite if the query is likely to benefit."""
    import re
    
    # Skip if query contains specific codes/identifiers
    if re.search(r'\b[A-Z]{2,}\d+\b|\b\d+\.\d+\.\d+\b', query):
        return False
    
    # Skip if query is already formal and specific (>= 5 words, no contractions)
    words = query.split()
    has_contractions = any("'" in w for w in words)
    if len(words) >= 5 and not has_contractions:
        return False
    
    # Skip very short queries — rewriting cannot add much
    if len(words) <= 2:
        return False
    
    return True
```

---

## Technique 2 — Query Expansion

Instead of replacing the query with one better version, generate multiple alternative queries and retrieve for all of them.

### Multi-Query Expansion

```python
async def expand_query(
    query: str,
    n_variants: int = 3,
    llm_client = None
) -> list[str]:
    """
    Generate n alternative phrasings of the query for retrieval.
    """
    
    prompt = f"""Generate {n_variants} different search queries that would retrieve 
documents relevant to answering this question. Each query should approach the 
topic from a slightly different angle or use different terminology.

Original question: {query}

Return exactly {n_variants} queries, one per line, numbered 1-{n_variants}.
Do not include any other text."""

    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,  # Moderate temperature for variety
        max_tokens=300
    )
    
    raw = response.choices[0].message.content.strip()
    
    # Parse numbered list
    variants = []
    for line in raw.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove "1. " prefix
            query_text = line.split('.', 1)[-1].strip()
            if query_text:
                variants.append(query_text)
    
    return variants[:n_variants]


async def retrieve_with_expansion(query: str, ...) -> list[dict]:
    variants = await expand_query(query, n_variants=3)
    all_queries = [query] + variants  # Include original
    
    # Retrieve for each query variant (can parallelize)
    all_results = await asyncio.gather(*[
        retrieve(q) for q in all_queries
    ])
    
    # RRF across all result lists
    return reciprocal_rank_fusion(list(all_results))
```

**Example expansion:**

```
Original: "what documents do I need to file for FMLA leave"

Variants:
1. "FMLA leave application required paperwork documentation process"
2. "Family Medical Leave Act forms employee submission HR requirements"
3. "medical leave of absence documentation certification physician forms"
```

Each variant captures a different vocabulary angle. Together they significantly increase recall compared to any single query.

### Step-Back Prompting

A specific expansion technique: generate a more general version of the query that retrieves broader context, then also retrieve for the specific query. The general version helps when the specific answer is embedded in a broader discussion.

```python
async def generate_step_back_query(query: str, llm_client) -> str:
    prompt = f"""Given a specific question, generate a more general question 
that covers the broader topic. The general question should retrieve background 
context that helps answer the specific question.

Specific question: {query}
General question:"""

    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.1
    )
    
    return response.choices[0].message.content.strip()
```

**Example:**

```
Specific: "what is the penalty for late payment under Section 8.3 of our vendor contract"
Step-back: "vendor contract payment terms late fees and penalties"
```

Retrieve for both. The step-back query finds the general payment terms section. The specific query finds the exact clause. Together they give complete context.

### When Expansion Hurts

**Noisy expansion.** If the query is ambiguous or the LLM generates off-topic variants, you flood retrieval with irrelevant signals. Five bad query variants plus the original retrieve 6× more noise.

**Latency.** Generating and retrieving for 4 queries takes 4× the retrieval time (partially mitigated by parallelization, but still adds latency). Plus one LLM call for expansion generation.

**The expansion quality check:** Before using expanded queries, verify that they are topically related to the original. A simple check: embed the original query and each variant, compute cosine similarity, discard variants that are too far from the original (similarity < 0.7).

```python
def filter_quality_variants(
    original_query: str,
    variants: list[str],
    embedding_model,
    min_similarity: float = 0.70
) -> list[str]:
    original_embedding = embedding_model.embed(original_query)
    quality_variants = []
    
    for variant in variants:
        variant_embedding = embedding_model.embed(variant)
        similarity = cosine_similarity(original_embedding, variant_embedding)
        
        if similarity >= min_similarity:
            quality_variants.append(variant)
    
    return quality_variants
```

---

## Technique 3 — Sub-Question Decomposition

For complex, multi-part queries, decompose into atomic sub-questions. Each sub-question is independently retrievable and targets a specific fact.

### When Decomposition Is Needed

A query contains multiple distinct information needs when:
- It asks to compare two or more things ("compare A and B").
- It asks about multiple attributes of one thing ("what are the eligibility requirements and the application process for X").
- It requires sequential reasoning ("if X is true, then what is the policy for Y").
- It asks for aggregation ("list all the cases where X applies").

Single-retrieval for these queries produces one chunk that partially addresses one part of the query while missing the rest.

### Implementation

```python
async def decompose_query(
    query: str,
    llm_client = None
) -> list[str]:
    """
    Break a complex query into atomic sub-questions.
    Returns a list of sub-questions, or [query] if no decomposition needed.
    """
    
    prompt = f"""Analyze the following question. If it contains multiple distinct 
information needs, break it into simpler atomic sub-questions. If it is already 
a simple, single-topic question, return it unchanged.

Question: {query}

Rules:
- Each sub-question should be independently answerable from a document
- Sub-questions should together cover all aspects of the original question
- If no decomposition is needed, return: SIMPLE: [original question]
- If decomposition is needed, return: DECOMPOSED: followed by numbered sub-questions

Response:"""

    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0
    )
    
    raw = response.choices[0].message.content.strip()
    
    if raw.startswith("SIMPLE:"):
        return [query]
    
    elif raw.startswith("DECOMPOSED:"):
        sub_questions = []
        for line in raw.split('\n')[1:]:  # Skip "DECOMPOSED:" line
            line = line.strip()
            if line and line[0].isdigit():
                sub_q = line.split('.', 1)[-1].strip()
                if sub_q:
                    sub_questions.append(sub_q)
        return sub_questions if sub_questions else [query]
    
    return [query]


async def retrieve_with_decomposition(query: str, ...) -> list[dict]:
    sub_questions = await decompose_query(query)
    
    if len(sub_questions) == 1:
        # No decomposition — single retrieval
        return await retrieve(query)
    
    # Retrieve for each sub-question
    sub_results = await asyncio.gather(*[
        retrieve(sq) for sq in sub_questions
    ])
    
    # Merge results — all sub-questions' results contribute to final ranking
    return reciprocal_rank_fusion(list(sub_results))
```

**Example decomposition:**

```
Complex query: "What are the maternity leave entitlements for part-time employees 
               in California, and how does the application process differ from 
               full-time employees?"

Sub-questions:
1. "maternity leave entitlement part-time employees California"
2. "maternity leave entitlement full-time employees California"
3. "maternity leave application process part-time employees"
4. "maternity leave application process full-time employees"
```

Each sub-question retrieves specific relevant chunks. The LLM receives all retrieved chunks and synthesizes the complete comparative answer.

### Decomposition + Answer Synthesis

For decomposed queries, the generation step changes. Instead of a single retrieval → single generation, you have multiple retrievals that feed one generation pass:

```python
async def answer_decomposed_query(
    original_query: str,
    sub_questions: list[str],
    sub_results: list[list[dict]],
    llm_client
) -> str:
    
    # Build context from all sub-question results
    context_parts = []
    for sub_q, results in zip(sub_questions, sub_results):
        context_parts.append(f"Sub-question: {sub_q}")
        for r in results[:3]:  # Top 3 results per sub-question
            context_parts.append(f"Context: {r['text']}")
        context_parts.append("")
    
    full_context = "\n".join(context_parts)
    
    prompt = f"""Using the provided context, answer the following question completely.
The context is organized by sub-questions that break down the original question.

Original question: {original_query}

Context:
{full_context}

Provide a comprehensive answer that addresses all aspects of the original question."""

    response = await llm_client.chat.completions.create(
        model="gpt-4o",  # Use capable model for synthesis
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    return response.choices[0].message.content
```

### When Decomposition Hurts

**Over-decomposition.** "What is the maternity leave policy?" decomposed into "what is maternity", "what is leave", "what is policy" — absurd. The LLM should recognize this is already atomic.

**Decomposition latency.** One LLM call for decomposition + N parallel retrievals + one LLM call for synthesis. Total: 2 LLM calls on the critical path plus N retrieval calls. For simple queries this is wasteful. Only decompose when the query genuinely requires it.

**Lost query coherence.** Some questions are holistic and cannot be answered well by assembling sub-answers. "Is our company culture a good fit for introverts?" — decomposing this into attribute sub-questions and recombining misses the holistic judgment.

---

## Technique 4 — Conversational Context Resolution

In multi-turn conversations, each user message may contain unresolved references to prior context. "What about the deductible?" only makes sense if the previous turn discussed insurance. "Can I still apply?" requires knowing what "apply" referred to.

The raw query "What about the deductible?" has essentially zero retrieval signal. It must be resolved against conversation history before retrieval.

### Standalone Query Generation

```python
async def resolve_conversational_query(
    current_query: str,
    conversation_history: list[dict],  # [{"role": "user/assistant", "content": "..."}]
    llm_client
) -> str:
    """
    Convert a context-dependent query into a standalone retrieval query.
    """
    
    # Format recent history (last 4 turns is usually enough)
    recent_history = conversation_history[-4:]
    history_text = "\n".join([
        f"{turn['role'].upper()}: {turn['content']}"
        for turn in recent_history
    ])
    
    prompt = f"""Given a conversation history and the user's latest question, 
rewrite the question as a complete, standalone query that can be understood 
without the conversation context. Include all necessary context from the history.

Conversation history:
{history_text}

User's latest question: {current_query}

Standalone query (do not include any preamble, just the query):"""

    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()
```

**Example:**

```
Conversation history:
USER: What is the health insurance deductible for the Premium plan?
ASSISTANT: The Premium plan has a $500 individual deductible and $1000 family deductible.

User's latest question: What about the standard plan?

Standalone query: "What is the health insurance deductible for the Standard plan?"
```

This standalone query is now a complete retrieval signal. Without resolution, "What about the standard plan?" retrieves almost nothing useful.

### When to Apply Conversational Resolution

Not every multi-turn query needs resolution. Apply it when:
- The query contains pronouns with no clear referent ("it", "that", "this", "they").
- The query uses relative terms ("what about X", "how does that compare", "and also").
- The query is a follow-up on a specific entity established in the conversation.

Simple factual follow-ups that are self-contained ("what's the phone number for HR?") do not need resolution even if they follow a longer conversation.

---

## Combining the Techniques

In production, these techniques are not mutually exclusive. A well-designed query understanding layer applies them in order:

```python
async def process_query(
    raw_query: str,
    conversation_history: list[dict],
    llm_client,
    embedding_model
) -> list[str]:
    """
    Full query understanding pipeline.
    Returns a list of queries to retrieve for (including original).
    """
    
    queries_to_retrieve = []
    
    # Step 1: Resolve conversational context (if multi-turn)
    if conversation_history and needs_context_resolution(raw_query):
        resolved = await resolve_conversational_query(
            raw_query, conversation_history, llm_client
        )
        primary_query = resolved
    else:
        primary_query = raw_query
    
    queries_to_retrieve.append(primary_query)
    
    # Step 2: Decompose if complex
    sub_questions = await decompose_query(primary_query, llm_client)
    if len(sub_questions) > 1:
        queries_to_retrieve.extend(sub_questions)
        # If decomposed, skip expansion — sub-questions already diversify retrieval
        return queries_to_retrieve
    
    # Step 3: Rewrite if needed (for simple, non-decomposed queries)
    if should_rewrite(primary_query):
        rewritten = await rewrite_query(primary_query, llm_client=llm_client)
        queries_to_retrieve.append(rewritten)
    
    # Step 4: Expand for moderate queries
    if len(primary_query.split()) >= 5:  # Only expand if enough signal to work with
        variants = await expand_query(primary_query, n_variants=2, llm_client=llm_client)
        quality_variants = filter_quality_variants(
            primary_query, variants, embedding_model
        )
        queries_to_retrieve.extend(quality_variants)
    
    return queries_to_retrieve


async def retrieve_all(queries: list[str], retriever) -> list[dict]:
    """Retrieve for all queries and merge with RRF."""
    all_results = await asyncio.gather(*[retriever.retrieve(q) for q in queries])
    return reciprocal_rank_fusion(list(all_results))
```

### Latency Budget for Query Understanding

Every technique adds LLM call latency. Here is a realistic budget:

| Technique | LLM size | Latency added |
|---|---|---|
| Context resolution | Small (mini/haiku) | 100–200ms |
| Query rewriting | Small | 100–200ms |
| Query decomposition | Small | 150–250ms |
| Query expansion (3 variants) | Small | 200–350ms |

With parallelization where possible, a full pipeline (resolution → decomposition OR rewrite+expand) adds 300–500ms. This is significant. For latency-sensitive applications (< 1s total response time), be selective about which techniques you apply per query.

> **Interview note:** A very common question is "walk me through your query processing pipeline." The answer interviewers are looking for: (1) conversational resolution first (without this, multi-turn retrieval fails silently), (2) decomposition for complex queries (different path from simple queries), (3) rewrite/expand for simple queries (not both — they address the same problem with different trade-offs), (4) all results merged with RRF, not concatenated. Show you understand the sequencing logic, not just the existence of each technique.

---

## Summary

- Raw user queries have vocabulary gaps, ambiguity, missing context, and mismatched complexity. Query understanding bridges these gaps before retrieval.
- Query rewriting converts colloquial queries into formal, document-vocabulary-aligned queries. Always run retrieval on both original and rewritten query; never discard the original. Use a small fast model.
- Query expansion generates multiple alternative phrasings. Increases recall but adds latency and noise risk. Filter low-quality variants by semantic similarity to original.
- Sub-question decomposition breaks multi-part queries into atomic retrievable units. Apply only to genuinely complex queries — over-decomposition wastes compute and degrades coherence.
- Conversational context resolution converts context-dependent multi-turn queries into standalone retrieval queries. Without this, multi-turn RAG silently fails on any reference-heavy follow-up question.
- Combine techniques in sequence: resolution → decomposition OR rewrite+expand. Merge all results with RRF.
- Every technique adds LLM call latency (100–350ms each). Design the pipeline to be selective: apply techniques only when the query characteristics justify the cost.

---

## What's Next

Lesson 3.5 covers HyDE (Hypothetical Document Embeddings) — a query-side augmentation technique that generates a hypothetical answer to embed instead of the query itself, significantly improving retrieval for short or vague queries.