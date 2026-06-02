# Lesson 5.5 — Multi-Hop and Multi-Document Reasoning

---

## What Multi-Hop Reasoning Is

A single-hop query has one retrieval step: find the document that answers the question. "What is the termination notice period in our vendor contract with Company X?" — retrieve the contract, find the clause, answer.

A multi-hop query requires chaining across multiple retrieval steps where the output of one step determines the input to the next. "What is the termination notice period for the vendor that supplies our core authentication infrastructure?" — this requires:

1. Identify which vendor supplies the core authentication infrastructure (from a technical architecture document or system inventory).
2. Find the contract for that vendor (from a contracts database).
3. Find the termination clause in that contract.

No single document contains all three pieces. The answer at step 1 determines which document to retrieve at step 2. The answer at step 2 determines which clause to look for at step 3.

This chain — where each retrieval step depends on the result of the previous one — is multi-hop reasoning. It cannot be handled by retrieving K documents in a single pass and hoping they contain everything.

---

## Types of Multi-Hop Queries

Understanding the structure of multi-hop queries helps design the right solution.

**Bridge queries:** The answer to one sub-question is needed to retrieve the next document. "What is the reporting structure of the executive who signed Contract X?" — first find who signed Contract X (bridge entity), then find that person's reporting structure.

**Comparison queries:** Retrieve separate facts from separate documents, then compare them. "How does our leave policy compare to the statutory minimum in California?" — retrieve internal policy (one document), retrieve California labor law (another source), then compare.

**Aggregation queries:** Retrieve multiple instances of the same type of fact, then aggregate. "What is the average payment term across all vendor contracts signed in 2023?" — retrieve all 2023 contracts, extract payment terms from each, compute average.

**Constraint-following queries:** A fact in one document constrains what to look for in another. "List all employees who meet the eligibility criteria for the senior engineer promotion." — retrieve eligibility criteria (one document), then retrieve employee records that match.

Each type has different retrieval patterns and solution approaches.

---

## Approach 1 — Iterative Retrieval

The most straightforward approach: detect that a query is multi-hop, decompose it into sequential sub-questions, and retrieve for each sub-question using the previous answer as context.

```python
async def iterative_multi_hop_retrieval(
    query: str,
    retriever,
    llm_client,
    max_hops: int = 4
) -> dict:
    """
    Iteratively retrieve and answer sub-questions until the full query is answered.
    """
    
    hop_results = []
    accumulated_context = ""
    
    current_question = query
    
    for hop in range(max_hops):
        # Step 1: Determine what to retrieve next
        retrieval_plan = await plan_next_hop(
            original_query=query,
            current_question=current_question,
            accumulated_context=accumulated_context,
            hop_number=hop,
            llm_client=llm_client
        )
        
        if retrieval_plan["is_answerable"]:
            # We have enough context to answer the original query
            break
        
        retrieval_query = retrieval_plan["retrieval_query"]
        
        # Step 2: Retrieve for this hop
        results = await retriever.retrieve(retrieval_query)
        
        if not results:
            hop_results.append({
                "hop": hop,
                "query": retrieval_query,
                "result": "No relevant documents found.",
                "success": False
            })
            break
        
        # Step 3: Extract the specific fact needed from retrieved content
        hop_context = "\n\n".join(r["text"] for r in results[:3])
        
        extracted = await extract_hop_answer(
            sub_question=retrieval_plan["sub_question"],
            context=hop_context,
            llm_client=llm_client
        )
        
        hop_results.append({
            "hop": hop,
            "query": retrieval_query,
            "sub_question": retrieval_plan["sub_question"],
            "retrieved_chunks": [r["chunk_id"] for r in results[:3]],
            "extracted_answer": extracted["answer"],
            "success": extracted["found"]
        })
        
        # Step 4: Add to accumulated context for the next hop
        accumulated_context += f"\n\nHop {hop + 1} finding: {extracted['answer']}"
        
        # Update the "current question" to reflect remaining unknown
        current_question = retrieval_plan.get("remaining_question", query)
    
    # Final generation using all accumulated context
    final_answer = await generate_final_answer(
        original_query=query,
        hop_results=hop_results,
        accumulated_context=accumulated_context,
        llm_client=llm_client
    )
    
    return {
        "answer": final_answer,
        "hop_count": len(hop_results),
        "hop_trace": hop_results
    }


async def plan_next_hop(
    original_query: str,
    current_question: str,
    accumulated_context: str,
    hop_number: int,
    llm_client
) -> dict:
    """
    Given what we know so far, plan the next retrieval step.
    """
    
    prompt = f"""You are solving a multi-step question by gathering information incrementally.

Original question: {original_query}

Information gathered so far:
{accumulated_context if accumulated_context else "Nothing yet."}

Based on what you know, determine:
1. Can the original question be answered with the current information? (yes/no)
2. If not, what is the next specific sub-question to answer?
3. What should be searched to answer that sub-question?

Return JSON:
{{
    "is_answerable": true/false,
    "sub_question": "the specific atomic question to answer next",
    "retrieval_query": "optimized search query to find documents answering the sub-question",
    "remaining_question": "what remains to be answered after this hop"
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=300,
        temperature=0.0
    )
    
    import json
    return json.loads(response.choices[0].message.content)


async def extract_hop_answer(
    sub_question: str,
    context: str,
    llm_client
) -> dict:
    """
    Extract the specific answer to a sub-question from retrieved context.
    """
    
    prompt = f"""From the following context, extract the specific answer to this question.

Sub-question: {sub_question}

Context: {context[:2000]}

Return JSON:
{{
    "found": true/false,
    "answer": "the specific extracted answer, or null if not found",
    "confidence": "high" | "medium" | "low"
}}

Be precise. Only extract what is explicitly in the context."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
        temperature=0.0
    )
    
    import json
    return json.loads(response.choices[0].message.content)


async def generate_final_answer(
    original_query: str,
    hop_results: list[dict],
    accumulated_context: str,
    llm_client
) -> str:
    """
    Generate the final answer from all hop findings.
    """
    
    hop_summary = "\n".join([
        f"Step {r['hop'] + 1}: {r.get('sub_question', 'Retrieval')} → {r.get('extracted_answer', 'Not found')}"
        for r in hop_results
    ])
    
    prompt = f"""Answer the following question using the step-by-step findings below.

Question: {original_query}

Reasoning steps:
{hop_summary}

Full context accumulated:
{accumulated_context}

Provide a direct, complete answer. Show your reasoning chain briefly."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1
    )
    
    return response.choices[0].message.content
```

---

## Approach 2 — IRCoT (Interleaved Retrieval with Chain-of-Thought)

IRCoT (Trivedi et al., 2022) interleaves chain-of-thought reasoning with retrieval. The model reasons step by step, and whenever the next reasoning step requires information it does not have, it retrieves.

The key insight: the model's partial reasoning trace is a better retrieval signal than the original query alone, because it captures what the model already knows and what it specifically needs next.

```python
async def ircot_reasoning(
    query: str,
    retriever,
    llm_client,
    max_steps: int = 6
) -> dict:
    """
    IRCoT: Interleaved Retrieval with Chain-of-Thought.
    The model reasons step by step, retrieving when needed.
    """
    
    reasoning_trace = []
    retrieved_documents = []
    
    system_prompt = """You are answering questions through step-by-step reasoning.
Think through the question one step at a time.

When you need information you don't have, output:
RETRIEVE: [specific query to search for the needed information]

When you have enough to answer, output:
ANSWER: [your final answer]

Think carefully before each step. Be specific about what information you need."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}\n\nBegin your step-by-step reasoning:"}
    ]
    
    for step in range(max_steps):
        # Generate next reasoning step
        response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
            temperature=0.1,
            stop=["RETRIEVE:", "ANSWER:"]  # Stop at these markers
        )
        
        reasoning_step = response.choices[0].message.content.strip()
        stop_reason = response.choices[0].finish_reason
        
        # Check what the model wants to do next
        if "ANSWER:" in reasoning_step:
            # Model has the answer
            final_answer = reasoning_step.split("ANSWER:")[-1].strip()
            return {
                "answer": final_answer,
                "reasoning_trace": reasoning_trace,
                "steps": step + 1,
                "retrieved_count": len(retrieved_documents)
            }
        
        if "RETRIEVE:" in reasoning_step:
            # Model needs to retrieve something
            retrieval_query = reasoning_step.split("RETRIEVE:")[-1].strip()
            
            # Retrieve documents
            results = await retriever.retrieve(retrieval_query, k=3)
            retrieved_documents.extend(results)
            
            # Format retrieved content
            retrieved_text = "\n\n".join([
                f"[Retrieved]: {r['text'][:500]}"
                for r in results
            ])
            
            reasoning_trace.append({
                "step": step,
                "reasoning": reasoning_step,
                "retrieval_query": retrieval_query,
                "retrieved_chunks": [r["chunk_id"] for r in results]
            })
            
            # Add retrieval result to conversation
            messages.append({"role": "assistant", "content": reasoning_step + " RETRIEVE: " + retrieval_query})
            messages.append({"role": "user", "content": f"Retrieved information:\n{retrieved_text}\n\nContinue your reasoning:"})
        
        else:
            # Pure reasoning step, no retrieval needed
            reasoning_trace.append({
                "step": step,
                "reasoning": reasoning_step,
                "retrieval_query": None
            })
            
            messages.append({"role": "assistant", "content": reasoning_step})
            messages.append({"role": "user", "content": "Continue:"})
    
    # Max steps reached — generate best effort answer
    all_context = "\n\n".join([r["text"] for r in retrieved_documents[:5]])
    final_response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Based on this context:\n{all_context}\n\nAnswer: {query}"}
        ],
        max_tokens=500,
        temperature=0.1
    )
    
    return {
        "answer": final_response.choices[0].message.content,
        "reasoning_trace": reasoning_trace,
        "steps": max_steps,
        "retrieved_count": len(retrieved_documents),
        "status": "max_steps_reached"
    }
```

**Why IRCoT outperforms simple iterative retrieval:** The chain-of-thought reasoning trace is a richer retrieval query than the raw sub-question. "Given that the authentication vendor is TechCorp and their contract number is TC-2023-A01, what is the termination clause?" retrieves much more precisely than just "termination clause."

---

## Approach 3 — FLARE (Forward-Looking Active Retrieval)

FLARE (Jiang et al., 2023) takes a different approach: generate the answer tentatively, detect sentences where the model is uncertain (low token probability), and retrieve specifically for those uncertain sentences.

The insight: the model's generation confidence is a signal for when it needs retrieval. High confidence = model knows this from parametric knowledge. Low confidence = model is uncertain and should retrieve.

```python
async def flare_generation(
    query: str,
    retriever,
    llm_client,
    confidence_threshold: float = 0.5,
    max_retrieval_rounds: int = 3
) -> dict:
    """
    FLARE: Forward-Looking Active REtrieval augmented generation.
    
    Generate tentatively, detect low-confidence sentences,
    retrieve for those sentences, regenerate with retrieved context.
    """
    
    context_so_far = ""
    retrieval_rounds = 0
    
    for round_num in range(max_retrieval_rounds):
        # Generate a tentative response
        tentative_prompt = f"Question: {query}\n\nContext so far:\n{context_so_far}\n\nContinue answering:"
        
        tentative_response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": tentative_prompt}],
            max_tokens=500,
            temperature=0.1,
            logprobs=True,  # Need token probabilities
            top_logprobs=1
        )
        
        tentative_text = tentative_response.choices[0].message.content
        token_logprobs = [
            t.logprob
            for t in tentative_response.choices[0].logprobs.content
        ]
        
        # Detect low-confidence sentences
        low_confidence_sentences = detect_low_confidence_sentences(
            text=tentative_text,
            token_logprobs=token_logprobs,
            threshold=confidence_threshold
        )
        
        if not low_confidence_sentences:
            # All sentences are high confidence — accept this response
            return {
                "answer": tentative_text,
                "retrieval_rounds": retrieval_rounds,
                "final_context": context_so_far
            }
        
        # Retrieve for the low-confidence sentences
        retrieval_queries = [
            build_retrieval_query_from_sentence(sent)
            for sent in low_confidence_sentences
        ]
        
        retrieval_results = await asyncio.gather(*[
            retriever.retrieve(q, k=2) for q in retrieval_queries
        ])
        
        # Add retrieved context
        new_context = "\n".join([
            chunk["text"]
            for results in retrieval_results
            for chunk in results
        ])
        
        context_so_far += f"\n\n[Retrieved context for: {', '.join(low_confidence_sentences[:2])}]\n{new_context[:2000]}"
        retrieval_rounds += 1
    
    # Final generation with accumulated context
    final_response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context_so_far}\n\nQuestion: {query}\n\nAnswer:"
            }
        ],
        max_tokens=600,
        temperature=0.1
    )
    
    return {
        "answer": final_response.choices[0].message.content,
        "retrieval_rounds": retrieval_rounds,
        "final_context": context_so_far
    }


def detect_low_confidence_sentences(
    text: str,
    token_logprobs: list[float],
    threshold: float = 0.5  # Average log probability threshold
) -> list[str]:
    """
    Identify sentences where the model had low generation confidence.
    """
    import nltk
    import numpy as np
    
    sentences = nltk.sent_tokenize(text)
    
    # Rough mapping of sentences to token probability ranges
    # This is approximate — proper implementation requires token-to-character alignment
    tokens_per_sentence = len(token_logprobs) // max(len(sentences), 1)
    
    low_confidence = []
    for i, sentence in enumerate(sentences):
        start = i * tokens_per_sentence
        end = min((i + 1) * tokens_per_sentence, len(token_logprobs))
        
        if start >= len(token_logprobs):
            break
        
        sentence_probs = token_logprobs[start:end]
        avg_prob = np.mean([np.exp(p) for p in sentence_probs])  # Convert log prob to prob
        
        if avg_prob < threshold:
            low_confidence.append(sentence)
    
    return low_confidence


def build_retrieval_query_from_sentence(sentence: str) -> str:
    """
    Build a retrieval query designed to find evidence for or against a sentence.
    Removes hedging language and focuses on the core claim.
    """
    # Remove hedging phrases that are generation artifacts
    hedging = ["I believe", "I think", "probably", "likely", "might", "may"]
    query = sentence
    for hedge in hedging:
        query = query.replace(hedge, "").strip()
    
    return query.strip(".,;:")
```

---

## Approach 4 — Sub-Question Parallel Retrieval

For comparison and aggregation queries, retrieve for all sub-questions in parallel (not sequentially) and synthesize the results.

```python
async def parallel_sub_question_retrieval(
    query: str,
    retriever,
    llm_client
) -> dict:
    """
    Decompose query into independent sub-questions, retrieve for each in parallel,
    then synthesize.
    
    Best for comparison and aggregation queries where sub-questions are independent.
    """
    
    # Decompose into independent sub-questions
    decomp_prompt = f"""Decompose this question into independent sub-questions 
that can each be answered by retrieving a different document.

Question: {query}

Requirements:
- Sub-questions must be independent (not dependent on each other's answers)
- Each sub-question should be answerable from a single document retrieval
- Include a "synthesis" step describing how to combine the answers

Return JSON:
{{
    "sub_questions": [
        {{
            "id": "sq1",
            "question": "specific sub-question",
            "retrieval_query": "optimized search query"
        }}
    ],
    "synthesis_instruction": "how to combine the sub-question answers"
}}"""
    
    decomp_response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": decomp_prompt}],
        response_format={"type": "json_object"},
        max_tokens=400,
        temperature=0.0
    )
    
    import json
    decomposition = json.loads(decomp_response.choices[0].message.content)
    sub_questions = decomposition.get("sub_questions", [])
    
    if not sub_questions:
        # No decomposition needed — standard single retrieval
        results = await retriever.retrieve(query)
        context = format_context(results)
        answer = await generate_from_context(query, context, llm_client)
        return {"answer": answer, "decomposed": False}
    
    # Retrieve for all sub-questions in parallel
    retrieval_tasks = [
        retriever.retrieve(sq["retrieval_query"], k=3)
        for sq in sub_questions
    ]
    all_results = await asyncio.gather(*retrieval_tasks)
    
    # Extract answers for each sub-question
    extraction_tasks = [
        extract_hop_answer(sq["question"], "\n\n".join(r["text"] for r in results[:3]), llm_client)
        for sq, results in zip(sub_questions, all_results)
    ]
    extracted_answers = await asyncio.gather(*extraction_tasks)
    
    # Build synthesis context
    sub_answers = "\n".join([
        f"Q: {sq['question']}\nA: {ea.get('answer', 'Not found')}"
        for sq, ea in zip(sub_questions, extracted_answers)
    ])
    
    synthesis_prompt = f"""Original question: {query}

Sub-question answers:
{sub_answers}

Synthesis instruction: {decomposition.get('synthesis_instruction', 'Combine the answers.')}

Provide the final synthesized answer:"""
    
    synthesis_response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": synthesis_prompt}],
        max_tokens=600,
        temperature=0.1
    )
    
    return {
        "answer": synthesis_response.choices[0].message.content,
        "decomposed": True,
        "sub_questions": [
            {"question": sq["question"], "answer": ea.get("answer")}
            for sq, ea in zip(sub_questions, extracted_answers)
        ]
    }
```

---

## Detecting Multi-Hop Queries

Not all queries need multi-hop retrieval. Apply it selectively — it is expensive and adds latency.

```python
async def classify_hop_requirement(query: str, llm_client) -> dict:
    """
    Classify whether a query requires multi-hop reasoning.
    """
    
    prompt = f"""Analyze this query to determine if it requires multi-step reasoning 
across multiple documents.

Query: {query}

A query requires MULTI-HOP reasoning if:
- The answer to one part determines what document to look at next
- Information must be gathered from 3+ different documents and synthesized
- The query compares facts across multiple distinct sources
- Finding an entity in one document leads to looking up that entity in another

A query requires SINGLE-HOP reasoning if:
- The complete answer likely exists in one document or chunk
- It asks about one specific fact or policy
- Simple lookups or definitions

Return JSON:
{{
    "requires_multi_hop": true/false,
    "hop_type": "bridge" | "comparison" | "aggregation" | "constraint" | "single",
    "estimated_hops": 1-5,
    "reason": "brief explanation"
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0.0
    )
    
    import json
    return json.loads(response.choices[0].message.content)


async def adaptive_retrieval(query: str, retriever, llm_client) -> dict:
    """
    Route to single-hop or multi-hop based on query classification.
    """
    
    classification = await classify_hop_requirement(query, llm_client)
    
    if not classification["requires_multi_hop"]:
        # Standard single-hop retrieval
        results = await retriever.retrieve(query)
        context = format_context(results)
        answer = await generate_from_context(query, context, llm_client)
        return {"answer": answer, "method": "single_hop"}
    
    hop_type = classification["hop_type"]
    
    if hop_type == "comparison":
        return await parallel_sub_question_retrieval(query, retriever, llm_client)
    
    elif hop_type in ["bridge", "constraint"]:
        return await iterative_multi_hop_retrieval(query, retriever, llm_client)
    
    elif hop_type == "aggregation":
        # Map-reduce over many retrievals
        return await map_reduce_multi_hop(query, retriever, llm_client)
    
    else:
        # Default to IRCoT for complex unknown patterns
        return await ircot_reasoning(query, retriever, llm_client)
```

---

## Choosing the Right Approach

| Query Type | Best Approach | Why |
|---|---|---|
| Bridge ("who signed X's contract, and what are their terms") | Iterative retrieval | Answer at each step gates next step — must be sequential |
| Comparison ("how does policy A differ from policy B") | Parallel sub-question | Independent retrievals, fast parallelism |
| Aggregation ("average payment terms across all 2023 contracts") | Map-reduce or agentic | Many independent retrievals + aggregation |
| Constraint-following ("employees matching criteria X") | Iterative or agentic | Criteria from one doc constrain retrieval in another |
| Unknown complex | IRCoT | Let the model reason and retrieve as needed |

---

## Cost and Latency Reality

Multi-hop retrieval is expensive. Be explicit about this in system design.

For a 3-hop iterative query:
- 3 retrieval calls × ~50ms = ~150ms retrieval
- 3 LLM extraction calls × ~200ms = ~600ms (parallelizable within each hop but not across hops)
- 1 final generation call × ~500ms = ~500ms
- Total: **~1.5–2.5s** beyond a standard single-hop

For IRCoT with 4 steps:
- 4+ LLM calls in sequence (each waits for the previous)
- Total: **3–6s** easily

> **Interview note:** "How would you handle a query that requires information from multiple documents?" — The structured answer: (1) first classify whether it's truly multi-hop or just multi-document (parallel retrieval handles multi-document without iteration), (2) for bridge/constraint queries use iterative retrieval, (3) for comparison queries use parallel sub-question decomposition, (4) for unknown complex patterns use IRCoT to let the reasoning guide retrieval, (5) always add hop classification to avoid the latency overhead on single-hop queries.

---

## Summary

- Multi-hop queries require chaining retrieval steps where one answer determines the next search. Single retrieval passes cannot handle them.
- Four query types: bridge (chain entities), comparison (parallel facts), aggregation (collect and compute), constraint-following (criteria from one doc applied to another).
- Iterative retrieval: plan → retrieve → extract → repeat. Sequential, handles bridge and constraint queries. Each hop depends on the previous.
- IRCoT: interleave chain-of-thought reasoning with retrieval. The reasoning trace is the retrieval query. Handles unknown complex patterns adaptively.
- FLARE: generate tentatively, detect low-confidence sentences by token probability, retrieve specifically for those sentences. Grounded in generation uncertainty.
- Parallel sub-question retrieval: for comparison and aggregation — decompose into independent sub-questions, retrieve in parallel, synthesize. Much faster than sequential for independent hops.
- Always classify queries before routing to multi-hop — the overhead is not justified for single-hop queries.
- Multi-hop adds 2–6× latency over single-hop. Build this into your latency budget and set user expectations accordingly.

---

## What's Next

Lesson 5.6 covers conversational RAG — managing multi-turn context, conversation history compression, session-level memory, and the specific challenges of maintaining coherent RAG behavior across a dialogue.