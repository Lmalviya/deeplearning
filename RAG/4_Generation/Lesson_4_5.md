# Lesson 4.5 — Hallucination: Causes, Detection, and Mitigation

---

## Defining Hallucination Precisely

"Hallucination" is an overloaded term. In RAG systems, it refers to several distinct phenomena that require different solutions. Being imprecise about which type you have leads to applying the wrong fix.

**Type 1 — Intrinsic hallucination (fabrication):**
The LLM generates information that directly contradicts the retrieved context. The context says "the penalty is $500." The LLM says "the penalty is $1,000." The context was retrieved and provided. The LLM fabricated a different answer.

**Type 2 — Extrinsic hallucination (addition):**
The LLM generates information that is not present in the retrieved context but is not directly contradicted by it either. The context describes a termination clause. The LLM adds "common practice in the industry is X" — which may or may not be true but was never in the retrieved context.

**Type 3 — Retrieval-induced hallucination:**
The retrieval system failed to retrieve the relevant context. The LLM was never given the right information. It generated a plausible answer from parametric knowledge. This looks like generation hallucination but is actually a retrieval failure — fixing the LLM prompt will not help.

**Type 4 — Boundary hallucination:**
The LLM correctly uses the retrieved context but extrapolates beyond what the context actually says. The context states an employee gets 16 weeks of leave. The LLM adds that this is "paid at 100% of salary" — which was not stated in the context but is a common assumption the LLM extrapolated from similar policies.

Each type has a different root cause and different fix. Mixing them up — treating retrieval failure as a prompt problem, or treating extrinsic addition as fabrication — wastes effort.

---

## Why Hallucination Happens in LLMs

Understanding the mechanism helps design better mitigations.

### The Probability Distribution View

An LLM generates tokens one at a time, each token sampled from a probability distribution over the vocabulary. This distribution is conditioned on everything the model has seen: its training data, the system prompt, the context, the conversation history, and all previously generated tokens.

The model does not have a separate "fact recall" system and "text generation" system. It is one unified system that produces the most statistically probable next token given all prior context. When it produces incorrect information, it is not "lying" — it is generating text that is statistically plausible given the patterns in its training data and the current context.

This means:
- If the training data frequently associates topic X with fact Y, the model will generate Y when discussing X, even if your retrieved context says something different.
- If the retrieved context is ambiguous, long, or poorly structured, the model's attention distributes across it unequally — and parts it attends to less contribute less to generation.
- As generation proceeds, the model conditions on its own previous outputs. Early errors compound — a wrong premise in sentence 2 leads to wrong conclusions in sentences 5 and 6.

### The Confidence Calibration Problem

LLMs are poorly calibrated in their expressed confidence. They frequently produce incorrect information with the same confident tone as correct information. They do not have internal uncertainty signals that surface in the output. A statement that is 90% likely to be correct sounds identical to one that is 20% likely to be correct.

This is why hallucination is dangerous at production scale — users cannot distinguish confident wrong answers from confident right answers without independent verification.

---

## Detection Method 1 — Self-Consistency Checking

Generate the same answer multiple times with non-zero temperature. If the answers are consistent, confidence is higher. If they diverge significantly, something is uncertain or hallucinated.

```python
async def self_consistency_check(
    query: str,
    context: str,
    llm_client,
    n_samples: int = 5,
    temperature: float = 0.7
) -> dict:
    """
    Generate n answers and check consistency.
    High variance in answers suggests potential hallucination.
    """
    
    messages = [
        {
            "role": "system",
            "content": "Answer questions using only the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
    
    # Generate multiple samples
    samples = await asyncio.gather(*[
        llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=500
        )
        for _ in range(n_samples)
    ])
    
    answers = [s.choices[0].message.content for s in samples]
    
    # Check consistency using embedding similarity
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = embedder.encode(answers)
    
    # Compute pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            similarities.append(sim)
    
    mean_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    
    # Extract the most common answer (mode by clustering)
    # Use the answer closest to the centroid as the "consensus" answer
    centroid = embeddings.mean(axis=0)
    distances_to_centroid = [np.linalg.norm(e - centroid) for e in embeddings]
    consensus_idx = int(np.argmin(distances_to_centroid))
    
    return {
        "consensus_answer": answers[consensus_idx],
        "all_answers": answers,
        "mean_consistency": float(mean_similarity),
        "min_consistency": float(min_similarity),
        "is_consistent": mean_similarity > 0.85,  # Threshold tunable per domain
        "confidence": "high" if mean_similarity > 0.90 else 
                      "medium" if mean_similarity > 0.75 else "low"
    }
```

**Trade-off:** Self-consistency requires N LLM calls. For N=5 with gpt-4o-mini, this is affordable for high-stakes queries but too expensive for routine retrieval. Use it selectively:
- For queries where the confidence score from re-ranking is low.
- For queries in high-stakes domains (legal, medical, financial).
- As an offline evaluation tool on sampled production queries.

---

## Detection Method 2 — NLI-Based Faithfulness Scoring

Natural Language Inference (NLI) models determine whether a hypothesis is supported by (entailed by), contradicted by, or neutral to a premise. You can use an NLI model to check whether each sentence in the LLM's response is entailed by the retrieved context.

```python
from transformers import pipeline

class NLIFaithfulnessChecker:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.nli = pipeline(
            "text-classification",
            model=model_name,
            device=0  # GPU
        )
    
    def check_faithfulness(
        self,
        context: str,
        response: str,
        threshold: float = 0.7
    ) -> dict:
        """
        Check each sentence in the response for faithfulness to the context.
        """
        import nltk
        sentences = nltk.sent_tokenize(response)
        
        results = []
        
        for sentence in sentences:
            if len(sentence.split()) < 5:
                continue  # Skip very short sentences
            
            # NLI: premise = context, hypothesis = response sentence
            nli_input = f"{context} [SEP] {sentence}"
            
            # Most NLI models have 3 classes: entailment, neutral, contradiction
            prediction = self.nli(
                nli_input,
                truncation=True,
                max_length=512
            )[0]
            
            results.append({
                "sentence": sentence,
                "label": prediction["label"],
                "score": prediction["score"],
                "is_faithful": prediction["label"] == "ENTAILMENT" and prediction["score"] > threshold
            })
        
        faithful_count = sum(1 for r in results if r["is_faithful"])
        total = len(results)
        
        faithfulness_score = faithful_count / total if total > 0 else 1.0
        
        unfaithful_sentences = [
            r["sentence"] for r in results 
            if not r["is_faithful"] and r["label"] == "CONTRADICTION"
        ]
        
        unsupported_sentences = [
            r["sentence"] for r in results 
            if not r["is_faithful"] and r["label"] == "NEUTRAL"
        ]
        
        return {
            "faithfulness_score": faithfulness_score,
            "sentence_results": results,
            "unfaithful_sentences": unfaithful_sentences,  # Contradictions
            "unsupported_sentences": unsupported_sentences,  # Not in context
            "is_faithful": faithfulness_score >= 0.85
        }
```

NLI-based checking is fast (no LLM call needed) and can run on GPU in real time. It works well for detecting direct contradictions. It is less reliable for detecting subtle extrinsic additions or boundary extrapolations.

**Model recommendations:**
- `cross-encoder/nli-deberta-v3-small` — fast, good quality for sentence-level NLI.
- `facebook/bart-large-mnli` — strong NLI quality, heavier.
- For RAG-specific faithfulness, the `TRUE` model (Honovich et al.) is trained specifically on faithfulness detection.

---

## Detection Method 3 — LLM-as-Judge

Ask a capable LLM to evaluate whether the response is faithful to the context. More nuanced than NLI but more expensive.

```python
async def llm_faithfulness_judge(
    query: str,
    context: str,
    response: str,
    judge_llm_client
) -> dict:
    """
    Use an LLM to judge faithfulness. The judge LLM should be different
    from (or a more capable version of) the generation LLM.
    """
    
    prompt = f"""You are a strict factual accuracy judge. Evaluate whether the
AI Response below is faithful to the provided Source Context.

Question: {query}

Source Context:
{context}

AI Response:
{response}

Evaluate each claim in the AI Response:
1. Is it directly supported by the source context?
2. Does it contradict the source context?
3. Does it add information not present in the source context?

Respond with JSON:
{{
    "overall_faithfulness": 0.0-1.0,
    "verdict": "faithful" | "mostly_faithful" | "unfaithful",
    "issues": [
        {{
            "claim": "the specific claim in the response",
            "issue_type": "contradiction" | "unsupported_addition" | "extrapolation",
            "explanation": "why this is an issue"
        }}
    ],
    "faithful_claims": ["list of claims that ARE supported by context"]
}}"""
    
    response_obj = await judge_llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1000
    )
    
    import json
    return json.loads(response_obj.choices[0].message.content)
```

Use LLM-as-Judge:
- In evaluation pipelines (offline, batch processing).
- For sampled production queries to track faithfulness trends.
- When NLI checking returns borderline results and you need a second opinion.

**Important:** The judge LLM must be different from or more capable than the generation LLM. Using the same model to judge its own output creates a self-consistency illusion — it will often judge its own hallucinations as faithful.

---

## Mitigation Layer 1 — Retrieval Quality

The first and most important mitigation is retrieval quality. Type 3 hallucination (retrieval-induced) accounts for a large fraction of production hallucination incidents. The LLM cannot faithfully answer from context it was never given.

Checklist:
- Is the relevant document indexed? (Coverage check, Lesson 3.8)
- Is the correct chunk retrieved in top-10? (Recall@K measurement)
- Is the re-ranker placing it in the final context? (Re-ranking demotion check)
- Is the context budget sufficient to include it? (Token budget check)

Before blaming generation for hallucination, verify retrieval was not the root cause.

---

## Mitigation Layer 2 — Prompt Grounding (Covered in Lesson 4.1)

Explicit grounding instructions, null handling for missing context, and parametric conflict instructions reduce but do not eliminate hallucination. They are necessary but not sufficient as the only mitigation layer.

Key additions beyond Lesson 4.1 for hallucination specifically:

```python
ANTI_HALLUCINATION_ADDITIONS = """
NEVER:
- Do not extrapolate beyond what the context explicitly states.
- Do not fill in details that seem logical but are not in the context.
  Example: If the context says "employees get 16 weeks leave" but does not 
  specify pay rate, do NOT assume it is paid — state that the pay rate is 
  not specified in the provided context.
- Do not combine information from context with general knowledge to produce
  a "more complete" answer. Completeness achieved through fabrication is worse
  than acknowledged incompleteness.

UNCERTAINTY MARKERS: When you are drawing on context that is ambiguous or 
incomplete, use explicit markers:
- "According to [1], ..." when you are stating what the document says
- "The document does not specify..." when a related detail is absent
- "This is based on [1] but [2] does not address this point..." when sources differ
"""
```

---

## Mitigation Layer 3 — Post-Generation Checking

Run a faithfulness check after generation and before returning the response to the user. For high-stakes applications, block unfaithful responses.

```python
async def generate_with_faithfulness_gate(
    query: str,
    context: str,
    llm_client,
    faithfulness_checker: NLIFaithfulnessChecker,
    min_faithfulness: float = 0.85,
    max_retries: int = 2
) -> dict:
    """
    Generate response and retry if faithfulness check fails.
    """
    
    for attempt in range(max_retries + 1):
        response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.1 * attempt,  # Slightly increase temperature on retry
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        
        # Check faithfulness
        faithfulness_result = faithfulness_checker.check_faithfulness(context, answer)
        
        if faithfulness_result["faithfulness_score"] >= min_faithfulness:
            return {
                "answer": answer,
                "faithfulness_score": faithfulness_result["faithfulness_score"],
                "attempts": attempt + 1,
                "status": "ok"
            }
        
        if attempt < max_retries:
            # Retry with explicit feedback about what was wrong
            unfaithful = faithfulness_result.get("unfaithful_sentences", [])
            if unfaithful:
                # Add the faithfulness failure as a negative example
                context += f"\n\n[CORRECTION NEEDED: Your previous response contained claims not supported by the context: {unfaithful[:2]}. Please regenerate, staying strictly within the provided context.]"
    
    # All retries failed — return with warning
    return {
        "answer": answer,
        "faithfulness_score": faithfulness_result["faithfulness_score"],
        "attempts": max_retries + 1,
        "status": "low_faithfulness_warning",
        "warning": "This response may contain information not fully supported by retrieved documents."
    }
```

**For real-time applications:** NLI-based checking is fast enough (20–50ms on GPU) to run inline before returning every response. LLM-based checking adds 300–1000ms — use only for high-stakes queries or a sampled subset.

---

## Mitigation Layer 4 — Uncertainty Quantification

Instead of a binary "hallucinated / not hallucinated" check, quantify and surface uncertainty to users.

```python
async def generate_with_uncertainty(
    query: str,
    context: str,
    llm_client
) -> dict:
    """
    Generate answer with explicit uncertainty markers.
    """
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Answer questions from the provided context.

For each claim in your answer, indicate your confidence level:
- Use [HIGH] for claims directly and explicitly stated in the context
- Use [MED] for claims that can be directly inferred from explicit statements  
- Use [LOW] for claims where you are uncertain whether the context supports them

Example: "The contract value is $500,000 [HIGH]. This likely includes installation 
costs [LOW] based on the scope description."

If you are unsure about something, say so explicitly rather than presenting it 
confidently."""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        max_tokens=800,
        temperature=0.1
    )
    
    answer = response.choices[0].message.content
    
    # Parse confidence markers to compute aggregate confidence
    import re
    high_count = len(re.findall(r'\[HIGH\]', answer))
    med_count = len(re.findall(r'\[MED\]', answer))
    low_count = len(re.findall(r'\[LOW\]', answer))
    total = high_count + med_count + low_count
    
    if total > 0:
        aggregate_confidence = (high_count * 1.0 + med_count * 0.6 + low_count * 0.3) / total
    else:
        aggregate_confidence = 0.5  # Unknown
    
    return {
        "answer": answer,
        "aggregate_confidence": aggregate_confidence,
        "high_confidence_claims": high_count,
        "low_confidence_claims": low_count
    }
```

Surfacing uncertainty markers directly in the response empowers users to apply their own judgment about which parts of the answer to trust without independent verification.

---

## Mitigation Layer 5 — Citation Enforcement and Verification

Require every factual claim to have a citation. Then verify that the citation actually supports the claim.

```python
async def verify_citations(
    answer: str,
    source_chunks: list[dict],
    llm_client
) -> dict:
    """
    Verify that each citation in the answer actually supports the cited claim.
    """
    import re
    
    # Extract all citations and their surrounding claims
    citation_pattern = r'(.{0,100})\[(\d+)\](.{0,50})'
    matches = re.findall(citation_pattern, answer)
    
    verification_results = []
    
    for pre_text, ref_num, post_text in matches:
        ref_idx = int(ref_num) - 1
        
        if ref_idx >= len(source_chunks):
            verification_results.append({
                "ref_num": ref_num,
                "claim": pre_text.strip(),
                "status": "INVALID_REF",
                "issue": f"Reference [{ref_num}] does not exist"
            })
            continue
        
        source_text = source_chunks[ref_idx]["text"]
        claim = pre_text.strip() + post_text.strip()
        
        # Ask LLM to verify the citation
        verify_prompt = f"""Does the source text support the claim?

Claim: {claim}
Source [{ref_num}]: {source_text[:500]}

Does the source directly support this claim?
Respond: SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED
Then briefly explain why."""
        
        v_response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": verify_prompt}],
            max_tokens=100,
            temperature=0.0
        )
        
        result_text = v_response.choices[0].message.content
        
        if "NOT_SUPPORTED" in result_text:
            status = "NOT_SUPPORTED"
        elif "PARTIALLY" in result_text:
            status = "PARTIALLY_SUPPORTED"
        else:
            status = "SUPPORTED"
        
        verification_results.append({
            "ref_num": ref_num,
            "claim": claim,
            "status": status,
            "explanation": result_text
        })
    
    unsupported = [r for r in verification_results if r["status"] == "NOT_SUPPORTED"]
    
    return {
        "verification_results": verification_results,
        "unsupported_citations": unsupported,
        "all_citations_valid": len(unsupported) == 0
    }
```

---

## The Hallucination Mitigation Stack

These mitigations are not alternatives — they are layers. Production systems apply them in combination:

```
Layer 1 (Retrieval):  Ensure right content is retrieved — prevents Type 3
Layer 2 (Prompt):     Explicit grounding, null handling — reduces Types 1, 2, 4  
Layer 3 (Generation): Self-consistency for uncertain queries — reduces Types 1, 4
Layer 4 (Post-check): NLI faithfulness gate — catches Types 1, 2
Layer 5 (Citations):  Citation enforcement + verification — catches Types 1, 2, 4
Layer 6 (Monitoring): LLM-as-judge on sampled production queries — measures all types
```

Not all layers apply to all systems. A customer support chatbot may use only Layers 1, 2, 4. A legal document analysis system may apply all six. Match the mitigation depth to the stakes and cost tolerance of your application.

> **Interview note:** "How do you handle hallucination in your RAG system?" — The answer structure interviewers want: (1) distinguish the four hallucination types (retrieval-induced vs. fabrication vs. addition vs. extrapolation), (2) explain that retrieval quality is the first mitigation (most hallucination is actually retrieval failure), (3) prompt grounding reduces but doesn't eliminate, (4) post-generation checking (NLI or LLM-as-judge) catches what slips through, (5) citations make the system auditable. Show you understand it is a multi-layer problem, not a prompt fix.

---

## Summary

- Hallucination has four distinct types: intrinsic fabrication (contradicts context), extrinsic addition (not in context), retrieval-induced (context never provided), and boundary extrapolation (goes beyond what context states).
- LLMs hallucinate because they generate statistically probable tokens, not verified facts. They are poorly calibrated — wrong answers sound as confident as correct ones.
- Detection methods: self-consistency checking (multiple samples, measure agreement), NLI-based faithfulness scoring (fast, no LLM call), LLM-as-judge (most nuanced, expensive).
- Mitigation layers: retrieval quality → prompt grounding → self-consistency → NLI faithfulness gate → citation enforcement → production monitoring.
- The most impactful mitigation is almost always improving retrieval quality — most production hallucination is retrieval-induced, not generation-induced.
- Apply detection and mitigation proportionally to stakes. All six layers for legal/medical/financial. Fewer layers for lower-stakes applications.

---

## What's Next

Part 4 is complete. Part 5 begins with Lesson 5.1 — Corrective RAG (CRAG): the self-assessment and fallback retrieval architecture that handles cases where local retrieval fails.