# Lesson 4.1 — Prompt Design for RAG: Grounding, Citation, and Handling Missing Context

---

## Why Prompt Design Is a First-Class RAG Concern

Most RAG literature focuses on retrieval. The assumption is: get the right chunks, and the LLM will produce the right answer. This is partially true but misses a critical fact — the same retrieved chunks, given to the same LLM with different prompts, can produce dramatically different quality responses.

The prompt is the interface between your retrieval system and the LLM. It determines:
- Whether the LLM reads and uses the retrieved context or ignores it.
- Whether the LLM cites specific sources or blends everything together.
- Whether the LLM says "I don't know" when context is insufficient or hallucinates a plausible-sounding answer.
- The format, length, and style of the response.

In production, poor prompt design causes generation failures even when retrieval is perfect. This lesson covers how to design prompts that reliably produce grounded, accurate, citable answers.

---

## The Three Grounding Problems

Before designing the prompt, understand the three grounding problems that RAG prompts must solve:

**Problem 1 — Parametric vs. contextual knowledge conflict.**
LLMs have strong parametric knowledge from pretraining. When retrieved context contradicts what the LLM "knows," it may ignore the context and answer from memory. This is dangerous — your retrieved context is authoritative for your domain; the LLM's parametric memory is not.

**Problem 2 — Context abandonment.**
When the retrieved context is long, complex, or poorly written, LLMs sometimes abandon it and answer from memory because it is easier. The lost-in-the-middle effect compounds this — context buried in the middle of a long block gets ignored.

**Problem 3 — Confident hallucination when context is insufficient.**
When the retrieved context does not contain the answer, LLMs often generate plausible-sounding answers anyway rather than acknowledging the gap. This is the most dangerous failure mode — users receive confident wrong answers with no indication of the problem.

A well-designed RAG prompt addresses all three.

---

## Anatomy of a RAG System Prompt

A production RAG system prompt has five components. Each serves a specific purpose.

### Component 1 — Role and Scope Definition

Sets who the LLM is and what it is allowed to answer.

```
You are a helpful assistant for [Company Name]'s HR department. 
You answer questions about employee policies, benefits, and procedures 
based exclusively on the official HR documentation provided to you.
```

The scope definition ("based exclusively on") primes the LLM to treat retrieved context as its authority rather than general knowledge. Being specific about the domain ("HR department") helps the LLM calibrate its uncertainty — it knows when a question is within scope.

### Component 2 — Grounding Instruction

Explicitly instructs the LLM to use the provided context and not to use outside knowledge.

```
Answer questions using ONLY the information provided in the context sections below.
Do not use your general knowledge or training data to answer questions.
If the answer is not explicitly present in the provided context, say so clearly.
```

This is the most critical component for preventing hallucination. The words "ONLY" and "explicitly" matter — vague instructions like "try to use the context" leave too much room for the LLM to fill gaps with parametric knowledge.

**The parametric conflict instruction:**
```
If the provided context contradicts information you may know from your training,
defer to the provided context — it represents the current, authoritative policy.
```

This directly addresses Problem 1. Without it, an LLM may override a recently changed policy with what it "knows" from training data that is now outdated.

### Component 3 — Citation Instruction

Tells the LLM how to cite sources.

```
When you provide information, cite the source using the reference number in brackets.
Example: "Employees are entitled to 16 weeks of parental leave [1]."
If multiple sources support a claim, cite all of them: "The policy was updated in 2024 [1][3]."
```

Citations serve two purposes: they make answers auditable (users can verify the source) and they force the LLM to be explicit about which context it is drawing from (reducing blending and fabrication).

### Component 4 — Insufficient Context Instruction

Explicitly tells the LLM what to do when the context does not contain the answer.

```
If the context does not contain enough information to answer the question:
- Say clearly: "I don't have information about this in the available documents."
- Do NOT guess, infer, or extrapolate beyond what the context states.
- You may suggest where the user might find this information if you know the appropriate department or resource.
```

This directly addresses Problem 3. Without this instruction, LLMs default to generating plausible-sounding answers even when context is insufficient. The explicit prohibition on guessing is important — "do not guess" is more effective than "be accurate."

### Component 5 — Output Format Specification

Defines how the answer should be structured.

```
Response format:
- Provide a direct answer to the question.
- Use bullet points for lists of items or steps.
- Use plain prose for explanations.
- Keep responses concise — do not restate information from the context unnecessarily.
- End with a brief note about the source document(s) if they are relevant to the user.
```

Format instructions reduce variance in output quality. Without them, LLMs produce inconsistently structured responses that are harder for users to parse and harder to test reliably.

---

## Complete System Prompt Example

Putting all five components together:

```python
RAG_SYSTEM_PROMPT = """You are a helpful assistant for Acme Corp's HR department.
You answer questions about employee policies, benefits, leave, compensation, and 
workplace procedures based exclusively on the official HR documentation provided to you.

GROUNDING RULES:
- Answer questions using ONLY the information in the context sections provided.
- Do not use your general knowledge or training data to supplement your answer.
- If the provided context contradicts information you know from elsewhere, defer 
  to the provided context — it represents Acme Corp's current authoritative policy.
- Preserve exact figures, dates, durations, and amounts as stated in the context.
  Do not round, estimate, or paraphrase numerical information.

CITATION RULES:
- Cite the source of each claim using the reference number in brackets: [1], [2], etc.
- If a claim is supported by multiple sources, cite all: [1][2].
- Do not fabricate citations. Only cite sources that are actually provided.

WHEN CONTEXT IS INSUFFICIENT:
- If the answer is not in the provided context, respond with:
  "I don't have information about this in the available HR documents."
- Do NOT guess, infer beyond what is stated, or generate plausible-sounding 
  information that is not present in the context.
- You may direct users to the appropriate team: "For this question, please contact 
  the HR team directly at hr@acmecorp.com."

RESPONSE FORMAT:
- Be direct and concise. Lead with the answer.
- Use bullet points only when listing multiple distinct items.
- For yes/no questions, state the answer first, then explain.
- Do not add disclaimers like "please consult HR" unless the context explicitly 
  warrants it — adding unnecessary disclaimers reduces user trust in correct answers.
"""
```

---

## The User Turn: Structuring the Retrieved Context

The system prompt defines the rules. The user turn contains the actual query and retrieved context. How you structure this matters significantly.

### Option 1 — Context First, Query Last

```python
def build_user_message_context_first(
    query: str,
    formatted_context: str
) -> str:
    return f"""Context documents:

{formatted_context}

---

Question: {query}"""
```

**Advantage:** The LLM reads all context before encountering the question. By the time it sees the question, it has already processed the relevant information.

**Disadvantage:** With a very long context block, the LLM may lose focus by the time it reaches the question.

### Option 2 — Query First, Context Second

```python
def build_user_message_query_first(
    query: str,
    formatted_context: str
) -> str:
    return f"""Question: {query}

Use the following context documents to answer the question:

{formatted_context}"""
```

**Advantage:** The LLM knows what it is looking for while reading the context. May improve attention to relevant passages.

**Disadvantage:** The LLM may start "thinking" about the answer before seeing the context, potentially anchoring on parametric knowledge.

### Option 3 — Query Framing + Context + Query Repetition

For long contexts, repeat the query after the context:

```python
def build_user_message_framed(
    query: str,
    formatted_context: str
) -> str:
    return f"""I need to answer this question: {query}

Here are the relevant documents:

{formatted_context}

---

Based on the documents above, answer this question:
{query}"""
```

The repeated query at the end combats the lost-in-the-middle effect — the LLM sees the query in the recency position, which receives stronger attention. This pattern consistently improves answer quality for long contexts.

**In practice:** For short contexts (< 2000 tokens), Option 1 or 2 works fine. For long contexts (> 5000 tokens), Option 3 with repeated query is measurably better.

---

## Handling the "I Don't Know" Problem

Getting LLMs to say "I don't know" reliably is harder than it sounds. LLMs are trained to be helpful, which creates a strong bias toward producing an answer even when the answer is not supported by the context.

### Testing Your Prompt's IDK Reliability

Before deploying, test with queries your corpus cannot answer:

```python
idk_test_queries = [
    "What is the CEO's personal email address?",
    "What will the 2026 health insurance rates be?",
    "How many employees does Acme Corp have in Brazil?",
    "What is the maximum overtime pay under California law?",  # Legal, not HR policy
]

async def test_idk_reliability(
    idk_queries: list[str],
    rag_pipeline,
    sample_context: str  # Context that does not contain answers to these queries
) -> float:
    """
    Measure what fraction of unanswerable queries correctly result in IDK responses.
    """
    correct_idk = 0
    
    for query in idk_queries:
        response = await rag_pipeline.answer(query)
        
        # Check if response appropriately acknowledges lack of information
        idk_signals = [
            "don't have information",
            "not in the available",
            "cannot find",
            "not covered in",
            "please contact",
            "no information about this"
        ]
        
        is_idk = any(signal in response.lower() for signal in idk_signals)
        if is_idk:
            correct_idk += 1
    
    return correct_idk / len(idk_queries)
```

A well-designed prompt should achieve > 90% IDK reliability on queries that are genuinely unanswerable from the corpus. If you are below 80%, your grounding instructions are not strong enough.

### Strengthening IDK Behavior

If the LLM is hallucinating instead of saying IDK, escalate the grounding instruction:

```python
# Weak (often insufficient):
"If you don't know the answer, say so."

# Stronger:
"If the answer is not explicitly stated in the numbered context sections,
respond with exactly: 'I don't have information about this in the provided documents.'
Do not infer, extrapolate, or reason beyond what is directly stated."

# Strongest (for high-stakes domains):
"You are ONLY allowed to state information that appears verbatim or as a direct
logical consequence in the provided context. Any information not traceable to a
specific [N] reference must NOT appear in your response."
```

The escalation strategy: start with the moderate version and test IDK reliability. Only use the strongest version if the moderate version fails — it can make the model overly restrictive and refuse to answer questions that are genuinely answerable from context.

---

## Domain-Specific Prompt Adaptations

Different domains require different prompt configurations.

### Legal and Compliance

```python
LEGAL_RAG_PROMPT_ADDITIONS = """
CRITICAL: Legal documents contain precise language where every word matters.
- Do not paraphrase legal terms — quote them exactly as they appear.
- When citing specific clauses, include the section number: "Per Section 8.3(b)..."
- If a clause has exceptions, conditions, or cross-references, state them explicitly.
- Do not provide legal advice. You provide factual summaries of document content.
  Always recommend consulting legal counsel for interpretation.
"""
```

### Financial Data

```python
FINANCIAL_RAG_PROMPT_ADDITIONS = """
PRECISION RULES for financial data:
- Report all monetary values exactly as stated (do not round $1,234,567 to "$1.2M").
- Always specify the currency, fiscal period, and whether figures are audited.
- When presenting financial ratios or percentages, include the calculation basis.
- If figures appear in multiple documents with discrepancies, report both and note
  the discrepancy rather than choosing one.
"""
```

### Medical and Clinical

```python
MEDICAL_RAG_PROMPT_ADDITIONS = """
SAFETY CRITICAL:
- This system provides information from clinical documentation only.
- It does NOT provide medical advice, diagnosis, or treatment recommendations.
- Always direct users to consult qualified healthcare professionals for medical decisions.
- Dosage, treatment protocols, and contraindications must be quoted exactly —
  never paraphrase, summarize, or abbreviate clinical instructions.
"""
```

### Customer Support

```python
SUPPORT_RAG_PROMPT_ADDITIONS = """
TONE: Be friendly and helpful. If the customer is frustrated, acknowledge it briefly 
before providing the answer.

ESCALATION: If the provided documents do not answer the customer's issue,
do not say "I don't know" alone — provide next steps:
"I don't have information about this specific issue. Please contact our support team
at support@company.com or call 1-800-XXX-XXXX for direct assistance."

NEVER: Do not promise outcomes you cannot guarantee from the documentation.
"""
```

---

## Multi-Turn Conversation Prompt Design

In multi-turn conversations, the system prompt persists but the context window fills with conversation history. This creates several prompt design challenges.

### Problem: Context Injection in Multi-Turn

In a single-turn RAG system, you inject retrieved context into each message. In multi-turn, should you re-inject context every turn?

**Strategy 1 — Inject context every turn:**
For every user message, retrieve fresh context and inject it into that turn's user message.

```python
async def multi_turn_rag(
    conversation_history: list[dict],
    new_user_message: str,
    retriever
) -> str:
    # Retrieve context for the new query
    context, sources = await retriever.retrieve_and_format(new_user_message)
    
    # Inject context into this turn's user message
    user_message_with_context = f"""Context for this question:
{context}

Question: {new_user_message}"""
    
    # Build messages for API call
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        *conversation_history,  # Previous turns
        {"role": "user", "content": user_message_with_context}
    ]
    
    response = await llm.chat(messages)
    return response
```

**Advantage:** Fresh, relevant context for every query. Handles topic changes between turns.
**Disadvantage:** Context window fills quickly with repeated context injections. Expensive.

**Strategy 2 — Retrieve only when needed:**
Detect whether the new query is a continuation (can be answered from conversation history) or a new information need (requires retrieval).

```python
async def adaptive_multi_turn_rag(
    conversation_history: list[dict],
    new_user_message: str,
    retriever,
    llm
) -> str:
    
    # Detect if retrieval is needed
    retrieval_needed = await should_retrieve(
        conversation_history, new_user_message, llm
    )
    
    if retrieval_needed:
        # Resolve conversational context and retrieve
        standalone_query = await resolve_conversational_query(
            new_user_message, conversation_history, llm
        )
        context, sources = await retriever.retrieve_and_format(standalone_query)
        
        user_content = f"[Retrieved context]\n{context}\n\n[Question]\n{new_user_message}"
    else:
        user_content = new_user_message
    
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": user_content}
    ]
    
    return await llm.chat(messages)

async def should_retrieve(history, query, llm) -> bool:
    """Use LLM to decide if retrieval is needed for this turn."""
    prompt = f"""Given this conversation and the new question, decide if new 
document retrieval is needed to answer the question, or if it can be answered 
from the conversation history.

Last 2 turns: {history[-4:]}
New question: {query}

Respond with only: RETRIEVE or NO_RETRIEVE"""
    
    response = await llm.complete(prompt, max_tokens=10)
    return "RETRIEVE" in response
```

---

## Prompt Versioning and Testing

Prompts are code. They should be versioned, tested, and deployed with the same rigor as application code.

```python
PROMPT_REGISTRY = {
    "v1.0": {
        "system_prompt": "...",
        "deployed_at": "2024-01-15",
        "eval_score": {"idk_reliability": 0.82, "faithfulness": 0.88}
    },
    "v1.1": {
        "system_prompt": "...",  # Strengthened IDK instruction
        "deployed_at": "2024-02-01",
        "eval_score": {"idk_reliability": 0.94, "faithfulness": 0.91}
    },
    "current": "v1.1"
}

def get_system_prompt(version: str = "current") -> str:
    key = PROMPT_REGISTRY.get(version, PROMPT_REGISTRY["current"])
    return PROMPT_REGISTRY[key]["system_prompt"]
```

Before deploying a new prompt version, run it against your evaluation set (Lesson 6.5) and compare metrics to the previous version. A prompt change that improves IDK reliability by 10% may reduce answer completeness by 5% — you need the numbers to make an informed decision.

---

## Common Prompt Design Mistakes

**Mistake 1: Vague grounding instructions.**
"Use the context to help answer" is not the same as "answer using ONLY the context." Vague instructions give the LLM latitude to fill gaps from memory.

**Mistake 2: No explicit IDK instruction.**
Without explicit instruction, LLMs hallucinate rather than acknowledge uncertainty. Every RAG prompt needs an IDK path.

**Mistake 3: Overly long system prompts.**
A 3000-token system prompt consumes context budget and may cause the LLM to lose focus on the retrieval context. Keep system prompts concise — under 500 tokens for most applications.

**Mistake 4: No format specification.**
Without format guidance, LLMs produce variable-length, inconsistently structured responses. This makes evaluation and testing harder and user experience inconsistent.

**Mistake 5: Same prompt for all query types.**
Factual lookup, synthesis, comparison, and procedural queries need different output structures. A single rigid prompt produces mediocre results across all types.

**Mistake 6: Never testing IDK behavior.**
Teams routinely test whether the system answers correctly but rarely test whether it handles unanswerable questions correctly. IDK reliability should be a first-class metric in your evaluation suite.

---

## Summary

- Prompt design is not an afterthought — it determines whether the LLM uses retrieved context faithfully, cites sources accurately, and handles missing information correctly.
- Five components of a production RAG prompt: role/scope, grounding instruction, citation instruction, IDK instruction, output format.
- The grounding instruction must be explicit: "ONLY the provided context" — vague instructions allow parametric knowledge contamination.
- The IDK instruction is the most important safety component. Test IDK reliability explicitly with unanswerable queries. Target > 90% reliability.
- Structure the user turn with context + repeated query for long contexts. The repeated query at the end combats the lost-in-the-middle effect.
- Domain-specific adaptations are necessary: legal (exact quoting), financial (precise figures), medical (no advice), support (escalation paths).
- Multi-turn conversations need context injection strategies — either inject every turn or use a retrieval necessity detector.
- Treat prompts as versioned code. Test against an evaluation set before deploying changes.

---

## What's Next

Lesson 4.2 covers the parametric vs. contextual knowledge conflict in depth — when the LLM's training knowledge conflicts with retrieved context, how to detect it, and strategies beyond prompt instructions to resolve these conflicts.