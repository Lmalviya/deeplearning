# Lesson 1.2 — The Full RAG Landscape: Naive → Advanced → Agentic → Graph

---

## Why You Need to Know the Landscape

When you build a RAG system, you are making architectural decisions at every step. The problem is that there is no single "correct" RAG architecture — the right design depends on your data, your query patterns, your accuracy requirements, and your scale.

Knowing the landscape means you can look at a problem and immediately know which class of RAG is appropriate, why simpler approaches will fail, and what trade-offs you are accepting with a more complex one.

Interviewers will frequently ask: "How would you improve your current RAG system?" or "Why did you choose this architecture over alternatives?" Without knowing the landscape, you cannot answer these well.

---

## The Four Generations of RAG

### 1. Naive RAG

This is where everyone starts. The pipeline is straightforward:

- **Index**: Split documents into fixed-size chunks → embed each chunk → store in vector DB.
- **Query**: Embed the user query → do a cosine similarity search → take top-K chunks → stuff into prompt → generate.

This works surprisingly well for simple demos and small, clean corpora. It is what most tutorials show you.

**Where it breaks:**

- **Fixed chunking destroys context.** A 512-token chunk splits mid-sentence, mid-table, mid-argument. The retrieved chunk is semantically incomplete.
- **Single-vector retrieval misses vocabulary mismatch.** If the user asks "myocardial infarction" but the document says "heart attack," cosine similarity of dense embeddings will often still find it — but edge cases abound, especially for rare terms, proper nouns, and codes (medical, legal, financial).
- **Top-K is arbitrary.** You pick K=5 with no principled reason. Too low — you miss relevant chunks. Too high — you flood the context with noise.
- **No query understanding.** The raw query goes directly to retrieval. Ambiguous, short, or poorly phrased queries retrieve garbage.
- **No re-ranking.** The order and selection of chunks is determined entirely by embedding similarity, which is a coarse signal.

Naive RAG is good enough to prove a concept. It is not good enough for production.

---

### 2. Advanced RAG

Advanced RAG is not a single technique — it is a collection of targeted improvements applied to the naive pipeline. Each improvement addresses a specific failure mode.

Think of it as upgrading each stage of the pipeline:

**Pre-retrieval improvements (query side):**

- **Query rewriting** — Use an LLM to rewrite the user's question into a cleaner, more retrieval-friendly form. "what does the contract say about getting out of it" → "contract termination clauses and exit conditions."
- **Query expansion** — Generate multiple versions of the query and retrieve for each, then merge results. Increases recall.
- **HyDE (Hypothetical Document Embeddings)** — Instead of embedding the query directly, ask the LLM to generate a hypothetical answer to the query, then embed that hypothetical answer. The hypothesis lives in "document space" rather than "query space," which significantly improves retrieval for short or vague queries. We cover this in depth in Lesson 3.5.
- **Sub-question decomposition** — Break a complex query into smaller sub-questions, retrieve for each, then synthesize. "Compare the refund policies of Product A and Product B" becomes two separate retrievals.

**Retrieval improvements:**

- **Hybrid search** — Combine dense (embedding-based) retrieval with sparse (BM25/keyword-based) retrieval. Dense handles semantic similarity; sparse handles exact keyword matches. They catch different things. Covered in depth in Lesson 3.3.
- **Metadata filtering** — Before or alongside vector search, apply hard filters (date range, document type, author, department). Dramatically reduces the search space and improves precision.
- **Parent-child chunking** — Index small chunks for retrieval precision, but return their larger parent chunk for generation context. You get the best of both: accurate retrieval, rich context.

**Post-retrieval improvements:**

- **Re-ranking** — After retrieving top-K candidates, run a more expensive but more accurate model (a cross-encoder) to re-score and re-order them. The top-K from vector search is not always the most relevant K — re-ranking fixes this. Covered in Lesson 3.6.
- **Contextual compression** — Remove irrelevant sentences from retrieved chunks before passing to the LLM. Reduces noise and saves context space.
- **Reciprocal Rank Fusion (RRF)** — When you have multiple retrieval sources (dense + sparse, or multiple queries), RRF merges their ranked lists in a principled way without needing to normalize scores.

**Your current pipeline is Advanced RAG.** Query rewrite/expansion → hybrid (dense + BM25) → RRF → cross-encoder re-ranking → generation. This is a solid, production-grade advanced RAG pipeline.

---

### <a href="https://arxiv.org/pdf/2310.11511" target="_blank">3. Modular / Self-RAG</a>

The key insight of Self-RAG (Asai et al., 2023) is: not every query needs retrieval.

In naive and advanced RAG, you always retrieve — every single query goes through the full pipeline. But if someone asks "what is 2 + 2?" or "write me a haiku about autumn," retrieval adds latency and noise without helping.

Self-RAG introduces **reflection tokens** — special tokens the LLM generates to decide:
- Should I retrieve at all? (ISREL token)
- Is the retrieved content actually relevant? (ISSUP token)
- Is my generated answer supported by the retrieved content? (ISUSE token)

<p align="center">
  <img src=".\assets\types_of_token_self_rag.png" alt="Centered Image", width="600">
</p>

The model essentially critiques its own retrieval and generation in real time.

**What this means in practice:** The system becomes adaptive. It retrieves when needed, skips retrieval when not, and can flag when the retrieved context is insufficient.

Self-RAG requires a specially trained model that understands these reflection tokens. In practice, most teams approximate this behavior with a router — a small classifier or LLM call that decides whether to retrieve before running the full pipeline.

---

### <a href="https://arxiv.org/pdf/2401.15884" target="_blank">4. Corrective RAG (CRAG)</a>

CRAG (Shi et al., 2024) addresses a specific failure mode: what happens when the retrieved documents are wrong or irrelevant?

In standard RAG, if retrieval fails (retrieves wrong chunks), the LLM either generates a wrong answer based on bad context, or ignores the context and hallucinates. Both are bad.

CRAG adds a **retrieval evaluator** — a lightweight model that scores the quality of retrieved documents. Based on the score, it takes one of three actions:

- **Score is high (correct retrieval):** Proceed normally. Use the retrieved chunks.
- **Score is low (wrong retrieval):** Discard the retrieved chunks and fall back to a web search or a broader knowledge source.
- **Score is ambiguous:** Use both the retrieved chunks and the fallback source, then combine.

The key addition is the **fallback mechanism**. When local retrieval fails, CRAG does not give up — it tries to find the answer elsewhere.

CRAG is particularly valuable for open-domain Q&A where your local corpus might not contain the answer, but you do not want the system to silently give a bad answer based on irrelevant retrieved content.

---

### 5. Agentic RAG

Agentic RAG is a fundamentally different mental model from the ones above. Instead of a fixed pipeline (query → retrieve → generate), an agent decides its own retrieval strategy at runtime.

The agent is an LLM equipped with tools. Tools can include:
- Vector database search
- Keyword search
- Web search
- SQL query execution
- API calls
- Code execution
- Calculator

When a query arrives, the agent reasons about what information it needs, calls the appropriate tools in whatever order makes sense, observes the results, and iterates until it has enough to answer.

**Example:** "What was the revenue growth of Acme Corp between Q1 2023 and Q1 2024, and how does it compare to the industry average?"

A fixed pipeline would try to retrieve all of this in one shot. An agent would:
1. Search the internal document store for Acme Corp Q1 2023 revenue.
2. Search for Acme Corp Q1 2024 revenue.
3. Recognize it needs industry average data, which might not be in the internal store.
4. Do a web search for industry average growth figures.
5. Calculate the comparison.
6. Generate the final answer.

**Frameworks:** LangChain, LlamaIndex, LangGraph, CrewAI all support agentic RAG patterns. The underlying mechanism is usually ReAct (Reason + Act) — the agent alternates between reasoning steps and tool-use steps.

**Trade-offs:**
- More flexible and capable than fixed pipelines.
- Harder to control, debug, and make deterministic.
- Higher latency (multiple LLM calls per query).
- Can loop, get stuck, or take unexpected paths.
- Much harder to evaluate — you cannot just measure retrieval quality, you have to evaluate the agent's entire reasoning trace.

Agentic RAG is the right choice when queries are complex, multi-step, or require reasoning across heterogeneous sources. It is the wrong choice when you need sub-second latency, deterministic behavior, or when your queries are well-structured and predictable.

---

### <a href="https://arxiv.org/pdf/2501.00309" target="_blank">6. Graph RAG</a>

Graph RAG (Edge et al., Microsoft, 2024) solves a problem that vector search fundamentally cannot: **reasoning over relationships between entities across many documents.**

Vector search is good at finding "what documents talk about X?" It is poor at answering "how are X, Y, and Z related?" or "what is the overall theme connecting these 500 documents?"

Graph RAG builds a **knowledge graph** from your document corpus:
- Entities are extracted (people, organizations, places, concepts, products).
- Relationships between entities are extracted ("Acme Corp acquired BetaCo in 2019").
- These entities and relationships are stored as nodes and edges in a graph.
- Community detection algorithms identify clusters of related entities.
- Each community gets a summary.

At query time, the system can traverse the graph to find paths between entities, retrieve the relevant community summaries, and answer relational and thematic questions that vector search would miss entirely.

**When Graph RAG wins over standard RAG:**

- "What are all the ways our products are connected to regulatory requirements?" — requires traversing relationships.
- "Summarize the key themes across our entire document corpus." — requires community-level summaries.
- "How did the relationship between Entity A and Entity B evolve over time?" — requires temporal graph traversal.
- Legal, biomedical, and financial domains where relationships between entities are the core of the answer.

**Cost of Graph RAG:** Building the knowledge graph is expensive — it requires many LLM calls to extract entities and relationships from every document. The graph also needs to be maintained as documents change. For most applications, standard advanced RAG is sufficient. Graph RAG is a specialized tool for specific use cases.

---

## How to Choose the Right Architecture

Here is a practical decision framework:

**Use Naive RAG when:**
- You are prototyping or proving a concept.
- Your corpus is small (< a few hundred documents) and well-structured.
- Accuracy requirements are low.

**Use Advanced RAG when:**
- You are building a production system.
- Your corpus is large or heterogeneous.
- You need high accuracy and can afford moderate latency.
- This covers 80%+ of real-world RAG use cases.

**Use Self-RAG / CRAG when:**
- You need adaptive behavior (not all queries need retrieval).
- Your local corpus is incomplete and you need fallback strategies.
- You can tolerate slightly higher complexity.

**Use Agentic RAG when:**
- Queries are complex, multi-step, or require reasoning across multiple sources.
- You need to call external APIs or tools as part of answering.
- Latency requirements are relaxed (users expect to wait a few seconds).

**Use Graph RAG when:**
- Your use case is inherently relational (connections between entities matter).
- You need global summaries or thematic understanding across a large corpus.
- Your domain is legal, biomedical, financial, or any field with rich entity relationships.

**Combine them when:**
- Real systems often combine approaches. For example: an agent that uses advanced RAG as one of its tools, falling back to Graph RAG for relational queries and web search for recency.

---

## Where Your Pipeline Sits

Your current pipeline — query rewrite/expansion → hybrid (dense + BM25) → RRF → cross-encoder re-ranking → generation — is solidly in the **Advanced RAG** category. It implements most of the key pre-retrieval and post-retrieval improvements.

The natural next evolutions from here would be:
- Adding a CRAG-style retrieval quality evaluator with fallback.
- Adding parent-child chunking to improve context richness.
- Wrapping the pipeline in an agent when complex multi-step queries arrive.

---

## Summary

- **Naive RAG** is fixed chunking + embedding search + generation. Works for demos, breaks in production.
- **Advanced RAG** adds query understanding, hybrid retrieval, metadata filtering, and re-ranking. This is the production baseline.
- **Self-RAG** makes retrieval adaptive — the model decides when to retrieve and critiques its own outputs.
- **Corrective RAG** adds a fallback mechanism for when local retrieval fails.
- **Agentic RAG** replaces fixed pipelines with a reasoning agent that uses retrieval as one of many tools.
- **Graph RAG** builds a knowledge graph for relational and thematic queries that vector search cannot handle.
- These are not competing options — they are layered capabilities you add as your requirements grow.

---