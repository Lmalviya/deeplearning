# RAG Architecture Variations — Case Study Notes

*Companion to the baseline "Conversational AI + RAG" system design doc. Each section covers: what's different from baseline RAG, the signals/clarifying-question-answers that point you toward this variant, key architectural decisions, and trade-off discussions.*

---

## How to Use This Doc

The baseline RAG design (single index, single-pass retrieval, single modality, single tenant) is your default. The interviewer will often start with a baseline problem statement and then **add a constraint** through your clarifying questions or their follow-up — and that constraint is the signal for which variant applies. The skill being tested is: **can you detect the signal and adapt the architecture, rather than forcing every problem into the same baseline design?**

Below, each variant starts with the **signal** — what answer to your clarifying questions should make you say "ah, this isn't baseline RAG anymore."

---

## 1. Agentic RAG

### Signal — When Does This Apply?

You ask: *"Is retrieval always needed, or could some queries be answered directly / require other actions (calculations, API calls, multiple data sources)?"*

If the interviewer says:
- "Users ask a mix of things — some need document lookup, some are general questions, some need real-time data (e.g., 'what's my current ticket status' from a live system)."
- "There are multiple knowledge sources (e.g., HR docs, IT docs, a ticketing system) and the system should figure out which to use."
- "Sometimes a single retrieval isn't enough — the system might need to look something up, realize it needs more info, and look up something else."

→ **This is Agentic RAG.** The "always retrieve once, then generate" baseline breaks because retrieval isn't always necessary or sufficient.

### What's Different from Baseline RAG

In baseline RAG, the flow is rigid: query → retrieve → generate. In agentic RAG, an LLM (or a lightweight planner) sits in a loop and **decides**:
- Do I need to retrieve at all? (e.g., "hello, how are you" needs no retrieval)
- Which source/index should I query? (routing across multiple knowledge bases or tools)
- Is what I retrieved sufficient, or do I need another round?
- Should I call a tool (calculator, API, code execution) instead of or in addition to retrieving?

### Key Architectural Decisions

- **Planner/router design**: A small classifier or the main LLM itself (via function-calling/tool-use) decides the next action. Function-calling-capable LLMs (Claude, GPT-4) can be given "tools" — e.g., `search_hr_docs`, `search_it_docs`, `get_ticket_status` — and decide which to invoke.
- **Loop termination**: Need a max-iteration cap (e.g., 3-5 retrieval/tool-call rounds) to prevent infinite loops and control latency/cost.
- **State tracking across iterations**: Each iteration's retrieved context needs to accumulate into the prompt for the next reasoning step — this grows context size, so summarization/compression of intermediate steps may be needed.
- **Latency implications**: Each agent "thought" + tool call adds a round trip. A query that used to take 2 seconds (baseline) might take 8-15 seconds with 3 agent iterations — this needs to be communicated to the user (e.g., "searching documents...", "checking ticket system...") via streaming status updates.

### Trade-off Discussions

- **Agentic vs baseline cost/latency**: Agentic RAG is more flexible but multiplies LLM calls (one for planning, one+ for retrieval decisions, one for final generation). For a system where 90% of queries are simple single-source lookups, full agentic overhead on every query is wasteful. **Hybrid approach**: a lightweight initial classifier decides "simple" vs "complex" queries; only complex ones go through the full agent loop.
- **Reliability vs flexibility**: A rigid pipeline is more predictable and easier to debug/test. An agent's decisions can be inconsistent (same query, different routing on different runs) — this needs to be tested with eval sets covering routing decisions specifically, not just final answer quality.

---

## 2. Multi-hop RAG

### Signal — When Does This Apply?

You ask: *"What kinds of questions will users typically ask — simple factual lookups, or things that might require combining information from multiple documents/sections?"*

If the interviewer says:
- "Users might ask comparative or analytical questions — e.g., 'How does our refund policy differ between the US and EU regions?' or 'What changed between the Q1 and Q2 product specs?'"
- "Some questions require connecting information that lives in different documents — e.g., a question about an employee's eligibility might require checking both the HR policy doc and their specific contract."

→ **This is Multi-hop RAG.** A single retrieval pass returns chunks relevant to *part* of the question but not all of it, and naive RAG would either miss half the answer or hallucinate the missing half.

### What's Different from Baseline RAG

Baseline RAG: one query → one embedding → one retrieval → one generation. Multi-hop RAG: the system recognizes that the question has **multiple sub-questions or requires sequential lookups**, where the second lookup may depend on the result of the first.

Example: "What's the revenue difference between the company's two largest product lines last year?"
- Hop 1: Identify the two largest product lines (requires retrieving overview/summary data)
- Hop 2: Retrieve revenue figures for product line A
- Hop 3: Retrieve revenue figures for product line B
- Final: Compute the difference and generate the answer

### Key Architectural Decisions

- **Query decomposition**: An LLM call breaks the original query into sub-queries. This could be done upfront (decompose everything, then retrieve for each sub-query in parallel) or iteratively (retrieve for sub-query 1, use that result to formulate sub-query 2).
- **Parallel vs sequential hops**: If sub-questions are independent (e.g., "compare X and Y" where X and Y don't depend on each other), retrieve in parallel for lower latency. If sub-question 2 depends on the answer to sub-question 1 (e.g., "find the two largest product lines, THEN get their revenue"), hops must be sequential.
- **Context aggregation**: Results from multiple hops need to be combined into a single context for final generation — risk of context window bloat if not summarized.
- **Stopping criteria**: How does the system know it has "enough" hops? Often capped at a fixed number (2-3) or determined by whether the decomposed sub-questions are all answered.

### Trade-off Discussions

- **Decomposition accuracy is the bottleneck**: If the LLM mis-decomposes the question (misses a sub-question, or decomposes a question that didn't actually need it), the whole pipeline fails. This makes multi-hop RAG harder to evaluate — you need eval sets specifically testing decomposition quality, not just end-to-end answer quality.
- **Latency cost is multiplicative**: Each hop adds a retrieval + possibly an LLM call. For a system with strict latency SLAs, multi-hop should be reserved for queries detected as "complex" (similar to agentic RAG's routing idea) — most queries are single-hop and shouldn't pay this cost.
- **Over-engineering risk**: If 95% of real user queries are single-hop, building elaborate multi-hop infrastructure for the 5% may not be worth the complexity — could instead handle multi-hop via a slightly larger top-k retrieval (retrieve more chunks, hope the union covers both "hops") as a cheaper approximation, accepting lower accuracy on truly complex questions.

---

## 3. RAG with Hallucination Detection / Self-Correction (CRAG-style)

### Signal — When Does This Apply?

You ask: *"How important is it that answers are strictly grounded in the retrieved documents? What happens if the retrieved documents don't actually contain the answer?"*

If the interviewer says:
- "This is for a regulated domain (legal, medical, finance, HR) — incorrect answers have real consequences, so we need high confidence the answer is grounded."
- "Sometimes the knowledge base might not have the answer — what should the system do then? We don't want it to make something up."
- "We need some way to know when the system is 'guessing' vs. confidently answering from source material."

→ **This is RAG with hallucination detection / self-correction.** Baseline RAG assumes retrieval always returns "good enough" context and the LLM will use it faithfully — neither assumption is safe in high-stakes domains.

### What's Different from Baseline RAG

Baseline RAG: retrieve → generate → return, with no check on whether (a) the retrieved chunks are actually relevant, or (b) the generated answer is actually supported by those chunks.

CRAG-style (Corrective RAG) adds an **evaluation step between retrieval and generation** (and sometimes after generation too):
1. Retrieve chunks as normal.
2. **Grade retrieval quality** — is this context actually relevant/sufficient to answer the question?
   - If **high confidence**: proceed to generation normally.
   - If **low confidence**: trigger a fallback — re-formulate the query and retry retrieval, expand the search (broader top-k, different index), or fall back to a web search / "I don't know" response.
   - If **ambiguous**: combine both retrieved context and fallback sources.
3. (Optional) **Grade the generated answer** — does it stay grounded in the provided context, or does it introduce claims not present in the retrieved chunks? If ungrounded, regenerate with a stricter prompt or flag the response to the user.

### Key Architectural Decisions

- **Relevance grading mechanism**: Could be a smaller/cheaper LLM call ("Is this passage relevant to answering this question? yes/no/maybe"), a cross-encoder re-ranker with a relevance score threshold, or an embedding-similarity threshold (cruder, less accurate).
- **Fallback strategies when retrieval is poor**:
  - Query rewriting/expansion (rephrase the user's question and retry)
  - Broaden retrieval (increase top-k, search across additional indices)
  - External fallback (web search, if appropriate for the domain)
  - Honest refusal ("I don't have information on this in the knowledge base")
- **Post-generation groundedness check**: An LLM call comparing the generated answer against the retrieved context — "Does every claim in this answer appear in the provided context?" If not, either regenerate with an instruction to only use provided context, or append a disclaimer.
- **Where to surface uncertainty to the user**: Confidence scores, citations linking specific claims to specific source chunks, or explicit "I'm not fully certain about this" framing.

### Trade-off Discussions

- **Added latency and cost vs trust**: Every extra grading/checking step is another LLM call, adding latency and cost. For a regulated domain, this is a worthwhile trade — a wrong answer has higher cost than a slow one. For a low-stakes internal FAQ bot, this overhead may not be justified.
- **False refusals**: An overly conservative relevance grader might reject context that's actually useful (e.g., context that requires some inference to connect to the question), leading to unnecessary "I don't know" responses and a frustrating user experience. This threshold needs tuning against real query distributions.
- **Citation granularity vs complexity**: Fine-grained citations (linking each sentence of the answer to a specific source) build user trust but are harder to implement reliably — the LLM needs to track provenance through generation, which most APIs don't natively support well, often requiring post-hoc matching (compare generated sentences against retrieved chunks via similarity).

---

## 4. Multi-tenant RAG

### Signal — When Does This Apply?

You ask: *"Is this knowledge base shared across all users, or does each user/organization/team have their own separate set of documents that others shouldn't see?"*

If the interviewer says:
- "This is a SaaS product — each customer company uploads their own documents, and obviously Customer A should never see Customer B's data."
- "Different departments within the company have access to different document sets — e.g., only HR can query HR policy docs."
- "We're building this as a platform that multiple internal teams will use, each with their own knowledge base."

→ **This is Multi-tenant RAG.** The core challenge shifts from "retrieve good context" to **"retrieve good context while guaranteeing strict data isolation"** — this is fundamentally a SaaS/access-control problem layered on RAG, not a retrieval-quality problem.

### What's Different from Baseline RAG

Baseline RAG assumes one shared knowledge base, accessible to all users equally. Multi-tenant RAG must ensure:
- A query from Tenant A's user **never** retrieves chunks belonging to Tenant B, even accidentally.
- Within a tenant, there may be further access control (role-based — e.g., HR docs visible only to HR staff).
- Each tenant might have different documents, different update frequencies, even different customization needs (different system prompts, different LLM tiers based on their subscription plan).

### Key Architectural Decisions

- **Isolation strategy** — three common approaches, each with different trade-offs:
  1. **Separate vector DB index/collection per tenant**: Strongest isolation guarantee (a bug in query filtering can't leak data, because there's no shared index to leak from). Downside: index sprawl — thousands of tenants means thousands of indices to provision, monitor, and keep warm; resource overhead if many tenants are small/inactive.
  2. **Shared index with metadata filtering** (e.g., every chunk tagged with `tenant_id`, every query filters `WHERE tenant_id = X`): Operationally simpler — one index to maintain, easier to scale. Downside: a missed or buggy filter is a serious data leak; also, **ANN search + post-filter can reduce result quality** — if you retrieve top-20 globally then filter to tenant X, and tenant X only has 2 of those 20, you effectively did a "top-2" retrieval instead of "top-20," potentially missing relevant chunks. (Some vector DBs support pre-filtering — filtering before or during the ANN search rather than after — which mitigates this but isn't universally available/performant.)
  3. **Hybrid — sharded by tenant tier**: Large/enterprise tenants get dedicated indices (isolation + performance guarantees worth the cost for big accounts); small tenants share a multi-tenant index with metadata filtering (cost-efficient for the long tail).
- **Embedding model sharing**: Generally one shared embedding model across tenants is fine (the model itself doesn't "know" whose data it's embedding) — but if a tenant has highly specialized vocabulary (e.g., a legal firm vs a gaming company), a shared general-purpose embedding model might underperform for both. Custom embeddings per tenant add significant operational complexity and are usually only justified for large enterprise tenants.
- **Noisy neighbor mitigation**: One tenant's traffic spike shouldn't degrade latency for others.
  - Per-tenant rate limiting (token bucket per tenant ID)
  - Separate request queues per tenant tier, or dedicated compute for high-tier tenants
  - Resource quotas on ingestion (one tenant bulk-uploading 10,000 documents shouldn't starve the ingestion pipeline for everyone else)
- **Tenant onboarding/offboarding**: New tenant signup needs to provision their index/namespace + ingest their initial documents without impacting existing tenants. Offboarding needs to cleanly delete all of a tenant's data (compliance requirement in many cases — "right to be forgotten").
- **Within-tenant access control**: Beyond tenant isolation, role-based filtering within a tenant — e.g., metadata tags like `department: HR` combined with the user's role at query time, similar filtering logic as tenant isolation but one layer deeper.

### Trade-off Discussions

- **Isolation strength vs operational cost**: Separate-index-per-tenant is the "safe by construction" choice but doesn't scale to thousands of tenants economically. Shared-index-with-filtering scales better but requires rigorous testing of filter logic (this is the kind of bug that causes major incidents — "Customer A saw Customer B's data" is a worst-case SaaS failure).
- **Cost allocation**: In a shared-index model, it's harder to attribute infra cost per tenant for billing purposes. Separate indices make cost attribution straightforward but at higher absolute infra cost.
- **Customization vs consistency**: Letting tenants customize chunking strategies, system prompts, or models increases their satisfaction but multiplies the testing/maintenance surface — every customization is a new "configuration" that needs to work correctly and be regression-tested.

---

## 5. Multi-modal RAG (Text + Images)

### Signal — When Does This Apply?

You ask: *"What format is the source content in — is it purely text, or does it include diagrams, images, charts, scanned pages, product photos, etc.? And would users want to ask questions about those visual elements?"*

If the interviewer says:
- "The documents are technical manuals with diagrams and schematics — a user might ask 'show me the wiring diagram for component X' or ask a question that's answered by a chart, not by text."
- "It's a product catalog — users ask things like 'find me a similar-looking product' or ask questions about product images."
- "Some source documents are scanned PDFs or contain embedded images that carry important information not captured in the surrounding text."

→ **This is Multi-modal RAG.** Baseline RAG (text embeddings, text retrieval, text generation) can't retrieve or reason about visual content at all — images are either ignored entirely (information loss) or need a separate pathway.

### What's Different from Baseline RAG

Baseline RAG's entire pipeline — chunking, embedding, indexing, retrieval, generation — is text-centric. Multi-modal RAG needs to handle images (and potentially other modalities) at each stage:

- **Ingestion**: documents now contain both text chunks and images that need to be extracted, stored, and made retrievable.
- **Embedding**: text and images need to end up in a retrievable form — either a shared vector space or parallel indices.
- **Retrieval**: a text query might need to surface relevant images (and vice versa — an image-based query retrieving related text).
- **Generation**: the LLM may need to actually "see" retrieved images to answer the question (requires a vision-capable LLM), or work from text descriptions of those images.

### Key Architectural Decisions

- **Embedding strategy — two main approaches**:
  1. **Unified multi-modal embedding model** (e.g., CLIP-style models that embed both text and images into the *same* vector space): A text query embedding can directly retrieve nearby image embeddings, and vice versa. Simpler retrieval (one index, one similarity search) but these models are often less precise for pure-text retrieval than dedicated text embedding models, and the "shared space" alignment between text and image embeddings can be imperfect for domain-specific imagery (e.g., technical schematics vs. the natural images CLIP was trained on).
  2. **Separate embeddings + caption bridge**: Generate a text caption/description for each image (using a vision-language model at ingestion time), embed that caption with a standard text embedding model alongside regular text chunks. Retrieval stays purely text-based (query → text embedding → search text+caption index), and the "image" is retrieved via its caption. Simpler to integrate into an existing text-RAG pipeline, but retrieval quality depends entirely on caption quality — a vague caption means the image is effectively invisible to retrieval.
- **Image storage**: Raw images don't belong in a vector DB — store images in object storage (e.g., S3), with the vector DB holding only embeddings/captions + a reference (URL/path) to the actual image.
- **Generation — does the LLM need to "see" the image?**
  - If using a **vision-capable LLM** (e.g., Claude or GPT-4 with vision), retrieved images can be passed directly into the prompt alongside text context — the model can reason about diagrams, charts, etc. directly. Higher cost per call, but much better for "visual reasoning" questions (e.g., "what does this chart show?").
  - If using a **text-only LLM**, retrieved images are represented only via their captions/descriptions in the prompt — cheaper, but the LLM can't verify or reason beyond what the caption captured, and the user doesn't see the actual image in context unless the frontend separately renders it.
- **Chunking for mixed-content documents**: When a document has interleaved text and images (e.g., a manual with a paragraph, then a diagram, then more text), chunk boundaries should ideally preserve this association — e.g., a chunk might be "paragraph + the diagram it refers to" rather than splitting them into unrelated chunks in different parts of the index.

### Trade-off Discussions

- **Unified embedding vs caption-bridge**: Unified multi-modal embeddings (CLIP-style) give more "native" cross-modal retrieval but may underperform on domain-specific text retrieval compared to specialized text embedding models — you might be trading off text-retrieval quality to gain image-retrieval capability. Caption-bridge keeps text retrieval quality intact (using your existing best text embedding model) but image retrieval is only as good as the captioning step — a two-stage lossy pipeline (image → caption → embedding) vs. unified models' single-stage (image → embedding) but less specialized.
- **Cost of vision-capable generation**: Including images directly in LLM prompts (vision models) is significantly more expensive per call than text-only. For a high-volume system, this pushes toward "only invoke vision model when retrieval surfaces an image as relevant" — i.e., conditional routing similar to agentic RAG's "decide what's needed" pattern, rather than always using a vision model by default.
- **User experience — showing vs describing**: Even if the LLM reasons about an image via caption only, the *frontend* should likely still display the actual retrieved image to the user (e.g., "here's the diagram referenced in this answer") — decoupling "what the LLM uses to reason" from "what the user sees" can give a good UX even with a cheaper text-only generation pipeline.

---

## Summary Table — Signal-to-Architecture Mapping

| Clarifying Question Answer | Points To | Core Architectural Shift |
|---|---|---|
| "Some queries need no retrieval, or need multiple sources/tools" | Agentic RAG | Add a planner/router that decides retrieval/tool actions per query |
| "Questions often require combining info from multiple places" | Multi-hop RAG | Decompose query into sub-queries; sequential or parallel retrieval hops |
| "This is high-stakes; wrong answers are costly; KB might lack the answer" | Self-correcting RAG (CRAG) | Add relevance grading + groundedness checking with fallback strategies |
| "Multiple customers/orgs/teams, each with their own documents" | Multi-tenant RAG | Isolation strategy (separate indices vs filtered shared index) + per-tenant resource controls |
| "Documents contain diagrams/images/charts users need to query" | Multi-modal RAG | Multi-modal embeddings or caption-bridge; vision-capable generation; image object storage |

---

## How These Combine in Practice

Real interview escalations often **stack** these. A common pattern:

> "Now what if this RAG system also needs to serve multiple customer organizations [multi-tenant], where questions sometimes require looking at both a policy doc and a contract doc [multi-hop], and we want to make sure the system says 'I don't know' rather than guessing when it's not confident [self-correcting]?"

When you hear a stacked scenario like this, **address each constraint as a separate architectural layer** rather than trying to redesign everything at once — e.g., "the tenant isolation happens at the retrieval/indexing layer, the multi-hop decomposition happens in the query-processing layer before retrieval, and the confidence-grading happens after retrieval but before generation — these are largely orthogonal concerns that compose without conflicting."