# Case Study 1 — Enterprise Document Q&A: Mixed PDF Types, 100K+ Documents, Multi-Tenant

---

## Problem Statement

A global financial services firm wants to build an internal Q&A system over their entire document corpus. Employees across legal, compliance, finance, HR, and engineering departments need to ask questions and get accurate answers.

The corpus:
- 120,000 documents of mixed types: policies (Word/PDF), contracts (PDF), regulatory filings (PDF), internal memos, Confluence wiki pages, and scanned legacy documents.
- Documents range from 1 page to 400 pages.
- Some documents are updated daily (market risk limits), others are static for years (legal contracts).
- Multiple language variants (English primary, German, French, Japanese for regional content).

The users:
- 5,000 employees across 8 departments and 12 countries.
- Access control is strict: legal cannot see HR compensation data; regional offices cannot see other regions' client data.
- Each department has a different mental model of how to phrase questions.
- Peak load: Monday mornings, 500 concurrent users.

The requirements:
- Answer accuracy: ≥ 90% on an internal evaluation set.
- Latency: p95 < 3 seconds.
- Freshness: policy changes reflected within 2 hours of publication.
- Multi-tenant: document-level access control enforced at retrieval time.
- Languages: English, German, French (Japanese as stretch goal).
- Auditability: every answer must be traceable to a source document.

---

## Architecture Design Decisions

### Decision 1 — Vector Database Choice: Qdrant (Self-Hosted)

**Why Qdrant over alternatives:**

The firm has strict data residency requirements — all data must remain in their EU data centers. Pinecone (managed-only, US/EU data centers with data residency SLA complexity) and Weaviate Cloud create compliance questions. Qdrant can be self-hosted in their existing Kubernetes infrastructure.

Qdrant-specific features that matter:
- **Sparse + dense vector support:** Allows hybrid search (BM25 sparse vectors alongside dense embeddings) in a single collection, avoiding a separate Elasticsearch cluster.
- **Payload indexing with complex filters:** Access control is implemented via payload filters on `access_groups` metadata field. Qdrant's payload index handles this efficiently.
- **Named vector collections:** Each language gets its own named vector (same collection, multiple vector spaces). English queries search the English embedding space, German queries search the German space.

```python
from qdrant_client.models import VectorParams, Distance, SparseVectorParams

client.create_collection(
    collection_name="enterprise_docs",
    vectors_config={
        "dense_en": VectorParams(size=1024, distance=Distance.COSINE),
        "dense_de": VectorParams(size=1024, distance=Distance.COSINE),
        "dense_fr": VectorParams(size=1024, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse_bm25": SparseVectorParams()
    }
)
```

### Decision 2 — Embedding Model: multilingual-e5-large

Given the multilingual requirement and the need for cross-lingual retrieval (a user asking in German should find relevant English documents if no German version exists), a multilingual model is essential.

**Evaluation on internal data:**
- `multilingual-e5-large` (self-hosted, 560M params): Recall@10 = 0.87 on English, 0.82 on German, 0.79 on French test sets.
- `text-embedding-3-large` (OpenAI API): Recall@10 = 0.91 English, 0.84 German, but data leaving the firm creates compliance concerns.
- `paraphrase-multilingual-mpnet-base-v2`: Faster but Recall@10 only 0.74 on German.

Decision: `multilingual-e5-large` self-hosted. Deploy on 2× NVIDIA A10G instances behind a load balancer for embedding serving.

**Important:** E5 models require instruction prefixes:
- Query: `"query: {query_text}"`
- Document: `"passage: {document_text}"`

Failing to apply these prefixes degrades recall by 8-12% on the internal evaluation set.

### Decision 3 — Chunking Strategy: Document-Aware + Parent-Child

Given the extreme diversity of document types, a single chunking strategy will not work:

```python
CHUNKING_CONFIG = {
    "policy_document": {
        "strategy": "structure_aware",
        "parent_size": 1000,    # Full section
        "child_size": 200,      # Individual paragraphs for retrieval
        "preserve_tables": True
    },
    "contract": {
        "strategy": "structure_aware",
        "parent_size": 1500,    # Contract clauses can be long
        "child_size": 300,
        "preserve_tables": True,
        "extract_defined_terms": True  # Special handling
    },
    "scanned_pdf": {
        "strategy": "fixed_with_overlap",
        "chunk_size": 400,      # OCR text is less structured
        "overlap": 80
    },
    "wiki_page": {
        "strategy": "structure_aware",  # Markdown headings
        "parent_size": 800,
        "child_size": 200
    },
    "long_report": {
        "strategy": "hierarchical",
        "section_level": True,
        "parent_size": 2000,
        "child_size": 300
    }
}
```

### Decision 4 — Access Control: Metadata-Based Multi-Tenant Filtering

Every chunk is tagged with the access groups that are allowed to retrieve it:

```python
# Example metadata for an HR policy chunk
{
    "doc_id": "hr-compensation-guide-2024",
    "document_type": "policy",
    "department": "hr",
    "access_groups": ["hr_all", "exec_compensation", "legal_employment"],
    "document_status": "active",
    "region": "global",
    "language": "en"
}
```

At query time, the user's access groups (from their identity provider via OIDC/SAML) are injected into the retrieval filter:

```python
def build_access_filter(user_context: dict) -> dict:
    """
    Build a Qdrant filter that enforces document-level access control.
    """
    user_groups = user_context["access_groups"]  # From JWT claims
    user_region = user_context["region"]
    user_language = user_context["preferred_language"]
    
    return {
        "must": [
            {"key": "document_status", "match": {"value": "active"}},
            {
                "should": [
                    # Document accessible to any of user's groups
                    {"key": "access_groups", "match": {"any": user_groups}},
                    # Or document is global (no access restriction)
                    {"key": "access_groups", "is_empty": True}
                ]
            }
        ],
        "should": [
            # Prefer documents in user's language or global
            {"key": "language", "match": {"value": user_language}},
            {"key": "language", "match": {"value": "en"}}  # English as fallback
        ]
    }
```

**Important:** Access control via metadata filtering must be verified with security. The filter is applied at the vector database level — it is not enforced by the LLM. The LLM only ever sees chunks that have already passed the access filter.

### Decision 5 — Pre-Processing Pipeline

Given the document diversity:

```
Document arrives
    ↓
Format detection (Python-magic)
    ↓
┌─────────────────────────────────────────────┐
│ Digital PDF → PyMuPDF + pdfplumber          │
│ Scanned PDF → AWS Textract                   │
│ Word/DOCX → python-docx                      │
│ HTML/Wiki → Trafilatura + BeautifulSoup      │
│ PPTX → python-pptx                           │
└─────────────────────────────────────────────┘
    ↓
Language detection (langdetect)
    ↓
Table extraction (pdfplumber or Textract)
    → NL serialization per row
    → Store markdown table in metadata
    ↓
Figure extraction (PyMuPDF image extraction)
    → GPT-4o vision captioning for figures > 200px
    ↓
Chunking (document-type-specific strategy)
    ↓
Metadata enrichment
    → Department classification (rule-based on source path)
    → Access group assignment (from DMS metadata)
    → Entity extraction (spaCy NER)
    ↓
Embedding (multilingual-e5-large, GPU batch)
    ↓
BM25 sparse encoding (SPLADE for better recall)
    ↓
Upsert to Qdrant
```

### Decision 6 — Retrieval Pipeline

```python
async def retrieve(query: str, user_context: dict) -> list[dict]:
    
    # Detect query language
    query_lang = detect_language(query)
    
    # Apply E5 instruction prefix
    query_for_embedding = f"query: {query}"
    
    # Choose appropriate vector space
    dense_vector_name = f"dense_{query_lang}" if query_lang in ["en", "de", "fr"] else "dense_en"
    
    # Build access filter
    access_filter = build_access_filter(user_context)
    
    # Parallel: dense retrieval + sparse (BM25/SPLADE) retrieval
    dense_task = client.search(
        collection_name="enterprise_docs",
        query_vector=(dense_vector_name, query_embedding),
        query_filter=access_filter,
        limit=50
    )
    
    sparse_task = client.search(
        collection_name="enterprise_docs",
        query_sparse_vector=("sparse_bm25", query_sparse_vector),
        query_filter=access_filter,
        limit=50
    )
    
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
    
    # RRF fusion
    merged = reciprocal_rank_fusion([
        [r.id for r in dense_results],
        [r.id for r in sparse_results]
    ])
    
    # Re-rank top-50
    reranked = cross_encoder.rerank(query, merged[:50])
    
    # Parent chunk expansion
    final_chunks = await expand_to_parents(reranked[:10])
    
    return final_chunks
```

### Decision 7 — Freshness: Webhook + Scheduled Reconciliation

Documents come from multiple sources (SharePoint, Confluence, file server, DMS). Each source has webhook support:

- **SharePoint:** Graph API webhooks fire on document create/modify/delete.
- **Confluence:** Space webhooks fire on page changes.
- **File server:** inotify-based watcher service.
- **DMS:** Event-driven via message queue.

Each webhook event goes to an SQS queue. An indexing worker fleet (10 workers, auto-scaled) processes events and updates the Qdrant index. Target freshness lag: < 30 minutes for policy documents.

Nightly reconciliation scan validates that every source document has a corresponding up-to-date index entry.

### Decision 8 — Language Model and Prompt Design

```python
SYSTEM_PROMPT = """You are an internal knowledge assistant for {company_name}.
Answer questions using ONLY the provided context from official company documents.

RULES:
- Cite every factual claim with its source number [1], [2], etc.
- If the answer differs by region or department, state all variants explicitly.
- If you find conflicting information, state both and note the conflict.
- If the answer is not in the provided context, say: "This information is not 
  available in the documents I have access to."
- Never speculate or use knowledge from outside the provided documents.
- For legal or financial matters, always recommend consulting the relevant team.

CONTEXT DOCUMENTS:
{formatted_context}

USER DEPARTMENT: {user_department}
USER REGION: {user_region}"""
```

Using GPT-4o for generation (quality requirement outweighs cost for this high-stakes enterprise use case). GPT-4o-mini for query rewriting and language detection (latency-sensitive, lower complexity tasks).

---

## Scaling to 500 Concurrent Users

At peak (Monday morning, 500 concurrent users), the system must handle 500 queries in parallel with p95 latency < 3 seconds.

**Bottleneck analysis:**
- Embedding generation: 50ms per query, GPU batch processing handles 500 concurrently with 2 A10G GPUs.
- Qdrant search: 30ms per query, Qdrant scales to thousands of concurrent searches per node.
- LLM generation: 800ms-2s per query. At 500 concurrent, you need enough API rate limit capacity.
- **Primary bottleneck: LLM API rate limits.**

**Solution:** Deploy with OpenAI Batch API for non-real-time queries, and implement a request queue with priority levels (real-time interactive queries get priority, background document re-summaries go to the batch queue).

For real-time queries:
```python
# Connection pool with rate limit management
llm_client = AsyncOpenAI(
    max_retries=3,
    timeout=10.0,
    # Distribute across multiple API keys if multiple accounts are available
)

# Request queue with priority
high_priority_queue = asyncio.Queue(maxsize=1000)   # User-facing queries
low_priority_queue = asyncio.Queue(maxsize=10000)   # Background tasks
```

---

## Evaluation Results

After 3 months of development and tuning:

| Metric | Target | Achieved |
|---|---|---|
| Recall@10 (English) | ≥ 0.85 | 0.89 |
| Recall@10 (German) | ≥ 0.80 | 0.83 |
| Faithfulness | ≥ 0.90 | 0.92 |
| Answer Relevancy | ≥ 0.85 | 0.87 |
| p95 Latency | < 3s | 2.4s |
| Freshness Lag | < 2h | ~25min |
| IDK Rate (correct) | N/A | 8% |
| False IDK Rate | < 5% | 2.1% |
| User Satisfaction (thumbs up) | N/A | 84% |

---

## Lessons Learned

**Lesson 1:** Access control must be designed into the metadata schema from day one. Retrofitting access control onto an existing index required re-indexing all 120,000 documents.

**Lesson 2:** Scanned legacy documents were the biggest quality issue. 15% of the corpus was scanned PDFs from before 2010. OCR errors in these documents caused retrieval failures. Investment in AWS Textract (vs. Tesseract) paid off — Textract reduced OCR error rate from 8% to 2% on these documents.

**Lesson 3:** The E5 instruction prefix was not applied consistently during the first deployment. Recall dropped 10% before the bug was found. Building a unit test that verifies the prefix is applied before every deployment caught this in staging.

**Lesson 4:** German legal documents have very long compound words that BM25 tokenizes poorly (splitting compound words incorrectly). Switching to a German-aware tokenizer for BM25 improved German recall@10 from 0.76 to 0.83.

**Lesson 5:** Users from different departments had very different query styles. Legal asked precise, formal questions. Engineering asked conversational questions. A single query rewriting prompt did not work well across both. Department-aware rewriting prompts (selected based on user_department in context) improved cross-department accuracy by 6%.

---

## Interview Questions This Case Study Prepares You For

**"How would you handle access control in a multi-tenant RAG system?"**
Answer: Metadata-based access filtering at the vector database level. Every chunk tagged with access_groups. User's JWT claims translated to a filter applied before any vector is retrieved. The LLM never sees unauthorized content.

**"How would you handle a multi-language corpus?"**
Answer: Multilingual embedding model (multilingual-e5-large), separate named vector spaces per language within one collection, language detection for query routing, cross-lingual retrieval as fallback when no same-language document exists.

**"How would you ensure freshness for a 120K document corpus?"**
Answer: Webhook-based ingestion for each source system, SQS queue for event buffering, async worker fleet for processing, nightly reconciliation scan as safety net. Track freshness lag per source system.

**"What would you do if accuracy is 95% on the eval set but 78% in production?"**
Answer: The eval set is not representative of production queries. Sample recent production queries, annotate them, add to eval set. Check offline-online metric correlation monthly.