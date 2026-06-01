# Lesson 2.3 — Metadata Design and Filtering Strategies

---

## Why Metadata Is Not Optional

Most RAG tutorials focus almost entirely on embeddings and vector search. Metadata is treated as an afterthought — "oh, also store the document title." This is a mistake that costs you in production.

Here is the core insight: **vector search finds semantically similar chunks, but it cannot enforce hard constraints.** Metadata filtering enforces hard constraints.

Consider this query: "What is the current refund policy?"

Vector search will find all chunks about refund policies — from 2019, 2021, 2023, and 2024. They are all semantically similar. Without metadata filtering, your system may retrieve and answer from an outdated 2019 policy while the 2024 policy is sitting right there in the index.

Metadata filtering lets you say: "only retrieve from documents with `effective_date >= 2024-01-01`" — a hard constraint that vector search cannot express.

This is why metadata is a first-class citizen in a production RAG system, not an afterthought.

---

## The Two Types of Metadata

**Source metadata** is extracted directly from the document and its surrounding context. It describes where the chunk came from.

**Derived metadata** is computed, inferred, or enriched after parsing. It describes what the chunk is about.

Both are important. They serve different purposes at retrieval time.

---

## Source Metadata: What to Always Capture

These fields should be captured for every chunk, regardless of document type. Missing any of them will cost you later.

### Document Identity

```python
{
  "doc_id": "hr-policy-2024-v3",        # Unique identifier for the document
  "doc_title": "Employee Handbook 2024", # Human-readable title
  "source_path": "s3://docs/hr/handbook_2024.pdf",  # Where the file lives
  "source_url": "https://intranet.company.com/hr/handbook",  # If web-sourced
  "file_type": "pdf",                   # pdf, docx, html, md, txt
}
```

The `doc_id` is the most important field. It is how you:
- Track which chunks belong to which document (needed for updates and deletes).
- Deduplicate retrieved chunks at query time (if two chunks share a `doc_id`, you may want only the most relevant one).
- Filter by document scope ("only search within this document").

### Temporal Metadata

```python
{
  "created_date": "2024-01-15",         # When the document was created
  "modified_date": "2024-06-01",        # When it was last modified
  "effective_date": "2024-01-01",       # When the policy/content takes effect
  "expiry_date": "2025-12-31",          # When it expires (for time-bound content)
  "indexed_at": "2024-06-15T10:32:00Z", # When this chunk was indexed
}
```

Temporal metadata is critical for:
- **Freshness filtering:** "only retrieve documents modified in the last 6 months."
- **Point-in-time queries:** "what was the policy as of March 2023?" — filter by `effective_date <= 2023-03-01` and `expiry_date >= 2023-03-01`.
- **Debugging:** knowing when a chunk was indexed helps diagnose stale data issues.

Store dates in ISO 8601 format and as actual date/datetime types in your vector database (not strings). String comparison of dates is fragile.

### Structural Metadata

```python
{
  "chunk_id": "hr-policy-2024-v3-chunk-042",  # Unique ID for this specific chunk
  "parent_id": "hr-policy-2024-v3-section-04", # For parent-child retrieval
  "section": "Benefits",                       # Top-level section
  "subsection": "Parental Leave",              # Subsection
  "page_number": 12,                           # Page in original document
  "chunk_index": 42,                           # Position of chunk in document
  "total_chunks": 187,                         # Total chunks in document
  "heading_path": "Benefits > Parental Leave > Eligibility",  # Full hierarchy
}
```

`chunk_index` and `total_chunks` enable position-based reasoning. If you retrieve chunk 42 of 187, you know it is from the middle of the document. If you retrieve chunk 1, it is likely an introduction. This can inform context assembly decisions.

`heading_path` is the document breadcrumb. It is shown to the LLM alongside the chunk text for citation and context.

### Authorship and Provenance

```python
{
  "author": "HR Department",
  "department": "Human Resources",
  "approved_by": "Legal Team",
  "version": "3.2",
  "document_status": "active",   # active, draft, archived, superseded
  "superseded_by": None,         # doc_id of the document that replaced this one
}
```

`document_status` is underused but powerful. Mark outdated documents as `archived` or `superseded` and filter them out by default. Users almost never want to retrieve from deprecated content — but naive vector search will happily return it.

`superseded_by` lets you build a chain of document versions. If a user retrieves an old chunk, you can programmatically check if a newer version exists.

---

## Derived Metadata: Enriching Chunks After Parsing

Derived metadata is computed at index time. It adds semantically rich fields that enable more precise filtering and retrieval.

### Topic and Category Classification

Classify each chunk (or document) into your taxonomy of topics.

```python
# Example: classify a chunk using an LLM
prompt = f"""
Classify the following text into one or more of these categories:
[Compensation, Benefits, Leave Policy, Code of Conduct, IT Security, 
 Compliance, Performance Management, Recruiting, Offboarding]

Return only a JSON list of matching categories.

Text: {chunk_text}
"""

response = llm.generate(prompt)
categories = json.loads(response)  # e.g., ["Benefits", "Leave Policy"]
```

Store this as a list field: `"categories": ["Benefits", "Leave Policy"]`

At retrieval time, if a user is browsing the "Leave Policy" section of your HR chatbot, you can pre-filter to only retrieve chunks with `"Leave Policy"` in `categories` before running vector search. This dramatically reduces noise.

### Named Entity Extraction

Extract and store entities mentioned in each chunk.

```python
{
  "entities": {
    "people": ["Jane Smith", "John Doe"],
    "organizations": ["Acme Corp", "Legal Department"],
    "locations": ["New York", "California"],
    "products": ["Enterprise Plan", "Pro Plan"],
    "dates": ["January 1, 2024", "Q1 2024"],
    "monetary_values": ["$50,000", "€1,200"]
  }
}
```

Use a NER (Named Entity Recognition) model: `spaCy` with the `en_core_web_trf` model for English, or a dedicated NER model for your domain (medical NER, legal NER).

This enables queries like: "find all documents mentioning Acme Corp and amounts over $10,000" — a structured filter that vector search alone cannot handle.

### Document Type and Intent

```python
{
  "document_type": "policy",      # policy, contract, invoice, manual, report, faq
  "content_type": "definition",   # definition, procedure, example, warning, summary
  "is_table": False,              # True if this chunk is from a table
  "is_figure_caption": False,     # True if this chunk describes a chart/figure
  "requires_human_review": False  # True if OCR confidence was low
}
```

`content_type` is particularly useful. A chunk classified as `"definition"` should be retrieved for "what is X?" queries. A chunk classified as `"procedure"` should be retrieved for "how do I do X?" queries. You can use this as a soft routing signal.

### Importance and Quality Scores

```python
{
  "importance_score": 0.85,   # 0-1, how important is this chunk in its document
  "ocr_confidence": 0.92,     # For scanned documents, OCR quality score
  "chunk_quality_score": 0.78 # Composite quality (length, completeness, etc.)
}
```

`importance_score` can be computed by looking at structural signals: introduction and conclusion paragraphs tend to be more important than mid-document filler. Headings with "Summary", "Key Points", "Conclusion" indicate high-importance content.

Use `chunk_quality_score` to filter out very low-quality chunks at retrieval time. A chunk that is a partial sentence, a page number, or a boilerplate footer has a low quality score and should not be retrieved.

---

## How Metadata Filtering Works in Vector Databases

Vector databases support metadata filtering in two ways: **pre-filtering** and **post-filtering**.

### Pre-filtering (Filtered ANN Search)

Apply metadata filters before the ANN (Approximate Nearest Neighbor) search. The vector search only considers vectors that pass the filter.

```python
# Qdrant example
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="document_status", match=MatchValue(value="active")),
            FieldCondition(key="department", match=MatchValue(value="Human Resources")),
            FieldCondition(
                key="effective_date",
                range=Range(gte="2024-01-01")
            )
        ]
    ),
    limit=10
)
```

**Pro:** Only returns chunks that satisfy all filters. Very precise.

**Con:** If the filtered subset is very small, ANN search quality degrades — the graph-based index (HNSW) is less accurate when searching over small subsets because the graph was built for the full collection. Some databases switch to brute-force search automatically when the filtered subset is small, which is slower.

### Post-filtering

Run ANN search first (retrieve top-K from the full index), then apply filters to the results.

**Pro:** ANN quality is unaffected — the search runs over the full index.

**Con:** If many top-K results fail the filter, your effective result count drops. Retrieving top-100 and filtering to 3 wastes compute and may miss relevant results that were ranked 101st.

### Hybrid approach

Most production vector databases (Qdrant, Weaviate, Pinecone) implement a hybrid: they maintain a separate index for metadata fields and use it to pre-filter before ANN search while compensating for small subset quality degradation. Qdrant's `payload index` and Weaviate's `inverted index` work this way.

**Best practice:** Index every metadata field you plan to filter on. An unindexed metadata filter forces a full scan of all chunks — extremely slow at scale.

```python
# Qdrant: create a payload index for frequently filtered fields
client.create_payload_index(
    collection_name="documents",
    field_name="document_status",
    field_schema="keyword"  # keyword, integer, float, datetime
)

client.create_payload_index(
    collection_name="documents",
    field_name="effective_date",
    field_schema="datetime"
)
```

---

## Metadata Schema Design: A Practical Example

Here is a complete metadata schema for an enterprise HR document RAG system:

```python
{
  # --- Source metadata ---
  "doc_id": "hr-leave-policy-2024",
  "doc_title": "Leave and Time-Off Policy 2024",
  "file_type": "pdf",
  "source_path": "s3://hr-docs/policies/leave_policy_2024.pdf",
  
  # --- Temporal ---
  "created_date": "2024-01-01",
  "modified_date": "2024-05-15",
  "effective_date": "2024-01-01",
  "document_status": "active",
  
  # --- Structural ---
  "chunk_id": "hr-leave-policy-2024-chunk-007",
  "parent_id": "hr-leave-policy-2024-section-02",
  "heading_path": "Leave Policy > Parental Leave > Eligibility",
  "page_number": 4,
  "chunk_index": 7,
  
  # --- Authorship ---
  "department": "Human Resources",
  "approved_by": "Legal",
  "version": "2.1",
  
  # --- Derived ---
  "categories": ["Leave Policy", "Benefits"],
  "document_type": "policy",
  "content_type": "procedure",
  "entities": {
    "locations": ["California", "New York"],
    "durations": ["16 weeks", "6 months"]
  },
  "is_table": False,
  "importance_score": 0.82,
  "chunk_quality_score": 0.91
}
```

---

## Metadata in the Query Pipeline

Metadata filtering is most powerful when it is applied intelligently based on the query context, not just as a static global filter.

### User Context Filters

If your system knows something about the user, use it:

```python
# If the user is in the US Engineering department
base_filter = {
    "department": ["Engineering", "All Employees"],
    "document_status": "active",
    "country": ["US", "Global"]
}
```

This ensures users only retrieve documents relevant to their department and location — critical for multi-national companies with different policies per region.

### Query-Derived Filters

An LLM can extract filter intent from the user's query:

```python
extraction_prompt = """
Given this user query, extract any explicit or implicit filters:
Query: "What was the remote work policy before 2023?"

Return JSON with these fields (null if not mentioned):
- time_filter: {"before": "2023-01-01"} 
- document_type: null
- department: null
- topic: "remote work policy"
"""
```

"Before 2023" becomes a date range filter. "Current policy" becomes `document_status = active`. "Engineering team policies" becomes a department filter.

This dynamic filter extraction converts natural language intent into structured metadata constraints.

### Recency Boosting

Instead of hard filtering by date, you can boost more recent documents in the ranking:

```python
# Qdrant supports score boosting based on payload values
# This is more flexible than hard cutoffs
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    score_threshold=0.6,
    limit=20
)

# Post-process: boost score based on recency
def boost_by_recency(results, decay_days=180):
    today = datetime.now()
    for r in results:
        doc_date = datetime.fromisoformat(r.payload['modified_date'])
        days_old = (today - doc_date).days
        recency_boost = max(0, 1 - (days_old / decay_days))
        r.score = r.score * (1 + 0.2 * recency_boost)  # up to 20% boost for recent docs
    return sorted(results, key=lambda x: x.score, reverse=True)
```

This is better than hard cutoffs when you want to prefer recent content without completely excluding older content that may still be relevant.

---

## Common Metadata Design Mistakes

**Mistake 1: Not indexing metadata fields for filtering.**
Storing metadata as JSON blobs without creating database indexes means every filter requires a full scan. Index every field you will filter on.

**Mistake 2: Using strings for dates.**
Storing `"2024-01-15"` as a string means range queries (`>= 2024-01-01`) require lexicographic comparison, which works for ISO 8601 but breaks for any other format. Use native datetime types.

**Mistake 3: Too many low-cardinality boolean flags.**
`"is_important": True/False`, `"is_recent": True/False`, `"is_approved": True/False` — these proliferate. Use a single `document_status` field with an enum instead.

**Mistake 4: Not capturing `doc_id` → `chunk_id` mapping.**
When you update a document, you need to find and delete all its chunks. Without a reliable `doc_id` field on every chunk, this becomes a table scan.

**Mistake 5: Deriving metadata at query time instead of index time.**
Computing category classification or entity extraction at query time adds 200–500ms per query. Do it at index time and store the result.

**Mistake 6: Designing metadata for the documents you have now, not the queries you will get.**
Before finalizing your schema, write out 20 realistic user queries and ask "what metadata would I need to filter or boost results for this query?" Design the schema to answer those queries.

---

## Summary

- Metadata enables hard constraints that vector search cannot express: date ranges, department filters, document status, content type.
- Source metadata (doc_id, dates, structure) should be captured for every chunk without exception. Missing it creates problems you cannot fix without re-indexing.
- Derived metadata (categories, entities, quality scores) adds semantic richness that improves retrieval precision. Compute it at index time, not query time.
- Index every metadata field you plan to filter on in your vector database. Unindexed filters = full table scans.
- Dynamic metadata filtering — extracting filter intent from the user query — is one of the highest-leverage improvements you can make to retrieval precision.
- Design your metadata schema by starting with realistic user queries and working backward to what constraints you need to express.

---

## What's Next

Lesson 2.4 covers document pre-processing pipelines in depth — OCR, layout parsing, table extraction, and handling the messy reality of real-world documents before they ever reach chunking and embedding.