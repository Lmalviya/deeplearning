# Lesson 1.4 — Anatomy of an Indexing Pipeline

---

## Overview

The indexing pipeline is everything that happens before a single user query arrives. It is the offline process that transforms raw documents into a searchable index. If this pipeline is done poorly, no amount of clever retrieval logic will save you — you cannot retrieve what was never indexed correctly.

The indexing pipeline runs:
- Once when you first ingest a document corpus.
- Again (partially or fully) whenever documents are added, updated, or deleted.
- Potentially on a schedule if your data changes frequently.

The full pipeline looks like this:

```
Raw Documents (PDF, DOCX, HTML, images, databases...)
    ↓
Document Ingestion & Parsing
    ↓
Cleaning & Normalization
    ↓
Chunking
    ↓
Metadata Extraction & Enrichment
    ↓
Embedding Generation
    ↓
Storage (Vector DB + optional keyword index)
```

Each stage has decisions that directly impact retrieval quality. We will go through all of them.

---

## Stage 1 — Document Ingestion & Parsing

Raw documents come in many formats and need to be converted to clean text before anything else can happen.

### Format-Specific Challenges

**Digital PDFs** — PDFs that were created from Word or other tools contain actual text in their structure. Parsers like PyMuPDF, pdfplumber, or PDFMiner can extract this text reliably. The challenge is that PDFs encode text in visual layout order, not reading order. Multi-column PDFs, sidebars, headers, and footers all confuse naive parsers. You end up with text that jumps between columns mid-sentence.

**Scanned PDFs** — These are images of pages with no embedded text. You need OCR (Optical Character Recognition) to extract text. Tools: Tesseract (open source, decent quality), AWS Textract, Google Document AI, Azure Document Intelligence (all managed services, much higher quality especially for complex layouts). Quality varies heavily based on scan quality, font, and layout complexity.

**Word documents (.docx)** — python-docx handles these well. Structure (headings, tables, lists) is preserved in the XML and can be used to guide chunking.

**HTML / web pages** — Beautiful Soup or Trafilatura for extraction. The challenge is boilerplate removal — navigation menus, footers, ads, cookie banners all get scraped along with the actual content and pollute the index.

**PowerPoint (.pptx)** — Each slide is a collection of text boxes in no guaranteed order. Speaker notes are often more information-dense than slide text. Both need to be captured.

**Tables** — Tables in any format (PDF, Word, HTML) are notoriously hard to extract correctly. A table row split across a page boundary becomes garbled. A merged cell confuses most parsers. This is a major source of indexing quality problems for financial and technical documents.

**Images within documents** — Charts, diagrams, and figures in documents are invisible to text-based parsers. You either ignore them (losing potentially critical information) or extract them as images and apply a vision model to generate captions or descriptions. This is the multimodal RAG problem, covered in depth in Lesson 2.5.

### What Good Parsing Produces

After parsing, you want clean, structured text that:
- Preserves reading order.
- Separates headings, body text, tables, and captions into identifiable sections.
- Removes boilerplate (headers, footers, page numbers, navigation).
- Preserves document structure signals (heading levels, list hierarchy) that will guide chunking.

---

## Stage 2 — Cleaning & Normalization

Raw parsed text is messy. Before chunking and embedding, you need to clean it.

**Common cleaning steps:**

- **Whitespace normalization** — Collapse multiple spaces, remove trailing/leading whitespace, normalize line endings.
- **Encoding fixes** — Fix mojibake (garbled characters from encoding mismatches), normalize Unicode (é vs. é can be the same character encoded two ways — normalize to one form).
- **Boilerplate removal** — Remove repeated headers, footers, page numbers, watermarks, legal disclaimers that appear on every page and add noise without information.
- **Table of contents removal** — TOCs in documents, if indexed as chunks, create misleading retrieval results (they mention every topic but contain no actual content about them).
- **Deduplication** — If you are ingesting from multiple sources, the same content may appear more than once. Exact deduplication (hash the text) and near-deduplication (MinHash/LSH) prevent inflating the index with redundant chunks.
- **Language detection** — If your system is monolingual, detect and filter out content in other languages. If multilingual, route to the appropriate embedding model.

**What not to over-clean:** Some "noise" is actually signal. Formatting signals like "IMPORTANT:", "WARNING:", "NOTE:" add context about the importance of following text. Removing them loses information.

---

## Stage 3 — Chunking

Chunking is the process of splitting documents into smaller pieces that can be individually embedded and retrieved. This is one of the most consequential decisions in the entire RAG system. Bad chunking cannot be fixed downstream.

The fundamental tension in chunking:

- **Chunks too small** — Each chunk lacks enough context for the embedding to be meaningful. Retrieved chunks may be incomplete fragments that confuse the LLM.
- **Chunks too large** — The chunk contains multiple topics. The embedding averages over all of them, becoming a weak signal for any specific topic. The retrieved chunk floods the context with irrelevant content.

There is no universal right answer. The right chunk size depends on your documents, your queries, and your embedding model.

### Fixed-Size Chunking

Split every N tokens, with an overlap of M tokens between adjacent chunks.

```
Document: [-----------------------------...]
Chunks:   [----chunk1----]
                    [----chunk2----]
                              [----chunk3----]
```

The overlap (typically 10-20% of chunk size) ensures that content near chunk boundaries appears in at least one complete chunk.

**Typical parameters:** 256–1024 tokens per chunk, 50–200 token overlap.

**Problems:**
- Splits mid-sentence, mid-paragraph, mid-table, mid-list item. The chunk boundary is arbitrary and has no relationship to the semantic structure of the document.
- A paragraph discussing one concept gets split into two chunks, each of which is semantically weaker than the whole.

Fixed-size chunking is simple to implement and works well enough for homogeneous text documents. It is a poor choice for structured documents.

### Recursive Character Splitting

An improvement over fixed-size. Split on a hierarchy of separators: first try to split on double newlines (paragraph boundaries), then single newlines, then sentences, then words, then characters. Stop when chunks are within the target size.

This respects natural text boundaries when possible. LangChain's `RecursiveCharacterTextSplitter` implements this. It is a reasonable default for most text documents.

Still document-structure-agnostic — it does not know what a heading, table, or list is.

### Document-Aware / Structure-Based Chunking

Instead of splitting by size, split by document structure. Use the parsed structure of the document to create chunks that correspond to meaningful semantic units.

For a document with headings:
- Each section (heading + body) becomes one chunk.
- Sub-sections become separate chunks with the parent heading prepended for context.
- Tables become their own chunks.
- Lists become chunks with the context sentence before the list prepended.

This produces chunks that correspond to what a human would consider a coherent unit of information. Retrieved chunks make sense on their own.

**Implementation:** Requires a parser that preserves structural signals (heading levels, tables, lists). Markdown documents are easy — `#`, `##`, `###` are unambiguous. PDFs and Word documents require more careful parsing.

**The context prepending trick:** When chunking by section, prepend the document title and section path to each chunk:

```
Document: "Employee Handbook 2024"
Section: "Benefits > Parental Leave > Eligibility"

[Chunk text]: "Employees are eligible for parental leave after 6 months of continuous employment..."
```

Stored as: "Employee Handbook 2024 > Benefits > Parental Leave > Eligibility\n\nEmployees are eligible for parental leave after 6 months..."

This ensures that the chunk's embedding captures its topical position in the document, not just its local text.

### Semantic Chunking

Instead of splitting on fixed boundaries or document structure, split where the *meaning* changes. Encode each sentence, then measure the cosine similarity between adjacent sentences. When similarity drops below a threshold, start a new chunk.

This produces chunks that are semantically coherent by construction — each chunk covers one topic or argument.

**Process:**
1. Split document into sentences.
2. Embed each sentence (or small window of sentences).
3. Compute similarity between consecutive sentence embeddings.
4. Split where similarity drops significantly (a "semantic break").
5. Merge small resulting segments until chunks are a reasonable size.

**Pros:** Chunks respect semantic boundaries, not arbitrary character counts.
**Cons:** Expensive — requires embedding every sentence at index time. Sensitive to the threshold parameter. Can produce very variable chunk sizes.

### Parent-Child Chunking (Small-to-Big)

This is one of the most practically valuable chunking strategies in production systems.

The idea: index small chunks for precise retrieval, but return larger chunks for context-rich generation.

- **Child chunks:** Small (128–256 tokens). Used for embedding and retrieval. Small chunks have focused embeddings that match specific queries precisely.
- **Parent chunks:** Large (512–2048 tokens). Stored separately. When a child chunk is retrieved, you look up its parent and return the parent to the LLM.

```
Parent chunk (stored, not indexed for retrieval):
[paragraph 1][paragraph 2][paragraph 3][paragraph 4]

Child chunks (indexed for retrieval):
[paragraph 1] [paragraph 2] [paragraph 3] [paragraph 4]
     ↑               ↑               ↑               ↑
     all point to the same parent chunk
```

**Why this works:** Short queries and specific questions match small, focused chunks well. But a small chunk (128 tokens) may lack the surrounding context the LLM needs to give a complete answer. The parent chunk provides that context.

This is implemented in LlamaIndex as the `ParentDocumentRetriever` pattern and in LangChain similarly.

### Agentic / Late Chunking

An emerging approach: do not chunk at all at index time. Store entire documents. At query time, use an agent to identify the relevant passages within the full document on the fly.

This avoids the chunking problem entirely but is much more expensive at query time — you need to process entire documents for every query. Only practical for small corpora or when query latency requirements are relaxed.

### Which Chunking Strategy to Use

| Document Type | Recommended Strategy |
|---|---|
| Clean markdown / structured text | Document-aware (section-based) |
| Homogeneous prose (articles, books) | Recursive character + overlap |
| Mixed structure (reports, manuals) | Document-aware + parent-child |
| Tables and structured data | Table-aware chunking (each row or logical group) |
| Short documents (< 1 page) | No chunking — index as single chunk |
| Very long documents (> 100 pages) | Hierarchical: section → paragraph → parent-child |

---

## Stage 4 — Metadata Extraction & Enrichment

Every chunk should be stored with metadata — structured fields that describe the chunk and can be used for filtering at retrieval time.

**Source metadata (extracted from the document):**
- Document title, author, creation date, last modified date
- Document type (policy, contract, manual, report)
- Section / chapter / page number
- URL or file path

**Derived metadata (computed or inferred):**
- Language
- Entity tags (people, organizations, locations, products mentioned) — extracted with NER
- Topic or category — classified with a model
- Importance score — some sections are more authoritative than others

**Why metadata matters:** Without metadata, retrieval is purely based on semantic similarity. With metadata, you can add hard filters: "only retrieve from documents modified in the last 6 months" or "only retrieve from policy documents, not draft proposals."

**Metadata filtering dramatically improves precision** in enterprise settings because users often have implicit filters in mind even when they do not state them — "what does our policy say" implies internal documents, not web search results.

**The metadata design trap:** Do not add too many metadata fields that you never filter on. Every field adds storage cost and index maintenance complexity. Add metadata fields that you will actually use in retrieval filters or that you will display in the UI (for source citations).

---

## Stage 5 — Embedding Generation

Each chunk is now passed through an embedding model to produce a dense vector representation.

### Choosing an Embedding Model

This is one of the most impactful decisions in your RAG system. The embedding model determines how well semantic similarity maps to actual relevance.

**Key dimensions:**
- **Vector dimension** — Higher dimension = more expressive, more storage. Common: 768, 1024, 1536, 3072.
- **Max token input** — Most models have a max of 512 or 8192 tokens. If your chunks exceed this, the model truncates and you lose information.
- **Domain** — General models (trained on web data) may perform poorly on specialized domains (medical, legal, code). Domain-specific or fine-tuned models are worth evaluating.
- **Multilingual** — If your corpus has multiple languages, you need a multilingual model.
- **Speed** — Embedding generation can be a bottleneck for large-scale indexing and for query-time embedding.

**Popular models (as of 2024–2025):**
- `text-embedding-3-large` (OpenAI) — Strong general performance, 3072 dimensions, managed API.
- `text-embedding-3-small` (OpenAI) — Faster, cheaper, 1536 dimensions, good quality.
- `BAAI/bge-large-en-v1.5` — Strong open-source general English model.
- `intfloat/e5-large-v2` — Strong open-source, good for retrieval tasks.
- `sentence-transformers/all-mpnet-base-v2` — Lightweight, fast, widely used.
- `Cohere embed-english-v3.0` — Managed, strong performance, supports compression.

**Evaluate before you commit.** Run your actual query-chunk pairs through candidate models and measure retrieval metrics (recall@K, NDCG). The benchmark rankings (MTEB leaderboard) do not always translate to your specific domain and query distribution.

### Embedding the Query vs. Embedding Chunks

The same model must embed both queries and chunks. If you switch embedding models after indexing, you must re-embed the entire corpus — the vectors are not compatible across models.

Some models are **asymmetric** — they use different instructions for query vs. document embedding:
- Query: "Represent this question for searching relevant passages: {query}"
- Document: "Represent this passage for retrieval: {chunk}"

E5 and BGE models use this pattern. If you use them, make sure you apply the right instruction prefix at both index time and query time. Forgetting this is a common bug that degrades retrieval quality silently.

### Batch Embedding at Index Time

At index time, embed chunks in batches, not one at a time. Most embedding APIs and models support batches of 64–512 chunks per call. Batching reduces overhead and dramatically speeds up large-scale indexing.

For a 100,000-chunk corpus, embedding one at a time at 50ms per call = 5,000 seconds. Batching 256 at a time at 200ms per batch = ~80 seconds.

---

## Stage 6 — Storage

Embeddings and their associated metadata go into storage. For most RAG systems, this means a vector database plus optionally a separate keyword index.

### Vector Database Responsibilities

A vector database stores vectors and supports:
- **ANN search** — Find the K vectors most similar to a query vector.
- **Metadata filtering** — Filter results by metadata fields before or after vector search.
- **CRUD operations** — Add, update, delete documents/chunks.
- **Persistence** — Store vectors durably to disk.
- **Scalability** — Handle millions to billions of vectors.

### Index Structures

The vector database builds an index structure over your vectors to enable fast ANN search. The two most common:

**HNSW (Hierarchical Navigable Small World):** A graph-based index. Builds a multi-layer graph where each node is connected to its approximate nearest neighbors. At query time, traverses the graph starting from an entry point, greedily moving toward the query vector. Very fast search, high recall, but high memory usage (the graph is held in RAM). Used by Qdrant, Weaviate, and many others.

**IVF (Inverted File Index):** Clusters vectors into groups (Voronoi cells). At query time, identifies the nearest cluster centers and searches only within those clusters. More memory-efficient than HNSW but requires a training step to build the clusters. Used by FAISS.

We cover these in depth in Lesson 3.1.

### Keyword Index for BM25

If you are doing hybrid retrieval (and you should be), you also need a BM25 index. Options:
- **Elasticsearch / OpenSearch** — Full-featured, battle-tested, supports BM25 natively and increasingly vector search too.
- **Qdrant's sparse vector support** — Store BM25 (or SPLADE) sparse vectors alongside dense vectors in the same system.
- **Tantivy / Meilisearch** — Lightweight alternatives for smaller scale.
- **In-memory BM25** — For small corpora, libraries like `rank_bm25` in Python work fine.

### Popular Vector Databases

| Database | Hosting | Strengths | Weaknesses |
|---|---|---|---|
| **Qdrant** | Self-hosted / Cloud | Fast, Rust-based, good filtering, sparse+dense support | Smaller ecosystem |
| **Pinecone** | Managed only | Simple API, scales easily | Expensive, vendor lock-in |
| **Weaviate** | Self-hosted / Cloud | Rich schema, hybrid search built-in, modules | Complex configuration |
| **Milvus** | Self-hosted / Cloud | Massive scale (billions), mature | Heavy infrastructure |
| **pgvector** | Self-hosted (PostgreSQL) | No new infra if already using Postgres | Slower ANN, limited filtering |
| **ChromaDB** | Self-hosted | Simple, good for prototyping | Not production-grade at scale |
| **FAISS** | Library (not a DB) | Fastest pure ANN, highly flexible | No persistence, no server |

We cover self-hosted vs. managed trade-offs, and when to choose which, in Lesson 8.1 and 8.2.

---

## The Full Indexing Pipeline with Failure Modes

| Stage | Purpose | Most Common Failure |
|---|---|---|
| Parsing | Convert raw docs to text | Lost content (tables, images, multi-column PDFs) |
| Cleaning | Remove noise | Over-cleaning removes meaningful signals |
| Chunking | Split into retrievable units | Wrong chunk size, chunks split mid-concept |
| Metadata | Enable filtering and citation | Missing metadata fields needed for retrieval filters |
| Embedding | Generate semantic vectors | Wrong model, wrong instruction prefix, truncated chunks |
| Storage | Persist and index | Index misconfigured, metadata not indexed for filtering |

---

## Incremental Indexing

When documents change, you do not want to re-index the entire corpus. You need an incremental update strategy.

**Add:** Embed and insert new chunks. Straightforward.

**Update:** Find existing chunks for the updated document (by document ID), delete them, re-parse, re-chunk, re-embed, re-insert. Requires storing a mapping from document ID to chunk IDs.

**Delete:** Find and delete all chunks belonging to the document.

**Version tracking:** Store a hash of the document content. At update time, compare the new hash against the stored one. Only re-index if the content actually changed.

This is the data freshness problem in practice. We cover it in depth in Lesson 2.6.

---

## Summary

- Parsing is format-specific. PDFs, scanned documents, tables, and images each have their own challenges. Bad parsing is the root cause of many retrieval failures that look like retrieval algorithm problems.
- Chunking strategy is one of the most impactful decisions in RAG. Match the strategy to the document type. Document-aware and parent-child chunking outperform fixed-size in most production scenarios.
- Every chunk needs metadata. Source metadata enables citation; derived metadata enables filtering. Design your metadata schema before you index.
- Embedding model choice matters. Evaluate on your specific domain and query distribution, not just benchmark rankings. Same model must be used at both index time and query time.
- A vector database is not optional — it provides ANN search, filtering, and persistence. Choose based on your scale, hosting preference, and whether you need hybrid search built in.
- Incremental indexing requires tracking document IDs, content hashes, and chunk-to-document mappings from the start.

---
