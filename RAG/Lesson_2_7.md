# Lesson 2.7 — Parent-Child Chunking and Hierarchical Indexing

---

## The Core Problem This Solves

Every time you set a chunk size, you are making a bet. Small chunks give precise retrieval — the embedding represents one tight idea, and short specific queries match it well. But small chunks often lack the surrounding context the LLM needs to give a complete answer. Large chunks give richer context but dilute the embedding across multiple ideas, reducing retrieval precision.

This is not a tuning problem you can solve by finding the "right" chunk size. It is a structural tension — precision and context richness pull in opposite directions. No single chunk size resolves it.

Parent-child chunking resolves the tension by decoupling retrieval from generation:

- **Retrieve** using small, precise child chunks.
- **Generate** using large, context-rich parent chunks.

This lesson covers the full implementation — not just the two-level case but multi-level hierarchies, variant strategies, and how to design retrieval logic that intelligently uses the hierarchy.

---

## The Two-Level Architecture

The basic parent-child setup has two layers:

**Child chunks** (small, 128–256 tokens):
- Indexed in the vector database.
- Each child has a focused embedding representing one specific idea.
- Used exclusively for retrieval — never passed directly to the LLM.

**Parent chunks** (large, 512–2048 tokens):
- Stored in a document store (key-value store, database, or even the vector database with a flag).
- Not indexed for retrieval — they are never searched directly.
- Retrieved by looking up the parent ID after child retrieval.
- Passed to the LLM as context.

```
Document text
│
├── Parent chunk 1 (stored, not indexed)
│   ├── Child chunk 1a (indexed for retrieval) ──── points to Parent 1
│   ├── Child chunk 1b (indexed for retrieval) ──── points to Parent 1
│   └── Child chunk 1c (indexed for retrieval) ──── points to Parent 1
│
├── Parent chunk 2 (stored, not indexed)
│   ├── Child chunk 2a (indexed for retrieval) ──── points to Parent 2
│   └── Child chunk 2b (indexed for retrieval) ──── points to Parent 2
│
└── Parent chunk 3 (stored, not indexed)
    ├── Child chunk 3a (indexed for retrieval) ──── points to Parent 3
    └── Child chunk 3b (indexed for retrieval) ──── points to Parent 3
```

At query time:
1. Query embedding matches Child chunk 2a (high similarity).
2. Look up `parent_id` from Child chunk 2a's metadata → Parent chunk 2.
3. Fetch Parent chunk 2 from document store.
4. Pass Parent chunk 2 to the LLM (not Child chunk 2a).

The LLM sees the full context of the paragraph or section, not just the specific sentence that matched the query.

---

## Implementation

### Indexing

```python
from dataclasses import dataclass
from typing import Optional
import uuid

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict
    parent_id: Optional[str] = None

def create_parent_child_chunks(
    document_text: str,
    doc_metadata: dict,
    parent_size: int = 1000,    # tokens
    child_size: int = 200,      # tokens
    child_overlap: int = 20     # tokens
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Returns (parent_chunks, child_chunks).
    Parent chunks are stored but not indexed.
    Child chunks are indexed for retrieval.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Step 1: Create parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=0  # Parents should not overlap — no duplicate context
    )
    parent_texts = parent_splitter.split_text(document_text)
    
    parents = []
    children = []
    
    for p_idx, parent_text in enumerate(parent_texts):
        parent_id = f"{doc_metadata['doc_id']}-parent-{p_idx:04d}"
        
        parent = Chunk(
            chunk_id=parent_id,
            text=parent_text,
            metadata={
                **doc_metadata,
                "chunk_type": "parent",
                "parent_index": p_idx
            }
        )
        parents.append(parent)
        
        # Step 2: Create child chunks within this parent
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap
        )
        child_texts = child_splitter.split_text(parent_text)
        
        for c_idx, child_text in enumerate(child_texts):
            child_id = f"{parent_id}-child-{c_idx:04d}"
            
            child = Chunk(
                chunk_id=child_id,
                text=child_text,
                parent_id=parent_id,
                metadata={
                    **doc_metadata,
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "child_index": c_idx,
                    "parent_index": p_idx
                }
            )
            children.append(child)
    
    return parents, children


async def index_with_parent_child(
    document_text: str,
    doc_metadata: dict,
    vector_db,
    document_store  # key-value store for parent chunks
):
    parents, children = create_parent_child_chunks(document_text, doc_metadata)
    
    # Store parents in document store (not vector DB)
    for parent in parents:
        await document_store.set(parent.chunk_id, {
            "text": parent.text,
            "metadata": parent.metadata
        })
    
    # Embed and index children in vector DB
    child_texts = [c.text for c in children]
    embeddings = await embed_batch(child_texts)
    
    points = [
        {
            "id": child.chunk_id,
            "vector": embedding,
            "payload": child.metadata  # includes parent_id
        }
        for child, embedding in zip(children, embeddings)
    ]
    
    await vector_db.upsert(collection="documents", points=points)
```

### Retrieval

```python
async def retrieve_with_parent_expansion(
    query: str,
    vector_db,
    document_store,
    k_children: int = 20,   # retrieve more children than you need parents
    k_parents: int = 5      # return this many unique parents
) -> list[dict]:
    """
    Retrieve child chunks, then return their parent chunks.
    Deduplicates parents — multiple children from the same parent
    count as one parent result.
    """
    # Step 1: Embed query and retrieve children
    query_embedding = await embed_query(query)
    
    child_results = await vector_db.search(
        collection="documents",
        query_vector=query_embedding,
        filter={"chunk_type": {"$eq": "child"}},
        limit=k_children
    )
    
    # Step 2: Collect unique parent IDs (preserve relevance ordering)
    seen_parent_ids = set()
    ordered_parent_ids = []
    
    for result in child_results:
        parent_id = result.payload["parent_id"]
        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            ordered_parent_ids.append(parent_id)
        
        if len(ordered_parent_ids) >= k_parents:
            break
    
    # Step 3: Fetch parent chunks from document store
    parents = []
    for parent_id in ordered_parent_ids:
        parent_data = await document_store.get(parent_id)
        if parent_data:
            parents.append(parent_data)
    
    return parents
```

---

## Three-Level Hierarchical Indexing

For long documents — books, large manuals, comprehensive reports — two levels are sometimes not enough. A three-level hierarchy gives more control:

```
Document
│
├── Section (H1 level, 2000–5000 tokens) — stored
│   │
│   ├── Passage (H2/paragraph level, 500–1000 tokens) — stored + optionally indexed
│   │   │
│   │   ├── Sentence group (128–256 tokens) — indexed for retrieval
│   │   └── Sentence group — indexed for retrieval
│   │
│   └── Passage — stored
│       ├── Sentence group — indexed for retrieval
│       └── Sentence group — indexed for retrieval
│
└── Section — stored
    └── ...
```

At retrieval time, you have options for which level to return:

- **Return passage level (middle):** Good default. Specific enough to be focused, large enough for context.
- **Return section level (top):** For queries requiring broad context or synthesis across a section.
- **Return sentence group (bottom):** For precise fact retrieval where compactness matters.

Implementing adaptive level selection:

```python
def select_retrieval_level(query: str, retrieved_child: dict) -> str:
    """
    Decide which hierarchy level to return to the LLM based on query type.
    """
    query_lower = query.lower()
    
    # Signals for broad context need
    broad_signals = [
        "explain", "describe", "summarize", "overview", 
        "how does", "what is the process", "compare"
    ]
    
    # Signals for precise fact need
    precise_signals = [
        "what is the exact", "how many", "when did", "who is",
        "what date", "what number", "list all"
    ]
    
    if any(signal in query_lower for signal in broad_signals):
        return "section"   # Return top-level section
    elif any(signal in query_lower for signal in precise_signals):
        return "passage"   # Return middle level (precise but with context)
    else:
        return "passage"   # Default to passage level
```

In practice, using a small LLM to classify query intent (rather than keyword heuristics) gives better results for level selection.

---

## Variant: Sentence Window Retrieval

A lighter alternative to full parent-child that is easier to implement:

Instead of defining explicit parent chunks, retrieve a child chunk and then grab its neighboring chunks from the document to build a window of context.

```python
async def retrieve_with_sentence_window(
    query: str,
    vector_db,
    chunk_store,      # stores all chunks by chunk_id with ordering info
    window_size: int = 2   # how many neighbors to include on each side
) -> list[dict]:
    """
    Retrieve matching chunks, then expand context by including
    neighboring chunks from the same document.
    """
    query_embedding = await embed_query(query)
    results = await vector_db.search(
        collection="documents",
        query_vector=query_embedding,
        limit=10
    )
    
    expanded_results = []
    
    for result in results:
        doc_id = result.payload["doc_id"]
        chunk_index = result.payload["chunk_index"]
        
        # Fetch neighboring chunks
        window_chunks = []
        for offset in range(-window_size, window_size + 1):
            neighbor_id = f"{doc_id}-chunk-{chunk_index + offset:04d}"
            neighbor = await chunk_store.get(neighbor_id)
            if neighbor:
                window_chunks.append(neighbor["text"])
        
        # Combine window into context
        window_text = "\n".join(window_chunks)
        
        expanded_results.append({
            "text": window_text,
            "matched_chunk": result.payload["text"],
            "score": result.score,
            "metadata": result.payload
        })
    
    return expanded_results
```

**Trade-offs vs. full parent-child:**

Sentence window is simpler — no separate document store, no explicit parent definition. You just store all chunks with sequential IDs and fetch neighbors at query time.

But it is less controlled. Neighbors may cross section boundaries — the chunk before a new section begins may belong to the previous topic. True parent-child respects document structure boundaries; sentence window does not.

Use sentence window for homogeneous prose documents where section boundaries matter less. Use true parent-child for structured documents where a parent should be confined to one section or topic.

---

## Variant: Document Summary Indexing

A third variant that is useful for high-level queries: index a summary of each document (or major section) alongside the detailed chunks.

```
Document
├── Summary chunk (indexed) — represents the whole document at high level
├── Child chunk 1 (indexed) — specific detail
├── Child chunk 2 (indexed) — specific detail
└── Child chunk 3 (indexed) — specific detail
```

The summary chunk is generated by an LLM at index time:

```python
async def generate_document_summary(document_text: str, doc_title: str) -> str:
    prompt = f"""
    Create a concise summary (150-200 words) of the following document.
    The summary should capture the main topics, key facts, and overall purpose.
    It will be used to help find this document when users ask broad questions.
    
    Document title: {doc_title}
    
    Document content:
    {document_text[:8000]}  # Use first ~8000 tokens for summary generation
    """
    
    response = await llm.generate(prompt)
    return response

# Index summary with special metadata
summary_chunk = {
    "text": await generate_document_summary(doc_text, doc_title),
    "metadata": {
        "doc_id": doc_id,
        "chunk_type": "document_summary",
        "importance_score": 1.0  # Always high importance
    }
}
```

At retrieval time, document summary chunks will naturally surface for broad queries ("what documents do you have about X?") while detailed child chunks surface for specific queries ("what is the exact deadline in the X policy?").

This is sometimes called **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) when extended recursively — summaries of summaries, building a tree where each level is more abstract than the one below.

---

## Deduplication at Context Assembly

A critical implementation detail that is easy to overlook: when multiple child chunks from the same parent are retrieved, you only want the parent once in the LLM's context.

Without deduplication:
```
Query retrieves: Child 1a, Child 1b, Child 2a (top 3 results)
Parent lookup:   Parent 1, Parent 1, Parent 2
LLM context:     [Parent 1 text][Parent 1 text][Parent 2 text]  ← Parent 1 duplicated!
```

This wastes context tokens and can confuse the LLM when it sees the same content twice.

With deduplication:
```python
def deduplicate_parents(parent_results: list[dict]) -> list[dict]:
    """Remove duplicate parents, keeping the one from the highest-scoring child."""
    seen = {}
    
    for result in parent_results:
        parent_id = result["parent_id"]
        
        if parent_id not in seen:
            seen[parent_id] = result
        else:
            # Keep the result from the higher-scoring child retrieval
            if result.get("child_score", 0) > seen[parent_id].get("child_score", 0):
                seen[parent_id] = result
    
    return list(seen.values())
```

---

## Metadata Design for Hierarchical Indexing

The metadata schema for parent-child must support:
- Looking up all children of a parent (for deletion when document updates).
- Looking up a parent from a child (for context expansion).
- Understanding position in the hierarchy (for ordering assembled context).

```python
# Child chunk metadata
{
    "chunk_id": "doc-001-parent-0003-child-0001",
    "chunk_type": "child",
    "doc_id": "doc-001",
    
    # Hierarchy links
    "parent_id": "doc-001-parent-0003",
    "grandparent_id": "doc-001-section-01",  # For 3-level hierarchies
    
    # Position information
    "parent_index": 3,       # Which parent (0-based)
    "child_index": 1,        # Position within parent (0-based)
    "global_chunk_index": 14, # Position in full document
    
    # Content metadata
    "doc_title": "Employee Handbook 2024",
    "section": "Benefits",
    "heading_path": "Benefits > Parental Leave",
    "page_number": 8
}

# Parent chunk (stored in document store, not vector DB)
{
    "chunk_id": "doc-001-parent-0003",
    "chunk_type": "parent",
    "doc_id": "doc-001",
    "text": "Full parent text content...",
    "child_ids": [
        "doc-001-parent-0003-child-0000",
        "doc-001-parent-0003-child-0001",
        "doc-001-parent-0003-child-0002"
    ],
    "parent_index": 3,
    "section": "Benefits",
    "heading_path": "Benefits > Parental Leave"
}
```

Storing `child_ids` in the parent record makes deletion efficient — when a document is updated, fetch all parent records for that `doc_id`, collect all `child_ids`, and delete them from the vector database in a single batch operation.

---

## Choosing the Right Hierarchy Depth

| Document type | Recommended hierarchy |
|---|---|
| Short articles (< 5 pages) | No hierarchy needed — single level |
| Medium documents (5–30 pages) | Two levels: parent (500–1000t) → child (128–256t) |
| Long documents (30–200 pages) | Three levels: section → passage → sentence group |
| Book-length (200+ pages) | Three levels + document summary index |
| Highly structured (legal, technical) | Two levels aligned to document structure (section → clause) |

The key rule: **the parent boundary should align with a meaningful semantic unit in the document** — a section, a paragraph, a procedure, a clause. If the parent boundary cuts across topics, the parent context is incoherent and the LLM gets confused.

---

## When Parent-Child Is Not the Right Choice

Parent-child adds complexity. It is not always justified.

**Skip parent-child when:**
- Documents are short enough that a single chunk captures the full answer (< 2 pages average).
- Query patterns are consistently broad, never requiring precise fact retrieval — just use large chunks.
- You are prototyping — add parent-child when you have evidence from evaluation that chunk size is hurting precision or context quality.
- Your document store infrastructure is not yet in place — parent-child requires storing and looking up parent chunks outside the vector database.

**Definitely use parent-child when:**
- Evaluation shows that retrieved chunks are correct topic but lack enough context for complete answers.
- Evaluation shows that embedding similarity scores are high but answers are wrong — often caused by large chunks with mixed topics diluting the embedding.
- Documents have clear structural hierarchy that you can align parents to.

---

## Summary

- Parent-child chunking resolves the tension between retrieval precision (small chunks) and generation context (large chunks) by decoupling the two.
- Child chunks (128–256 tokens) are indexed for retrieval. Parent chunks (500–2000 tokens) are stored and returned to the LLM.
- At retrieval time: embed query → find matching children → look up their parents → deduplicate parents → pass parents to LLM.
- Deduplication is essential — multiple children from the same parent must not result in duplicate parent text in the LLM context.
- Three-level hierarchies (sentence group → passage → section) work well for long documents. Adaptive level selection based on query type provides additional precision.
- Sentence window retrieval is a simpler alternative that expands context by including neighboring chunks. Less structurally precise but easier to implement.
- Document summary indexing (RAPTOR) adds high-level summary chunks that surface for broad queries while detailed chunks serve specific ones.
- Align parent boundaries to meaningful document structure. An arbitrary parent boundary that cuts across topics produces incoherent context.

---

## What's Next

Part 2 is complete. Part 3 begins with Lesson 3.1 — dense retrieval internals: how HNSW and IVF indexes work, what approximate nearest neighbor search actually does, and how to configure vector indexes for production performance.