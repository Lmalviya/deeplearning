# Lesson 2.1 — Chunking Strategies: Fixed, Recursive, Semantic, Late Chunking, and Hierarchical

---

## Why Chunking Deserves Its Own Deep Dive

Chunking was introduced in Lesson 1.4 as one stage of the indexing pipeline. But it deserves a full lesson because it is the single decision that most practitioners get wrong, and it is the failure that is hardest to diagnose.

Here is why it is hard to diagnose: bad chunking does not throw an error. The system indexes successfully, retrieval runs, the LLM generates an answer. The answer is just wrong or incomplete — and tracing that back to a chunking decision requires deliberately inspecting what chunks were actually retrieved and what they contain.

In this lesson we go deep on every major chunking strategy — not just what they are, but how to implement them, when they fail, and how to choose between them for a given situation.

---

## What a Good Chunk Looks Like

Before discussing strategies, define the target. A good chunk has three properties:

**1. Semantic completeness.** The chunk expresses a complete idea. It does not start mid-sentence or end mid-argument. A reader who sees only this chunk understands what it is saying without needing surrounding context.

**2. Appropriate density.** The chunk is about one topic or one coherent set of related points. It is not so broad that it covers three different subjects (making its embedding a confused average), and not so narrow that it is a single sentence with no supporting context.

**3. Self-contained for retrieval.** The chunk contains enough context to be retrievable on its own. It does not rely on a previous chunk to establish what "it" refers to, or what the defined term means, or what table column a number belongs to.

Every chunking strategy is an attempt to produce chunks with these three properties. They all make different trade-offs.

---

## Strategy 1 — Fixed-Size Chunking

### How It Works

Split the document every N tokens (or characters), with an overlap of M tokens between consecutive chunks.

```
Token stream: [t1 t2 t3 ... t512 | t463 t464 ... t1024 | ...]
                                   ↑ overlap starts here
```

The overlap window (t463–t512 appears in both chunk 1 and chunk 2) ensures that content near boundaries is not permanently split — it appears whole in at least one chunk.

### Implementation

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,       # tokens per chunk
    chunk_overlap=64,     # overlap tokens
    encoding_name="cl100k_base"  # tiktoken encoding (matches GPT-4/OpenAI)
)

chunks = splitter.split_text(document_text)
```

Use token-based splitting (not character-based) because embedding models and LLMs have token limits, not character limits. A character limit of 2000 will produce very different chunk counts across English vs. Chinese text.

### Choosing Chunk Size

There is no universal right answer. A useful heuristic:

- **128–256 tokens:** Good for precise fact retrieval. Each chunk is a tight, focused statement. Bad for questions requiring context.
- **512 tokens:** A reasonable default for general-purpose Q&A over prose documents. Roughly 1–2 paragraphs.
- **1024 tokens:** Better for complex reasoning questions. More context per chunk. Worse retrieval precision — one chunk now covers multiple ideas.
- **2048+ tokens:** Rarely a good chunk size. The embedding model's ability to represent a 2048-token chunk meaningfully degrades. Use parent-child instead.

The overlap should be 10–20% of chunk size. Less than 10% and boundary splits are not fully healed. More than 20% and you have significant redundancy across chunks.

### Where It Fails (and Why)

**Mid-sentence splits.** A paragraph about a concept gets split at token 512 regardless of sentence boundaries. The first chunk ends mid-sentence; the second begins mid-sentence. Both embeddings are degraded by the incomplete statement.

**Mid-concept splits.** A multi-paragraph argument about one topic gets split into two chunks. Each chunk captures half the argument. Neither chunk alone is semantically complete. A query about the topic may retrieve only one half.

**Boundary sensitivity.** Moving the chunk size from 512 to 513 changes which content falls in which chunk across the entire document. There is nothing principled about where boundaries land.

**Overlap is not a full fix.** Overlap heals the sentence split for the overlapping tokens. It does not help when a coherent concept spans 600 tokens — the split still cuts the concept in two, overlap or not.

### When to Use It

- Homogeneous, prose-heavy documents without clear structural signals.
- When you need a simple, predictable baseline to compare against.
- When documents are already short (< 2 pages) — chunking artifacts matter less.
- As a fallback when structure-based chunking fails (malformed documents, corrupted parsing).

Fixed-size chunking is a reasonable baseline, not a production-quality solution for complex documents.

---

## Strategy 2 — Recursive Character Splitting

### How It Works

Instead of splitting mechanically every N tokens, try to split at natural text boundaries. Use a priority-ordered list of separators:

```
Priority 1: "\n\n"  (paragraph break)
Priority 2: "\n"    (line break)
Priority 3: ". "    (sentence end)
Priority 4: " "     (word boundary)
Priority 5: ""      (character — last resort)
```

For each segment, try the highest-priority separator first. If the resulting piece is still larger than `chunk_size`, recurse with the next separator. Only go to character-level splitting if nothing else works.

### Implementation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # characters (not tokens — note the difference)
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(document_text)
```

Note: LangChain's default implementation uses characters, not tokens. For token-accurate splitting, use `RecursiveCharacterTextSplitter.from_tiktoken_encoder()`.

### Why It Is Better Than Fixed-Size

Splits happen at paragraph boundaries when possible. This means:
- Chunks are much more likely to be semantically complete (paragraphs are natural semantic units).
- Mid-sentence splits are rare (only happen when a single sentence exceeds `chunk_size`).
- The resulting chunks are more readable and more embeddable.

### Where It Still Fails

- It is still structure-agnostic. It does not know what a heading is, what a table is, or what a list is.
- A long paragraph about two different topics does not get split — the whole paragraph lands in one chunk, diluting the embedding.
- Tables and code blocks get split at whatever character boundary the separator finds — typically mid-row or mid-function.
- Custom separators for specific formats (Markdown headers, HTML tags) require explicit configuration.

### Custom Separators for Specific Formats

For Markdown:
```python
md_separators = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
```

For code:
```python
code_separators = ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
```

LangChain provides `Language` enum with pre-configured separators for Python, JavaScript, Java, C++, and others via `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`.

### When to Use It

- Prose documents without rich structural markup (plain text, some PDFs).
- When you want better-than-fixed-size chunking without building a full document-aware parser.
- Good default for unstructured text in production when you do not know the document format in advance.

---

## Strategy 3 — Document-Aware / Structure-Based Chunking

### How It Works

Instead of splitting by size, split by the document's logical structure. Parse the document to extract its structural elements — headings, sections, paragraphs, tables, lists, code blocks — then create chunks that correspond to those elements.

### For Markdown Documents

Markdown is the easiest case because the structure is explicit in the text:

```python
import re

def chunk_markdown(text: str) -> list[dict]:
    """Split markdown by headers, preserving hierarchy."""
    chunks = []
    
    # Split on H2 headers
    sections = re.split(r'\n(?=## )', text)
    
    for section in sections:
        lines = section.strip().split('\n')
        heading = lines[0].replace('## ', '')
        body = '\n'.join(lines[1:]).strip()
        
        # If section is too long, split further at H3
        if len(body) > 2000:
            subsections = re.split(r'\n(?=### )', body)
            for sub in subsections:
                sublines = sub.strip().split('\n')
                subheading = sublines[0].replace('### ', '') if sublines[0].startswith('###') else ''
                subbody = '\n'.join(sublines[1:] if subheading else sublines).strip()
                chunks.append({
                    'text': f"{heading} > {subheading}\n\n{subbody}" if subheading else f"{heading}\n\n{subbody}",
                    'metadata': {'section': heading, 'subsection': subheading}
                })
        else:
            chunks.append({
                'text': f"{heading}\n\n{body}",
                'metadata': {'section': heading}
            })
    
    return chunks
```

The key technique: **prepend the heading path to each chunk's text**. This means the chunk's embedding captures both where it lives in the document and what it says.

### For PDFs and Word Documents

These require a parser that extracts structural signals:

**python-docx for Word:**
```python
from docx import Document

def extract_structure(docx_path):
    doc = Document(docx_path)
    elements = []
    
    for para in doc.paragraphs:
        if para.style.name.startswith('Heading'):
            level = int(para.style.name.split(' ')[-1])
            elements.append({'type': 'heading', 'level': level, 'text': para.text})
        elif para.text.strip():
            elements.append({'type': 'paragraph', 'text': para.text})
    
    for table in doc.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        elements.append({'type': 'table', 'rows': rows})
    
    return elements
```

Once you have structured elements, group them into chunks by heading level — each H1 or H2 section becomes the parent chunk, paragraphs within it become child chunks.

### The Context Enrichment Pattern

A critical technique for structure-based chunking: enrich each chunk with its document context before embedding.

Instead of embedding just the chunk text, embed:
```
[Document Title]
[Section Path: H1 > H2 > H3]
[Chunk text]
```

Example:
```
Employee Handbook 2024
Benefits > Health Insurance > Eligibility Requirements

Employees become eligible for health insurance coverage on the first day 
of the month following 30 days of continuous employment...
```

This context enrichment ensures the embedding captures where the chunk lives in the document, not just what it says locally. A query about "health insurance eligibility" will match this chunk much more precisely because the section path is in the embedding.

**Important:** Store the raw chunk text (without the enrichment prefix) as the content to show the LLM. Only use the enriched version for embedding. Showing the LLM repetitive prefixes wastes context tokens.

### When to Use It

- Whenever your documents have consistent, parseable structure (headings, sections).
- Technical documentation, policy documents, handbooks, reports.
- When document structure maps well to the topics users ask about.

---

## Strategy 4 — Semantic Chunking

### How It Works

Split where the meaning changes, not where a character boundary or heading appears.

```
Algorithm:
1. Split document into sentences (use spaCy or NLTK sentence tokenizer).
2. Embed each sentence (or small window of 2–3 sentences for stability).
3. Compute cosine similarity between embedding[i] and embedding[i+1] for all i.
4. Identify "semantic breakpoints" — positions where similarity drops significantly.
5. Group sentences between breakpoints into chunks.
6. Merge very small groups with their neighbors.
```

The "semantic breakpoint" detection can be done several ways:
- **Percentile threshold:** Find the bottom Xth percentile of similarity values and declare those positions as breakpoints. LangChain's `SemanticChunker` uses this approach.
- **Gradient-based:** Find positions where the drop in similarity is steepest (local minima of the similarity curve).
- **Fixed threshold:** Split wherever similarity falls below an absolute value (e.g., 0.7). Requires tuning per domain.

### Implementation

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

chunker = SemanticChunker(
    embedder,
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95           # split at bottom 5% similarity values
)

chunks = chunker.split_text(document_text)
```

### Why It Is Appealing

The chunks produced are semantically coherent by construction — the algorithm guarantees that sentences within a chunk are meaningfully related. This is exactly what you want for embedding quality.

For documents with topic drift (an article that starts with background, moves to technical details, then discusses implications), semantic chunking produces chunks that each cover one phase of the argument cleanly.

### Where It Fails

**Expensive at index time.** You embed every sentence (or sentence window) of every document just to determine chunk boundaries, then embed the final chunks again. For a large corpus this is 3–5x the embedding cost of fixed-size chunking.

**Threshold sensitivity.** The right threshold is domain and document-specific. A threshold that works for news articles may produce chunks that are too large for dense technical papers or too small for conversational text.

**Variable chunk sizes.** Some chunks may be 1 sentence, some may be 20 sentences. Very short chunks have weak embeddings. Very long chunks have diluted embeddings. You need a minimum/maximum chunk size enforcement on top of the semantic split.

**Does not handle structure.** Semantic chunking works on meaning transitions in running text. It does not understand that a heading starts a new section, that a table is a distinct unit, or that a code block should not be split.

### When to Use It

- Long-form prose where topic transitions are gradual and do not align with headings (research papers, long essays, narrative documents).
- When you have the indexing budget for the extra embedding cost.
- As a refinement step on top of coarse structural chunking — semantically split within each section.

---

## Strategy 5 — Parent-Child Chunking (Small-to-Big Retrieval)

### The Core Insight

Retrieval precision and generation context have opposite requirements:

- **Retrieval wants small chunks.** A 128-token chunk has a focused embedding — it represents one specific fact or idea. Short, specific queries match it well.
- **Generation wants large chunks.** A 128-token chunk is a sentence or two. It may not give the LLM enough context to generate a complete, accurate answer.

Parent-child chunking decouples these two requirements:
- **Child chunks** (small, 128–256 tokens): indexed in the vector database, used for retrieval.
- **Parent chunks** (large, 512–2048 tokens): stored separately (in a document store), returned to the LLM when a child is retrieved.

Each child chunk has a pointer to its parent. When retrieval returns child chunk IDs, a lookup step fetches the corresponding parent chunks.

### Implementation

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Child splitter — small chunks for retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Parent splitter — larger chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# Vector store for child embeddings
vectorstore = Qdrant(...)

# Document store for parent chunks (key-value store)
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)
```

At query time:
```python
# Returns parent chunks even though retrieval was by child similarity
results = retriever.get_relevant_documents("your query here")
```

### Hierarchical Variants

You can extend this to more than two levels:

```
Document (entire document, stored)
    ↓ parent
Section (H1/H2 section, stored + optionally indexed)
    ↓ parent
Paragraph (indexed for retrieval)
    ↓ parent
Sentence (optionally indexed for very precise retrieval)
```

At retrieval time, you can choose which level to return to the LLM based on query complexity:
- Simple factual queries → return the paragraph (small context, precise)
- Complex reasoning queries → return the section (large context, more complete)

### The Deduplication Challenge

Multiple child chunks may belong to the same parent. If three child chunks from the same parent are all retrieved, you only include the parent once in the LLM context — not three times. Implement deduplication by parent ID before context assembly.

```python
retrieved_children = vectorstore.similarity_search(query, k=20)
parent_ids = set()
parents = []
for child in retrieved_children:
    pid = child.metadata['parent_id']
    if pid not in parent_ids:
        parent_ids.add(pid)
        parents.append(docstore.get(pid))
```

### When to Use It

- Almost always, for any document type longer than a page.
- Especially valuable for long documents where precise retrieval and rich generation context are both required.
- For technical documentation and legal documents where a sentence may be the precise answer but requires surrounding sections for full understanding.

---

## Strategy 6 — Late Chunking

### The Problem It Solves

Standard chunking embeds each chunk independently. This means the embedding of a chunk does not "know" about the rest of the document it came from. A chunk saying "the policy was amended in 2023" does not encode which policy — that context was in a previous chunk.

Late chunking addresses this with a different approach: embed the entire document first (using a long-context embedding model), then chunk the resulting token-level embeddings.

### How It Works

```
Full document text → Long-context embedding model → Per-token embeddings
                                                           ↓
                                         Chunk the token embeddings by position
                                                           ↓
                                         Pool each chunk's token embeddings → chunk vector
```
![Late Chunking Image](..\assets\late_chunking.png)

```
Note: The late chunking approach reverses this process. Instead of dividing the document into chunks and then computing the embeddings, the whole text of the document is first passed through a Transformer model. This generates an embedding representation for each token, and these tokens now contain contextual information not only limited to a single chunk, but encompassing the entire document. After this step, the chunking process is performed, where the original text is divided into chunks, and the corresponding tokens are used to compute the mean pooling, resulting in the final representation.
```

Because the embedding model processes the full document before chunking, each token's embedding already incorporates context from the surrounding document. When you pool a chunk's token embeddings into a single chunk vector, that vector encodes both the local text and its document-level context.

### Requirements and Limitations

- Requires an embedding model that operates on token level and supports long contexts. **JinA AI's jina-embeddings-v2** and **ColBERT-style models** support this. Standard OpenAI or sentence-transformer models do not — they produce a single pooled vector per input, not per-token vectors.
- The entire document must fit in the model's context window. For very long documents (100+ pages), this is still not feasible.
- More computationally expensive than chunking first and embedding chunks independently.

### When to Use It

- When contextual continuity across chunks is critical and the document fits in the model's context window.
- Particularly useful for documents with heavy pronoun and reference use ("the aforementioned clause", "as defined above") where chunk-level embeddings lose the referent.
- Still an emerging technique — evaluate carefully against simpler approaches on your specific data before committing to the infrastructure complexity.

---

## Choosing a Strategy: Decision Framework

Work through these questions in order:

**1. Does my document have parseable structure (headings, sections)?**
- Yes → Start with document-aware structure-based chunking.
- No → Go to question 2.

**2. Is the document predominantly prose?**
- Yes → Recursive character splitting with sentence-boundary preference.
- No (tables, code, forms) → Type-specific handling (see Lesson 1.5).

**3. Do I need both precise retrieval and rich generation context?**
- Yes → Add parent-child on top of whatever base strategy you chose.
- No → Base strategy alone may suffice.

**4. Are there significant topic shifts within sections?**
- Yes → Add semantic chunking within sections as a refinement.
- No → Skip it (not worth the cost).

**5. Is cross-chunk context loss causing retrieval failures?**
- Yes → Evaluate late chunking or context enrichment (prepending document/section title).
- No → Proceed with current strategy.

---

## Evaluating Your Chunking Quality

Do not guess — measure. Two practical evaluation approaches:

**Manual inspection:** Take 20 random chunks from your index. Ask: is each chunk semantically complete? Does it make sense in isolation? Would a retrieval of this chunk answer a plausible question? If more than 20–30% fail this check, your chunking needs work.

**Retrieval-based evaluation:** Create a test set of 50–100 question-answer pairs where you know which document section contains the answer. Run retrieval and check whether the retrieved chunks contain the answer. If recall@5 is below 80%, suspect chunking (or embedding model quality) first.

We cover retrieval evaluation metrics in depth in Lesson 6.2.

---

## Summary

- Fixed-size chunking is simple but creates arbitrary splits that degrade embedding quality. Use as a baseline only.
- Recursive character splitting respects natural text boundaries — a reliable default for prose.
- Structure-based chunking produces semantically complete chunks aligned with document organization. Best for documents with clear structure.
- Semantic chunking splits on meaning transitions. Produces high-quality chunks for prose at higher indexing cost.
- Parent-child chunking decouples retrieval precision (small chunks) from generation context (large chunks). Valuable in almost all production systems.
- Late chunking preserves document-level context within chunk embeddings. Emerging technique with specific model requirements.
- The right strategy depends on document type, query patterns, and budget. Often the right answer is a combination.

---

## What's Next

Lesson 2.2 goes deep into embedding models — how they work internally, how to choose between them, how to fine-tune them for your domain, and the emerging techniques of Matryoshka embeddings and late interaction models like ColBERT.