# Lesson 1.5 — Choosing the Right RAG Design for Your Data Type

---

## Why Data Type Changes Everything

Most RAG tutorials assume your documents are clean, well-structured text articles. In the real world, you rarely get that. You get scanned PDFs from the 1990s, financial reports full of tables and charts, legal contracts with complex cross-references, invoices with handwritten fields, codebases with interdependent files, and PowerPoint decks where half the information is in the slide visuals.

The same RAG pipeline that works beautifully for a knowledge base of markdown articles will fail silently on any of these. The failures are not obvious — the system still produces answers, they are just wrong because the indexing lost the critical information.

This lesson maps the five most common document types you will encounter in production to the specific RAG design decisions they require. After this lesson, you should be able to look at a document type and immediately know where the hard problems are.

---

## Type 1 — Clean Text Documents (Articles, Wikis, Manuals)

**Examples:** Knowledge base articles, Wikipedia dumps, product documentation in Markdown, blog posts, internal wikis.

**Characteristics:**
- Predominantly prose.
- Clear paragraph and section structure.
- Minimal tables, few or no images.
- Written to be read linearly.

**Where this is easy:** This is the happy path for RAG. Text is clean, structure is clear, semantic meaning is well-captured by standard embedding models.

**Indexing approach:**
- Parser: direct text extraction (no OCR needed).
- Chunking: recursive character splitting or document-aware section-based chunking. Section-based is better when headings are present and consistent.
- Chunk size: 512–1024 tokens works well.
- Metadata: title, section, date, author, URL.
- Embedding: any general-purpose embedding model.

**Retrieval approach:** Standard hybrid search + re-ranking. No special handling needed.

**The one subtle problem:** Long articles where the key answer is buried in one paragraph while the rest is context. Fixed-size chunking may split the key paragraph or dilute it with surrounding context. Parent-child chunking helps — small child chunks retrieve precisely, parent provides context.

---

## Type 2 — Financial Documents (Reports, Earnings Calls, Filings)

**Examples:** Annual reports (10-K, 10-Q), earnings call transcripts, investor presentations, regulatory filings (SEC, SEBI), financial statements.

**Characteristics:**
- Mix of dense prose, complex tables, and charts/graphs.
- Tables may span multiple pages.
- Numbers are the most important content — exact values, not paraphrases.
- Charts convey trends that are not in the text.
- Highly structured with standardized sections (MD&A, Risk Factors, Financial Statements).
- Cross-references: "see Note 14 on page 87."

**Where this breaks naive RAG:**

Tables are the biggest problem. A balance sheet or income statement spread over a page looks like this when naively parsed:

```
Assets Current assets Cash and cash equivalents 1,234 1,456 
Accounts receivable net 5,678 4,321 Total current assets...
```

The structure — which number belongs to which row, which column is 2023 vs. 2022 — is completely lost. The embedding of this garbled text is meaningless. A user asking "what was the cash position at end of 2023?" will get a retrieved chunk that contains the number, but neither the LLM nor the retrieval system can make sense of the relationship between label and value.

Charts and graphs are completely invisible to text parsers. Revenue growth charts, margin trend lines, market share pie charts — all lost.

Cross-references break retrieval. A chunk says "as discussed in Note 14" but Note 14 is in a different part of the document, in a different chunk. Retrieval returns one without the other.

**Indexing approach:**

*For tables:*
- Use a table-aware parser: pdfplumber (good for digital PDFs), AWS Textract (good for scanned/complex), Camelot (specifically for PDF tables).
- Convert each logical table into a structured representation. Two good options:
  - **Markdown table:** Preserves row/column relationships, embeds reasonably well, LLM can read it.
  - **Natural language serialization:** "In fiscal year 2023, Cash and cash equivalents were $1,234 million, compared to $1,456 million in 2022." This embeds much better than raw table structure because it resembles how documents describe numbers.
- Index tables as their own chunks. Do not mix table content with surrounding prose.
- Store the raw table structure (HTML or JSON) in metadata so the LLM gets the full structured version even if the embedding was from the NL serialization.

*For charts and figures:*
- Extract images from the PDF (PyMuPDF can do this).
- Pass each image through a vision-capable model (GPT-4o, Claude 3, Gemini) with a prompt like: "Describe the key data points, trends, and conclusions shown in this chart. Be specific about numbers and time periods."
- Index the generated caption as a chunk, with metadata indicating it is derived from a figure.
- Store the original image linked to the chunk so the LLM can reference it if needed.

*For cross-references:*
- At index time, detect cross-reference patterns ("see Note X", "as described in Section Y").
- Link the referencing chunk to the referenced chunk in metadata.
- At retrieval time, when a cross-referenced chunk is retrieved, also fetch its linked chunks. This is a form of graph-based retrieval at the document level.

**Retrieval approach:**
- Metadata filtering by document section is critical. A query about "revenue growth" should filter to the MD&A section, not retrieve from Risk Factors.
- Exact number queries benefit from BM25 — if a user asks about a specific dollar figure, BM25 will find chunks containing that exact number faster than semantic search.
- Consider a specialized financial embedding model or a model fine-tuned on financial text (FinBERT-based embeddings).

**Generation approach:**
- Explicitly instruct the LLM to cite specific figures and their source (table name, year).
- For numerical answers, ask the LLM to show its reasoning — do not trust it to do arithmetic correctly from multiple retrieved chunks. Structure the prompt to provide all the numbers and ask for the calculation explicitly.

---

## Type 3 — Scanned Documents and Invoices

**Examples:** Scanned invoices, receipts, purchase orders, historical documents, handwritten forms, contracts from before the digital era.

**Characteristics:**
- The document is an image. There is no embedded text.
- Layout carries semantic meaning: position on the page tells you what a value refers to (invoice number is always top-right, line items are always in the central table, total is always bottom-right).
- Fields are often unlabeled by text proximity — "TOTAL" and "€4,521.00" are related because they are spatially close, not because there is an explicit data structure linking them.
- Handwriting, stamps, and low scan quality degrade OCR accuracy.
- Highly templated within a vendor but wildly different across vendors.

**Where this breaks naive RAG:**

Plain OCR reads text in a linear scan, losing spatial relationships. An invoice line item table:

```
Widget A    10    $25.00    $250.00
Widget B    5     $40.00    $200.00
```

OCR might produce: "Widget A 10 $25.00 $250.00 Widget B 5 $40.00 $200.00" which loses the column structure. A query "what was the unit price of Widget B?" cannot be answered from this.

**Indexing approach:**

*OCR is not enough — you need layout-aware document understanding.*

Tools in this space:
- **AWS Textract:** Returns text plus bounding boxes and table/form structure. A "table" in Textract output is a proper row-column grid, not raw text.
- **Google Document AI:** Similar capabilities, strong on forms and specialized document types.
- **Azure Document Intelligence (formerly Form Recognizer):** Excellent for specific document types (invoices, receipts, tax forms) with pre-built models.
- **LayoutLM / LayoutLMv3:** Open-source models that combine OCR text with position information (the bounding box of each word) to understand document structure. Can be fine-tuned for specific document types.
- **Donut (Document Understanding Transformer):** End-to-end model that reads document images directly without a separate OCR step. Strong for templated documents.

*For invoices specifically:*

The goal is structured extraction, not free-text retrieval. You want to extract:
```json
{
  "invoice_number": "INV-2024-0042",
  "vendor": "Acme Supplies Ltd",
  "date": "2024-03-15",
  "line_items": [
    {"description": "Widget A", "quantity": 10, "unit_price": 25.00, "total": 250.00},
    {"description": "Widget B", "quantity": 5, "unit_price": 40.00, "total": 200.00}
  ],
  "subtotal": 450.00,
  "tax": 45.00,
  "total": 495.00
}
```

Once you have structured JSON, you store it in a database (not just a vector database) and answer questions with a combination of:
- **SQL/structured queries** for exact lookups ("show all invoices from Acme Supplies over $1000 in March 2024").
- **Vector search** over natural language descriptions generated from the structured data ("Invoice INV-2024-0042 from Acme Supplies dated March 15 2024 for Widget A x10 at $25 each and Widget B x5 at $40 each, total $495 including tax").

This is a hybrid structured + vector retrieval architecture. It is fundamentally different from pure vector search.

*For historical/low-quality scans:*
- Pre-process images: deskew, denoise, increase contrast before OCR.
- Use multiple OCR passes with different settings and pick the highest confidence result.
- For critical fields, use confidence scores from OCR to flag low-confidence extractions for human review.

**Retrieval approach:**
- For invoices and forms, structured query against extracted fields is more reliable than vector search.
- Vector search is useful for "find invoices similar to this description" or when the query is semantic ("invoices with shipping disputes").
- Combine both: structured filter narrows the candidate set, vector search ranks within it.

---

## Type 4 — Long Documents (Contracts, Books, Technical Manuals)

**Examples:** Legal contracts (100+ pages), technical reference manuals, regulatory compliance documents, academic papers with extensive appendices.

**Characteristics:**
- Much longer than the LLM's optimal context window.
- Contain dense, precise language where every word matters (especially legal).
- Heavy use of cross-references, defined terms, and section dependencies.
- A single question may require synthesizing information from multiple sections.
- Reading order matters — later sections often modify or override earlier ones.

**Where this breaks naive RAG:**

A 200-page contract chunked into 512-token pieces produces hundreds of chunks. A query like "what are the termination conditions?" might be answered by five different sections (general termination, termination for cause, termination for convenience, survival clauses, notice requirements). Retrieving top-5 chunks means you likely miss some of them.

Cross-references are particularly destructive. Section 8.2 says "subject to the limitations in Section 14.5." The two sections end up in different chunks. If Section 8.2 is retrieved without Section 14.5, the answer is incomplete or wrong.

Defined terms are another landmine. A contract defines "Confidential Information" in Section 1. Every other section uses this term. The definition lives in one chunk. If a query asks about handling Confidential Information and retrieves Section 7 (which uses the term) but not Section 1 (which defines it), the LLM is missing the critical definition.

**Indexing approach:**

*Hierarchical chunking:*
- Chunk at multiple granularities simultaneously: section level (coarse), paragraph level (fine).
- Index both. Use fine chunks for retrieval, coarse chunks for context.
- This is parent-child chunking applied to the document hierarchy.

*Defined terms handling:*
- Extract all defined terms and their definitions at index time.
- For every chunk, identify which defined terms appear in it.
- Store the relevant definitions as metadata or append them to the chunk before embedding.
- At retrieval time, when a chunk is returned, also return the definitions of any defined terms it contains.

*Cross-reference graph:*
- Parse cross-references from the document ("Section 8.2 references Section 14.5").
- Build a dependency graph.
- When a chunk is retrieved, also retrieve its direct dependencies (one hop in the graph).

*For very long documents:*
- Consider a two-stage retrieval: first retrieve at section level to identify the relevant sections, then retrieve at paragraph level within those sections.
- This hierarchical retrieval reduces noise from irrelevant chunks while maintaining precision.

**Retrieval approach:**
- Multi-query retrieval is important here. Complex legal questions need multiple sub-questions to catch all relevant sections.
- Re-ranking is critical — with many potentially relevant chunks, precise ordering matters.
- Consider increasing K (number of retrieved chunks) for long document Q&A compared to shorter document systems.

**Generation approach:**
- For legal documents, faithfulness is paramount. Instruct the LLM strongly to quote directly rather than paraphrase.
- Ask the LLM to flag when an answer might be incomplete because of missing context ("this answer is based on Section 8.2 but the complete answer may depend on other sections I was not provided").

---

## Type 5 — Code and Technical Documentation

**Examples:** Codebases (GitHub repositories), API documentation, SDK references, internal technical runbooks.

**Characteristics:**
- Two types of content: code and documentation. They need different handling.
- Code has structure: functions, classes, modules, imports. The call graph defines relationships.
- A function's meaning depends on the functions it calls and the classes it uses.
- Documentation often references code elements by name. Code often has docstrings.
- Queries are often extremely specific: "how to authenticate with the Payments API using OAuth2."
- Token distribution is very different from prose — code has many repeated structural tokens (def, return, class, import) that dilute embedding quality.

**Where this breaks naive RAG:**

Chunking code by token count splits functions mid-body. A retrieved half-function is useless. The LLM cannot understand what a function does without seeing the whole thing.

Code dependencies: a retrieved function calls three helper functions that are in different files. Without those helper functions, the retrieved code is incomplete context for understanding behavior.

Documentation-code mismatch: documentation may be outdated relative to the code. RAG may retrieve documentation that describes behavior that the actual code no longer implements.

**Indexing approach:**

*For code:*
- Parse code using AST (Abstract Syntax Tree) parsers, not character-based chunkers. Python: `ast` module, `tree-sitter`. JavaScript: `@babel/parser`. These give you the exact boundaries of functions, classes, and methods.
- Each function/method is one chunk. Never split a function.
- Prepend the file path, module name, and class name to each function chunk: "File: payments/auth.py, Class: OAuthHandler, Method: exchange_token — [function body]"
- Include the function signature and docstring prominently — these are the highest-signal parts for retrieval.
- For functions that call other functions, include the signatures of called functions as metadata.

*For documentation:*
- Use document-aware chunking as with clean text.
- Link documentation chunks to the code chunks they describe using metadata.

*Embedding code:*
- Code-specific embedding models significantly outperform general text embedders for code retrieval: `code-search-net`, `CodeBERT`, `UniXcoder`, OpenAI's text-embedding models (which were trained on code too).
- Query embedding and code embedding have different distributions — some models handle this asymmetry better than others. Evaluate on your specific languages.

**Retrieval approach:**
- Hybrid search is important: a developer asking "how does the authenticate function work" needs both semantic search (for the concept) and exact keyword search (to find the function named `authenticate`).
- Consider **call graph expansion**: when a code chunk is retrieved, also retrieve the chunks for functions it calls (one hop in the dependency graph). This is analogous to the cross-reference expansion for legal documents.

---

## Summary Decision Table

| Document Type | Key Challenge | Chunking Strategy | Special Handling |
|---|---|---|---|
| Clean text | None major | Section-based or recursive | Parent-child for long articles |
| Financial docs | Tables + charts | Table-aware + figure captioning | NL serialization of tables, vision models for charts |
| Scanned/invoices | No text layer, layout dependency | Structured extraction first | Layout-aware OCR, hybrid structured + vector |
| Long documents | Cross-references, defined terms, length | Hierarchical + dependency graph | Defined terms injection, reference graph retrieval |
| Code | AST structure, call graph | AST-based (function boundaries) | Code embedding models, call graph expansion |

---

## The Universal Principle

Every document type challenge reduces to the same root problem: **information that requires context to understand gets separated from that context during chunking.**

Tables need column headers. Code needs function signatures. Legal text needs defined terms. Charts need their axes and labels. Cross-referenced sections need each other.

The solution is always some variant of: preserve the relationship at index time (by encoding it in the chunk or in metadata) so it can be restored at retrieval time.

When you encounter a new document type in the future, ask yourself: "what context is required to make a chunk of this document meaningful?" That context needs to travel with the chunk.

---