# Case Study 5 — Legal/Compliance Document Search: Precision-Critical, Citation Required, Audit Trail

---

## Problem Statement

A multinational corporation's legal department needs a system to answer compliance questions, review contracts, and search regulatory requirements across their document library. The legal team currently employs 12 paralegals who spend 60% of their time searching documents.

The corpus:
- 8,500 vendor contracts (PDF, 5-200 pages each).
- 2,200 regulatory compliance documents (GDPR, CCPA, SOX, HIPAA, ISO 27001, local regulations across 23 countries).
- 1,800 internal legal policies and procedures.
- 450 litigation-related documents (most are access-restricted).
- Legal opinions and memos: 3,400 documents.

The requirements — and they are unusually strict:

**Precision over recall:** In legal, a wrong answer can trigger a compliance violation, a missed contractual obligation, or a lawsuit. A false IDK ("I couldn't find information") is much safer than a confidently wrong answer. The system must be calibrated toward precision.

**Verbatim citation:** Legal professionals need the exact language of a clause, not a paraphrase. "The clause says approximately X" is inadequate — they need the exact text.

**Audit trail:** Every answer must be traceable to its source: which document, which section, which page. This audit trail must be immutable and accessible for regulatory review.

**No hallucination tolerance:** In medical RAG, hallucination is dangerous. In legal RAG, hallucination is professional malpractice. Zero tolerance.

**Cross-jurisdictional awareness:** The same question ("what is our data retention obligation?") has different answers for data about EU citizens vs. California residents vs. data in China. The system must be jurisdiction-aware.

---

## Architecture Design Decisions

### Decision 1 — Extractive Over Abstractive Generation

The fundamental architectural choice: the system primarily retrieves and presents exact text from source documents rather than generating paraphrased answers.

This is the opposite of most RAG systems. Standard RAG: retrieve chunks → LLM synthesizes an answer. Legal RAG: retrieve chunks → present exact text → LLM adds minimal synthesis to connect the dots.

```python
LEGAL_GENERATION_MODES = {
    "citation_only": "Return exact quoted text from source documents with source citations. No synthesis.",
    "citation_plus_summary": "Return exact quoted text followed by a one-sentence plain-language summary.",
    "comparison": "Present the relevant text from each applicable source side-by-side.",
    "gap_analysis": "Identify what the documents do NOT address about this query."
}

async def generate_legal_response(
    query: str,
    retrieved_chunks: list[dict],
    mode: str = "citation_plus_summary",
    llm_client = None
) -> dict:
    """
    Legal-mode response generation: exact text citation first.
    """
    
    # Format chunks as exact citations
    citations = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        citation = {
            "ref": i,
            "document": chunk["metadata"].get("doc_title"),
            "section": chunk["metadata"].get("heading_path"),
            "page": chunk["metadata"].get("page_number"),
            "exact_text": chunk["text"],  # VERBATIM — no modification
            "effective_date": chunk["metadata"].get("effective_date"),
            "jurisdiction": chunk["metadata"].get("jurisdiction")
        }
        citations.append(citation)
    
    if mode == "citation_only":
        return {
            "answer_type": "citations",
            "citations": citations,
            "synthesis": None
        }
    
    # Minimal synthesis prompt — heavy emphasis on NOT paraphrasing
    synthesis_prompt = f"""A legal professional asked: {query}

The following are EXACT text excerpts from source documents.
Your task:
1. Quote the most relevant portion of each relevant source verbatim using [N] citations
2. Provide ONE sentence of plain-language summary connecting the citations
3. Note any apparent conflicts between sources
4. Note any gaps: aspects of the query NOT addressed by the provided text

CRITICAL: Do NOT paraphrase legal text. Only quote exact language from the sources below.
Do NOT state legal conclusions. Present what the documents say, not what they mean.

Sources:
{chr(10).join(f"[{c['ref']}] {c['document']} | {c['section']} | Page {c['page']}:{chr(10)}{c['exact_text'][:600]}" for c in citations)}

Response:"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": synthesis_prompt}],
        max_tokens=800,
        temperature=0.0  # Zero temperature — no creativity in legal context
    )
    
    return {
        "answer_type": "citations_with_synthesis",
        "citations": citations,
        "synthesis": response.choices[0].message.content
    }
```

### Decision 2 — Jurisdiction-Aware Retrieval

Legal questions are almost always jurisdiction-specific. A GDPR-related question should retrieve EU regulatory documents; a CCPA question should retrieve California-specific documents; and both should retrieve global internal policies that apply everywhere.

```python
JURISDICTION_HIERARCHY = {
    # When user asks about EU data, retrieve in this order
    "eu": ["gdpr", "eu_member_states", "global"],
    "us_california": ["ccpa", "us_federal", "global"],
    "us_federal": ["us_federal", "global"],
    "china": ["pipl", "china_regulations", "global"],
    "global": ["global"]
}

JURISDICTION_KEYWORDS = {
    "gdpr": ["gdpr", "eu", "european union", "data protection regulation"],
    "ccpa": ["ccpa", "california", "california consumer privacy"],
    "pipl": ["pipl", "china", "personal information protection law"],
    "hipaa": ["hipaa", "health", "healthcare", "medical records"],
    "sox": ["sox", "sarbanes", "financial controls", "audit"]
}

async def jurisdiction_aware_retrieval(
    query: str,
    user_context: dict,
    retriever,
    llm_client
) -> dict:
    """
    Retrieve documents respecting jurisdiction hierarchy.
    """
    
    # Step 1: Detect jurisdictions relevant to the query
    jurisdiction_prompt = f"""Identify which legal jurisdictions are relevant to this query.

Query: {query}
User's primary jurisdiction: {user_context.get('jurisdiction', 'global')}
Data subjects' jurisdictions (if applicable): {user_context.get('data_subjects_jurisdictions', [])}

Return JSON:
{{
    "primary_jurisdiction": "eu|us_california|us_federal|china|global|...",
    "secondary_jurisdictions": ["list"],
    "regulatory_frameworks": ["gdpr|ccpa|hipaa|sox|pipl|..."],
    "requires_multi_jurisdiction": true/false
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": jurisdiction_prompt}],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0.0
    )
    
    import json
    jurisdiction_info = json.loads(response.choices[0].message.content)
    
    primary = jurisdiction_info["primary_jurisdiction"]
    regulatory_frameworks = jurisdiction_info["regulatory_frameworks"]
    
    # Step 2: Build jurisdiction-aware filter
    applicable_jurisdictions = JURISDICTION_HIERARCHY.get(primary, ["global"])
    
    metadata_filter = {
        "must": [
            {"key": "document_status", "match": {"value": "active"}},
            {
                "should": [
                    {"key": "jurisdiction", "match": {"any": applicable_jurisdictions}},
                    {"key": "regulatory_framework", "match": {"any": regulatory_frameworks}}
                ]
            }
        ]
    }
    
    # Step 3: Retrieve with jurisdiction filter
    primary_results = await retriever.retrieve(
        query=query,
        metadata_filter=metadata_filter,
        k=10
    )
    
    # Step 4: If multi-jurisdiction query, also retrieve for secondary jurisdictions
    if jurisdiction_info.get("requires_multi_jurisdiction"):
        secondary_results_by_jurisdiction = {}
        
        for sec_jurisdiction in jurisdiction_info.get("secondary_jurisdictions", [])[:2]:
            sec_applicable = JURISDICTION_HIERARCHY.get(sec_jurisdiction, ["global"])
            sec_filter = {
                "must": [
                    {"key": "document_status", "match": {"value": "active"}},
                    {"key": "jurisdiction", "match": {"any": sec_applicable}}
                ]
            }
            sec_results = await retriever.retrieve(
                query=query,
                metadata_filter=sec_filter,
                k=5
            )
            secondary_results_by_jurisdiction[sec_jurisdiction] = sec_results
        
        return {
            "primary_results": primary_results,
            "secondary_results": secondary_results_by_jurisdiction,
            "jurisdiction_info": jurisdiction_info
        }
    
    return {
        "primary_results": primary_results,
        "secondary_results": {},
        "jurisdiction_info": jurisdiction_info
    }
```

### Decision 3 — Defined Terms and Cross-Reference Handling

Legal documents define terms that appear throughout the document. A contract defining "Confidential Information" in Section 1.1 and using it in 47 other clauses must have that definition available when any of those 47 clauses are retrieved.

```python
class DefinedTermsRegistry:
    """
    Maintains a registry of defined terms and their definitions
    for each document. Injected into retrieved chunks at query time.
    """
    
    def __init__(self, db_client):
        self.db = db_client
    
    async def extract_and_store_defined_terms(
        self,
        doc_id: str,
        document_text: str,
        llm_client
    ):
        """
        Extract defined terms from a legal document and store them.
        """
        
        # Legal documents typically define terms in specific ways
        import re
        
        # Pattern: "Term" means/refers to/shall mean...
        # Or: "Term" as used herein means...
        definition_patterns = [
            r'"([^"]+)"\s+(?:means|refers to|shall mean|is defined as)\s+([^.]+\.)',
            r'"([^"]+)"\s+(?:as used (?:herein|in this Agreement))\s+(?:means|refers to)\s+([^.]+\.)',
            r'(?:The term|")\s*([A-Z][A-Za-z\s]+)(?:"|)\s+(?:means|refers to)\s+([^.]+\.)'
        ]
        
        extracted_terms = {}
        for pattern in definition_patterns:
            matches = re.finditer(pattern, document_text)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                extracted_terms[term] = definition
        
        # Store in database
        if extracted_terms:
            await self.db.upsert("defined_terms", {
                "doc_id": doc_id,
                "terms": extracted_terms
            })
    
    async def get_relevant_definitions(
        self,
        chunk_text: str,
        doc_id: str
    ) -> dict:
        """
        Find defined terms that appear in a chunk and return their definitions.
        """
        
        doc_terms = await self.db.get("defined_terms", doc_id)
        if not doc_terms:
            return {}
        
        relevant = {}
        for term, definition in doc_terms["terms"].items():
            if term.lower() in chunk_text.lower():
                relevant[term] = definition
        
        return relevant


async def enrich_chunk_with_definitions(
    chunk: dict,
    terms_registry: DefinedTermsRegistry
) -> dict:
    """
    When a chunk is retrieved, inject relevant defined terms.
    """
    
    doc_id = chunk["metadata"].get("doc_id")
    if not doc_id:
        return chunk
    
    relevant_terms = await terms_registry.get_relevant_definitions(
        chunk_text=chunk["text"],
        doc_id=doc_id
    )
    
    if relevant_terms:
        # Append definitions to the chunk text for LLM context
        definitions_text = "\n\n[Defined Terms from this document]\n"
        for term, definition in relevant_terms.items():
            definitions_text += f'"{term}" means: {definition}\n'
        
        chunk = {
            **chunk,
            "text": chunk["text"] + definitions_text,
            "metadata": {
                **chunk["metadata"],
                "defined_terms_injected": list(relevant_terms.keys())
            }
        }
    
    return chunk
```

### Decision 4 — Immutable Audit Trail

Every query and its response must be logged immutably. Legal departments face regulatory audits where they must demonstrate what advice was given, based on what sources, at what time.

```python
import hashlib
from datetime import datetime

class LegalAuditLogger:
    """
    Immutable audit log for all legal RAG queries.
    Written to append-only storage (Postgres with row-level security,
    or a WORM-compliant object store like S3 with Object Lock).
    """
    
    def __init__(self, audit_store):
        self.store = audit_store
    
    async def log_query(
        self,
        query: str,
        user_id: str,
        user_role: str,
        retrieved_chunks: list[dict],
        response: dict,
        session_id: str
    ) -> str:
        """
        Log a complete query-response event immutably.
        Returns audit_id for reference.
        """
        
        # Compute a content hash of the response for integrity verification
        response_hash = hashlib.sha256(
            str(response).encode()
        ).hexdigest()
        
        audit_record = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            
            # User information
            "user_id": user_id,
            "user_role": user_role,
            
            # Query
            "query": query,
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            
            # Retrieved sources (immutable record of what was consulted)
            "sources_consulted": [
                {
                    "doc_id": c["metadata"].get("doc_id"),
                    "doc_title": c["metadata"].get("doc_title"),
                    "section": c["metadata"].get("heading_path"),
                    "page": c["metadata"].get("page_number"),
                    "chunk_id": c.get("chunk_id"),
                    "doc_version": c["metadata"].get("version"),
                    "doc_effective_date": c["metadata"].get("effective_date")
                }
                for c in retrieved_chunks
            ],
            
            # Response
            "response_citations": response.get("citations", []),
            "synthesis_provided": response.get("synthesis"),
            "response_hash": response_hash,
            
            # System state at time of query
            "pipeline_version": PIPELINE_VERSION,
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL
        }
        
        # Write to append-only store
        await self.store.append(audit_record)
        
        return audit_record["audit_id"]
    
    async def export_audit_trail(
        self,
        start_date: str,
        end_date: str,
        user_id: str = None,
        doc_id: str = None
    ) -> list[dict]:
        """
        Export audit trail for regulatory review.
        Supports filtering by date range, user, or document.
        """
        filters = {
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }
        if user_id:
            filters["user_id"] = user_id
        if doc_id:
            filters["sources_consulted.doc_id"] = doc_id
        
        return await self.store.query(filters)
```

### Decision 5 — Contract Comparison Mode

A primary use case: "Compare the termination clauses across all our contracts with Vendor X."

```python
async def contract_comparison(
    aspect: str,                    # e.g., "termination clause"
    contracts: list[str],           # doc_ids to compare
    vector_db,
    embedding_model,
    llm_client,
    terms_registry: DefinedTermsRegistry
) -> dict:
    """
    Compare a specific clause or aspect across multiple contracts.
    """
    
    comparison_results = {}
    
    # For each contract, retrieve the relevant section
    for doc_id in contracts:
        # Search within this specific document
        aspect_embedding = await embedding_model.embed(aspect)
        
        results = await vector_db.search(
            query_vector=aspect_embedding,
            filter={
                "must": [
                    {"key": "doc_id", "match": {"value": doc_id}}
                ]
            },
            limit=3
        )
        
        if results:
            # Enrich with defined terms
            enriched_chunks = [
                await enrich_chunk_with_definitions(
                    {"text": r.payload.get("text"), "metadata": r.payload, "chunk_id": r.id},
                    terms_registry
                )
                for r in results
            ]
            
            comparison_results[doc_id] = {
                "doc_title": results[0].payload.get("doc_title"),
                "relevant_text": [c["text"][:800] for c in enriched_chunks[:2]],
                "page_references": [r.payload.get("page_number") for r in results[:2]],
                "confidence": results[0].score
            }
        else:
            comparison_results[doc_id] = {
                "doc_title": doc_id,
                "relevant_text": None,
                "note": f"No relevant content found for '{aspect}' in this contract"
            }
    
    # Build comparison summary
    found_contracts = {k: v for k, v in comparison_results.items() if v["relevant_text"]}
    
    if not found_contracts:
        return {
            "aspect": aspect,
            "results": comparison_results,
            "summary": f"No contracts contained relevant text for '{aspect}'"
        }
    
    # Use LLM to summarize differences (verbatim quotes required)
    comparison_text = "\n\n".join([
        f"CONTRACT: {v['doc_title']}\nRelevant text:\n{v['relevant_text'][0]}"
        for k, v in found_contracts.items()
    ])
    
    comparison_prompt = f"""Compare how these contracts address: {aspect}

For each contract, quote the EXACT relevant language.
Then note:
1. Key differences in language, scope, or conditions
2. Which contract is most favorable/restrictive
3. Any missing provisions (contracts that do not address this aspect)

CONTRACTS:
{comparison_text}

Format your response as:
CONTRACT 1 [name]: "[exact quote]"
CONTRACT 2 [name]: "[exact quote]"
...
DIFFERENCES: [plain English comparison]
GAPS: [aspects not addressed]"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": comparison_prompt}],
        max_tokens=1000,
        temperature=0.0
    )
    
    return {
        "aspect": aspect,
        "results": comparison_results,
        "comparison_summary": response.choices[0].message.content
    }
```

### Decision 6 — Confidence Calibration: Prefer IDK

Legal RAG must be calibrated toward IDK when uncertain. A legal professional who gets a false confident answer and acts on it faces worse consequences than one who gets an IDK and does further research.

```python
LEGAL_IDK_THRESHOLD = 0.75  # Much higher than typical 0.5-0.6

async def legal_confidence_check(
    query: str,
    retrieved_chunks: list[dict],
    generated_response: dict,
    llm_client
) -> dict:
    """
    Legal-specific confidence assessment. Errs toward IDK.
    """
    
    signals = {}
    
    # Signal 1: Retrieval confidence
    if retrieved_chunks:
        top_score = retrieved_chunks[0].get("rerank_score", 0)
        signals["retrieval_confidence"] = top_score
    else:
        signals["retrieval_confidence"] = 0
        return {
            "should_show_answer": False,
            "reason": "No relevant documents found",
            "recommendation": "Consult legal counsel or search manually"
        }
    
    # Signal 2: Are the retrieved documents current?
    import datetime
    today = datetime.date.today()
    docs_current = all(
        c["metadata"].get("document_status") == "active"
        for c in retrieved_chunks
    )
    signals["all_docs_current"] = docs_current
    
    # Signal 3: Does the response contain uncertain language?
    uncertain_phrases = [
        "may", "might", "could", "generally", "typically",
        "in most cases", "it depends", "subject to"
    ]
    synthesis = generated_response.get("synthesis", "")
    uncertainty_count = sum(
        1 for phrase in uncertain_phrases
        if phrase.lower() in synthesis.lower()
    )
    signals["uncertainty_count"] = uncertainty_count
    
    # Signal 4: Does the query span multiple jurisdictions?
    # Multi-jurisdiction answers are inherently less certain
    signals["multi_jurisdiction"] = len(
        generated_response.get("jurisdiction_info", {}).get("secondary_jurisdictions", [])
    ) > 0
    
    # Compute legal confidence score (conservative)
    base_confidence = signals["retrieval_confidence"]
    
    if not signals["all_docs_current"]:
        base_confidence -= 0.20  # Heavy penalty for stale documents
    
    if signals["uncertainty_count"] > 2:
        base_confidence -= 0.10
    
    if signals["multi_jurisdiction"]:
        base_confidence -= 0.10
    
    final_confidence = max(0.0, min(1.0, base_confidence))
    
    should_show = final_confidence >= LEGAL_IDK_THRESHOLD
    
    return {
        "confidence": final_confidence,
        "should_show_answer": should_show,
        "signals": signals,
        "recommendation": (
            "Suitable for reference — verify with counsel before relying on for decisions"
            if should_show
            else "Confidence insufficient — recommend manual document review or legal counsel"
        )
    }
```

---

## Evaluation Metrics for Legal RAG

Standard RAG metrics are necessary but insufficient. Legal RAG requires additional metrics:

```python
LEGAL_EVAL_METRICS = {
    # Standard
    "retrieval_recall@5": "Target: > 0.90 (higher than typical)",
    "faithfulness": "Target: > 0.98 (near-perfect — zero hallucination tolerance)",
    
    # Legal-specific
    "verbatim_citation_accuracy": "% of citations that quote exact document text",
    "citation_traceability": "% of answers where source can be located in original document",
    "jurisdiction_accuracy": "% of answers that correctly identify applicable jurisdiction",
    "defined_term_injection_rate": "% of relevant defined terms included in context",
    "false_idk_rate": "% of IDK responses where answer was actually in the corpus",
    "false_confidence_rate": "% of confident answers that were actually wrong",
    
    # Audit
    "audit_completeness": "% of queries with complete immutable audit records",
    "source_freshness": "% of cited documents that are current (not superseded)"
}
```

---

## Lessons Learned

**Lesson 1:** Extractive generation (verbatim citation) was initially resisted by the product team who wanted a more "helpful" paraphrased answer. After a pilot where a paraphrased answer omitted a critical exception clause and a paralegal almost missed it, the team embraced verbatim citation as the only acceptable mode.

**Lesson 2:** Defined term injection was the single biggest quality improvement. Before it, queries about "Confidential Information" in contract clause 8.3 returned text that used the term but not its definition, requiring the paralegal to manually look up Section 1.1. After injection, the definition traveled with the clause automatically.

**Lesson 3:** The audit trail has already been used in two regulatory inquiries. Its existence was more valuable than expected — regulators were satisfied when the company could demonstrate exactly what documents were consulted and when, down to the specific clause.

**Lesson 4:** Jurisdiction detection errors are more dangerous than retrieval failures. If the system retrieves the correct clause for the wrong jurisdiction, the answer is confidently wrong. Investing in accurate jurisdiction detection (and defaulting to "multi-jurisdiction" when uncertain) paid off.

---

## Interview Questions This Case Study Prepares You For

**"How do you prevent hallucination in a high-stakes legal RAG system?"**
Answer: Three layers: (1) extractive generation — verbatim citation mode where the LLM quotes rather than paraphrases, (2) temperature=0.0 for all generation, (3) high IDK threshold (0.75 vs typical 0.5) so the system defaults to IDK when retrieval confidence is below that bar. Hallucination tolerance is literally zero.

**"How do you build an immutable audit trail for a RAG system?"**
Answer: Append-only storage (PostgreSQL with no DELETE permissions, or S3 with Object Lock). Every query logs: user identity, query text, source documents consulted (doc_id, version, page), exact response content, response hash, pipeline version, and timestamp. Immutability is enforced at the storage layer, not the application layer.

**"How do you handle a question that has different legal answers in different jurisdictions?"**
Answer: Jurisdiction detection step before retrieval — classify the query by applicable legal frameworks (GDPR, CCPA, PIPL, etc.). Build a filter that retrieves from the correct jurisdictional documents. For multi-jurisdiction queries, retrieve separately for each jurisdiction and present them side-by-side. Never blend conflicting jurisdictional requirements into one answer.