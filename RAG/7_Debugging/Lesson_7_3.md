# Lesson 7.3 — Data Conflicts and Knowledge Inconsistency Resolution

---

## The Conflict Problem in Enterprise RAG

In an ideal world, your document corpus is a consistent, authoritative, non-contradictory body of knowledge. In practice, enterprise document corpora are anything but:

- A policy was updated in March but the old version was not removed from the intranet.
- Two departments independently wrote guidelines for the same process with different rules.
- An FAQ document summarizes a policy but the summary contains a subtle error not present in the original.
- A regulatory document was superseded but still indexed because the deletion was not propagated.
- Regional variants of the same policy contradict each other (California law vs. federal law).
- A training slide deck contains an outdated figure that contradicts the current annual report.

When retrieval surfaces chunks from conflicting documents, the LLM faces an impossible task: it cannot give a single correct answer because the corpus itself does not contain one. Its options are to pick one source arbitrarily (wrong), blend them (wrong), refuse to answer (safe but unhelpful), or explicitly surface the conflict (correct).

Without deliberate design, most RAG systems do the first or second. This lesson covers how to detect conflicts, communicate them to users, and prevent them from degrading answer quality.

---

## Type 1 — Version Conflicts (Same Document, Multiple Versions)

The most common conflict type: an older version of a document coexists in the index alongside a newer version. Both versions appear relevant to queries about that topic. The old version may contain outdated facts, superseded policies, or wrong numbers.

### Detection

At index time, track document lineage:

```python
@dataclass
class DocumentVersion:
    doc_id: str
    doc_family_id: str          # Shared across versions of the same document
    version: str                 # "v1.0", "2024-Q1", etc.
    effective_date: date
    document_status: str         # "active", "superseded", "draft", "archived"
    superseded_by: Optional[str] # doc_id of newer version, if superseded

async def detect_version_conflicts(
    registry,
    doc_family_id: str
) -> list[DocumentVersion]:
    """
    Find all versions of a document and identify if multiple active versions exist.
    """
    all_versions = await registry.get_by_family(doc_family_id)
    
    active_versions = [
        v for v in all_versions
        if v.document_status == "active"
    ]
    
    if len(active_versions) > 1:
        # Multiple active versions = version conflict
        return sorted(active_versions, key=lambda v: v.effective_date, reverse=True)
    
    return []  # No conflict
```

### Prevention

**Mark documents as superseded when a new version is published.** This is the primary prevention mechanism — old versions should never be "active" once a new version exists.

```python
async def publish_new_document_version(
    new_doc_path: str,
    doc_family_id: str,
    registry,
    vector_db
):
    """
    Publish a new version and supersede all previous versions.
    """
    # Step 1: Find all currently active versions of this document family
    existing_versions = await registry.get_active_by_family(doc_family_id)
    
    # Step 2: Mark them as superseded
    for version in existing_versions:
        await registry.update(version.doc_id, {
            "document_status": "superseded",
            "superseded_by": None  # Will be set after new version is indexed
        })
        
        # Update metadata in vector DB so retrieval filters work
        await vector_db.set_payload(
            collection="documents",
            payload={"document_status": "superseded"},
            filter={"doc_id": version.doc_id}
        )
    
    # Step 3: Index new version as active
    new_doc_id = await index_document(new_doc_path, {
        "doc_family_id": doc_family_id,
        "document_status": "active",
        "effective_date": date.today().isoformat()
    })
    
    # Step 4: Update superseded_by pointers
    for version in existing_versions:
        await registry.update(version.doc_id, {
            "superseded_by": new_doc_id
        })
    
    return new_doc_id
```

**Default retrieval filter:** Always filter to `document_status = active` unless a user explicitly requests historical content.

```python
DEFAULT_RETRIEVAL_FILTER = {
    "must": [
        {"key": "document_status", "match": {"value": "active"}}
    ]
}
```

### Runtime Detection

Even with good prevention, version conflicts may slip through. Detect them at retrieval time:

```python
def detect_version_conflicts_in_results(
    retrieved_chunks: list[dict]
) -> list[dict]:
    """
    Check retrieved chunks for multiple versions of the same document family.
    """
    from collections import defaultdict
    
    family_chunks = defaultdict(list)
    
    for chunk in retrieved_chunks:
        family_id = chunk["metadata"].get("doc_family_id")
        if family_id:
            family_chunks[family_id].append(chunk)
    
    conflicts = []
    
    for family_id, chunks in family_chunks.items():
        if len(chunks) > 1:
            # Multiple chunks from the same document family
            versions = set(c["metadata"].get("version") for c in chunks)
            if len(versions) > 1:
                conflicts.append({
                    "doc_family_id": family_id,
                    "conflicting_versions": list(versions),
                    "chunks": chunks
                })
    
    return conflicts
```

---

## Type 2 — Content Conflicts (Different Documents, Contradicting Facts)

Different documents in your corpus contain directly contradictory factual claims. Neither is "wrong" in the sense of being an old version — they genuinely disagree.

**Common causes:**
- Department A's guidelines contradict Department B's guidelines on the same topic.
- An FAQ summarizes a policy incorrectly.
- A vendor specification contradicts your internal specification.
- Regional legal variations (GDPR requirements differ from CCPA requirements for the same data type).

### Detection at Query Time

```python
async def detect_content_conflicts(
    query: str,
    retrieved_chunks: list[dict],
    llm_client,
    text_key: str = "text"
) -> dict:
    """
    Check whether retrieved chunks contain contradicting information
    about the topic of the query.
    """
    
    if len(retrieved_chunks) < 2:
        return {"conflicts_detected": False}
    
    # Take top chunks for conflict check
    top_chunks = retrieved_chunks[:5]
    chunks_text = "\n\n".join([
        f"[Source {i+1}: {c['metadata'].get('doc_title', 'Unknown')}]\n{c[text_key][:500]}"
        for i, c in enumerate(top_chunks)
    ])
    
    prompt = f"""You are checking retrieved document sections for factual contradictions.

Query: {query}

Retrieved sections:
{chunks_text}

Do any of these sections directly contradict each other on facts relevant 
to answering the query?

Focus on: numerical values, dates, procedures, eligibility criteria, 
limits, permissions (what is/is not allowed).

Ignore: stylistic differences, different levels of detail, complementary 
information.

Return JSON:
{{
    "conflicts_detected": true/false,
    "conflicts": [
        {{
            "claim_a": "what source X says",
            "source_a": "Source number",
            "claim_b": "what source Y says (contradicting claim A)",
            "source_b": "Source number",
            "topic": "what this conflict is about"
        }}
    ],
    "can_be_reconciled": true/false,
    "reconciliation_note": "how to interpret if they can be reconciled, or null"
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=600,
        temperature=0.0
    )
    
    import json
    result = json.loads(response.choices[0].message.content)
    
    return result
```

### Handling Detected Conflicts in Generation

When conflicts are detected, the generation prompt must change. The LLM should not silently pick one source or blend them — it must surface the conflict to the user.

```python
async def generate_with_conflict_awareness(
    query: str,
    retrieved_chunks: list[dict],
    conflict_result: dict,
    llm_client
) -> str:
    """
    Generate a response that appropriately handles detected conflicts.
    """
    
    context = format_context(retrieved_chunks)
    
    if not conflict_result.get("conflicts_detected"):
        # Standard generation
        return await generate_standard(query, context, llm_client)
    
    conflicts = conflict_result.get("conflicts", [])
    can_reconcile = conflict_result.get("can_be_reconciled", False)
    reconciliation = conflict_result.get("reconciliation_note")
    
    # Build conflict-aware system prompt
    conflict_instruction = f"""
IMPORTANT: The retrieved documents contain contradicting information on this topic.
Conflicts detected:
{chr(10).join([f"- {c['source_a']} says: {c['claim_a']}" + chr(10) + f"  {c['source_b']} says: {c['claim_b']}" for c in conflicts])}

{"The conflict can be reconciled as follows: " + reconciliation if can_reconcile else "These claims appear to directly contradict. Do not pick one — surface both to the user."}

Your response MUST:
1. Answer the question to the extent possible
2. Explicitly note the contradicting information
3. Cite which sources say what
4. If unresolvable, recommend the user verify with the authoritative team/document
"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions from provided context.\n" + conflict_instruction
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        max_tokens=800,
        temperature=0.1
    )
    
    return response.choices[0].message.content
```

### Example Output with Conflict Surfaced

Instead of silently giving the wrong answer, the system should produce something like:

> "I found conflicting information in our documents about the remote work stipend:
> - **[HR Policy Manual, Section 4.3]** states the monthly stipend is $75.
> - **[Benefits FAQ, Updated January 2024]** states the monthly stipend is $100.
>
> I recommend verifying the current amount with HR directly or checking the most recently updated policy document. If the January 2024 FAQ is the most recent, $100 may be the correct current amount."

This is far more useful than confidently citing either wrong amount.

---

## Type 3 — Regional or Jurisdictional Conflicts

The same policy topic has genuinely different rules depending on jurisdiction. "What is the maximum overtime pay?" has different answers for California, New York, and federal employees.

These are not errors — they are legitimate variations. But without jurisdiction-awareness, retrieval returns a mix of regional variants and the LLM produces a confused average.

### Handling Jurisdictional Conflicts

**Metadata-based routing:** Tag every document with its jurisdiction. When a user asks a question, extract the relevant jurisdiction from the query or user context, then filter retrieval accordingly.

```python
async def jurisdiction_aware_retrieval(
    query: str,
    user_context: dict,  # {location, employment_type, etc.}
    retriever,
    llm_client
) -> list[dict]:
    """
    Extract jurisdiction from query context and filter retrieval accordingly.
    """
    
    # Extract jurisdiction from query + user context
    jurisdiction_prompt = f"""Based on this question and user context, determine the relevant jurisdiction.

Question: {query}
User location: {user_context.get('location', 'Unknown')}
User employment type: {user_context.get('employment_type', 'Unknown')}

Return JSON:
{{
    "jurisdiction": "federal" | "california" | "new_york" | "texas" | ... | "unknown",
    "confidence": "high" | "medium" | "low",
    "multiple_jurisdictions": ["list if query spans multiple"]
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": jurisdiction_prompt}],
        response_format={"type": "json_object"},
        max_tokens=100,
        temperature=0.0
    )
    
    import json
    jurisdiction_info = json.loads(response.choices[0].message.content)
    
    jurisdiction = jurisdiction_info.get("jurisdiction", "unknown")
    
    # Build filter
    if jurisdiction != "unknown":
        metadata_filter = {
            "should": [
                {"key": "jurisdiction", "match": {"value": jurisdiction}},
                {"key": "jurisdiction", "match": {"value": "global"}}  # Always include global docs
            ]
        }
    else:
        metadata_filter = None
    
    return await retriever.retrieve(
        query=query,
        metadata_filter=metadata_filter,
        k=10
    )
```

**Explicit multi-jurisdiction queries:** When a user asks "how does our overtime policy differ between California and federal employees?", the system should retrieve content for both jurisdictions and explicitly compare them.

---

## Type 4 — Temporal Conflicts (Outdated Facts)

Some documents contain information that was accurate when written but is now outdated — not because a new version exists, but because the underlying fact changed.

Example: A legal guide written in 2021 citing a specific regulation that was amended in 2023. The 2021 document is not superseded (it may still be useful for other content), but one specific claim in it is now wrong.

### Detection via LLM Temporal Reasoning

```python
async def check_temporal_consistency(
    chunk_text: str,
    chunk_metadata: dict,
    query: str,
    llm_client,
    current_date: str = None
) -> dict:
    """
    Check if a chunk's content may be outdated relative to the current date.
    """
    
    import datetime
    if current_date is None:
        current_date = datetime.date.today().isoformat()
    
    doc_date = chunk_metadata.get("effective_date") or chunk_metadata.get("created_date", "Unknown")
    
    prompt = f"""Check if this content might be outdated.

Current date: {current_date}
Document date: {doc_date}
Query: {query}

Content:
{chunk_text[:800]}

Questions:
1. Does this content contain time-sensitive information (regulations, rates, dates, deadlines)?
2. Could this information have changed since {doc_date}?
3. Are there any statements that are likely no longer accurate?

Return JSON:
{{
    "may_be_outdated": true/false,
    "time_sensitive_claims": ["specific claims that may have changed"],
    "risk_level": "high" | "medium" | "low" | "none",
    "recommendation": "add_disclaimer" | "use_with_caution" | "safe_to_use"
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=300,
        temperature=0.0
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

---

## The Conflict Resolution Pipeline

Putting it all together into a coherent conflict detection and handling pipeline:

```python
class ConflictAwareRAGPipeline:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm = llm_client
    
    async def answer(
        self,
        query: str,
        user_context: dict = None,
        check_conflicts: bool = True
    ) -> dict:
        """
        RAG pipeline with conflict detection and appropriate handling.
        """
        
        # Step 1: Retrieve
        chunks = await self.retriever.retrieve(query, k=10)
        
        if not check_conflicts:
            answer = await generate_standard(query, format_context(chunks), self.llm)
            return {"answer": answer, "conflicts": None}
        
        # Step 2: Check for version conflicts in retrieved chunks
        version_conflicts = detect_version_conflicts_in_results(chunks)
        
        # Step 3: If version conflicts, prefer the most recent active version
        if version_conflicts:
            chunks = self._resolve_version_conflicts(chunks, version_conflicts)
        
        # Step 4: Check for content conflicts
        content_conflict_result = await detect_content_conflicts(
            query=query,
            retrieved_chunks=chunks,
            llm_client=self.llm
        )
        
        # Step 5: Generate with appropriate conflict handling
        if content_conflict_result.get("conflicts_detected"):
            answer = await generate_with_conflict_awareness(
                query=query,
                retrieved_chunks=chunks,
                conflict_result=content_conflict_result,
                llm_client=self.llm
            )
            conflict_summary = {
                "type": "content_conflict",
                "details": content_conflict_result["conflicts"]
            }
        else:
            answer = await generate_standard(query, format_context(chunks), self.llm)
            conflict_summary = None
        
        return {
            "answer": answer,
            "conflicts": conflict_summary,
            "version_conflicts_resolved": len(version_conflicts) > 0,
            "chunks_used": [c["chunk_id"] for c in chunks]
        }
    
    def _resolve_version_conflicts(
        self,
        chunks: list[dict],
        version_conflicts: list[dict]
    ) -> list[dict]:
        """
        When multiple versions of a document are retrieved,
        keep only the most recent active version's chunks.
        """
        
        resolved_chunks = []
        conflicted_family_ids = {
            vc["doc_family_id"] for vc in version_conflicts
        }
        
        # For each conflicted family, find the most recent version
        best_versions = {}
        for conflict in version_conflicts:
            family_id = conflict["doc_family_id"]
            # Sort by effective_date, pick most recent
            sorted_chunks = sorted(
                conflict["chunks"],
                key=lambda c: c["metadata"].get("effective_date", "1900-01-01"),
                reverse=True
            )
            best_versions[family_id] = sorted_chunks[0]["metadata"].get("doc_id")
        
        for chunk in chunks:
            family_id = chunk["metadata"].get("doc_family_id")
            
            if family_id in conflicted_family_ids:
                # Only include chunk if it's from the best version
                if chunk["metadata"].get("doc_id") == best_versions.get(family_id):
                    resolved_chunks.append(chunk)
            else:
                resolved_chunks.append(chunk)
        
        return resolved_chunks
```

---

## Building a Conflict Audit Report

Run a periodic audit across the corpus to proactively identify conflicts before they affect users.

```python
async def run_corpus_conflict_audit(
    vector_db,
    registry,
    llm_client,
    sample_topics: list[str],   # High-level topics to audit
    max_topics: int = 20
) -> dict:
    """
    Proactive audit: find conflicting information in the corpus
    before it reaches users.
    """
    
    import asyncio
    
    audit_results = []
    
    for topic in sample_topics[:max_topics]:
        # Find all chunks about this topic
        topic_embedding = await embed(topic)
        relevant_chunks = await vector_db.search(
            query_vector=topic_embedding,
            limit=10
        )
        
        if len(relevant_chunks) < 2:
            continue
        
        # Check for conflicts within this topic cluster
        conflict_result = await detect_content_conflicts(
            query=topic,
            retrieved_chunks=[
                {"text": r.payload.get("text", ""), "metadata": r.payload}
                for r in relevant_chunks
            ],
            llm_client=llm_client
        )
        
        if conflict_result.get("conflicts_detected"):
            audit_results.append({
                "topic": topic,
                "conflicts": conflict_result["conflicts"],
                "can_reconcile": conflict_result.get("can_be_reconciled"),
                "affected_chunks": [r.id for r in relevant_chunks[:5]]
            })
    
    # Also check for version conflicts across all document families
    all_families = await registry.get_all_document_families()
    version_conflicts_found = []
    
    for family_id in all_families[:100]:  # Sample
        conflicts = await detect_version_conflicts(registry, family_id)
        if conflicts:
            version_conflicts_found.append({
                "doc_family_id": family_id,
                "active_versions": [v.version for v in conflicts]
            })
    
    return {
        "content_conflicts": audit_results,
        "version_conflicts": version_conflicts_found,
        "total_content_conflicts": len(audit_results),
        "total_version_conflicts": len(version_conflicts_found),
        "recommended_actions": _prioritize_conflict_fixes(audit_results, version_conflicts_found)
    }


def _prioritize_conflict_fixes(
    content_conflicts: list[dict],
    version_conflicts: list[dict]
) -> list[dict]:
    """
    Prioritize which conflicts to fix first based on severity and prevalence.
    """
    
    actions = []
    
    # Version conflicts are always high priority — automatic fix is possible
    if version_conflicts:
        actions.append({
            "priority": "P0",
            "type": "version_conflicts",
            "count": len(version_conflicts),
            "action": "Run supersede workflow on all document families with multiple active versions",
            "automated": True
        })
    
    # Content conflicts need human review
    for conflict in content_conflicts:
        actions.append({
            "priority": "P1" if not conflict.get("can_reconcile") else "P2",
            "type": "content_conflict",
            "topic": conflict["topic"],
            "action": (
                "Manual review required — irreconcilable conflict"
                if not conflict.get("can_reconcile")
                else "Add clarifying note or update one document to resolve ambiguity"
            ),
            "automated": False
        })
    
    return sorted(actions, key=lambda x: x["priority"])
```

---

## Summary

- Enterprise document corpora almost always contain conflicts. Pretending otherwise leads to confident wrong answers that erode user trust.
- Four conflict types: version conflicts (old and new versions coexist), content conflicts (different documents contradict each other), jurisdictional conflicts (legitimate regional variations), temporal conflicts (accurate when written, outdated now).
- Version conflicts are best prevented by marking superseded documents during new version publication. Default retrieval filter to `document_status = active`.
- Content conflicts should be detected at retrieval time using LLM-based conflict checking, and surfaced explicitly to users rather than silently resolved.
- When conflicts are detected, generate a response that names both conflicting claims, cites their sources, and recommends verification — do not silently pick one or blend them.
- Jurisdictional conflicts are handled by extracting jurisdiction from user context and filtering retrieval to the appropriate regional documents.
- Run a periodic corpus conflict audit to catch conflicts proactively before they reach users.
- The conflict pipeline: retrieve → check version conflicts (auto-resolve by recency) → check content conflicts → generate with appropriate conflict awareness.

---

## What's Next

Lesson 7.4 covers retrieval accuracy degradation at scale in more depth — specifically embedding drift, query distribution shift effects on retrieval, and how to maintain accuracy as both the corpus and the user base grow.