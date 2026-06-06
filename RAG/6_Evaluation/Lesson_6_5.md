# Lesson 6.5 — Building Evaluation Datasets: Golden Sets, Synthetic QA, and Human Annotation

---

## Why Evaluation Dataset Quality Is the Foundation

Your evaluation is only as good as your evaluation dataset. A poorly constructed evaluation set leads to false confidence — the system scores well on the eval set but fails in production. This is the most common mistake in RAG development.

Three failure modes of bad evaluation datasets:

**Too easy:** Questions are simple factual lookups that any reasonable system handles correctly. The eval set does not expose the hard cases where systems actually differ.

**Not representative:** Questions were written by engineers who know the corpus well. Real users ask different kinds of questions — more vague, more colloquial, with different assumptions about what is in the system.

**Overfitted:** You have looked at the eval set so many times while developing that you have unconsciously optimized for it. The system has effectively "memorized" the evaluation, not improved generally.

A good evaluation dataset is hard to build. It is also one of the highest-leverage investments you can make in your RAG system — a representative, challenging eval set is worth more than any single algorithmic improvement.

---

## The Three Sources of Evaluation Data

### Source 1 — LLM-Generated Synthetic QA

The fastest way to create evaluation data at scale. You use an LLM to generate questions from your actual corpus chunks and provide ground truth answers.

**Advantages:**
- Fast: thousands of pairs in hours.
- No human annotation required.
- Covers all document types in your corpus.
- Consistent quality (no annotator fatigue).

**Disadvantages:**
- Questions look like document content, not real user queries (vocabulary match is too high, making retrieval artificially easy).
- Tends to produce simple factual questions (harder question types require more sophisticated generation).
- Lacks the contextual framing real users bring (user knows their department, their situation, their history with the system).

```python
async def generate_synthetic_eval_dataset(
    corpus_chunks: list[dict],
    llm_client,
    target_size: int = 500,
    question_types: list[str] = None
) -> list[dict]:
    """
    Generate a synthetic evaluation dataset from corpus chunks.
    """
    
    if question_types is None:
        question_types = ["factual", "inferential", "multi_fact", "negative", "comparative"]
    
    # Sample chunks proportionally from different document types
    sampled_chunks = smart_sample_chunks(corpus_chunks, target_size)
    
    eval_pairs = []
    
    for chunk in sampled_chunks:
        # Randomly select a question type for variety
        import random
        q_type = random.choice(question_types)
        
        pairs = await generate_questions_for_chunk(
            chunk=chunk,
            question_type=q_type,
            llm_client=llm_client
        )
        eval_pairs.extend(pairs)
    
    return eval_pairs[:target_size]


async def generate_questions_for_chunk(
    chunk: dict,
    question_type: str,
    llm_client
) -> list[dict]:
    """
    Generate evaluation questions of a specific type from a chunk.
    """
    
    type_instructions = {
        "factual": """Generate a simple factual question that has a specific, 
unambiguous answer in the text. The answer should be a fact, date, number, or name.""",
        
        "inferential": """Generate a question that requires inference from the text — 
the answer is not stated directly but can be reasoned from what is stated.""",
        
        "multi_fact": """Generate a question that requires combining multiple 
pieces of information from the text to answer.""",
        
        "negative": """Generate a question about something that is NOT in the text — 
for testing that the system correctly says 'I don't know' rather than hallucinating.
The question should be plausible given the document's topic.""",
        
        "comparative": """If the text compares two things (policies, options, 
time periods, etc.), generate a question asking to compare them."""
    }
    
    instruction = type_instructions.get(question_type, type_instructions["factual"])
    
    prompt = f"""Generate ONE question and answer pair from this document chunk.

Document: {chunk['metadata'].get('doc_title', 'Unknown')}
Section: {chunk['metadata'].get('heading_path', '')}

Text:
{chunk['text']}

Question type: {question_type}
Instructions: {instruction}

Return JSON:
{{
    "question": "the question",
    "answer": "the expected answer (or null for negative questions)",
    "question_type": "{question_type}",
    "answerable": true/false,
    "answer_spans": ["exact quotes from the text supporting the answer"]
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=400,
        temperature=0.5
    )
    
    import json
    pair = json.loads(response.choices[0].message.content)
    
    # Attach metadata
    pair["source_chunk_id"] = chunk.get("chunk_id")
    pair["source_doc_id"] = chunk["metadata"].get("doc_id")
    pair["relevant_chunk_ids"] = [chunk["chunk_id"]]  # Ground truth: this chunk is relevant
    
    return [pair]


def smart_sample_chunks(
    chunks: list[dict],
    target_size: int
) -> list[dict]:
    """
    Sample chunks proportionally by document type and topic to ensure
    the eval dataset covers the full corpus distribution.
    """
    import random
    from collections import defaultdict
    
    # Group by document type
    by_type = defaultdict(list)
    for chunk in chunks:
        doc_type = chunk["metadata"].get("document_type", "unknown")
        by_type[doc_type].append(chunk)
    
    # Sample proportionally from each type
    sampled = []
    per_type = max(1, target_size // len(by_type))
    
    for doc_type, type_chunks in by_type.items():
        n = min(per_type, len(type_chunks))
        sampled.extend(random.sample(type_chunks, n))
    
    return sampled[:target_size]
```

**Question type distribution for a balanced eval set:**

| Type | Fraction | Why Include |
|---|---|---|
| Simple factual | 30% | Most common user query type |
| Multi-fact/synthesis | 20% | Tests multi-hop retrieval |
| Inferential | 20% | Tests reasoning, not just retrieval |
| Unanswerable (out of scope) | 15% | Tests IDK behavior |
| Comparative | 10% | Tests retrieval of multiple relevant chunks |
| Adversarial | 5% | Tests robustness |

---

### Source 2 — Real User Query Logs

If your system is already deployed, your user query logs are the best source of evaluation data. Real queries are automatically representative of actual usage patterns.

**Collection pipeline:**

```python
class EvalDataCollector:
    """
    Collects and processes user queries for evaluation dataset creation.
    Requires human annotation of answers.
    """
    
    def __init__(self, query_log_store, annotation_store):
        self.query_logs = query_log_store
        self.annotations = annotation_store
    
    async def collect_for_annotation(
        self,
        n_queries: int = 200,
        time_window_days: int = 30,
        stratify_by: list[str] = None
    ) -> list[dict]:
        """
        Select a stratified sample of real queries for human annotation.
        """
        # Fetch recent queries
        recent_queries = await self.query_logs.get_recent(
            days=time_window_days,
            limit=n_queries * 10  # Oversample, then stratify
        )
        
        # Filter out personal/sensitive queries
        filtered = [
            q for q in recent_queries
            if not self._is_sensitive(q["query"])
        ]
        
        # Stratify sample
        stratified = self._stratify_sample(filtered, n_queries, stratify_by)
        
        # Prepare annotation tasks
        annotation_tasks = []
        for query in stratified:
            task = {
                "task_id": generate_id(),
                "query": query["query"],
                "system_answer": query.get("response"),
                "retrieved_chunks": query.get("context_chunks"),
                "annotation_fields": [
                    "reference_answer",
                    "relevant_chunk_ids",
                    "answer_quality_rating",
                    "notes"
                ]
            }
            annotation_tasks.append(task)
        
        return annotation_tasks
    
    def _stratify_sample(
        self,
        queries: list[dict],
        n: int,
        stratify_by: list[str]
    ) -> list[dict]:
        """
        Ensure diverse coverage across query types, lengths, topics.
        """
        import random
        
        # Simple stratification: by query length bucket
        short = [q for q in queries if len(q["query"].split()) <= 5]
        medium = [q for q in queries if 5 < len(q["query"].split()) <= 15]
        long = [q for q in queries if len(q["query"].split()) > 15]
        
        n_short = int(n * 0.3)
        n_medium = int(n * 0.5)
        n_long = n - n_short - n_medium
        
        return (
            random.sample(short, min(n_short, len(short))) +
            random.sample(medium, min(n_medium, len(medium))) +
            random.sample(long, min(n_long, len(long)))
        )
    
    def _is_sensitive(self, query: str) -> bool:
        """Filter out queries that might contain PII or sensitive content."""
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email
        ]
        import re
        return any(re.search(p, query, re.IGNORECASE) for p in sensitive_patterns)
```

### Source 3 — Expert-Written Questions

Domain experts write questions based on their knowledge of what real users ask and what the system should be able to answer. This is the highest quality source but requires expert time.

**Expert annotation guidelines:**

```markdown
## Annotation Guidelines for RAG Evaluation

### Writing Good Questions

1. Write questions as a real user would ask them — use natural language, not document language
2. Include a mix of:
   - Specific factual lookups ("What is the maximum reimbursement for business travel meals?")
   - Process questions ("How do I submit an expense report for international travel?")
   - Policy interpretation questions ("Does the work-from-home policy apply to contractors?")
   - Edge cases ("Can I use parental leave if I adopt a child over 5 years old?")
   - Out-of-scope questions ("What is the salary band for Senior Engineers?") — mark these as unanswerable

3. For each question, provide:
   - The ideal answer (what a perfect response would say)
   - The specific document section(s) that contain the answer
   - Whether the answer is definitively in the corpus or requires saying IDK
   - Difficulty rating (1=trivial, 3=moderate, 5=requires deep domain knowledge)

### What Makes a Bad Question

- Questions that could only be answered by someone who wrote the documents
- Questions about specific individuals (use roles/departments instead)
- Questions with ambiguous answers where different reasonable people would disagree
- Questions that are trivially easy (the answer is the first sentence of the obvious document)
```

---

## Annotation Tooling

For questions requiring human annotation, you need tooling that makes the process efficient and consistent.

```python
# Simple annotation interface using a spreadsheet-compatible format

def create_annotation_template(tasks: list[dict]) -> str:
    """
    Create a CSV template for human annotation.
    """
    import csv
    import io
    
    output = io.StringIO()
    fieldnames = [
        "task_id",
        "query",
        "system_answer",  # Show annotator what the system said
        "reference_answer",  # Annotator fills this in
        "relevant_chunk_ids",  # Annotator confirms/corrects
        "answerable",  # true/false
        "difficulty",  # 1-5
        "answer_quality",  # 1-5 rating of system_answer
        "notes"
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for task in tasks:
        writer.writerow({
            "task_id": task["task_id"],
            "query": task["query"],
            "system_answer": task.get("system_answer", ""),
            "reference_answer": "",  # To be filled
            "relevant_chunk_ids": "",  # To be filled
            "answerable": "",  # To be filled
            "difficulty": "",  # To be filled
            "answer_quality": "",  # To be filled
            "notes": ""
        })
    
    return output.getvalue()
```

For larger annotation projects, dedicated tools like Label Studio, Argilla, or Prodigy provide better interfaces and inter-annotator agreement tracking.

---

## Inter-Annotator Agreement

When multiple humans annotate the same queries, measure how often they agree. Low agreement means the annotation guidelines are ambiguous or the questions are genuinely subjective.

```python
def compute_inter_annotator_agreement(
    annotations_a: list[dict],
    annotations_b: list[dict],
    metric: str = "cohen_kappa"
) -> float:
    """
    Compute agreement between two annotators on the same tasks.
    """
    from sklearn.metrics import cohen_kappa_score
    import numpy as np
    
    # Align by task_id
    a_by_id = {a["task_id"]: a for a in annotations_a}
    b_by_id = {b["task_id"]: b for b in annotations_b}
    
    shared_ids = set(a_by_id.keys()) & set(b_by_id.keys())
    
    # Compare answerable labels (binary)
    a_labels = [1 if a_by_id[tid]["answerable"] else 0 for tid in shared_ids]
    b_labels = [1 if b_by_id[tid]["answerable"] else 0 for tid in shared_ids]
    
    kappa = cohen_kappa_score(a_labels, b_labels)
    
    # Compare answer quality ratings (ordinal)
    a_quality = [int(a_by_id[tid].get("answer_quality", 3)) for tid in shared_ids]
    b_quality = [int(b_by_id[tid].get("answer_quality", 3)) for tid in shared_ids]
    
    quality_agreement = np.mean([a == b for a, b in zip(a_quality, b_quality)])
    
    return {
        "cohen_kappa_answerable": kappa,
        "exact_agreement_quality": quality_agreement,
        "n_shared_tasks": len(shared_ids),
        "interpretation": (
            "Substantial agreement" if kappa > 0.6
            else "Moderate agreement" if kappa > 0.4
            else "Fair agreement — review guidelines"
        )
    }
```

Target: Cohen's Kappa > 0.6 for answerable/unanswerable labels. Below 0.4 means your annotation guidelines need revision.

---

## Negative Examples: Testing IDK Behavior

Unanswerable questions are as important as answerable ones. A system that confidently answers everything is worse than one that correctly says "I don't know" when the answer is not in the corpus.

**Sources of negative examples:**

1. Questions about topics adjacent to but outside your corpus scope.
2. Questions about events or policies that will happen in the future.
3. Questions about specific individuals or private information not in documents.
4. Questions about competing organizations or external entities.
5. Deliberately ambiguous questions without a clear answer.

```python
async def generate_negative_examples(
    corpus_metadata: dict,  # What topics the corpus covers
    llm_client,
    n: int = 50
) -> list[dict]:
    """
    Generate unanswerable questions that test IDK behavior.
    """
    
    prompt = f"""Generate {n} questions that cannot be answered from the following 
document corpus but are plausible questions a user might ask.

Corpus covers: {corpus_metadata['topics']}
Corpus does NOT cover: {corpus_metadata.get('exclusions', 'external data, future events, personal information')}

Generate questions that:
- Are reasonable for a user to ask
- Seem like they might be in the corpus
- Cannot actually be answered from the corpus

Return JSON array:
[
    {{
        "question": "question text",
        "why_unanswerable": "brief explanation",
        "expected_response_type": "idk | partial | redirect"
    }}
]"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=1000,
        temperature=0.7
    )
    
    import json
    raw = json.loads(response.choices[0].message.content)
    negatives = raw if isinstance(raw, list) else raw.get("questions", [])
    
    for neg in negatives:
        neg["answerable"] = False
        neg["relevant_chunk_ids"] = []
        neg["reference_answer"] = None
    
    return negatives
```

---

## Adversarial Examples

Adversarial examples specifically test robustness to tricky inputs:

```python
ADVERSARIAL_TEMPLATES = [
    # Misleading surface similarity — sounds relevant but is not
    "What is the policy for {topic_adjacent_to_actual}?",
    
    # Ambiguous reference — "the policy" could mean multiple things
    "What does the policy say about {ambiguous_topic}?",
    
    # Temporal trick — asks about something that changed
    "What was the {fact} before the 2024 update?",
    
    # Double negative — "not prohibited" vs "permitted"
    "Is {action} not prohibited under our guidelines?",
    
    # Compound with one false premise — "since X is true, what about Y?"
    "Given that {slightly_wrong_premise}, what is the policy on {real_topic}?",
]
```

Adversarial examples with false premises are particularly valuable — they test whether the LLM will correct the false premise or compound the error.

---

## Evaluating Your Evaluation Dataset

Before using an evaluation dataset, validate it:

```python
def validate_eval_dataset(
    eval_dataset: list[dict]
) -> dict:
    """
    Check evaluation dataset quality before using it.
    """
    total = len(eval_dataset)
    
    issues = []
    
    # Check for required fields
    required_fields = ["question", "reference_answer", "answerable"]
    for i, item in enumerate(eval_dataset):
        for field in required_fields:
            if field not in item or item[field] is None:
                if item.get("answerable", True):  # Only require answer for answerable questions
                    issues.append(f"Item {i}: missing required field '{field}'")
    
    # Check distribution
    answerable_count = sum(1 for item in eval_dataset if item.get("answerable", True))
    unanswerable_count = total - answerable_count
    
    # Check question type distribution
    type_counts = {}
    for item in eval_dataset:
        q_type = item.get("question_type", "unspecified")
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    # Check for duplicate questions
    questions = [item["question"].lower().strip() for item in eval_dataset]
    unique_questions = set(questions)
    duplicate_count = total - len(unique_questions)
    
    # Check answer length distribution
    import numpy as np
    answer_lengths = [
        len(item.get("reference_answer", "").split())
        for item in eval_dataset
        if item.get("answerable") and item.get("reference_answer")
    ]
    
    warnings = []
    if unanswerable_count / total < 0.10:
        warnings.append("Less than 10% unanswerable questions — IDK behavior undertested")
    if unanswerable_count / total > 0.40:
        warnings.append("More than 40% unanswerable — may be too many negatives")
    if duplicate_count > 0:
        warnings.append(f"{duplicate_count} duplicate questions detected")
    if "factual" in type_counts and type_counts.get("factual", 0) / total > 0.6:
        warnings.append("Over 60% factual questions — may not test synthesis/reasoning")
    
    return {
        "total": total,
        "answerable": answerable_count,
        "unanswerable": unanswerable_count,
        "unanswerable_rate": unanswerable_count / total,
        "question_type_distribution": type_counts,
        "duplicate_questions": duplicate_count,
        "avg_answer_length": float(np.mean(answer_lengths)) if answer_lengths else 0,
        "issues": issues,
        "warnings": warnings,
        "is_valid": len(issues) == 0
    }
```

---

## Keeping the Evaluation Dataset Fresh

An evaluation dataset becomes stale when:
- The corpus changes significantly (new document types, major policy updates).
- User query distribution shifts (new use cases, new user segments).
- The system has improved so much that most eval questions are trivially easy.

**Refresh strategy:**

```python
def plan_dataset_refresh(
    current_dataset: list[dict],
    current_system_scores: dict,
    production_query_sample: list[str]
) -> dict:
    """
    Determine whether and how to refresh the evaluation dataset.
    """
    
    # Indicators that refresh is needed
    indicators = []
    
    # 1. Performance is too high (dataset too easy)
    if current_system_scores.get("hit_rate@5", 0) > 0.97:
        indicators.append("hit_rate too high — dataset may be too easy")
    if current_system_scores.get("faithfulness", 0) > 0.97:
        indicators.append("faithfulness too high — may have overfit to dataset")
    
    # 2. Production queries look different from eval queries
    # Compare embedding distributions of eval vs production queries
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    eval_queries = [item["question"] for item in current_dataset]
    eval_embeddings = model.encode(eval_queries)
    prod_embeddings = model.encode(production_query_sample[:200])
    
    # Simple divergence: cosine similarity of centroids
    eval_centroid = eval_embeddings.mean(axis=0)
    prod_centroid = prod_embeddings.mean(axis=0)
    centroid_similarity = float(np.dot(
        eval_centroid / np.linalg.norm(eval_centroid),
        prod_centroid / np.linalg.norm(prod_centroid)
    ))
    
    if centroid_similarity < 0.80:
        indicators.append(f"Query distribution diverged (similarity={centroid_similarity:.2f})")
    
    return {
        "refresh_recommended": len(indicators) > 0,
        "indicators": indicators,
        "recommended_additions": 100,  # Add this many new questions
        "recommended_replacements": 50,  # Replace this many easy/stale ones
        "keep_existing": len(indicators) < 2  # Keep most if only minor refresh needed
    }
```

---

## Summary

- Evaluation dataset quality determines the validity of all evaluation results. A bad dataset creates false confidence.
- Three sources: LLM-generated synthetic QA (fast, scalable, vocabulary-biased), real user query logs (most representative, requires annotation), expert-written questions (highest quality, expensive).
- A balanced eval set should include ~30% factual, ~20% inferential, ~20% multi-fact, ~15% unanswerable, ~10% comparative, ~5% adversarial questions.
- Negative examples (unanswerable questions) are critical — a system that always answers is more dangerous than one that correctly says IDK.
- Adversarial examples with false premises test whether the LLM corrects errors or compounds them.
- Validate the dataset before use: check required fields, distribution, duplicates, and answer length.
- Keep the dataset fresh — refresh when system performance is too high (dataset too easy), when query distribution shifts, or after major corpus changes.
- Track inter-annotator agreement (Cohen's Kappa > 0.6) to ensure annotation quality and guideline clarity.

---

## What's Next

Lesson 6.6 covers online evaluation — A/B testing for RAG systems, implicit feedback signals, and how to run controlled experiments to validate that pipeline improvements actually help real users.