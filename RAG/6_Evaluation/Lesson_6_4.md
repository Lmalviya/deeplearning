# Lesson 6.3 — Generation Metrics in Depth: Exact Match, F1, BLEU, ROUGE, METEOR, BERTScore, Semantic Similarity

---

## The Challenge of Evaluating Generated Text

Measuring retrieval quality is relatively clean: either the right chunk was retrieved or it was not. Measuring generation quality is fundamentally harder because there is no single "correct" answer for most questions.

"What is the parental leave policy?" might be correctly answered as:
- "Employees are entitled to 16 weeks of paid leave."
- "The policy provides 16 weeks of fully paid parental leave for all employees."
- "Our parental leave program offers 4 months of leave at full salary."

All three are correct, but they share very few words. A metric that requires exact string matching would score all three as wrong against any given reference answer.

This is the core challenge of generation evaluation: measuring quality without requiring verbatim reproduction of a reference answer.

Different metrics make different trade-offs between computational cost, linguistic sophistication, and correlation with human judgment. This lesson covers each metric, its strengths, its failure modes, and when it is appropriate for RAG evaluation.

---

## Metric 1 — Exact Match (EM)

**Definition:** The generated answer exactly matches the reference answer (after normalization).

```
EM = 1 if normalize(generated) == normalize(reference) else 0
```

Normalization typically involves: lowercase, strip punctuation, strip articles ("the", "a", "an"), normalize whitespace.

```python
import re
import string

def normalize_answer(text: str) -> str:
    """Standard normalization for Exact Match evaluation."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()

def exact_match(generated: str, reference: str) -> float:
    return 1.0 if normalize_answer(generated) == normalize_answer(reference) else 0.0

# Example
print(exact_match("The policy is 16 weeks.", "16 weeks"))  # → 0.0 (different after norm)
print(exact_match("16 weeks", "16 weeks"))                  # → 1.0
```

**When EM is appropriate:**
- Questions with short, unambiguous factual answers: dates, numbers, names, short phrases.
- Questions from structured data: "What is the revenue?" → "$4.2B" is the answer.
- Yes/no questions where the answer is definitively one of two options.

**When EM fails:**
- Any question with natural language answers. "What are the termination conditions?" cannot have a single reference answer that all correct responses must exactly match.
- When the answer is a fact stated differently than the reference.
- When the answer is a summary or explanation.

**EM in RAG:** Useful for evaluating factual extraction tasks (extractive QA) but not for conversational or explanatory responses. Do not use EM as your primary metric for a conversational RAG system.

---

## Metric 2 — Token-Level F1

**Definition:** The overlap of tokens between the generated answer and the reference answer, measured as F1 (harmonic mean of precision and recall).

```
Token Precision = |tokens(generated) ∩ tokens(reference)| / |tokens(generated)|
Token Recall    = |tokens(generated) ∩ tokens(reference)| / |tokens(reference)|
Token F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

```python
from collections import Counter

def token_f1(generated: str, reference: str) -> float:
    """
    Compute token-level F1 between generated and reference answers.
    Standard approach used in SQuAD evaluation.
    """
    gen_tokens = normalize_answer(generated).split()
    ref_tokens = normalize_answer(reference).split()
    
    # Token overlap
    gen_counter = Counter(gen_tokens)
    ref_counter = Counter(ref_tokens)
    
    # Intersection: for each token, min of count in gen and ref
    common = gen_counter & ref_counter
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(gen_tokens)
    recall = num_common / len(ref_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Example
gen = "The employee is entitled to 16 weeks of parental leave"
ref = "Employees get 16 weeks of paid parental leave"
print(token_f1(gen, ref))  # Moderate F1 — shares "16", "weeks", "parental", "leave"
```

**When F1 is appropriate:**
- Extractive QA where the answer is a span from the document.
- Short factual answers where partial overlap is meaningful.
- Comparison baselines (it is widely reported in QA benchmarks).

**When F1 fails:**
- Long answers where word overlap is a poor proxy for semantic correctness.
- Answers with synonyms ("payment" vs "compensation" vs "remuneration" — zero overlap despite same meaning).
- Answers with different structure ("16 weeks" vs "four months" — no token overlap, same fact).

**F1 vs. EM:** F1 is always at least as high as EM and often much higher. Use F1 when you want to reward partial correctness, EM when you require complete correctness.

---

## Metric 3 — BLEU (Bilingual Evaluation Understudy)

**Definition:** Measures n-gram precision between generated text and one or more reference texts. Originally designed for machine translation.

```
BLEU = BP × exp(Σ wₙ × log pₙ)

Where:
- pₙ = modified n-gram precision for n-grams of order n
- wₙ = weight for n-gram order n (typically uniform: 1/4 each for n=1,2,3,4)
- BP = brevity penalty (penalizes outputs shorter than references)
```

Modified n-gram precision clips the count of each n-gram in the generated text by its maximum count in any reference:

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

def compute_bleu(generated: str, references: list[str]) -> float:
    """
    Compute BLEU score for a generated text against multiple references.
    """
    # Tokenize
    gen_tokens = generated.lower().split()
    ref_tokens_list = [ref.lower().split() for ref in references]
    
    # BLEU with smoothing (prevents zero for short texts)
    smoother = SmoothingFunction().method1
    
    score = sentence_bleu(
        references=ref_tokens_list,
        hypothesis=gen_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4
        smoothing_function=smoother
    )
    
    return score

# Example
gen = "The policy provides sixteen weeks of parental leave to all employees"
refs = [
    "Employees are entitled to 16 weeks of parental leave",
    "The parental leave policy grants 16 weeks to eligible employees"
]
print(compute_bleu(gen, refs))  # Low score — "sixteen" vs "16", different structure
```

**When BLEU is appropriate:**
- Machine translation and text generation tasks where surface-form similarity matters.
- As a historical baseline for comparison with other systems (many benchmarks report BLEU).
- When you have multiple valid reference answers for the same question.

**When BLEU fails for RAG:**
- BLEU is notoriously poor at measuring semantic correctness in RAG. "The penalty is 10 dollars" scores 0 against "The fine is $10" despite conveying the same information.
- It penalizes paraphrase and penalizes responses that are longer than the reference even when the extra content is correct.
- It requires multiple reference answers to work well — a single reference gives a very noisy signal.
- BLEU was designed for translation, not Q&A. Its correlation with human judgment is weak for RAG outputs.

**Recommendation:** Do not use BLEU as a primary metric for RAG evaluation. It is mentioned here because interviewers may ask about it, and you should know why it is inadequate.

---

## Metric 4 — ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Definition:** A family of recall-oriented n-gram overlap metrics, originally designed for summarization evaluation.

Three main variants:

**ROUGE-1:** Unigram (word) recall/precision/F1 between generated and reference.

**ROUGE-2:** Bigram (2-consecutive-word) recall/precision/F1.

**ROUGE-L:** Longest Common Subsequence (LCS) recall/precision/F1. More flexible than n-gram matching — handles word reordering better.

```python
from rouge_score import rouge_scorer

def compute_rouge(generated: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True  # Stem words for better matching (run → running, runs)
    )
    
    scores = scorer.score(reference, generated)
    
    return {
        "rouge1_f1": scores['rouge1'].fmeasure,
        "rouge1_precision": scores['rouge1'].precision,
        "rouge1_recall": scores['rouge1'].recall,
        "rouge2_f1": scores['rouge2'].fmeasure,
        "rougeL_f1": scores['rougeL'].fmeasure
    }

# Example
gen = "Employees can take 16 weeks of parental leave which is fully paid"
ref = "The parental leave policy gives employees 16 weeks of paid leave"
scores = compute_rouge(gen, ref)
print(scores)
# rouge1: decent (shares many words)
# rouge2: lower (fewer shared bigrams)
# rougeL: moderate (shares subsequence "parental leave" and "16 weeks")
```

**When ROUGE is appropriate:**
- Summarization evaluation where coverage of key facts matters (ROUGE-L is common).
- When you have multiple references and want a flexible overlap measure.
- As a secondary metric alongside semantic similarity metrics.

**When ROUGE fails for RAG:**
- Same synonym problem as BLEU: different words, same meaning → low ROUGE.
- Does not distinguish factually correct summaries from plausible but wrong ones.
- ROUGE-1 can be gamed by repeating source document words.
- Better than BLEU for RAG but still primarily surface-level.

**ROUGE in practice:** If you must use n-gram metrics, ROUGE-L is the most robust (LCS handles word order flexibility). Report alongside semantic similarity metrics.

---

## Metric 5 — METEOR (Metric for Evaluation of Translation with Explicit ORdering)

**Definition:** Improves on BLEU by including stemming, synonym matching, and an ordering penalty.

METEOR matches generated tokens to reference tokens using:
1. Exact match
2. Stem match (run = running = runs)
3. Synonym match (using WordNet: car = automobile)
4. Paraphrase match

Then computes an F-mean (9× recall + precision) with a fragmentation penalty for non-contiguous matches.

```python
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet', quiet=True)

def compute_meteor(generated: str, references: list[str]) -> float:
    """Compute METEOR score."""
    gen_tokens = generated.split()
    ref_tokens_list = [ref.split() for ref in references]
    
    # METEOR takes the best match across references
    scores = [meteor_score([ref], gen_tokens) for ref in ref_tokens_list]
    return max(scores)

# Example
gen = "Workers receive four months of paid family leave"
ref = "Employees get 16 weeks of parental leave which is fully compensated"
print(compute_meteor(gen, [ref]))
# Better than BLEU because: "workers" matches "employees" via synonym, 
# "family" relates to "parental", "compensated" relates to "paid"
```

**When METEOR is appropriate:**
- When synonym handling matters and you want better coverage than BLEU/ROUGE.
- As an intermediate metric between surface-overlap and semantic similarity.

**Limitation:** WordNet synonym coverage is limited and English-only. Does not handle domain-specific synonyms well.

---

## Metric 6 — BERTScore

**Definition:** Computes semantic similarity between generated and reference text using contextual BERT embeddings.

Instead of n-gram overlap, BERTScore embeds each token in context and finds the best matching token in the other text using cosine similarity.

```
For each token in generated text:
    Find the most similar token in reference text (cosine similarity)
    
Precision: average of best reference matches for each generated token
Recall: average of best generated matches for each reference token
F1: harmonic mean of precision and recall
```

```python
from bert_score import score as bert_score_fn

def compute_bertscore(
    generated_texts: list[str],
    reference_texts: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli"
) -> dict:
    """
    Compute BERTScore for a batch of generated/reference pairs.
    deberta-xlarge-mnli is the recommended model for BERTScore.
    """
    P, R, F1 = bert_score_fn(
        cands=generated_texts,
        refs=reference_texts,
        model_type=model_type,
        lang="en",
        rescale_with_baseline=True  # Normalize scores to be more interpretable
    )
    
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
        "per_example_f1": F1.tolist()
    }

# Example (batched)
gens = ["Employees get 16 weeks of paid parental leave"]
refs = ["Workers receive four months of family leave which is fully compensated"]
scores = compute_bertscore(gens, refs)
print(f"BERTScore F1: {scores['f1']:.3f}")
# Higher than n-gram metrics — captures that "16 weeks" ≈ "four months" 
# and "paid" ≈ "compensated" in embedding space
```

**Why BERTScore is better than n-gram metrics for RAG:**
- Captures semantic similarity, not just surface overlap.
- "Car" and "automobile" get high similarity even though no n-gram overlap.
- "16 weeks" and "four months" get high similarity because their contextual embeddings are close.
- Handles paraphrase well.

**When BERTScore is appropriate:**
- The single best off-the-shelf metric for measuring semantic correctness of generated text.
- When you need more than surface overlap but do not have an LLM evaluation budget.
- As a primary metric for measuring generation quality in RAG systems.

**Limitations:**
- Computationally expensive (BERT forward passes for every evaluation).
- Still measures similarity to a reference — if your reference is poor, BERTScore is a poor signal.
- Does not measure faithfulness to retrieved context, only similarity to a reference answer.

---

## Metric 7 — Semantic Similarity

**Definition:** Embed both generated answer and reference answer using a sentence embedding model, compute cosine similarity.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSimilarityScorer:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
    
    def score(self, generated: str, reference: str) -> float:
        """Compute cosine similarity between generated and reference embeddings."""
        embeddings = self.model.encode([generated, reference], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))
    
    def score_batch(self, generated: list[str], references: list[str]) -> list[float]:
        """Score multiple pairs efficiently."""
        all_texts = generated + references
        all_embeddings = self.model.encode(all_texts, normalize_embeddings=True)
        
        gen_embeddings = all_embeddings[:len(generated)]
        ref_embeddings = all_embeddings[len(generated):]
        
        # Pairwise cosine similarity (diagonal of the matrix)
        return [float(np.dot(g, r)) for g, r in zip(gen_embeddings, ref_embeddings)]

scorer = SemanticSimilarityScorer()

gen = "The penalty for late payment is $500 per month"
ref = "Late payment incurs a monthly fine of five hundred dollars"
print(scorer.score(gen, ref))  # High similarity despite no n-gram overlap
```

**When semantic similarity is appropriate:**
- Quick, lightweight semantic correctness measurement.
- When you want a coarser signal than BERTScore without the compute cost.
- As a threshold filter: responses below 0.7 similarity are likely wrong.

**Limitations:**
- Sentence embedding models pool the entire text into one vector. For long texts, important specific details get diluted.
- Less sensitive to factual errors than BERTScore — "The penalty is $500" and "The penalty is $5000" may have high semantic similarity despite different facts.
- Does not capture faithfulness to retrieved context.

---

## Choosing Metrics for RAG Evaluation

The right metric depends on what you are measuring:

**For factual extraction (short answers, specific values):**
- Exact Match as the primary metric.
- Token F1 as the secondary metric.
- Ignore BLEU/ROUGE/BERTScore for this use case.

**For explanatory answers (medium-length prose):**
- BERTScore F1 as the primary metric.
- ROUGE-L as a secondary metric.
- Semantic similarity as a fast sanity check.

**For long summaries and reports:**
- ROUGE-L for coverage measurement.
- BERTScore for semantic correctness.
- Human evaluation as the gold standard (no automated metric handles long summaries reliably).

**For RAG-specific evaluation (faithfulness, grounding):**
- The metrics above measure similarity to a reference answer, not faithfulness to retrieved context. This requires the RAG-specific metrics covered in Lesson 6.4 (RAGAS: faithfulness, context precision, context recall, answer relevancy).

---

## Building the Generation Evaluation Pipeline

```python
class GenerationEvaluator:
    def __init__(self, use_bertscore: bool = True):
        self.semantic_scorer = SemanticSimilarityScorer()
        self.use_bertscore = use_bertscore
        
        if use_bertscore:
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(
                model_type="microsoft/deberta-xlarge-mnli",
                rescale_with_baseline=True
            )
    
    def evaluate_batch(
        self,
        generated_answers: list[str],
        reference_answers: list[str],
        query_types: list[str] = None  # "factual", "explanatory", "summary"
    ) -> dict:
        """
        Evaluate a batch of generated answers against references.
        """
        assert len(generated_answers) == len(reference_answers)
        
        n = len(generated_answers)
        
        # Compute all metrics
        em_scores = [
            exact_match(g, r)
            for g, r in zip(generated_answers, reference_answers)
        ]
        
        f1_scores = [
            token_f1(g, r)
            for g, r in zip(generated_answers, reference_answers)
        ]
        
        rouge_scores = [
            compute_rouge(g, r)
            for g, r in zip(generated_answers, reference_answers)
        ]
        
        semantic_scores = self.semantic_scorer.score_batch(
            generated_answers, reference_answers
        )
        
        results = {
            "exact_match": float(np.mean(em_scores)),
            "token_f1": float(np.mean(f1_scores)),
            "rouge1_f1": float(np.mean([r["rouge1_f1"] for r in rouge_scores])),
            "rougeL_f1": float(np.mean([r["rougeL_f1"] for r in rouge_scores])),
            "semantic_similarity": float(np.mean(semantic_scores)),
            "n_evaluated": n
        }
        
        if self.use_bertscore:
            P, R, F1 = self.bert_scorer.score(generated_answers, reference_answers)
            results["bertscore_f1"] = float(F1.mean())
        
        # Per-example analysis for the worst performers
        combined_scores = [
            (semantic_scores[i] + f1_scores[i]) / 2
            for i in range(n)
        ]
        
        worst_idx = sorted(range(n), key=lambda i: combined_scores[i])[:5]
        results["worst_examples"] = [
            {
                "generated": generated_answers[i][:200],
                "reference": reference_answers[i][:200],
                "semantic_similarity": semantic_scores[i],
                "token_f1": f1_scores[i]
            }
            for i in worst_idx
        ]
        
        return results
```

---

## The Reference Answer Problem

All the metrics above require reference answers. For RAG systems, creating high-quality reference answers at scale is expensive. Strategies:

**LLM-generated references:** Ask a capable LLM to generate reference answers by reading the relevant chunks. Fast and scales. Risk: the LLM may generate answers that are factually different from what the system produces (even if both are correct), penalizing correct-but-different responses.

**Expert-written references:** Domain experts write ground truth answers. Highest quality but expensive and slow.

**Multiple references:** The best mitigation for the evaluation brittleness of single-reference metrics. Provide 3-5 valid reference answers per question. BLEU and ROUGE both accept multiple references.

**LLM-as-judge (next section):** Use an LLM to directly evaluate whether the generated answer is correct, without requiring a fixed reference answer. This is often better than reference-based metrics for RAG.

---

## LLM-as-Judge for Generation Evaluation

For RAG systems, LLM-based evaluation often outperforms reference-based metrics because it can reason about correctness without requiring exact or near-exact match to a reference.

```python
async def llm_evaluate_answer(
    query: str,
    generated_answer: str,
    reference_answer: str,
    retrieved_context: str,
    llm_client
) -> dict:
    """
    Use an LLM to evaluate the quality of a generated answer.
    """
    
    prompt = f"""Evaluate this AI-generated answer to a question.

Question: {query}

Reference answer (correct answer): {reference_answer}

AI-generated answer: {generated_answer}

Retrieved context used: {retrieved_context[:1000]}

Evaluate on these dimensions (score 1-5 each):
1. Correctness: Does the generated answer convey the same correct information as the reference?
2. Completeness: Does it cover all the key points from the reference?
3. Faithfulness: Is it grounded in the retrieved context?
4. Conciseness: Is it appropriately concise without unnecessary information?

Return JSON:
{{
    "correctness": 1-5,
    "completeness": 1-5,
    "faithfulness": 1-5,
    "conciseness": 1-5,
    "overall": 1-5,
    "notes": "brief explanation of any major issues"
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=300,
        temperature=0.0
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

LLM-as-judge correlates better with human judgment than most automated metrics for RAG evaluation tasks. The downside is cost — each evaluation requires an LLM API call. Use it on a sample (10-20% of your evaluation set) rather than every query.

---

## Summary

- Exact Match: strict equality after normalization. Best for short, unambiguous factual answers. Too strict for most RAG responses.
- Token F1: token overlap as F1. Better than EM for partial credit, still sensitive to synonym differences.
- BLEU: n-gram precision with brevity penalty. Designed for translation, poor for RAG. Know it exists, do not use it as a primary RAG metric.
- ROUGE: n-gram recall-oriented metrics. ROUGE-L (LCS-based) is the most useful variant for RAG summarization evaluation.
- METEOR: extends n-gram matching with stemming and synonyms via WordNet. Better than BLEU/ROUGE but limited synonym coverage.
- BERTScore: contextual token embedding similarity. Best off-the-shelf automated metric for semantic correctness in RAG. Computationally expensive.
- Semantic Similarity: sentence embedding cosine similarity. Fast, coarser than BERTScore. Good as a sanity check threshold.
- For RAG: BERTScore + semantic similarity for explanatory answers; Exact Match + F1 for factual extraction; LLM-as-judge for highest-quality evaluation on a sample.
- None of these metrics directly measure faithfulness to retrieved context — that requires RAG-specific metrics covered in Lesson 6.4.

---

## What's Next

Lesson 6.4 covers RAG-specific metrics — faithfulness, answer relevancy, context precision, context recall, and answer correctness — as implemented in the RAGAS framework and its alternatives.