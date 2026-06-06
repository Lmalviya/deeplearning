# Lesson A.8 — Embedding Model Fine-Tuning: Full Pipeline [OPTIONAL — Cover Later]

> **Tag: OPTIONAL** — This lesson covers the end-to-end fine-tuning workflow in production depth. Skip if your current priority is interview preparation. Come back when you actually need to fine-tune a model for a production system.

---

## When Fine-Tuning Is Actually Worth It

Before building a fine-tuning pipeline, confirm you actually need one. Fine-tuning adds infrastructure complexity and ongoing maintenance cost. It is worth it when:

- Domain evaluation (Lesson A.7) shows recall@10 below 0.80 on your domain with the best general model.
- You have domain vocabulary that general models represent poorly (verified via OOV analysis from Lesson 7.4).
- You have sufficient training data: minimum 1,000 pairs, ideally 10,000+.
- The quality improvement justifies the compute cost and maintenance overhead.

If domain recall@10 is already above 0.85 with a general model, fine-tuning rarely gets you more than 3-5% improvement. Usually not worth it.

---

## Step 1 — Training Data Construction at Scale

### Source 1 — Existing User Interactions (Best)

```python
async def extract_training_pairs_from_logs(
    query_logs: list[dict],
    feedback_store,
    min_confidence: float = 0.8
) -> list[dict]:
    """
    Extract (query, positive_chunk) pairs from production logs
    where user feedback indicates a good answer.
    """
    
    training_pairs = []
    
    for log in query_logs:
        query = log["query"]
        retrieved_chunks = log.get("retrieved_chunks", [])
        
        if not retrieved_chunks:
            continue
        
        # High-confidence positive: user gave thumbs up AND top chunk was used
        if (log.get("user_feedback") == "thumbs_up" and 
            log.get("answer_was_grounded") == True):
            
            top_chunk = retrieved_chunks[0]
            training_pairs.append({
                "query": query,
                "positive": top_chunk["text"],
                "source": "user_feedback_positive"
            })
        
        # Negative signal: user reformulated (previous answer was bad)
        elif log.get("was_reformulation") == True:
            previous_log = log.get("previous_query_log")
            if previous_log and previous_log.get("retrieved_chunks"):
                # The previous retrieval was wrong
                bad_chunk = previous_log["retrieved_chunks"][0]
                # This chunk is a hard negative for the CURRENT query
                training_pairs.append({
                    "query": query,
                    "hard_negative": bad_chunk["text"],
                    "source": "reformulation_negative"
                })
    
    return training_pairs
```

### Source 2 — LLM-Generated Synthetic Pairs at Scale

```python
import asyncio
from asyncio import Semaphore

async def generate_training_data_at_scale(
    corpus_chunks: list[dict],
    llm_client,
    target_pairs: int = 10000,
    max_concurrent: int = 20
) -> list[dict]:
    """
    Generate synthetic training pairs from corpus chunks at scale.
    Uses semaphore to control concurrency (API rate limits).
    """
    
    semaphore = Semaphore(max_concurrent)
    
    async def generate_for_chunk(chunk: dict) -> list[dict]:
        async with semaphore:
            prompt = f"""Generate 3 diverse queries that this passage would answer.
Include: 1 simple factual query, 1 inferential query, 1 multi-concept query.

Passage: {chunk['text'][:600]}

Return JSON:
{{
    "pairs": [
        {{"query": "...", "type": "factual"}},
        {{"query": "...", "type": "inferential"}},
        {{"query": "...", "type": "multi_concept"}}
    ]
}}"""
            
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=250,
                temperature=0.5
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return [
                {
                    "query": pair["query"],
                    "positive": chunk["text"],
                    "chunk_id": chunk["chunk_id"],
                    "query_type": pair["type"],
                    "source": "synthetic"
                }
                for pair in result.get("pairs", [])
            ]
    
    # Sample chunks proportionally by document type
    n_per_chunk = max(1, target_pairs // len(corpus_chunks))
    selected_chunks = corpus_chunks[:target_pairs // 3]  # Assume 3 pairs per chunk
    
    tasks = [generate_for_chunk(chunk) for chunk in selected_chunks]
    
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    training_pairs = []
    for result in all_results:
        if isinstance(result, Exception):
            continue  # Skip failed generations
        training_pairs.extend(result)
    
    return training_pairs[:target_pairs]
```

---

## Step 2 — Hard Negative Mining Pipeline

Hard negatives are the most important ingredient. This is a multi-stage pipeline:

```python
class HardNegativeMiner:
    """
    Automated pipeline for mining hard negatives.
    Uses the current production model to find challenging negatives.
    """
    
    def __init__(self, retriever, embedding_model, cross_encoder=None):
        self.retriever = retriever
        self.embedder = embedding_model
        self.cross_encoder = cross_encoder
    
    async def mine(
        self,
        training_pairs: list[dict],  # [{query, positive, chunk_id}]
        n_hard_negatives: int = 5,
        min_retrieval_rank: int = 3,   # Negatives must appear in top-3 at minimum
        max_retrieval_rank: int = 20,  # But not be too easy (rank > 20 = easy negative)
        cross_encoder_score_max: float = 0.4  # Cross-encoder should score low
    ) -> list[dict]:
        """
        Mine hard negatives for each training pair.
        
        Hard negatives: chunks that the retrieval model ranks highly
        but the cross-encoder (or ground truth) confirms are not relevant.
        """
        
        enriched_pairs = []
        
        for pair in training_pairs:
            query = pair["query"]
            positive_id = pair.get("chunk_id")
            
            # Retrieve top-20 candidates with current model
            candidates = await self.retriever.retrieve(query, k=20)
            
            # Filter: exclude the true positive, take ranks 3-20
            hard_neg_candidates = [
                c for i, c in enumerate(candidates, 1)
                if (c.get("chunk_id") != positive_id and 
                    min_retrieval_rank <= i <= max_retrieval_rank)
            ]
            
            if self.cross_encoder and hard_neg_candidates:
                # Use cross-encoder to confirm negatives are not relevant
                # (prevents accidentally using near-positives as negatives)
                pairs_to_score = [(query, c["text"]) for c in hard_neg_candidates[:10]]
                scores = self.cross_encoder.predict(pairs_to_score)
                
                # Keep only chunks with low cross-encoder score
                confirmed_negatives = [
                    c for c, score in zip(hard_neg_candidates[:10], scores)
                    if score < cross_encoder_score_max
                ]
                
                hard_negatives = confirmed_negatives[:n_hard_negatives]
            else:
                hard_negatives = hard_neg_candidates[:n_hard_negatives]
            
            if hard_negatives:
                enriched_pairs.append({
                    **pair,
                    "hard_negatives": [n["text"] for n in hard_negatives]
                })
        
        return enriched_pairs
```

---

## Step 3 — Training with LoRA for Large Models

For embedding models with 1B+ parameters (E5-mistral-7b, GTE-Qwen2), full fine-tuning is expensive. LoRA (Low-Rank Adaptation) reduces memory requirements by 4-8×.

```python
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer
import torch

def prepare_model_with_lora(
    base_model_name: str,
    lora_r: int = 16,          # Rank of LoRA matrices
    lora_alpha: int = 32,      # Scaling factor (usually 2× rank)
    lora_dropout: float = 0.1,
    target_modules: list[str] = None  # Which layers to apply LoRA to
) -> tuple:
    """
    Load base embedding model and apply LoRA for memory-efficient fine-tuning.
    """
    
    # Load base model
    base_model = SentenceTransformer(base_model_name)
    
    if target_modules is None:
        # Apply LoRA to query and value projection layers (standard choice)
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none"
    )
    
    # Apply LoRA to the underlying transformer model
    transformer_model = base_model._first_module()  # Access underlying transformer
    peft_model = get_peft_model(transformer_model, lora_config)
    
    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    # For a 7B model with LoRA r=16: ~0.5% of parameters are trainable
    
    return base_model, peft_model


def fine_tune_with_lora(
    base_model,
    peft_model,
    training_data: list[dict],
    output_path: str,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    batch_size: int = 32
):
    """
    Fine-tune with LoRA adapters.
    Only the LoRA adapter weights are updated; base model is frozen.
    """
    from sentence_transformers import losses, InputExample
    from torch.utils.data import DataLoader
    
    # Build training examples
    train_samples = []
    for example in training_data:
        texts = [example["query"], example["positive"]]
        texts.extend(example.get("hard_negatives", [])[:3])
        train_samples.append(InputExample(texts=texts))
    
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(base_model)
    
    # Training (only LoRA parameters are updated)
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Standard training loop with LoRA
    base_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        optimizer_params={"lr": learning_rate},
        output_path=output_path
    )
    
    # Save LoRA adapter separately (small file — only adapter weights)
    peft_model.save_pretrained(output_path + "/lora_adapter")
    
    print(f"LoRA adapter saved. Size: only the adapter weights, not the full model.")
```

**LoRA memory savings:**

| Model | Full Fine-Tune VRAM | LoRA (r=16) VRAM |
|---|---|---|
| BERT-base (110M) | ~6 GB | ~4 GB |
| E5-large (335M) | ~18 GB | ~8 GB |
| E5-mistral-7b (7B) | ~80 GB | ~20 GB |
| GTE-Qwen2-7b (7B) | ~80 GB | ~20 GB |

For 7B models, full fine-tuning requires 4-8× A100 GPUs. LoRA fits on a single A100 (40GB).

---

## Step 4 — Evaluation After Fine-Tuning

```python
def evaluate_fine_tuned_model(
    base_model_name: str,
    fine_tuned_path: str,
    eval_pairs: list[dict],
    corpus_chunks: list[dict]
) -> dict:
    """
    Compare base model vs. fine-tuned model on domain evaluation set.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    base_model = SentenceTransformer(base_model_name)
    fine_tuned_model = SentenceTransformer(fine_tuned_path)
    
    results = {}
    
    for model_name, model in [("base", base_model), ("fine_tuned", fine_tuned_model)]:
        corpus_texts = [c["text"] for c in corpus_chunks]
        corpus_embeddings = model.encode(corpus_texts, normalize_embeddings=True)
        
        recalls_at_10 = []
        mrrs = []
        
        for pair in eval_pairs:
            query_emb = model.encode(pair["query"], normalize_embeddings=True)
            sims = corpus_embeddings @ query_emb
            
            top_10_indices = np.argsort(sims)[-10:][::-1]
            retrieved_ids = [corpus_chunks[i]["chunk_id"] for i in top_10_indices]
            relevant_id = pair["relevant_chunk_id"]
            
            recall = 1.0 if relevant_id in retrieved_ids else 0.0
            recalls_at_10.append(recall)
            
            if relevant_id in retrieved_ids:
                mrrs.append(1.0 / (retrieved_ids.index(relevant_id) + 1))
            else:
                mrrs.append(0.0)
        
        results[model_name] = {
            "recall@10": float(np.mean(recalls_at_10)),
            "mrr": float(np.mean(mrrs))
        }
    
    improvement = results["fine_tuned"]["recall@10"] - results["base"]["recall@10"]
    
    return {
        **results,
        "improvement_recall@10": improvement,
        "deploy_fine_tuned": improvement > 0.01  # Only deploy if >1% improvement
    }
```

---

## Step 5 — Deployment: Re-Embed the Corpus

Once the fine-tuned model is validated, you must re-embed the entire corpus. The fine-tuned model's embedding space is different from the base model's — existing vectors are incompatible.

```python
async def re_embed_corpus_with_new_model(
    new_model_path: str,
    all_chunks: list[dict],
    vector_db,
    new_collection_name: str,
    batch_size: int = 256
):
    """
    Re-embed entire corpus with fine-tuned model.
    Uses blue-green approach: write to new collection, validate, switch traffic.
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(new_model_path)
    vector_dim = model.get_sentence_embedding_dimension()
    
    # Create new collection
    await vector_db.create_collection(
        collection_name=new_collection_name,
        vector_size=vector_dim
    )
    
    # Re-embed in batches
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        
        points = [
            {
                "id": chunk["chunk_id"],
                "vector": emb.tolist(),
                "payload": {
                    **chunk["metadata"],
                    "embedding_model": new_model_path,
                    "embedding_model_version": "fine_tuned_v1"
                }
            }
            for chunk, emb in zip(batch, embeddings)
        ]
        
        await vector_db.upsert(
            collection_name=new_collection_name,
            points=points
        )
        
        print(f"Re-embedded {i + len(batch)}/{len(all_chunks)} chunks")
    
    print(f"Re-embedding complete. Validate {new_collection_name} before switching traffic.")
```

---

## The Full Fine-Tuning Decision Tree

```
1. Measure domain recall@10 with best general model
   ├── > 0.85 → Do NOT fine-tune. Focus on retrieval pipeline improvements.
   └── < 0.85 → Continue
   
2. Check training data availability
   ├── < 500 pairs available → Generate synthetic pairs first (Lesson A.8 Step 1)
   └── >= 500 pairs → Continue
   
3. Choose model size
   ├── < 500M params (BERT, e5-large) → Full fine-tuning on single GPU
   └── > 500M params (e5-mistral, GTE-Qwen2) → LoRA fine-tuning
   
4. Train with hard negatives
   └── Mine hard negatives from current retrieval system
   
5. Evaluate
   ├── Improvement < 1% → Do NOT deploy. Return to data quality.
   └── Improvement > 1% → Deploy with blue-green corpus re-embedding.
```

---

## Summary

- Fine-tuning is worth it when domain recall@10 < 0.80 with the best general model and you have ≥ 1,000 training pairs.
- Training data sources: user interaction logs (best quality), LLM-generated synthetic pairs (scalable), cross-encoder-labeled silver data.
- Hard negative mining from your retrieval system is the most important step — it finds the examples where the model currently makes mistakes.
- LoRA reduces memory by 4-8× for large models (7B+) — essential for fine-tuning E5-mistral or similar without 8× A100 setup.
- After fine-tuning, always re-embed the full corpus (blue-green) before deploying.
- Only deploy if improvement exceeds 1% on domain evaluation — otherwise the infrastructure cost is not justified.