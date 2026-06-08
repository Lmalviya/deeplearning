# Lesson 8.8 — CI/CD for RAG Systems: Index Versioning, Model Versioning, and Regression Testing

---

## Why RAG CI/CD Is Different

Traditional software CI/CD deploys code. RAG CI/CD deploys code, models, and data — three separate artifacts that can each introduce regressions independently.

A code change to the chunking strategy does not break unit tests, but it can degrade retrieval quality. A new embedding model may improve general performance but regress on your specific domain. A corpus update may introduce conflicting documents that cause hallucination. None of these are caught by standard software testing.

RAG CI/CD must:
- Test changes to code, models, and index configurations before deploying to production.
- Version each artifact (code, model, index) independently.
- Allow rollback of any artifact without requiring rollback of others.
- Measure quality metrics as a gate, not just functional tests.

---

## The Four RAG Artifacts to Version

```
Artifact            Changes when              Versioning
─────────────────────────────────────────────────────────
Application code    Engineer pushes code       Git SHA / semver
Embedding model     New model released         Model name + date
Index configuration Chunking, metadata schema  Schema version
Corpus              Documents added/updated    Content hash + timestamp
```

Each artifact needs its own versioning scheme and its own CI/CD path.

---

## Code CI/CD: Standard but with RAG-Specific Quality Gates

```yaml
# .github/workflows/rag-ci.yml
name: RAG CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      qdrant:
        image: qdrant/qdrant:v1.9.0
        ports:
          - 6333:6333
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run integration tests
        env:
          QDRANT_URL: http://localhost:6333
          REDIS_URL: redis://localhost:6379/0
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_TEST }}
        run: pytest tests/integration/ -v --timeout=120

  retrieval-quality-gate:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run retrieval quality evaluation
        env:
          QDRANT_URL: ${{ secrets.STAGING_QDRANT_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_PROD }}
        run: |
          python scripts/eval/run_retrieval_eval.py \
            --eval-set data/eval/retrieval_eval.json \
            --output results/retrieval_metrics.json
      
      - name: Check quality gates
        run: |
          python scripts/eval/check_quality_gates.py \
            --metrics results/retrieval_metrics.json \
            --thresholds config/quality_thresholds.yaml
      
      - name: Upload eval results
        uses: actions/upload-artifact@v4
        with:
          name: retrieval-eval-results
          path: results/
```

**The retrieval quality gate script:**

```python
# scripts/eval/check_quality_gates.py
import json
import yaml
import sys

def check_quality_gates(metrics_path: str, thresholds_path: str) -> bool:
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    with open(thresholds_path) as f:
        thresholds = yaml.safe_load(f)
    
    failures = []
    
    for metric, threshold in thresholds["required"].items():
        actual = metrics.get(metric, 0)
        if actual < threshold:
            failures.append(
                f"FAILED: {metric} = {actual:.3f} (required >= {threshold})"
            )
    
    if failures:
        print("Quality gate FAILED:")
        for f in failures:
            print(f"  {f}")
        return False
    
    print("Quality gate PASSED:")
    for metric, threshold in thresholds["required"].items():
        print(f"  {metric}: {metrics[metric]:.3f} >= {threshold} ✓")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--thresholds", required=True)
    args = parser.parse_args()
    
    passed = check_quality_gates(args.metrics, args.thresholds)
    sys.exit(0 if passed else 1)
```

```yaml
# config/quality_thresholds.yaml
required:
  hit_rate_at_5: 0.88
  recall_at_10: 0.80
  mrr: 0.70
  faithfulness: 0.90
  answer_relevancy: 0.85

warning:  # These don't block deployment but generate alerts
  precision_at_5: 0.60
  ndcg_at_10: 0.75
```

---

## Index Versioning

The index (the Qdrant collection) is not just a database — it is a versioned artifact. When the chunking strategy, embedding model, or metadata schema changes, you need a new index version.

### Index Version Registry

```python
# src/indexing/version_registry.py

INDEX_VERSIONS = {
    "v1.0": {
        "created": "2024-01-15",
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "embedding_dim": 1024,
        "chunking_strategy": "recursive_character",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "metadata_schema": "v1",
        "collection_name": "documents_v1",
        "status": "archived"
    },
    "v2.0": {
        "created": "2024-03-20",
        "embedding_model": "multilingual-e5-large",
        "embedding_dim": 1024,
        "chunking_strategy": "structure_aware",
        "chunk_size": 256,  # Parent-child
        "metadata_schema": "v2",
        "collection_name": "documents_v2",
        "status": "production"
    },
    "v2.1": {
        "created": "2024-06-01",
        "embedding_model": "multilingual-e5-large",
        "embedding_dim": 1024,
        "chunking_strategy": "structure_aware",
        "chunk_size": 256,
        "metadata_schema": "v2.1",  # Added jurisdiction field
        "collection_name": "documents_v2_1",
        "status": "staging"  # Being validated
    }
}

CURRENT_PRODUCTION_VERSION = "v2.0"
STAGING_VERSION = "v2.1"
```

### Index Migration Pipeline

When introducing a new index version, the migration follows a blue-green pattern:

```python
# scripts/index/migrate_index.py

async def migrate_to_new_index_version(
    source_version: str,
    target_version: str,
    registry,
    embedding_model,
    vector_db
):
    """
    Migrate corpus from one index version to another.
    Runs offline — no production traffic affected.
    """
    
    source_config = INDEX_VERSIONS[source_version]
    target_config = INDEX_VERSIONS[target_version]
    
    print(f"Migrating {source_version} → {target_version}")
    print(f"  Source embedding: {source_config['embedding_model']}")
    print(f"  Target embedding: {target_config['embedding_model']}")
    
    # Step 1: Create new collection
    await vector_db.create_collection(
        collection_name=target_config["collection_name"],
        vectors_config=VectorParams(
            size=target_config["embedding_dim"],
            distance=Distance.COSINE
        )
    )
    
    # Step 2: Re-index all documents with new configuration
    all_docs = await registry.get_all_active_documents()
    
    print(f"Re-indexing {len(all_docs)} documents...")
    
    batch_size = 100
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        
        # Re-process each document with new configuration
        for doc in batch:
            source_path = doc["source_path"]
            
            # Re-parse and re-chunk with new strategy
            content = await parse_document(source_path)
            chunks = chunk_with_strategy(content, target_config["chunking_strategy"])
            
            # Re-embed with new model
            embeddings = await embedding_model.embed_batch(
                [c["text"] for c in chunks]
            )
            
            # Upsert to new collection
            points = [
                PointStruct(
                    id=generate_chunk_id(doc["doc_id"], i),
                    vector=embedding.tolist(),
                    payload={**chunk["metadata"], "index_version": target_version}
                )
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]
            
            await vector_db.upsert(
                collection_name=target_config["collection_name"],
                points=points
            )
        
        print(f"  Processed {min(i + batch_size, len(all_docs))}/{len(all_docs)}")
    
    print("Re-indexing complete. Running validation...")
    
    # Step 3: Validate new index quality
    eval_results = await run_eval_on_collection(
        collection_name=target_config["collection_name"],
        embedding_model=embedding_model,
        eval_dataset=load_eval_dataset()
    )
    
    if eval_results["recall@10"] >= QUALITY_THRESHOLDS["recall_at_10"]:
        print(f"✓ Quality gate passed: recall@10 = {eval_results['recall@10']:.3f}")
        print("Migration successful. Promote with: switch_active_collection()")
    else:
        print(f"✗ Quality gate FAILED: recall@10 = {eval_results['recall@10']:.3f}")
        print("Migration aborted. Deleting new collection.")
        await vector_db.delete_collection(target_config["collection_name"])
```

### Switching Active Collections

```python
# src/config.py

class CollectionConfig:
    """
    Runtime configuration that can be updated without code deployment.
    Stored in Redis so all pods pick up the change simultaneously.
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_active_collection(self) -> str:
        collection = await self.redis.get("active_collection")
        return collection.decode() if collection else "documents_v2"
    
    async def set_active_collection(self, collection_name: str):
        """
        Atomically switch all pods to use a new collection.
        Zero-downtime: all pods read from Redis on every request.
        """
        await self.redis.set("active_collection", collection_name)
        print(f"Switched active collection to: {collection_name}")
        print("All pods will pick up the change within 1-2 requests.")
```

---

## Model Versioning: Tracking What Was Used

Every query must be traceable to the exact model versions used. This enables debugging ("was this answer from before or after the model upgrade?") and rollback.

```python
# src/config/model_config.py

MODEL_REGISTRY = {
    "embedding": {
        "current": {
            "name": "multilingual-e5-large",
            "version": "2024-03-20",
            "deployed_at": "2024-04-01",
            "index_collection": "documents_v2"
        },
        "previous": {
            "name": "BAAI/bge-large-en-v1.5",
            "version": "2024-01-15",
            "deployed_at": "2024-01-20",
            "index_collection": "documents_v1"
        }
    },
    "reranker": {
        "current": {
            "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "version": "2024-02-10"
        }
    },
    "llm": {
        "current": {
            "name": "gpt-4o-2024-05-13",    # Pinned version
            "provider": "openai"
        }
    }
}

# Attach model versions to every trace
def get_current_model_versions() -> dict:
    return {
        "embedding_model": MODEL_REGISTRY["embedding"]["current"]["name"],
        "embedding_model_version": MODEL_REGISTRY["embedding"]["current"]["version"],
        "reranker_model": MODEL_REGISTRY["reranker"]["current"]["name"],
        "llm_model": MODEL_REGISTRY["llm"]["current"]["name"]
    }
```

---

## Regression Testing Suite

The regression test suite catches known failure cases before they reach production. Every diagnosed production failure should become a regression test.

```python
# tests/regression/test_known_failures.py

import pytest
from src.pipeline import RAGPipeline

REGRESSION_CASES = [
    {
        "id": "REG-001",
        "description": "Scoped termination clause — regression from 2024-03-15",
        "query": "What is the notice period for contract termination in the Vendor X agreement?",
        "expected_answer_contains": ["90 days", "written notice"],
        "expected_chunk_ids": ["vendor-x-contract-section-8.3"],
        "date_discovered": "2024-03-15",
        "root_cause": "Re-ranker was demoting the relevant clause due to length"
    },
    {
        "id": "REG-002",
        "description": "IDK for answerable question — regression from 2024-04-02",
        "query": "What is the maximum expense reimbursement for international travel?",
        "expected_behavior": "not_idk",  # Should answer, not say IDK
        "expected_answer_contains": ["$500", "per diem"],
        "date_discovered": "2024-04-02",
        "root_cause": "Context budget was too small, relevant chunk was truncated"
    },
    {
        "id": "REG-003",
        "description": "Wrong answer from parametric memory — 2024-04-20",
        "query": "What is the current parental leave policy?",
        "expected_answer_contains": ["16 weeks"],  # Our policy
        "expected_answer_does_not_contain": ["12 weeks"],  # Industry norm (wrong)
        "date_discovered": "2024-04-20",
        "root_cause": "LLM answered from parametric memory (12 weeks) instead of context (16 weeks)"
    }
]

@pytest.mark.parametrize("case", REGRESSION_CASES)
async def test_regression_case(case, rag_pipeline: RAGPipeline):
    """
    Run each known regression case and verify it does not regress.
    """
    result = await rag_pipeline.answer(case["query"])
    
    answer = result["answer"].lower()
    
    # Check expected content
    if "expected_answer_contains" in case:
        for expected in case["expected_answer_contains"]:
            assert expected.lower() in answer, (
                f"Regression {case['id']}: Expected '{expected}' in answer.\n"
                f"Got: {result['answer'][:200]}"
            )
    
    # Check prohibited content
    if "expected_answer_does_not_contain" in case:
        for prohibited in case["expected_answer_does_not_contain"]:
            assert prohibited.lower() not in answer, (
                f"Regression {case['id']}: '{prohibited}' should not be in answer.\n"
                f"Got: {result['answer'][:200]}"
            )
    
    # Check IDK behavior
    if case.get("expected_behavior") == "not_idk":
        idk_phrases = ["don't have information", "cannot find", "not available"]
        is_idk = any(phrase in answer for phrase in idk_phrases)
        assert not is_idk, f"Regression {case['id']}: Expected non-IDK but got IDK response"
    
    # Check source chunks
    if "expected_chunk_ids" in case:
        retrieved_ids = [c["chunk_id"] for c in result.get("retrieved_chunks", [])]
        for expected_id in case["expected_chunk_ids"]:
            assert expected_id in retrieved_ids, (
                f"Regression {case['id']}: Expected chunk {expected_id} not retrieved.\n"
                f"Retrieved: {retrieved_ids}"
            )
```

---

## Complete CI/CD Pipeline

```
Developer pushes code
        │
        ▼
[Unit Tests]
  - pytest: chunking, parsing, metadata
  - No external services needed
  - Must pass: block merge if fails
        │
        ▼
[Integration Tests]  
  - Against local Qdrant + Redis (Docker Compose)
  - Test: retrieval pipeline, embedding server, re-ranker
  - Must pass: block merge if fails
        │
        ▼
[Regression Tests]
  - Against staging Qdrant (real data)
  - Run all known-failure regression cases
  - Must pass: block deployment if any regression
        │
        ▼
[Retrieval Quality Gate]
  - Run full eval set against staging
  - Measure: recall@10, hit_rate@5, faithfulness
  - Must meet thresholds: block deployment if below
        │
        ▼
[Blue-Green Deploy to Production]
  - Deploy new pods alongside old pods
  - Run smoke tests against new pods
  - Switch load balancer if smoke tests pass
  - Old pods removed after 5-minute drain period
        │
        ▼
[Post-Deploy Validation]
  - Run eval set against production (sampled)
  - Compare to pre-deploy baseline
  - Alert if any metric regresses > 3%
        │
        ▼
[Continuous Monitoring]
  - IDK rate, reformulation rate, p95 latency
  - Automated rollback if metrics degrade
```

### Automated Rollback

```python
# scripts/monitoring/auto_rollback.py

async def monitor_post_deploy_metrics(
    deployment_id: str,
    pre_deploy_metrics: dict,
    rollback_fn,
    check_interval_seconds: int = 60,
    monitoring_duration_seconds: int = 600
):
    """
    Monitor metrics after deployment. Auto-rollback if regression detected.
    """
    
    print(f"Monitoring deployment {deployment_id} for {monitoring_duration_seconds}s...")
    
    start_time = time.time()
    checks_passed = 0
    
    while time.time() - start_time < monitoring_duration_seconds:
        await asyncio.sleep(check_interval_seconds)
        
        current_metrics = await compute_live_metrics(window_minutes=5)
        
        regressions = []
        
        for metric, pre_value in pre_deploy_metrics.items():
            current_value = current_metrics.get(metric, 0)
            regression_pct = (pre_value - current_value) / pre_value
            
            if regression_pct > 0.05:  # > 5% regression
                regressions.append({
                    "metric": metric,
                    "before": pre_value,
                    "after": current_value,
                    "regression_pct": regression_pct
                })
        
        if regressions:
            print(f"REGRESSION DETECTED after {int(time.time() - start_time)}s:")
            for r in regressions:
                print(f"  {r['metric']}: {r['before']:.3f} → {r['after']:.3f} ({r['regression_pct']*100:.1f}% drop)")
            
            print("Initiating automatic rollback...")
            await rollback_fn(deployment_id)
            
            await alert_team(f"Auto-rollback triggered for {deployment_id}: {regressions}")
            return False
        
        checks_passed += 1
        print(f"Check {checks_passed}: All metrics healthy")
    
    print(f"Deployment {deployment_id} validated. All metrics stable.")
    return True
```

---

## Summary

- RAG CI/CD must version and test four artifacts independently: application code, embedding model, index configuration, and corpus.
- Quality gates (recall@K, faithfulness, answer relevancy) are mandatory gates in the deployment pipeline — not just functional tests.
- Index migration uses blue-green switching: build new collection, validate quality, atomically switch via Redis config, clean up old collection.
- Store model versions in every trace to enable debugging and rollback attribution.
- Regression tests encode every known production failure. Run them on every deployment to prevent re-introduction.
- Automated post-deploy monitoring watches live metrics and rolls back if any metric degresses more than 5% within 10 minutes of deployment.
- The complete pipeline: unit → integration → regression → quality gate → blue-green deploy → post-deploy validation → continuous monitoring.

---

## Part 8 Complete

You now have coverage of the full deployment stack: vector database selection, self-hosted vs. managed trade-offs, Docker containerization, AWS deployment patterns, Kubernetes orchestration, retrieval layer scaling, LLM serving options, and CI/CD with quality gates.

The next parts (9 and 10) cover scale, reliability at millions of users, and the system design case studies.