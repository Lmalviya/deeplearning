# Lesson 9.3 — Async Indexing Pipelines: Queues, Workers, and Retry Logic

---

## Why Async Indexing

Synchronous document indexing — "wait until the document is fully indexed before returning" — works for small corpora. At scale, it creates three problems:

**Problem 1: Latency.** Indexing a 100-page PDF with OCR, chunking, and embedding takes 30-120 seconds. Blocking the caller for 2 minutes is unacceptable.

**Problem 2: Failure handling.** If indexing fails mid-way (embedding API timeout, Qdrant write error), you have no way to retry just the failed step without re-running everything.

**Problem 3: Throughput.** You need to index 50,000 documents. Sequential processing at 30 seconds each takes 17 days. Parallel workers reduce this to hours.

Async indexing with a message queue solves all three: the caller is acknowledged immediately, failures are retried automatically, and multiple workers process documents in parallel.

---

## Queue-Based Indexing Architecture

```
Document Source (S3, SharePoint, webhook)
          │
          ▼
    [Ingestion API]     ← Accepts document, returns immediately
          │
          ▼ enqueue
    [SQS / Celery Queue]
          │
     ┌────┴────┐
     ▼         ▼
 [Worker 1] [Worker 2]  ... [Worker N]   ← Process in parallel
     │
     ▼
[Parse → Chunk → Embed → Upsert]
     │
     ▼
[Registry Update]
     │
     ▼
[DLQ if failed]  ← Dead letter queue for permanent failures
```

---

## Implementation: Celery + SQS

Celery is the standard Python task queue. With SQS as the broker, it is durable, scalable, and cloud-native.

```python
# src/indexing/celery_app.py
from celery import Celery
from kombu import Exchange, Queue

celery_app = Celery(
    "rag_indexing",
    broker="sqs://",   # Uses AWS credentials from environment
    backend="redis://redis:6379/1"
)

celery_app.conf.update(
    # SQS configuration
    broker_transport_options={
        "region": "us-east-1",
        "predefined_queues": {
            "indexing": {
                "url": "https://sqs.us-east-1.amazonaws.com/account/rag-indexing",
                "access_key_id": None,      # Use IAM role
                "secret_access_key": None
            },
            "indexing_urgent": {
                "url": "https://sqs.us-east-1.amazonaws.com/account/rag-indexing-urgent"
            }
        },
        "visibility_timeout": 300,  # 5 minutes — task must complete within this
        "polling_interval": 1,      # Poll SQS every 1 second
        "wait_time_seconds": 20,    # Long polling (reduces empty receives)
    },
    
    # Task configuration
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # Retry configuration
    task_acks_late=True,          # ACK only after successful completion
    task_reject_on_worker_lost=True,  # Requeue if worker crashes
    
    # Routing: urgent documents get their own queue
    task_routes={
        "indexing.tasks.index_document_urgent": {"queue": "indexing_urgent"},
        "indexing.tasks.index_document": {"queue": "indexing"},
    }
)
```

### Task Definition with Retry Logic

```python
# src/indexing/tasks.py
from celery import Task
from celery.utils.log import get_task_logger
import time

logger = get_task_logger(__name__)

class IndexingTask(Task):
    """Base class for indexing tasks with shared infrastructure."""
    abstract = True
    
    _embedding_model = None
    _vector_db = None
    
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            # Initialize once per worker process, not per task
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("multilingual-e5-large")
        return self._embedding_model
    
    @property
    def vector_db(self):
        if self._vector_db is None:
            from qdrant_client import QdrantClient
            self._vector_db = QdrantClient(url=QDRANT_URL)
        return self._vector_db


@celery_app.task(
    base=IndexingTask,
    bind=True,
    
    # Retry configuration
    max_retries=3,
    default_retry_delay=60,      # Wait 60s before first retry
    retry_backoff=True,          # Exponential backoff: 60s, 120s, 240s
    retry_backoff_max=600,       # Cap backoff at 10 minutes
    retry_jitter=True,           # Add randomness to prevent thundering herd
    
    # Timeout
    soft_time_limit=240,         # Raises SoftTimeLimitExceeded at 4 minutes
    time_limit=300,              # Hard kill at 5 minutes
    
    # Error handling
    autoretry_for=(
        ConnectionError,
        TimeoutError,
        # Add specific retryable exceptions
    ),
    dont_autoretry_for=(
        ValueError,              # Bad document — don't retry
        UnicodeDecodeError,      # Corrupt file — don't retry
    )
)
def index_document(
    self,
    source_path: str,
    doc_id: str,
    metadata: dict,
    force_reindex: bool = False
) -> dict:
    """
    Full document indexing pipeline as a Celery task.
    """
    
    start_time = time.time()
    logger.info(f"Starting indexing: {doc_id} from {source_path}")
    
    try:
        # Step 1: Update registry to "indexing"
        update_registry(doc_id, {"status": "indexing", "started_at": start_time})
        
        # Step 2: Fetch document from source
        document_bytes = fetch_document(source_path)  # S3, filesystem, etc.
        
        # Step 3: Parse
        parsed = parse_document(document_bytes, source_path)
        
        # Step 4: Chunk
        chunks = chunk_document(parsed, metadata)
        
        # Step 5: Delete existing chunks (for re-indexing)
        if force_reindex:
            delete_chunks_for_doc(doc_id, self.vector_db)
        
        # Step 6: Embed in batches
        all_points = []
        batch_size = 64
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            
            # This can fail with timeout — will be retried
            embeddings = self.embedding_model.encode(
                [f"passage: {t}" for t in texts],
                normalize_embeddings=True
            )
            
            for chunk, embedding in zip(batch, embeddings):
                all_points.append(build_point(chunk, embedding.tolist()))
        
        # Step 7: Upsert to vector DB
        # Batch upsert — if this fails, retry will re-embed and re-upsert
        self.vector_db.upsert(
            collection_name="documents",
            points=all_points,
            wait=True  # Wait for indexing to complete
        )
        
        # Step 8: Update registry to "indexed"
        elapsed = time.time() - start_time
        update_registry(doc_id, {
            "status": "indexed",
            "indexed_at": time.time(),
            "chunk_count": len(chunks),
            "elapsed_seconds": elapsed,
            "pipeline_version": PIPELINE_VERSION
        })
        
        logger.info(f"Indexed {doc_id}: {len(chunks)} chunks in {elapsed:.1f}s")
        
        return {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "elapsed_seconds": elapsed
        }
    
    except self.SoftTimeLimitExceeded:
        # Task is taking too long — mark as failed and notify
        update_registry(doc_id, {
            "status": "failed",
            "error_message": "Task timed out after 4 minutes"
        })
        raise
    
    except Exception as exc:
        # Log the failure
        logger.error(f"Indexing failed for {doc_id}: {str(exc)}")
        
        # Determine if retryable
        retry_count = self.request.retries
        
        if retry_count < self.max_retries:
            update_registry(doc_id, {
                "status": "retrying",
                "error_message": str(exc),
                "retry_count": retry_count + 1
            })
            # Raise for Celery to handle retry
            raise self.retry(exc=exc)
        else:
            # Max retries exceeded — move to DLQ happens automatically
            update_registry(doc_id, {
                "status": "failed",
                "error_message": f"Failed after {self.max_retries} retries: {str(exc)}"
            })
            raise
```

---

## Dead Letter Queue (DLQ) Handling

Documents that fail all retries go to the Dead Letter Queue. The DLQ is not a trash bin — it is a queue of documents that need human attention.

```python
# src/indexing/dlq_processor.py

async def process_dlq_messages():
    """
    Periodically review DLQ messages and determine the appropriate action.
    """
    
    sqs = boto3.client("sqs")
    dlq_url = "https://sqs.us-east-1.amazonaws.com/account/rag-indexing-dlq"
    
    response = sqs.receive_message(
        QueueUrl=dlq_url,
        MaxNumberOfMessages=10,
        AttributeNames=["ApproximateReceiveCount", "SentTimestamp"]
    )
    
    for message in response.get("Messages", []):
        body = json.loads(message["Body"])
        receive_count = int(message["Attributes"]["ApproximateReceiveCount"])
        
        doc_id = body.get("doc_id")
        source_path = body.get("source_path")
        error = body.get("error_message", "Unknown error")
        
        # Classify the failure
        failure_type = classify_failure(error)
        
        if failure_type == "transient":
            # Network/service issue — retry manually after delay
            await retry_with_delay(body, delay_hours=1)
            sqs.delete_message(
                QueueUrl=dlq_url,
                ReceiptHandle=message["ReceiptHandle"]
            )
        
        elif failure_type == "document_corrupt":
            # Document is corrupted — alert team, mark as permanently failed
            await alert_team(
                f"Document cannot be indexed: {source_path}\nError: {error}"
            )
            await update_registry(doc_id, {"status": "permanently_failed"})
            sqs.delete_message(
                QueueUrl=dlq_url,
                ReceiptHandle=message["ReceiptHandle"]
            )
        
        elif failure_type == "quota_exceeded":
            # API quota hit — requeue for next day
            await retry_with_delay(body, delay_hours=24)
            sqs.delete_message(
                QueueUrl=dlq_url,
                ReceiptHandle=message["ReceiptHandle"]
            )
        
        else:
            # Unknown failure — alert team for manual investigation
            await alert_team(
                f"Unknown DLQ failure for {doc_id}:\nSource: {source_path}\nError: {error}"
            )


def classify_failure(error_message: str) -> str:
    """Classify failure type for appropriate DLQ handling."""
    
    error_lower = error_message.lower()
    
    if any(x in error_lower for x in ["timeout", "connection", "network", "temporary"]):
        return "transient"
    
    if any(x in error_lower for x in ["corrupt", "invalid", "parse error", "unsupported"]):
        return "document_corrupt"
    
    if any(x in error_lower for x in ["quota", "rate limit", "too many requests"]):
        return "quota_exceeded"
    
    return "unknown"
```

---

## Idempotency: Safe Retries

Retries are safe only if the task is **idempotent** — running it multiple times produces the same result as running it once. Indexing tasks must be idempotent.

```python
# The key to idempotent indexing:
# 1. Delete old chunks BEFORE inserting new ones
# 2. Use deterministic chunk IDs (not random UUIDs)
# 3. Use upsert (not insert) for the vector DB operation

def generate_deterministic_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate a chunk ID that is always the same for the same doc+position.
    Enables safe retries — re-inserting the same chunk_id is an upsert.
    """
    import hashlib
    return hashlib.sha256(f"{doc_id}:chunk:{chunk_index}".encode()).hexdigest()[:32]


async def idempotent_upsert(
    doc_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    vector_db
):
    """
    Upsert chunks idempotently:
    1. Delete all existing chunks for this doc
    2. Insert new chunks
    
    If the task fails between step 1 and step 2, a partial delete
    is visible to queries. This is acceptable — the registry shows
    'indexing' status, and retrieval filters on status='indexed'.
    """
    
    # Step 1: Delete existing chunks atomically
    vector_db.delete(
        collection_name="documents",
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )
    )
    
    # Step 2: Insert new chunks
    points = [
        PointStruct(
            id=generate_deterministic_chunk_id(doc_id, i),
            vector=embedding,
            payload={**chunk["metadata"], "doc_id": doc_id}
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    
    vector_db.upsert(
        collection_name="documents",
        points=points,
        wait=True
    )
```

---

## Priority Queues for Urgent Documents

Not all documents have equal urgency. A just-published policy change needs immediate indexing. A historical archive document can wait hours.

```python
# Priority levels
PRIORITY_LEVELS = {
    "urgent": {
        "queue": "indexing_urgent",
        "description": "Recent policy changes, breaking news",
        "target_lag_minutes": 5
    },
    "normal": {
        "queue": "indexing",
        "description": "Regular documents",
        "target_lag_minutes": 60
    },
    "bulk": {
        "queue": "indexing_bulk",
        "description": "Historical documents, batch imports",
        "target_lag_minutes": 1440  # 24 hours
    }
}

def determine_priority(doc_metadata: dict) -> str:
    """Determine indexing priority based on document metadata."""
    
    # Release notes and policy changes are urgent
    if doc_metadata.get("document_type") in ["release_note", "policy_update"]:
        return "urgent"
    
    # Documents older than 1 year can be bulk-processed
    if doc_metadata.get("created_date"):
        from datetime import datetime, timedelta
        created = datetime.fromisoformat(doc_metadata["created_date"])
        if datetime.utcnow() - created > timedelta(days=365):
            return "bulk"
    
    return "normal"
```

---

## Monitoring the Indexing Pipeline

```python
INDEXING_PIPELINE_METRICS = {
    # Throughput
    "documents_indexed_per_hour": "Target: consistent with ingestion rate",
    "queue_depth": "Alert if growing — workers falling behind",
    "dlq_depth": "Alert if > 0 — permanent failures need attention",
    
    # Latency (freshness)
    "avg_indexing_lag_minutes": "Time from document upload to searchable",
    "urgent_queue_lag_minutes": "Must be < 5 minutes",
    
    # Quality
    "indexing_success_rate": "Target: > 99%",
    "retry_rate": "Should be < 5% of tasks",
    "dlq_rate": "Should be < 0.1% of tasks",
    
    # Workers
    "active_workers": "Should match expected concurrency",
    "worker_memory_mb": "Alert if growing over time (memory leak)"
}
```

---

## Summary

- Async indexing with message queues provides acknowledgment-immediate responses, automatic retry on failure, and parallel processing throughput.
- Celery + SQS is the production-ready combination for Python RAG systems. SQS provides durable, at-least-once delivery; Celery provides retry logic and worker management.
- Retry with exponential backoff handles transient failures (API timeouts, network errors). Hard-code which exceptions are retryable and which are not.
- Dead Letter Queue captures permanent failures for human review and re-processing. Classify DLQ failures (transient, corrupt, quota) for appropriate handling.
- Idempotency is required for safe retries: use deterministic chunk IDs, delete-then-insert, and upsert operations.
- Priority queues ensure urgent documents (policy changes, release notes) are indexed within minutes while bulk historical documents can process over hours.

---

## What's Next

Lesson 9.4 covers cost management at scale — token budgets, caching ROI, batching strategies, and tiered retrieval to keep LLM costs from growing linearly with usage.