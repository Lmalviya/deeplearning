# Lesson 2.6 — Incremental Indexing and Data Freshness Strategies

---

## The Data Freshness Problem

A RAG system is only as good as its index. If your index is stale, users get answers based on outdated information — and unlike a broken system, a stale system does not announce itself. It silently returns confident, wrong answers.

The data freshness problem has two dimensions:

**Freshness at query time:** When a user asks a question, does the retrieved content reflect the current state of the documents, or an older snapshot?

**Freshness lag:** How long does it take from when a document changes to when that change is reflected in the index and retrievable by users?

For a legal department's contract management system, a policy that changed yesterday needs to be reflected in the index today — not next week when someone runs a manual re-index. For a customer support system, pricing changes need to propagate within hours, not days.

The naive solution — re-index everything from scratch whenever anything changes — is correct for small corpora but breaks at scale. Re-indexing 500,000 documents takes hours and consumes significant compute. This lesson covers how to build an incremental indexing system that maintains freshness efficiently.

---

## Why Full Re-indexing Does Not Scale

Let us be precise about the cost. Assume:
- 100,000 documents, average 50 chunks per document = 5,000,000 chunks.
- Embedding model throughput: 1,000 chunks per second (batched, GPU-accelerated).
- Full re-index time: 5,000 seconds ≈ 83 minutes of compute, plus storage write time.

If 500 documents change per day (0.5% of corpus), full re-indexing wastes 99.5% of compute on documents that have not changed. At 10,000 documents per day (10%), the waste is still 90%.

More importantly, during re-indexing, your system is either serving stale data (if you keep the old index live) or has no index at all (if you rebuild in place). Neither is acceptable for production systems with uptime requirements.

Incremental indexing solves this by processing only the documents that have changed.

---

## The Foundation: Document Change Detection

Before you can incrementally index, you need to know which documents changed. This requires a change detection mechanism.

### Content Hashing

Store a hash of each document's content when it is indexed. On each check cycle, compute the hash of the current document and compare to the stored hash.

```python
import hashlib

def compute_document_hash(file_path: str) -> str:
    """Compute SHA-256 hash of document content."""
    sha256 = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files without loading all into memory
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    
    return sha256.hexdigest()

def has_document_changed(file_path: str, stored_hash: str) -> bool:
    current_hash = compute_document_hash(file_path)
    return current_hash != stored_hash
```

**Advantages:** Catches any content change regardless of metadata. Works even if the file system timestamps are unreliable (common with NFS mounts, some cloud storage configurations, or files copied from external sources).

**Disadvantages:** Requires reading the entire file to compute the hash. For very large files (100MB+ PDFs), this adds up. Optimization: hash only the first 64KB + file size for a fast approximate check, then do a full hash only if the fast check suggests a change.

### Timestamp-Based Detection

Compare the file's `last_modified` timestamp against the timestamp recorded at last indexing.

```python
import os
from datetime import datetime

def get_file_modified_time(file_path: str) -> datetime:
    mtime = os.path.getmtime(file_path)
    return datetime.fromtimestamp(mtime)

def has_been_modified_since(file_path: str, last_indexed_at: datetime) -> bool:
    modified_at = get_file_modified_time(file_path)
    return modified_at > last_indexed_at
```

**Advantages:** Very fast — no file reading required, just a stat() system call.

**Disadvantages:** Timestamps are not always reliable. Files synced from cloud storage may have their timestamps reset. Files restored from backup retain old timestamps. Files touched without content change (metadata updates, permissions changes) trigger unnecessary re-indexing.

**Best practice:** Use timestamps for a fast first pass, then verify with content hash before triggering a full re-index. This avoids reading most files while still catching the cases where timestamps mislead.

### Source System Webhooks and Events

For documents stored in managed systems (SharePoint, Google Drive, Confluence, S3, Dropbox), subscribe to change events rather than polling.

```python
# Example: AWS S3 Event Notification
# Configure S3 to send events to SQS when objects are created, modified, or deleted

import boto3
import json

def process_s3_event(event: dict):
    """Process an S3 change event from SQS."""
    for record in event['Records']:
        event_type = record['eventName']  # ObjectCreated, ObjectRemoved, etc.
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        s3_path = f"s3://{bucket}/{key}"
        
        if event_type.startswith('ObjectCreated'):
            queue_for_indexing(s3_path, operation='upsert')
        elif event_type.startswith('ObjectRemoved'):
            queue_for_deletion(s3_path, operation='delete')
```

Webhook-based detection has near-zero latency — changes trigger indexing within seconds. This is how you achieve low freshness lag (minutes rather than hours) without continuous polling.

For Google Drive:
```python
# Google Drive Push Notifications
# Set up a webhook that Google calls when files change in a watched folder
from googleapiclient.discovery import build

def setup_drive_watch(folder_id: str, webhook_url: str):
    service = build('drive', 'v3')
    
    watch_response = service.files().watch(
        fileId=folder_id,
        body={
            'id': 'unique-channel-id',
            'type': 'web_hook',
            'address': webhook_url,
            'expiration': '3600000'  # 1 hour in milliseconds, then renew
        }
    ).execute()
    
    return watch_response
```

---

## The Document Registry

To support incremental indexing, you need a registry that tracks the state of every indexed document. This is a database table (not the vector database) that stores:

```sql
CREATE TABLE document_registry (
    doc_id          VARCHAR(255) PRIMARY KEY,
    source_path     TEXT NOT NULL,
    content_hash    VARCHAR(64),           -- SHA-256 of content at last index
    last_modified   TIMESTAMP,             -- File modification time at last index
    indexed_at      TIMESTAMP NOT NULL,    -- When we last indexed this document
    index_version   INTEGER DEFAULT 1,     -- Version of indexing pipeline used
    chunk_count     INTEGER,               -- How many chunks were created
    status          VARCHAR(50),           -- indexed, pending, failed, deleted
    error_message   TEXT,                  -- If status = failed, why
    metadata        JSONB                  -- Document-level metadata
);

CREATE INDEX idx_registry_status ON document_registry(status);
CREATE INDEX idx_registry_indexed_at ON document_registry(indexed_at);
CREATE INDEX idx_registry_source_path ON document_registry(source_path);
```

Every indexing operation — add, update, delete — updates this registry. It is your source of truth for what is in the index and when it was last updated.

---

## The Incremental Update Operations

### Adding a New Document

```python
async def index_new_document(file_path: str, metadata: dict):
    """Full indexing pipeline for a document not yet in the index."""
    
    doc_id = generate_doc_id(file_path)  # deterministic from path
    
    try:
        # 1. Parse and pre-process
        content = await parse_document(file_path)
        
        # 2. Chunk
        chunks = chunk_document(content, metadata)
        
        # 3. Embed
        embeddings = await embed_chunks_batch(chunks)
        
        # 4. Store in vector database
        points = [
            build_vector_point(chunk, embedding, doc_id)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        await vector_db.upsert(collection="documents", points=points)
        
        # 5. Update registry
        await registry.upsert({
            "doc_id": doc_id,
            "source_path": file_path,
            "content_hash": compute_document_hash(file_path),
            "last_modified": get_file_modified_time(file_path),
            "indexed_at": datetime.utcnow(),
            "chunk_count": len(chunks),
            "status": "indexed",
            "metadata": metadata
        })
        
    except Exception as e:
        await registry.upsert({
            "doc_id": doc_id,
            "source_path": file_path,
            "status": "failed",
            "error_message": str(e),
            "indexed_at": datetime.utcnow()
        })
        raise
```

### Updating a Changed Document

Updating is not simply re-adding. You must delete the old chunks before inserting new ones. The chunk count, chunk boundaries, and chunk IDs may all change when a document changes.

```python
async def update_document(file_path: str, doc_id: str):
    """Re-index a document that has changed."""
    
    # Step 1: Delete all existing chunks for this document
    # Vector databases support filtering by metadata field
    await vector_db.delete(
        collection="documents",
        filter={"doc_id": {"$eq": doc_id}}
    )
    
    # Step 2: Run the full indexing pipeline for the updated content
    await index_new_document(file_path, metadata=get_metadata(doc_id))
    
    # Registry is updated inside index_new_document
```

**Critical detail:** Delete first, then insert. Do not insert first and then delete — there is a window during which both old and new chunks coexist, and queries may retrieve a mix of old and new content.

For systems that cannot tolerate even a brief window of stale data, use a blue-green index pattern (described below).

### Deleting a Removed Document

```python
async def delete_document(doc_id: str):
    """Remove all chunks for a deleted document from the index."""
    
    # Delete from vector database
    await vector_db.delete(
        collection="documents",
        filter={"doc_id": {"$eq": doc_id}}
    )
    
    # Mark in registry (keep the record for auditing, don't hard delete)
    await registry.update(
        doc_id=doc_id,
        updates={
            "status": "deleted",
            "indexed_at": datetime.utcnow()
        }
    )
```

---

## The Incremental Indexing Worker

Production incremental indexing runs as a background worker, continuously processing a queue of change events.

```
Document Change Event (S3, webhook, poll)
    ↓
Change Queue (SQS, Kafka, Redis Queue, Celery)
    ↓
Indexing Worker (picks up events, processes them)
    ↓
Vector Database (updated)
    ↓
Document Registry (updated)
```

```python
import asyncio
from dataclasses import dataclass
from enum import Enum

class ChangeOperation(Enum):
    UPSERT = "upsert"   # Add or update
    DELETE = "delete"

@dataclass
class ChangeEvent:
    source_path: str
    operation: ChangeOperation
    detected_at: datetime

async def indexing_worker(queue):
    """Continuously process change events from the queue."""
    
    while True:
        try:
            event = await queue.get(timeout=5)
            
            doc_id = generate_doc_id(event.source_path)
            existing = await registry.get(doc_id)
            
            if event.operation == ChangeOperation.DELETE:
                if existing and existing['status'] != 'deleted':
                    await delete_document(doc_id)
                    
            elif event.operation == ChangeOperation.UPSERT:
                if existing is None:
                    # New document
                    await index_new_document(event.source_path, {})
                else:
                    # Check if actually changed (avoid duplicate processing)
                    if has_document_changed(event.source_path, existing['content_hash']):
                        await update_document(event.source_path, doc_id)
            
            await queue.task_done()
            
        except asyncio.TimeoutError:
            # No events in queue, do a brief reconciliation scan
            await reconcile_scan(limit=100)
        except Exception as e:
            log_error(f"Indexing worker error: {e}")
            await asyncio.sleep(5)  # backoff before retrying
```

### Reconciliation Scan

Even with webhooks, events can be missed (network failures, service restarts, webhook delivery failures). A periodic reconciliation scan walks through all source documents, compares their hash/timestamp against the registry, and queues anything that looks out of sync.

```python
async def reconcile_scan(limit: int = 1000):
    """
    Scan source documents and find any that are out of sync with the index.
    Runs periodically as a safety net alongside event-driven updates.
    """
    source_paths = await list_all_source_documents()  # From S3, filesystem, etc.
    
    scanned = 0
    for path in source_paths[:limit]:
        doc_id = generate_doc_id(path)
        registry_entry = await registry.get(doc_id)
        
        if registry_entry is None:
            # Document exists in source but not in index
            await queue_for_indexing(path, ChangeOperation.UPSERT)
            
        elif registry_entry['status'] == 'indexed':
            # Document is indexed — check if still current
            if has_been_modified_since(path, registry_entry['last_modified']):
                await queue_for_indexing(path, ChangeOperation.UPSERT)
        
        scanned += 1
    
    # Also check for documents in registry that no longer exist in source
    indexed_paths = await registry.get_all_active_paths()
    source_path_set = set(source_paths)
    
    for path in indexed_paths:
        if path not in source_path_set:
            doc_id = generate_doc_id(path)
            await queue_for_deletion(doc_id)
```

Run the full reconciliation scan nightly. Run a partial scan (recent documents only) every few hours as a safety net.

---

## Blue-Green Index Pattern for Zero-Downtime Updates

For systems where even a brief window of inconsistency is not acceptable (financial systems, compliance-critical applications), use a blue-green index pattern.

Instead of updating the index in place, maintain two complete index collections: blue (currently serving traffic) and green (being updated).

```
Current state:
  Blue index: serving all queries (current)
  Green index: empty or outdated (standby)

Update process:
  1. Re-index updated documents into Green
  2. Verify Green index quality (run eval queries)
  3. Switch traffic from Blue to Green (atomic pointer swap)
  4. Blue is now standby
  5. Next update cycle: repeat with Blue and Green swapped
```

```python
class BlueGreenIndexManager:
    def __init__(self, vector_db_client):
        self.client = vector_db_client
        self.active_collection = "documents_blue"
        self.standby_collection = "documents_green"
    
    async def prepare_update(self, changed_docs: list[str]):
        """Index changes into the standby collection."""
        
        # Copy active collection to standby (or sync only changed docs)
        await self.sync_standby(changed_docs)
        
        # Verify standby quality
        quality_ok = await self.verify_standby_quality()
        
        if not quality_ok:
            raise Exception("Standby index quality check failed — not switching")
    
    async def switch_traffic(self):
        """Atomically swap active and standby collections."""
        self.active_collection, self.standby_collection = (
            self.standby_collection, self.active_collection
        )
        # Update config/service discovery so queries route to new active
        await self.update_routing_config(self.active_collection)
    
    def get_active_collection(self) -> str:
        return self.active_collection
```

Blue-green adds operational complexity and doubles storage costs. Use it when freshness lag and consistency windows are hard requirements, not as a default.

---

## Handling Specific Freshness Challenges

### Versioned Documents

Some documents are explicitly versioned. A policy document may have v1.0, v2.0, v2.1. You need to decide:

- **Keep only the latest version:** Simple, but users cannot query historical state.
- **Keep all versions:** Increases index size. Requires version-aware retrieval (filter to latest by default, allow historical queries with explicit version filter).
- **Keep N most recent versions:** A compromise.

Store version in metadata and use `document_status` to mark superseded versions:

```python
# When a new version is uploaded:
async def handle_new_version(new_path: str, doc_family_id: str):
    # Mark all previous versions as superseded
    await registry.update_where(
        condition={"doc_family_id": doc_family_id, "status": "indexed"},
        updates={"document_status": "superseded"}
    )
    
    # Update vector DB metadata for old chunks
    await vector_db.update_payload(
        filter={"doc_family_id": doc_family_id},
        payload={"document_status": "superseded"}
    )
    
    # Index new version as active
    await index_new_document(new_path, {
        "doc_family_id": doc_family_id,
        "document_status": "active"
    })
```

Default retrieval filter: `document_status = active`. Historical queries: remove the filter or specify version explicitly.

### Embedding Model Updates

When you upgrade your embedding model, every existing vector in the index becomes incompatible with new query vectors. You must re-embed the entire corpus.

This is expensive and cannot be done incrementally in the traditional sense. Strategies:

**Full re-index with blue-green:** Build the new index from scratch in the standby collection using the new model, then switch traffic. This is the cleanest approach but requires 2x storage during the transition.

**Rolling re-embed:** Split the corpus into batches. Re-embed and update each batch. Keep track of which chunks use the old model vs. the new model. At query time, embed the query with both models and merge results, routing old-model chunks to old-model search and new-model chunks to new-model search. This works but is complex to maintain.

**Dual-write period:** For a transition period, write new documents with both the old and new embedding models. Gradually re-embed old documents in the background. Once all documents are re-embedded with the new model, remove the old vectors and switch fully.

Store the embedding model name and version in chunk metadata — you will need it:

```python
{
  "embedding_model": "text-embedding-3-large",
  "embedding_model_version": "2024-01-15",
  "embedding_dimension": 3072
}
```

### Pipeline Version Changes

When you change chunking strategy, metadata schema, or pre-processing logic, old chunks in the index reflect the old pipeline. Track the pipeline version:

```python
CURRENT_PIPELINE_VERSION = 4  # Increment when pipeline changes

# In registry:
{
  "pipeline_version": 3,  # This document was indexed with pipeline v3
  "status": "indexed"
}

# At startup, find documents indexed with old pipeline versions:
stale_pipeline_docs = await registry.find_where(
    condition={"pipeline_version": {"$lt": CURRENT_PIPELINE_VERSION}}
)
# Queue them for re-indexing with current pipeline
```

---

## Monitoring Data Freshness

Build explicit freshness monitoring into your system from the start.

**Freshness lag metric:** Track the time from document modification to successful indexing. Alert when this exceeds your SLA (e.g., alert if any document has been waiting more than 2 hours to be indexed).

```python
async def compute_freshness_metrics():
    # Documents modified but not yet re-indexed
    pending_updates = await registry.find_where(
        condition={
            "last_modified": {"$gt": "indexed_at"},  # modified after last index
            "status": "indexed"
        }
    )
    
    if pending_updates:
        max_lag_hours = max(
            (doc['last_modified'] - doc['indexed_at']).total_seconds() / 3600
            for doc in pending_updates
        )
        metrics.gauge("rag.freshness.max_lag_hours", max_lag_hours)
        
        if max_lag_hours > 2:
            alert("RAG index freshness SLA breached: some documents are "
                  f"{max_lag_hours:.1f} hours stale")
    
    # Failed indexing jobs
    failed_docs = await registry.count_where(condition={"status": "failed"})
    metrics.gauge("rag.indexing.failed_count", failed_docs)
```

**Index coverage metric:** What percentage of known source documents are currently indexed and active? Drops in coverage indicate bulk deletion events, source system migrations, or indexing failures.

**Queue depth metric:** How many documents are waiting to be indexed? A growing queue means your indexing workers cannot keep up with the rate of change.

---

## Summary

- Data freshness is a silent failure mode. Stale indexes return confident wrong answers without any error signal.
- Full re-indexing does not scale. Incremental indexing processes only changed documents, reducing compute waste by 90%+ in typical corpora.
- Change detection requires a document registry tracking content hashes, timestamps, and indexing state. Build this database table from day one.
- The three change detection approaches — content hashing, timestamp comparison, and source system webhooks — are complementary. Use webhooks for low latency, hashing for correctness, and a reconciliation scan as a safety net.
- Update operations must delete old chunks before inserting new ones. Never insert-then-delete — it creates a consistency window where mixed old/new content is retrievable.
- Blue-green indexing provides zero-downtime updates at the cost of 2x storage and operational complexity. Use it when consistency is a hard requirement.
- Embedding model upgrades require full corpus re-embedding. Store embedding model version in metadata and plan for this from the start.
- Build freshness lag, queue depth, and coverage monitoring from day one. Without visibility, you will not know your index is stale until users complain.

---

## What's Next

Lesson 2.7 covers parent-child chunking and hierarchical indexing in depth — the implementation details, variant strategies, and how to design retrieval logic that intelligently chooses which level of the hierarchy to return based on query type.