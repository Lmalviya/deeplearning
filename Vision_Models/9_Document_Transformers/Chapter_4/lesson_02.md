# 4.2 Postgres Schema Design

## Problem

Given the placement decisions from Lesson 4.1, the relational database needs an actual schema
capable of representing: documents (which may be resubmitted, so need dedup support),
multi-page documents where each page may have gone through a different extraction path,
document-level predictions tied to a specific taxonomy version, a class taxonomy that grows
over time, and a human review/correction trail. Designing this schema loosely ("we'll figure it
out as we go") leads to exactly the kind of retrofit pain Lesson 4.1 warned against.

## Solution / Concept: Core Tables

```sql
-- Class taxonomy: classes are DATA, not a hardcoded enum or column set.
-- This is the schema-level decision that makes 5 -> 50+ class growth a data operation,
-- not a migration (elaborated in Lesson 4.3).
CREATE TABLE classes (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,                 -- e.g. 'invoice', 'contract'
    taxonomy_version TEXT NOT NULL,                 -- version this class was introduced in
    status          TEXT NOT NULL DEFAULT 'active', -- 'active' | 'deprecated'
    parent_class_id INTEGER REFERENCES classes(id), -- nullable; enables hierarchy later (Ch 9.2)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    deprecated_at   TIMESTAMPTZ
);
CREATE UNIQUE INDEX idx_classes_name_version ON classes(name, taxonomy_version);

-- One row per submitted document (not per page).
CREATE TABLE documents (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash      TEXT NOT NULL,                 -- SHA-256 of raw bytes, drives idempotency (Ch 3.2)
    source_lane       TEXT NOT NULL,                 -- 'batch' | 'realtime' (Ch 1.1 traffic split)
    batch_id          UUID REFERENCES batches(id),    -- null for standalone real-time submissions
    object_storage_uri TEXT NOT NULL,                 -- raw file location (Ch 4.1)
    status            TEXT NOT NULL DEFAULT 'queued', -- 'queued'|'processing'|'completed'|'failed'
    page_count        INTEGER,
    submitted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at      TIMESTAMPTZ
);
CREATE UNIQUE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_batch_id ON documents(batch_id);

-- One row per page actually processed (not necessarily every page — early-exit means
-- some pages of a document are never touched, per the aggregation design in Ch 2.4).
CREATE TABLE pages (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id        UUID NOT NULL REFERENCES documents(id),
    page_number        INTEGER NOT NULL,
    extraction_method  TEXT NOT NULL,   -- 'direct_text' | 'ocr' | 'htr'
    extracted_text     TEXT,
    ocr_confidence     REAL,            -- null for direct_text pages
    rendered_image_uri TEXT,            -- object storage; null if no rendering was needed
    embedding          VECTOR(768),     -- pgvector; see Ch 4.1 on embedding placement
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX idx_pages_document_page ON pages(document_id, page_number);

-- Document-level prediction — the actual output of the system.
CREATE TABLE predictions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id       UUID NOT NULL REFERENCES documents(id),
    class_id          INTEGER NOT NULL REFERENCES classes(id),
    confidence        REAL NOT NULL,
    taxonomy_version  TEXT NOT NULL,     -- redundant with classes.taxonomy_version, stored
                                          -- directly so historical predictions remain
                                          -- interpretable even if the class row later changes
    model_version     TEXT NOT NULL,     -- which embedding backbone / model build produced this
    pages_used        INTEGER NOT NULL,  -- how many pages early-exit actually consumed (Ch 2.4)
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_reviewed        BOOLEAN NOT NULL DEFAULT false,
    reviewer_class_id  INTEGER REFERENCES classes(id),  -- corrected label, if reviewed & wrong
    reviewed_at         TIMESTAMPTZ
);
CREATE INDEX idx_predictions_document_id ON predictions(document_id);
CREATE INDEX idx_predictions_class_confidence ON predictions(class_id, confidence);
CREATE INDEX idx_predictions_needs_review ON predictions(is_reviewed) WHERE is_reviewed = false;

-- Batch submissions (Ch 3.1's async batch contract).
CREATE TABLE batches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status          TEXT NOT NULL DEFAULT 'queued', -- 'queued'|'processing'|'completed'|'partial_failure'
    document_count  INTEGER NOT NULL,
    completed_count INTEGER NOT NULL DEFAULT 0,
    callback_url    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);
```

## Key Design Decisions Explained

- **Classes as a table, not an enum or hardcoded set of columns.** This is the single most
  consequential decision in this schema — it's what makes Lesson 4.3's "no-downtime class
  growth" possible at all. An `ENUM` type in Postgres requires `ALTER TYPE ... ADD VALUE` for
  every new class (and historically required careful handling around transactions); a
  foreign-key reference to a normal table just needs an `INSERT`.
- **`content_hash` unique index on `documents`** directly implements the idempotency design
  from Chapter 3.2 at the data layer — a duplicate submission is caught by a unique constraint
  violation, not application-level logic alone.
- **`taxonomy_version` stored redundantly on `predictions`**, not just referenced via
  `classes.taxonomy_version`, so that a historical prediction remains fully interpretable even
  if the referenced class row's own version field is later updated for unrelated reasons — this
  guards the auditability requirement from Chapter 1.1.
- **`pages_used` on `predictions`** records how much of the early-exit budget (Ch 2.4) was
  actually consumed — valuable both for auditing and for tuning the early-exit confidence
  threshold against real data later.
- **Soft deprecation (`status`, `deprecated_at`) on `classes`, never a hard `DELETE`** — a
  deleted class row would break every historical prediction's foreign key; deprecation must be
  reversible/inspectable, not destructive.

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Separate `pages` table (one row per processed page) vs. storing all page data as JSON inside `documents` | Enables proper indexing, querying, and joins per page (e.g., "all OCR pages with confidence below X") | More tables to join for a full document view; requires the application layer to assemble the full picture from multiple queries |
| `pgvector` column directly on `pages` vs. a separate embeddings table | Simpler schema, one less join for the common case (fetch a page and its embedding together) | Couples embedding storage lifecycle to page storage lifecycle — if embeddings are later moved to a dedicated vector index (Ch 9.3), this column needs a migration |
| Redundant `taxonomy_version` on `predictions` | Protects historical auditability even against future changes to the `classes` table | Minor denormalization — one more field to keep consistent at write time |

## Summary

The schema's central design choice is representing the class taxonomy as ordinary relational
data (a `classes` table with a foreign key from `predictions`), not as a hardcoded type or
column set — this single decision is what allows Chapter 4.3's "add class #51 without downtime"
to be true. Beyond that, the schema separates documents from their (possibly partial, due to
early-exit) processed pages, keeps predictions auditable via redundant taxonomy versioning, and
supports the idempotency and batch-submission contracts already established in Chapter 3
directly at the data layer via constraints, not just application logic.