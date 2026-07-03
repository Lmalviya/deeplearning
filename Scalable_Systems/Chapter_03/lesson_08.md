# Lesson 3.8 — Schema Design for Scale

> **Chapter 3 — The Data Layer**
> Previous: [Lesson 3.7 — CAP Theorem](./lesson-3.7-cap-theorem.md) | Next: [Lesson 3.9 — Database Anti-Patterns](./lesson-3.9-database-anti-patterns.md)

---

## What this lesson covers

- Normalization vs denormalization and when to use each
- Designing for query patterns, not just entities
- Data types that matter for performance (IDs, timestamps, enums)
- Schema migrations at scale without downtime
- Soft deletes and why they cause problems

---

## 1. Normalization vs Denormalization

**Normalization** means organizing data to eliminate redundancy. Each piece of data lives in exactly one place. Changes are made once. Relationships are maintained through foreign keys.

**Denormalization** means intentionally duplicating data to make reads faster. You trade storage and write complexity for read speed.

### Fully normalized schema

```sql
CREATE TABLE users (
    id          BIGSERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    email       VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE products (
    id          BIGSERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    price       DECIMAL(10,2) NOT NULL
);

CREATE TABLE orders (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT REFERENCES users(id),
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE order_items (
    id          BIGSERIAL PRIMARY KEY,
    order_id    BIGINT REFERENCES orders(id),
    product_id  BIGINT REFERENCES products(id),
    quantity    INT NOT NULL,
    price_at_purchase DECIMAL(10,2) NOT NULL  -- snapshot of price at time of purchase
);
```

To show an order summary, you JOIN four tables. This is correct and fast when:
- Tables are reasonably sized
- Indexes are in place (on foreign keys and join columns)
- You are not running millions of these queries per second

### When to denormalize

Denormalize when:
- A specific query pattern is too slow despite good indexes
- The query runs extremely frequently (hundreds of times per second)
- The join cost is measurably hurting performance

```sql
-- Denormalized: store user name and email directly on orders
-- (redundant — same data in users table)
CREATE TABLE orders (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT REFERENCES users(id),
    user_name     VARCHAR(255),  -- duplicated from users
    user_email    VARCHAR(255),  -- duplicated from users
    created_at    TIMESTAMP DEFAULT NOW()
);
```

Now showing an order does not require joining to the users table. But if Alice changes her name, you must update every order row that contains it — or accept that old orders show the name at time of order (which is often correct for order history anyway).

**The golden rule:** denormalize for read performance, but maintain the source of truth in a normalized form. Know which copy is canonical.

---

## 2. Designing for Query Patterns

A schema that perfectly models your business domain but ignores your query patterns will be slow. Design schema and indexes together.

### Start by listing your most common queries

For a messaging app:
```
1. Get all conversations for user X (sorted by last_message_at DESC)
2. Get all messages in conversation Y (sorted by created_at ASC, paginated)
3. Get unread count for user X
4. Search messages containing keyword Z in conversation Y
```

Now design the schema to make these fast:

```sql
CREATE TABLE conversations (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMP DEFAULT NOW(),
    last_message_at TIMESTAMP DEFAULT NOW()  -- denormalized for sort
);

CREATE TABLE conversation_members (
    conversation_id BIGINT REFERENCES conversations(id),
    user_id         BIGINT REFERENCES users(id),
    last_read_at    TIMESTAMP,
    PRIMARY KEY (conversation_id, user_id)
);

CREATE TABLE messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT REFERENCES conversations(id),
    sender_id       BIGINT REFERENCES users(id),
    body            TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Index for query 1: conversations for user X, sorted by last_message_at
CREATE INDEX idx_conv_members_user ON conversation_members(user_id);
-- (JOIN with conversations on conversation_id, ORDER BY last_message_at)

-- Index for query 2: messages in a conversation, sorted by time
CREATE INDEX idx_messages_conv_time ON messages(conversation_id, created_at);

-- Query 3: unread count — compute from last_read_at vs message created_at
-- (no extra index needed if the above indexes are in place)
```

Notice `last_message_at` in the conversations table — this is a denormalization. The last message timestamp lives on messages, but we store a copy on conversations so we can sort conversations without a subquery.

---

## 3. Choosing the Right Data Types

Data type choices affect storage size, index size, and query performance.

### IDs — BIGINT vs UUID

```sql
-- BIGINT (8 bytes) — auto-increment
id BIGSERIAL PRIMARY KEY

-- UUID (16 bytes) — globally unique
id UUID DEFAULT gen_random_uuid() PRIMARY KEY
```

| | BIGINT | UUID |
|---|---|---|
| Size | 8 bytes | 16 bytes |
| Index size | Smaller | 2× larger |
| Insert speed | Fast (sequential) | Slower (random — causes B-tree fragmentation) |
| Globally unique | No (only within this DB) | Yes (safe across DBs, shards, services) |
| Human readable | Yes (42) | No (550e8400-e29b-41d4...) |

**Use BIGINT when:** Single database, no sharding, no need to merge IDs from multiple sources.

**Use UUID when:** Microservices where multiple services generate IDs, sharding (no central ID generator), or when IDs must be unguessable (UUIDs are harder to enumerate than sequential integers).

**UUID v7 (recommended for new systems):** UUID v7 is time-ordered — UUIDs generated later sort after UUIDs generated earlier. This avoids the B-tree fragmentation of random UUIDs while keeping global uniqueness.

```sql
-- UUID v7 gives you sorted UUIDs — better index performance
-- Available in PostgreSQL 17+ or via extension in earlier versions
```

### Timestamps

```sql
-- TIMESTAMP (no timezone) — stores local time, causes bugs across timezones
created_at TIMESTAMP

-- TIMESTAMPTZ (with timezone) — stores UTC, converts on display
created_at TIMESTAMPTZ DEFAULT NOW()
```

**Always use TIMESTAMPTZ.** Store everything in UTC. Convert to the user's timezone only at display time. Mixing timezones in the database is a source of extremely hard-to-debug bugs.

### Enums vs VARCHAR vs Lookup Tables

```sql
-- Option 1: PostgreSQL ENUM type
CREATE TYPE order_status AS ENUM ('pending', 'processing', 'shipped', 'delivered', 'cancelled');
ALTER TABLE orders ADD COLUMN status order_status;

-- Option 2: VARCHAR with constraint
ALTER TABLE orders ADD COLUMN status VARCHAR(20) CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled'));

-- Option 3: Lookup table (most flexible)
CREATE TABLE order_statuses (id INT PRIMARY KEY, name VARCHAR(20) UNIQUE);
INSERT INTO order_statuses VALUES (1,'pending'),(2,'processing'),(3,'shipped'),(4,'delivered'),(5,'cancelled');
ALTER TABLE orders ADD COLUMN status_id INT REFERENCES order_statuses(id);
```

| | PostgreSQL ENUM | VARCHAR + CHECK | Lookup table |
|---|---|---|---|
| Adding new value | Requires ALTER TYPE (table rewrite in old PG versions) | ALTER TABLE | INSERT a row |
| Storage | Tiny (4 bytes) | VARCHAR size | 4-byte INT + join |
| Query readability | `WHERE status = 'pending'` | Same | Requires JOIN |
| Flexibility | Low | Medium | High |

**Recommendation:** For small, stable value sets (order statuses, user roles), PostgreSQL ENUM is fine. For frequently changing value sets, a lookup table gives you flexibility. Avoid putting all your enums in a single generic "lookup" table — it becomes a god table.

---

## 4. Schema Migrations at Scale Without Downtime

As your application evolves, you need to change the schema. On a small database, `ALTER TABLE` completes in milliseconds. On a 500M row table, the same command might lock the table for hours.

### Why ALTER TABLE locks the table

```
ALTER TABLE orders ADD COLUMN refund_amount DECIMAL;
```

In older PostgreSQL versions, adding a nullable column required rewriting every row in the table. The table was locked for writes during the entire rewrite. For 500M rows, this could take hours — total downtime.

**PostgreSQL 11+ improvement:** Adding a nullable column with no default is now instant (no rewrite). But adding a column with a non-null default still rewrites the table.

### The safe migration pattern for large tables

**Rule:** Never run a long-blocking migration on a production table without a plan.

#### Adding a non-null column with a default

```sql
-- WRONG on large table — rewrites all rows while table is locked
ALTER TABLE orders ADD COLUMN fee_rate DECIMAL NOT NULL DEFAULT 0.05;

-- RIGHT — three steps, no downtime
-- Step 1: Add column as nullable (instant)
ALTER TABLE orders ADD COLUMN fee_rate DECIMAL;

-- Step 2: Backfill in batches (background process, no lock)
UPDATE orders SET fee_rate = 0.05 WHERE fee_rate IS NULL AND id BETWEEN 1 AND 100000;
UPDATE orders SET fee_rate = 0.05 WHERE fee_rate IS NULL AND id BETWEEN 100001 AND 200000;
-- ... repeat in batches, overnight

-- Step 3: Add NOT NULL constraint with default (still fast after backfill)
ALTER TABLE orders ALTER COLUMN fee_rate SET NOT NULL;
ALTER TABLE orders ALTER COLUMN fee_rate SET DEFAULT 0.05;
```

#### Adding an index without locking

```sql
-- WRONG — locks the table for the entire index build duration
CREATE INDEX idx_orders_user ON orders(user_id);

-- RIGHT — uses CONCURRENTLY (slower, but does not lock)
CREATE INDEX CONCURRENTLY idx_orders_user ON orders(user_id);
```

`CREATE INDEX CONCURRENTLY` builds the index in the background while allowing reads and writes. It takes longer but does not block.

#### Dropping a column safely

```sql
-- Step 1: Remove all code references to the column
-- Step 2: Ignore the column at the application level for one deploy cycle
-- Step 3: Drop the column (PostgreSQL marks it invisible instantly; space reclaimed by VACUUM)
ALTER TABLE orders DROP COLUMN old_column;
```

Dropping a column in PostgreSQL does not immediately reclaim disk space — it marks it as invisible. VACUUM reclaims the space later.

---

## 5. Soft Deletes — The Pattern and Its Problems

**Soft delete** means adding a `deleted_at` column instead of actually removing rows. "Deleted" rows are filtered out of queries with `WHERE deleted_at IS NULL`.

```sql
-- Hard delete (permanent)
DELETE FROM users WHERE id = 42;

-- Soft delete
UPDATE users SET deleted_at = NOW() WHERE id = 42;

-- All queries must filter soft-deleted rows
SELECT * FROM users WHERE deleted_at IS NULL;
```

**Why teams use soft deletes:**
- Recovery (undelete a mistakenly deleted record)
- Audit trail (see what was deleted and when)
- Referential integrity (other tables reference deleted rows; hard delete would cascade)

**The problems soft deletes cause:**

| Problem | Description |
|---------|-------------|
| Index bloat | Every index must include both active and deleted rows, even though deleted rows are almost never queried. Index size grows over time. |
| Query complexity | Every WHERE clause must include `AND deleted_at IS NULL`. Forgetting it in one query returns deleted data. |
| Unique constraints break | `UNIQUE (email)` prevents a new user from reusing a deleted user's email. You must include `deleted_at` in the constraint. |
| Reporting complexity | Aggregates include deleted rows unless explicitly filtered |
| Performance degradation | Over time, many deleted rows accumulate and degrade query performance despite the filter |

**Better alternatives to soft deletes:**

```sql
-- Option 1: Archive table — move deleted rows to a separate table
CREATE TABLE users_deleted AS SELECT * FROM users WHERE 1=0;
-- On delete: INSERT INTO users_deleted SELECT * FROM users WHERE id = 42; DELETE FROM users WHERE id = 42;

-- Option 2: Audit log table — append-only event log
CREATE TABLE audit_events (
    id          BIGSERIAL PRIMARY KEY,
    table_name  VARCHAR(50),
    row_id      BIGINT,
    action      VARCHAR(10),  -- 'INSERT', 'UPDATE', 'DELETE'
    old_data    JSONB,
    new_data    JSONB,
    changed_at  TIMESTAMPTZ DEFAULT NOW(),
    changed_by  BIGINT
);

-- Option 3: Partial index — if you must use soft deletes
CREATE UNIQUE INDEX idx_users_active_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_active ON users(id) WHERE deleted_at IS NULL;
-- Active queries use these partial indexes and are not slowed by deleted rows
```

---

## Summary

- Normalize first, denormalize later — only when a specific query pattern is measurably slow despite good indexes
- Design your schema around your most frequent queries, not just the data model
- BIGINT for single-DB sequential IDs; UUID v7 for globally unique sortable IDs across services
- Always use TIMESTAMPTZ — store UTC, display in user's timezone
- Schema migrations on large tables must be done in batches: add nullable, backfill, add constraint
- Use `CREATE INDEX CONCURRENTLY` to avoid locking large tables
- Soft deletes cause index bloat, query complexity, and unique constraint problems — prefer archive tables or audit logs

---

## ⚠️ Common Mistakes

- Using `TIMESTAMP` instead of `TIMESTAMPTZ` — causes timezone bugs in production that only appear for users in non-UTC timezones
- Running `ALTER TABLE ... ADD COLUMN NOT NULL DEFAULT ...` on a large table without testing the lock duration first
- Forgetting `CREATE INDEX CONCURRENTLY` and locking a production table for 30 minutes
- Soft deleting everything "for safety" without realizing the long-term cost to index performance and query complexity
- Designing a schema around the data model without considering the queries that will be run against it

---

> Next: [Lesson 3.9 — Database Anti-Patterns](./lesson-3.9-database-anti-patterns.md)