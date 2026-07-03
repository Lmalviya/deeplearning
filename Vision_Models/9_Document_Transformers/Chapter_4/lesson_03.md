# 4.3 Schema Evolution: 5 → 50+ Classes Without Downtime

## Problem

Chapter 1.1 states the class taxonomy must grow from 5 toward ~50 classes without full
retraining *or* downtime. Chapter 2.3 solved the retraining half of this (embedding +
prototype/KNN architecture). This lesson addresses the other half: what actually has to happen
at the data layer when a new class is introduced, and why a naive schema design would force a
migration — and likely downtime — every time.

## Solution / Concept: Adding a Class Is an INSERT, Not a Migration

Because the schema from Lesson 4.2 represents classes as rows in a `classes` table rather than
as an `ENUM` type or a fixed set of columns, adding class #51 is:

```sql
INSERT INTO classes (name, taxonomy_version, status)
VALUES ('bank_statement', '2027-01-15', 'active');
```

No `ALTER TABLE`, no schema migration, no application redeploy required to make this new class
exist in the system. The classification pipeline (Ch 2.3) picks it up as soon as reference
embeddings are computed and added for it — a data operation, not a code or schema change.

**Contrast with what a naive schema would require:**

```sql
-- If class had been modeled as an ENUM type:
ALTER TYPE document_class ADD VALUE 'bank_statement';
-- Historically required running outside a transaction block in Postgres, and depending on
-- version/usage, could require careful sequencing around concurrent readers/writers —
-- exactly the kind of operational risk a "no downtime" requirement rules out.

-- If class had been modeled as a fixed column per class (e.g., a wide predictions table
-- with one boolean/score column per class):
ALTER TABLE predictions ADD COLUMN is_bank_statement BOOLEAN;
-- Requires a schema migration and, at 100M-row-scale tables (Ch 1.2), can be a slow,
-- lock-heavy operation depending on Postgres version and migration tooling.
```

Both alternatives turn "add a class" into a schema-migration event — exactly the operational
risk the chosen design (classes as ordinary rows) avoids entirely.

## Full Lifecycle of Adding a Class, End to End

1. **Insert the new class row** (`status = 'active'`, new `taxonomy_version`) — no downtime,
   takes effect immediately for new submissions.
2. **Collect reference examples** for the new class and compute their embeddings using the
   existing, unchanged embedding backbone (Ch 2.3) — this can happen entirely offline, with no
   impact on the live serving path until step 3.
3. **Add the new reference embeddings** to the comparison set used at inference time — at MVP
   scale (Ch 4.1), this is simply new rows; at larger scale, this is an update to the vector
   index (Ch 9.3).
4. **Existing predictions are unaffected** — they reference the `taxonomy_version` they were
   made under (Ch 4.2's redundant `taxonomy_version` field), so historical data remains
   interpretable exactly as before.
5. **New submissions are now eligible to be classified into the new class**, without any
   deploy, migration, or downtime having occurred.

## Removing or Renaming a Class — Not Symmetric With Adding

Adding a class is safe and cheap; removing or renaming one is not, and needs an explicit
deprecation process rather than a `DELETE` or in-place rename:

```sql
UPDATE classes SET status = 'deprecated', deprecated_at = now()
WHERE name = 'old_class_name';
```

A hard delete would violate every historical prediction's foreign key reference. An in-place
rename would silently change the meaning of historical predictions that referenced the old
name/id, breaking the auditability requirement from Chapter 1.1. The correct pattern —
soft-deprecate, never delete or silently rename, communicate a deprecation window to API
consumers (Ch 3.2) — trades a small amount of ongoing "dead row" storage for correctness and
auditability, which is the right trade given the stated requirements.

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Classes as table rows (chosen) vs. ENUM type or fixed columns | Zero-downtime, zero-migration class additions; directly satisfies the Ch 1.1 requirement | Slightly more complex queries (a join to `classes` instead of reading a literal value) — a negligible cost against the alternative's operational risk |
| Soft deprecation vs. hard delete/rename for removing a class | Preserves historical auditability and referential integrity | Requires ongoing bookkeeping (deprecated rows never truly go away, deprecation windows must be tracked and communicated) |

## When to Use / When Not To

- **This pattern (classes as data) should be adopted from the very first version of the
  schema**, even at 5 classes — retrofitting it after an ENUM-based or fixed-column design is
  already in production, with real historical data referencing it, is a much harder migration
  than simply starting with the flexible design.
- **The deprecation-window process** becomes operationally relevant the first time a class
  actually needs to be removed or split — worth having the process documented and agreed with
  API consumers (Ch 3.2) before that first real occurrence, not improvised in the moment.

## Summary

The "no downtime to add a class" requirement is satisfied entirely by a data-modeling decision
made back in Lesson 4.2 — representing classes as rows in a table rather than as a rigid type
or column set — combined with the embedding-based classification architecture from Chapter 2.3
that doesn't require retraining to recognize a new class. Adding class #51 becomes an `INSERT`
plus an offline embedding-computation step; removing or renaming a class is handled via
soft deprecation, never a destructive schema or data change, to preserve the auditability
requirement stated in Chapter 1.1.