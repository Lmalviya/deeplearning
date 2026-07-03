# 3.2 Versioning, Idempotency, and Backward Compatibility

## Problem

Three distinct failure modes show up as the system evolves, and each needs a deliberate
contract decision made early, because retrofitting any of them onto an API already in use by
real clients is painful: (1) a client retries a submission after a network timeout and gets
double-processed/double-billed, (2) the API's request/response shape needs to change over time
without breaking existing integrations, and (3) the **class taxonomy itself** changes — classes
get added (5 → 50, per Ch 1.1), and clients need a well-defined way to know which taxonomy
version a given prediction was made against.

## Solution / Concept: Three Separate Contracts

### Idempotency

**Chosen approach: content-hash-based deduplication, with an optional client-supplied
idempotency key for cases where the same content is intentionally resubmitted.**

```
On submission: compute a content hash (e.g., SHA-256) of the uploaded document bytes.
If a document with the same hash was already submitted within a defined dedup window
(e.g., 24 hours), return the existing document_id and its current status/result
instead of creating a new processing record.
```

This handles the most common real cause of duplicate submissions — a client retrying after a
timeout without knowing whether the first request succeeded — automatically, without requiring
the client to do anything extra. For the less common case where a client *deliberately* wants
to reprocess identical content as a distinct request (e.g., testing, or legitimate resubmission
of a corrected version with the same bytes), an explicit `Idempotency-Key` header can be
supplied, which takes precedence over the content-hash check.

| Approach | Gain | Cost |
|---|---|---|
| Content-hash dedup only | Zero client burden — works automatically on retries | Cannot distinguish an accidental retry from an intentional resubmission of identical bytes |
| Client-supplied idempotency key only | Full client control over what counts as a duplicate | Requires every client integration to correctly generate and manage keys — a common source of bugs if omitted or reused incorrectly |
| Both, with explicit key taking precedence | Safe by default (content-hash), full control when needed | Slightly more logic in the dedup-check path — negligible cost for the safety gained |

### API Versioning

**Chosen approach: URL path versioning (`/v1/documents`), not header-based versioning.**

URL versioning is visible in logs, in browser/curl testing, and in API documentation without
needing to inspect headers — a real operational convenience, at the cost of being slightly less
"pure REST." Header-based versioning is more elegant in principle but adds friction to
debugging and is easy for client integrations to get subtly wrong (e.g., forgetting the
header and silently hitting a default version). For an API primarily consumed by backend
integrations rather than public API purists, the debugging convenience wins.

### Backward Compatibility, Specifically for the Class Taxonomy

This is the compatibility concern most specific to this system, and the one most likely to be
overlooked. Every prediction response should carry a `taxonomy_version` field:

```json
{ "document_id": "...", "label": "invoice", "confidence": 0.94, "taxonomy_version": "2026-07-03" }
```

**Rules that make taxonomy growth backward-compatible:**

- **Never repurpose an existing class identifier.** If "receipt" is ever split into
  "retail_receipt" and "restaurant_receipt," the old "receipt" label is deprecated, not
  reassigned — old predictions remain valid under the taxonomy version they were made with.
- **Additions are always backward-compatible; removals/renames are not** — adding class #51
  doesn't invalidate any client's existing integration, since old classes still mean what they
  meant. Removing or renaming a class requires a taxonomy version bump and a deprecation
  window, communicated explicitly, not silently.
- **Confidence scores are not comparable across taxonomy versions.** Adding new, potentially
  visually-similar classes can lower confidence scores for existing classes purely because the
  reference/decision space got more crowded (relevant once the embedding+prototype/KNN
  architecture from Ch 2.3 is in play) — any client logic that hardcodes a confidence threshold
  needs to be aware it may need retuning after a taxonomy version change, not just after an API
  version change.

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Separate `taxonomy_version` from API version (`/v1/`) | Class taxonomy can evolve frequently (expected, per Ch 1.1's growth requirement) without forcing API version bumps for every new class | Requires clients to actually read and handle `taxonomy_version` correctly, rather than assuming a fixed class list forever — a real integration contract to document clearly |
| Deprecation window for class removal/rename, rather than an immediate change | Existing client integrations don't break unexpectedly | Requires maintaining support for a deprecated class's meaning for some defined period — real operational overhead, not free |

## When to Use / When Not To

- **Content-hash dedup** should be in place from the very first version of the API — the cost
  of adding it later, after clients have built retry logic around its absence, is much higher
  than building it in from day one.
- **`taxonomy_version` in every response** should also be present from day one, even at 5
  classes — retrofitting it once client integrations already assume a fixed class list is a
  breaking change in itself.
- **URL versioning** is the right default unless there's a specific reason (e.g., an existing
  client ecosystem convention) to prefer headers — not a decision worth relitigating per
  project without a concrete reason.

## Summary

Idempotency, API versioning, and taxonomy backward-compatibility are three separate contracts
that all need to be decided before real client integrations exist, not after. Content-hash-based
deduplication (with an optional explicit override) handles the common retry case automatically;
URL-based API versioning trades minor REST purity for real debugging convenience; and a
separate, explicit `taxonomy_version` field — with strict rules against repurposing class IDs
and a deprecation window for removals — is what allows the class list to actually grow from 5
toward 50 (Ch 1.1's stated requirement) without breaking every existing integration each time.