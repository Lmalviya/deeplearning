# Lesson 8.1 — Consistency vs Availability (CAP in Practice)

> **Chapter 8 — Core Tradeoffs**
> Previous: [Lesson 7.5 — 10M+ DAU](../chapter-7/lesson-7.5-10m-plus.md) | Next: [Lesson 8.2 — Latency vs Throughput](./lesson-8.2-latency-vs-throughput.md)

---

## What this lesson covers

- The real meaning of this tradeoff beyond CAP theory
- Concrete scenarios where you must choose one over the other
- The spectrum of consistency models and when each is appropriate
- How to make the consistency vs availability decision for each piece of data in your system
- The common mistake of applying one consistency model to everything

---

## 1. The Tradeoff Restated Simply

When your system experiences a failure (network partition, node crash, slow network), you face a choice for every request:

**Choose Consistency:** Return an error or refuse to answer. Do not risk returning stale or wrong data.

**Choose Availability:** Return the best answer you have, even if it might be slightly stale. Never return an error.

Outside of failure scenarios, this becomes a latency vs freshness tradeoff:
- **Strong consistency** requires coordination across nodes → adds latency
- **Eventual consistency** serves immediately from local state → fast but may be stale

---

## 2. The Consistency Spectrum in Practice

Do not think of this as binary. Every piece of data you store sits somewhere on this spectrum:

```
Strongest ←──────────────────────────────────────── Weakest
                                                              
Linearizable → Sequential → Causal → Read-your-writes → Eventual
     │               │          │            │               │
 "Instant"       "Ordered"  "Causally    "I see my      "Eventually
  globally       globally    related"     own writes"    consistent"
                              ordered
     │               │          │            │               │
~180ms cross     ~50ms       ~10ms         ~5ms           ~1ms
region round     region      local         local          local
trip             coord       coord
```

### Linearizability — the gold standard (and most expensive)

Every operation appears to execute instantaneously at some point in time. All clients see the same ordering.

```
User A reads balance: $500
User B simultaneously transfers $200 to User A
User A reads balance again: must see $700 (or $500 if transfer not yet committed)
No client ever sees $500 after seeing $700.
```

**Cost:** Every read must contact a quorum of nodes. In a multi-region setup, adds cross-region round trip (~180ms).

**Use for:** Financial balances, inventory counts, unique constraint enforcement.

### Read-your-writes — the practical minimum for user-facing writes

After you write something, your subsequent reads see your write. Other users may see stale data temporarily.

```
User updates their profile name
User's next request: always sees the new name ✅
Other users: may see old name for up to TTL period
```

**Cost:** Route reads for the same user to the same node/region as their write, or use a primary for reads after writes.

**Use for:** Almost all user-facing writes. The user expects to see their own change immediately.

### Eventual consistency — the default for read-heavy shared data

All nodes will eventually agree on the same value. During the window of inconsistency, different clients may see different values.

```
Post goes viral: view count incremented by 1,000,000 users simultaneously
Node A shows: 4,231,847 views
Node B shows: 4,230,991 views
Both are "right" — propagation is in progress
In 50ms, both nodes agree
```

**Cost:** Near zero — serve from local state immediately.

**Use for:** View counts, like counts, social feeds, recommendations.

---

## 3. The Decision Framework — Per Data Type

The mistake most teams make is applying one consistency model to all their data. The right approach is to categorize each piece of data by the cost of inconsistency:

### Category 1 — Inconsistency is catastrophic

**Examples:** Account balance, inventory count, payment status, access permissions, password hash, unique username.

**Failure mode:** Showing $500 balance when actually $200. Selling 10 items when only 1 is in stock.

**Required consistency:** Strong (linearizable or at least read-your-writes with synchronous replication).

**Design decision:** Use synchronous replication. Accept higher write latency. Never cache these values without a write-through strategy.

```python
# Payment check — never serve stale data
def check_payment_status(payment_id: str) -> str:
    # Always read from primary — no replica, no cache
    return primary_db.query(
        "SELECT status FROM payments WHERE id = ?", payment_id
    )
```

### Category 2 — Inconsistency is annoying but recoverable

**Examples:** User profile name/avatar, friend list, notification preferences, post content after editing.

**Failure mode:** User updates their avatar, sees old avatar for 30 seconds. Friend added, does not appear in list immediately.

**Required consistency:** Read-your-writes (user sees their own changes) + eventual consistency for others.

**Design decision:** Serve from cache with short TTL. Invalidate cache on write. Accept that other users may see stale data for the TTL period.

```python
def update_profile(user_id: int, data: dict):
    db.execute("UPDATE users SET name = ? WHERE id = ?", data['name'], user_id)
    redis.delete(f"user:{user_id}")  # invalidate immediately

    # Flag: this user must read from primary for next 5 seconds
    redis.setex(f"primary_read:{user_id}", 5, "1")

def get_profile(user_id: int, requesting_user_id: int):
    if user_id == requesting_user_id:
        # User viewing their own profile — must see their changes
        if redis.exists(f"primary_read:{user_id}"):
            return primary_db.query("SELECT * FROM users WHERE id = ?", user_id)

    # Others viewing this profile — eventual consistency OK
    return get_from_cache_or_replica(user_id)
```

### Category 3 — Inconsistency is invisible or acceptable

**Examples:** View counts, like counts, trending lists, recommendations, search results, activity feeds.

**Failure mode:** Showing 4.2M views instead of 4.2001M views. Recommendation engine suggests something you saw yesterday.

**Required consistency:** Eventual. Serve from nearest node immediately.

**Design decision:** No coordination. Accept that different users see different values. Reconcile in background.

```python
def increment_view_count(post_id: int):
    # Write to local Redis (no coordination with other regions)
    redis.incr(f"views:{post_id}")
    # Background job periodically syncs all regions to DB

def get_view_count(post_id: int) -> int:
    # Read from local Redis — may differ from other regions by seconds
    return int(redis.get(f"views:{post_id}") or 0)
```

---

## 4. Consistency Under Failure — The Hard Scenario

The previous examples deal with normal operation. The real CAP tradeoff appears during failures.

### Scenario: Network partition between US and EU regions

```
Normal:
  US ←──── 80ms ────→ EU
  (both regions can communicate)

Partition:
  US ✗✗✗✗✗✗✗✗✗✗✗✗✗✗ EU
  (network link is down, cannot communicate)

A user in India is trying to change their password.
Their request hits the EU region (nearest to India).
```

**If you chose CP (Consistency over Availability):**
```
EU region: "I cannot confirm this write will reach US primary. Refusing."
User: gets an error → "Service unavailable, try again later"

Cost: user is frustrated, but data is always correct.
Appropriate for: password changes, payment operations.
```

**If you chose AP (Availability over Consistency):**
```
EU region: "I'll accept this write and sync to US when partition heals."
User: password changed in EU region ✅
Meanwhile: US region still has old password
User logs in from US: can use OLD password ← security issue ❌

After partition heals: conflict must be resolved (last-write-wins? EU version? US version?)

Cost: user experience is good, but data may be wrong.
Appropriate for: view counts, social feeds, non-critical preferences.
```

### The practical rule

For any data where inconsistency could cause security issues, financial loss, or regulatory violation: choose CP. Accept that during a partition, some operations fail rather than risk incorrect data.

For everything else: choose AP. Users prefer a slightly stale social feed to an error page.

---

## 5. Practical Patterns for Each Choice

### Pattern 1 — Synchronous replication (CP for critical writes)

```sql
-- PostgreSQL: synchronous replication to at least one replica
-- In postgresql.conf:
synchronous_commit = on
synchronous_standby_names = 'replica-1'  -- at least one must confirm

-- Now every write waits for replica-1 to confirm before ACK
-- If replica-1 is unavailable: writes block
-- Trade: higher write latency (~5ms extra per write) for zero data loss on failover
```

### Pattern 2 — Write to quorum (AP with tunable consistency)

```python
# Cassandra: write to quorum of nodes
session.execute(
    "INSERT INTO accounts (id, balance) VALUES (%s, %s)",
    (account_id, balance),
    consistency_level=ConsistencyLevel.QUORUM  # majority must confirm
)

# Read from quorum: at least one node that confirmed write will respond
# → Consistent reads even with eventual replication
row = session.execute(
    "SELECT balance FROM accounts WHERE id = %s",
    (account_id,),
    consistency_level=ConsistencyLevel.QUORUM
)
```

### Pattern 3 — Optimistic concurrency (detect conflicts rather than prevent them)

```python
# Instead of locking, detect if data changed since you read it
def transfer_funds(from_id, to_id, amount):
    from_account = db.query("SELECT balance, version FROM accounts WHERE id = ?", from_id)

    if from_account.balance < amount:
        raise InsufficientFunds()

    # Conditional update: only update if version has not changed
    rows_updated = db.execute("""
        UPDATE accounts
        SET balance = balance - ?, version = version + 1
        WHERE id = ? AND version = ?
    """, amount, from_id, from_account.version)

    if rows_updated == 0:
        # Another transaction modified this account concurrently
        raise ConflictError("Account modified concurrently, please retry")

    db.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", amount, to_id)
```

This allows concurrent reads without locking, detects conflicts on write, and asks the caller to retry on conflict. Correct behavior without sacrificing availability.

---

## Summary

- Consistency vs availability is not binary — every piece of data has the right consistency level for its failure cost
- **Catastrophic inconsistency** (financial, security, unique constraints): strong consistency, synchronous replication
- **Annoying inconsistency** (user profile, friend list): read-your-writes consistency, short-TTL caching
- **Invisible inconsistency** (view counts, feeds): eventual consistency, serve immediately from local state
- During a network partition: CP systems refuse (error), AP systems serve stale (wrong but available)
- The common mistake: applying strong consistency to everything (adds unnecessary latency) or eventual consistency to everything (causes financial and security bugs)
- Pattern: use quorum writes/reads for tunable consistency, optimistic concurrency for conflict detection without locking

---

## ⚠️ Common Mistakes

- Using eventual consistency for inventory counts — overselling products is a real financial and reputational cost
- Using strong consistency for social feed view counts — adds 180ms cross-region round trip to every feed load for no user-visible benefit
- Not accounting for the read-your-writes requirement — users who cannot see their own writes believe the system is broken
- Assuming multi-region replication is synchronous by default — it almost never is; you must configure synchronous replication explicitly

---

> Next: [Lesson 8.2 — Latency vs Throughput](./lesson-8.2-latency-vs-throughput.md)