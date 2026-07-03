# Lesson 4.6 — Redis Deep Dive

> **Chapter 4 — The Caching Layer**
> Previous: [Lesson 4.5 — Eviction Policies](./lesson-4.5-eviction-policies.md) | Next: [Lesson 4.7 — Distributed Caching Problems](./lesson-4.7-distributed-caching-problems.md)

---

## What this lesson covers

- Redis data structures and their real-world use cases
- Redis Cluster vs Redis Sentinel — when to use each
- Redis Pub/Sub for event broadcasting
- Atomic operations and Lua scripts
- Redis as a rate limiter, distributed lock, and session store
- Redis persistence — RDB vs AOF

---

## 1. Redis Data Structures

Redis is not just a key-value store. It is a data structure server. Each type has specific commands and use cases.

### 1.1 String

The most basic type. Stores any binary data up to 512MB. Counters are implemented as strings with atomic increment.

```python
# Basic set/get
redis.set("greeting", "hello")
redis.get("greeting")  # "hello"

# Atomic increment — race-condition-free counter
redis.set("page_views:post:42", 0)
redis.incr("page_views:post:42")   # → 1
redis.incrby("page_views:post:42", 5)  # → 6

# Set with TTL
redis.setex("session:abc123", 1800, user_id)  # expires in 30 minutes

# Set only if not exists (used for distributed locking)
redis.setnx("lock:payment:42", "1")  # → 1 if set, 0 if already exists
```

**Use cases:** Session tokens, counters, rate limit counts, feature flags, simple caching.

---

### 1.2 Hash

A map of field-value pairs stored under one key. Like a row in a database — one key holds multiple fields.

```python
# Store user as a hash
redis.hset("user:42", mapping={
    "name": "Alice Chen",
    "email": "alice@example.com",
    "city": "Bangalore",
    "plan": "premium"
})

# Get specific fields (not the whole object)
redis.hget("user:42", "name")        # "Alice Chen"
redis.hmget("user:42", "name", "plan")  # ["Alice Chen", "premium"]
redis.hgetall("user:42")             # all fields

# Atomic increment on a specific field
redis.hincrby("user:42:stats", "login_count", 1)
```

**Memory advantage:** Redis uses a compact encoding (ziplist) for hashes with fewer than 128 fields and values under 64 bytes each — much more memory-efficient than one string key per field.

**Use cases:** User profiles, session data with multiple fields, per-user counters (login count, points, credits).

---

### 1.3 List

An ordered list of strings. Supports push/pop from both ends. Implements stacks (LIFO) and queues (FIFO).

```python
# Queue (FIFO): push to tail, pop from head
redis.rpush("notifications:user:42", json.dumps(notification))  # enqueue
redis.lpop("notifications:user:42")                             # dequeue

# Stack (LIFO): push and pop from same end
redis.lpush("recent_pages:user:42", "/products/99")  # push
redis.lpop("recent_pages:user:42")                   # pop (most recent)

# Keep only last N items — rolling window
redis.lpush("activity:user:42", event)
redis.ltrim("activity:user:42", 0, 99)  # keep only 100 most recent

# Blocking pop — worker waits for items
redis.blpop("job_queue", timeout=30)  # blocks up to 30 seconds
```

**Use cases:** Activity feeds, notification queues, recent history (last N items), simple task queues.

---

### 1.4 Set

An unordered collection of unique strings. Fast membership testing, union, intersection, difference.

```python
# Followers of user:42
redis.sadd("followers:42", "user:1", "user:2", "user:3")
redis.sismember("followers:42", "user:2")  # True
redis.smembers("followers:42")             # {"user:1", "user:2", "user:3"}
redis.scard("followers:42")                # 3 (count)

# Friends in common between user:42 and user:99
mutual = redis.sinter("followers:42", "followers:99")

# Who user:42 follows that user:99 does not
redis.sdiff("following:42", "following:99")

# Unique visitors today (add user_id, cardinality = unique count)
redis.sadd("visitors:2024-01-15", user_id)
redis.scard("visitors:2024-01-15")  # unique visitor count
```

**Use cases:** Social graph (followers/following), unique item collections, tag systems, set operations.

---

### 1.5 Sorted Set (ZSet) — the most powerful structure

Like a set but each member has a **score** (floating point). Members are kept sorted by score. O(log n) for most operations.

```python
# Leaderboard — score is the player's points
redis.zadd("leaderboard:game:1", {"alice": 9500, "bob": 8200, "carol": 9800})

# Top 10 players (highest score first)
redis.zrevrange("leaderboard:game:1", 0, 9, withscores=True)
# [("carol", 9800.0), ("alice", 9500.0), ("bob", 8200.0)]

# Alice's rank (0-indexed, descending)
redis.zrevrank("leaderboard:game:1", "alice")  # 1 (second place)

# Increment score atomically
redis.zincrby("leaderboard:game:1", 200, "bob")  # bob now has 8400

# Rate limiting: sliding window
now = time.time()
window_start = now - 60  # 60-second window
redis.zadd(f"rate:user:42", {str(now): now})           # add current timestamp
redis.zremrangebyscore(f"rate:user:42", 0, window_start)  # remove old entries
count = redis.zcard(f"rate:user:42")                   # requests in last 60s
```

**Use cases:** Leaderboards, priority queues, time-series with range queries, sliding window rate limiting, ranked recommendations.

---

### 1.6 Stream

Append-only log. Designed for event streaming — similar to Kafka but simpler and in-memory.

```python
# Produce events
redis.xadd("events:orders", {
    "order_id": "ord_123",
    "user_id": "42",
    "amount": "299.98",
    "status": "created"
})

# Consumer group — multiple consumers share the stream
redis.xgroup_create("events:orders", "notification-service", id="0")

# Consume and acknowledge
messages = redis.xreadgroup("notification-service", "consumer-1",
                             {"events:orders": ">"}, count=10, block=5000)
for stream, events in messages:
    for event_id, data in events:
        process_event(data)
        redis.xack("events:orders", "notification-service", event_id)
```

**Use cases:** Lightweight event streaming within a system, activity logs, real-time data pipelines where Kafka would be overkill.

---

## 2. Redis as a Rate Limiter

Two patterns — fixed window (simple) and sliding window (accurate):

```python
# Fixed window rate limiter
def is_rate_limited_fixed(user_id: str, limit: int = 100, window: int = 60) -> bool:
    key = f"rate:fixed:{user_id}:{int(time.time() // window)}"
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, window)  # set TTL on first request
    return count > limit

# Sliding window rate limiter (sorted set)
def is_rate_limited_sliding(user_id: str, limit: int = 100, window: int = 60) -> bool:
    key = f"rate:sliding:{user_id}"
    now = time.time()
    window_start = now - window

    pipeline = redis.pipeline()
    pipeline.zremrangebyscore(key, 0, window_start)  # remove old entries
    pipeline.zadd(key, {str(now): now})               # add current request
    pipeline.zcard(key)                               # count in window
    pipeline.expire(key, window)
    results = pipeline.execute()

    count = results[2]
    return count > limit
```

The sliding window is more accurate (no burst at window boundary) but uses more memory.

---

## 3. Redis as a Distributed Lock

The standard pattern is SETNX (SET if Not eXists) with an expiry:

```python
import uuid

def acquire_lock(resource: str, timeout: int = 10) -> str | None:
    lock_id = str(uuid.uuid4())  # unique token to identify this lock holder
    key = f"lock:{resource}"

    # SET key value NX EX timeout
    # NX = only set if not exists (atomic)
    # EX = expire in `timeout` seconds (prevents deadlock if holder crashes)
    acquired = redis.set(key, lock_id, nx=True, ex=timeout)
    return lock_id if acquired else None

def release_lock(resource: str, lock_id: str) -> bool:
    key = f"lock:{resource}"
    # Must verify we own the lock before releasing (Lua for atomicity)
    lua_script = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """
    result = redis.eval(lua_script, 1, key, lock_id)
    return result == 1

# Usage
lock_id = acquire_lock("payment:user:42", timeout=30)
if lock_id:
    try:
        process_payment(user_id=42)
    finally:
        release_lock("payment:user:42", lock_id)
else:
    raise Exception("Could not acquire lock — payment already in progress")
```

The Lua script for release is crucial — it checks and deletes atomically, preventing a race condition where lock expires between the GET check and the DEL.

---

## 4. Redis Sentinel vs Redis Cluster

Two different problems, two different solutions:

### Redis Sentinel — High Availability (HA)

Sentinel monitors a primary + replicas and performs automatic failover if the primary fails.

```
Architecture:
  Primary (1)
  Replicas (2+)
  Sentinel nodes (3) — monitor and coordinate failover

When primary fails:
  1. Sentinels detect failure (via ping)
  2. Sentinels hold an election (quorum: 2 of 3 must agree)
  3. Winning sentinel promotes the most up-to-date replica to primary
  4. Other replicas begin replicating from the new primary
  5. Clients are notified of new primary address (via Sentinel API)
```

**Use Sentinel when:** You need HA (automatic failover) but your dataset fits on one server. Data volume up to ~100GB, single-digit GB/s throughput.

### Redis Cluster — Horizontal Sharding

Splits data across multiple Redis nodes. Each node holds a subset of the keyspace (16,384 hash slots distributed among nodes).

```
3 master nodes × 16,384 slots = all slots covered
  Node A: slots 0–5460     (keys hashing to these slots)
  Node B: slots 5461–10922
  Node C: slots 10923–16383

Each master has 1+ replicas for HA.

Key routing:
  CLUSTER KEYSLOT user:42 → 4775  → Node A handles this key
  CLUSTER KEYSLOT user:99 → 8093  → Node B handles this key
```

**Use Cluster when:** Dataset does not fit on one server (>100GB), or you need throughput beyond what a single server provides.

**Cluster limitations:**
- Multi-key commands only work if all keys are on the same node
- Use hash tags to force related keys to the same slot: `{user:42}:profile` and `{user:42}:cart` both route to the same slot
- Lua scripts must only access keys on one node

---

## 5. Redis Pub/Sub — Event Broadcasting

Publish a message, all subscribers receive it. Fire-and-forget — if a subscriber is offline, it misses the message.

```python
# Publisher (one service)
redis.publish("notifications:new_message", json.dumps({
    "to_user": 42,
    "from_user": 99,
    "message": "Hey!"
}))

# Subscriber (another service, runs in separate thread)
pubsub = redis.pubsub()
pubsub.subscribe("notifications:new_message")

for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        send_push_notification(data['to_user'], data['message'])
```

**Use cases:** Real-time notifications across multiple app servers (WebSocket message routing), cache invalidation broadcast (tell all app servers to clear local cache), live dashboards.

**Limitation:** No persistence. No delivery guarantee. If all subscribers are offline or slow, messages are lost. For reliable delivery, use Redis Streams or Kafka.

---

## 6. Redis Persistence — RDB vs AOF

By default, Redis is purely in-memory. On restart, all data is lost. For production, configure persistence.

### RDB (Redis Database) — Point-in-time snapshots

```
save 900 1      # save if at least 1 key changed in 900 seconds
save 300 10     # save if at least 10 keys changed in 300 seconds
save 60 10000   # save if at least 10000 keys changed in 60 seconds
```

Redis forks a child process, which writes the entire dataset to disk as a binary snapshot. Fast to restore. Low overhead during operation.

**Downside:** Data since last snapshot is lost on crash. If Redis crashes 5 minutes after a 10-minute snapshot, you lose up to 5 minutes of data.

### AOF (Append-Only File) — Log every write

```
appendonly yes
appendfsync everysec  # flush to disk every second (balanced)
# appendfsync always  # flush every write (safe but slow)
# appendfsync no      # let OS decide (fast but risky)
```

Every write command is appended to a log file. On restart, Redis replays the log to reconstruct the dataset.

**Downside:** AOF files grow large. Redis compacts them periodically (`BGREWRITEAOF`). Slightly slower than RDB.

### Recommendation

```
# Use both for production:
save 900 1
appendonly yes
appendfsync everysec
```

RDB for fast restarts (replaying AOF of 100GB takes time). AOF for minimizing data loss (at most 1 second of writes lost with `everysec`).

---

## Summary

- **String:** counters, sessions, flags, simple cache values
- **Hash:** structured objects (user profiles), memory-efficient for multiple related fields
- **List:** queues, activity feeds, recent N items
- **Set:** unique collections, social graph, set operations
- **Sorted Set:** leaderboards, priority queues, sliding window rate limiting
- **Stream:** lightweight event log, consumer groups
- **Sentinel:** HA and automatic failover for single-shard Redis
- **Cluster:** horizontal sharding when data exceeds single-node capacity
- **Pub/Sub:** fire-and-forget event broadcasting across services
- **Persistence:** use both RDB + AOF (everysec) for production

---

## ⚠️ Common Mistakes

- Using Redis List as a production task queue — no acknowledgement, no dead letter queue, no retry. Use a real queue (Lesson 5.4).
- No persistence configured on a Redis instance holding session data — server restart logs out all users
- Using Pub/Sub when you need guaranteed delivery — offline subscribers miss messages silently
- Not using hash tags in Redis Cluster for related keys — multi-key operations fail with CROSSSLOT error
- Storing large objects (>1MB) in Redis — wastes memory, slows serialization, blocks single-threaded command processing

---

> Next: [Lesson 4.7 — Distributed Caching Problems](./lesson-4.7-distributed-caching-problems.md)