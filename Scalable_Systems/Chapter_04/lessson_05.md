# Lesson 4.5 — Eviction Policies

> **Chapter 4 — The Caching Layer**
> Previous: [Lesson 4.4 — Cache Stampede](./lesson-4.4-cache-stampede.md) | Next: [Lesson 4.6 — Redis Deep Dive](./lesson-4.6-redis-deep-dive.md)

---

## What this lesson covers

- What happens when a cache runs out of memory
- The six eviction policies Redis supports
- Which policy fits which workload
- How to detect that evictions are hurting your hit ratio
- Memory management best practices

---

## 1. What Happens When Cache is Full

Redis stores all data in RAM. RAM is finite. When Redis reaches its configured `maxmemory` limit, it must make a decision: what to delete to make room for new data?

This is the **eviction policy** — the rule that decides which keys to remove.

Without a configured policy (`maxmemory-policy noeviction`), Redis returns an error on write when memory is full:

```
OOM command not allowed when used memory > 'maxmemory'
```

This causes your application to fail on cache writes. Almost always the wrong behavior.

---

## 2. The Six Eviction Policies

Redis supports six policies, split into two groups: policies that apply to **all keys** and policies that apply only to **keys with a TTL** (volatile keys).

```
Volatile = keys that have an expiry time set (SET key value EX 300)
All Keys = every key in Redis, regardless of whether it has a TTL
```

### Policy 1 — allkeys-lru (most common choice)

**LRU = Least Recently Used.** Evict the key that has not been accessed for the longest time.

```
Cache contains: [A accessed 5min ago, B accessed 1min ago, C accessed 10min ago, D accessed 2min ago]
Cache is full, need to evict one key
allkeys-lru evicts: C (least recently accessed — 10 minutes ago)
```

**Best for:** General-purpose caching where recently accessed items are most likely to be accessed again (temporal locality). This is correct for most web application caches.

### Policy 2 — allkeys-lfu (best for skewed access)

**LFU = Least Frequently Used.** Evict the key that has been accessed the fewest times.

```
Cache contains: [A accessed 1000x, B accessed 5x, C accessed 2x, D accessed 800x]
Cache is full, need to evict one key
allkeys-lfu evicts: C (accessed only 2 times)
```

**Best for:** Workloads with highly skewed access patterns — a small number of "hot" keys are accessed far more than others (Zipf distribution, which is common in web traffic). LFU keeps the hot keys and evicts the cold ones, giving better hit ratios than LRU.

**Added in Redis 4.0.** Use this if you are on a recent Redis version and your workload is read-heavy with popular items.

### Policy 3 — allkeys-random

Evict a random key regardless of when it was last accessed or how often.

**Best for:** Almost never the right choice for application caches. Useful for uniform access patterns where all keys are equally likely to be accessed next — very rare.

### Policy 4 — volatile-lru

LRU, but only among keys that have an expiry time. Keys without TTL are never evicted.

**Best for:** When you have two categories of data:
- Critical data (no TTL) that must never be evicted — configuration, session lookup tables
- Cache data (with TTL) that can be evicted — user profiles, product catalog

```python
# Never evict this:
redis.set("config:feature_flags", json.dumps(flags))  # no TTL

# Can be evicted:
redis.setex("user:42", 300, json.dumps(user))  # has TTL
```

With `volatile-lru`, the config key is protected; user cache keys are the eviction candidates.

### Policy 5 — volatile-lfu

LFU, but only among keys with a TTL. Same use case as volatile-lru but with frequency-based eviction.

### Policy 6 — volatile-ttl

Among keys with a TTL, evict the key whose TTL is closest to expiring (it will be gone soon anyway).

**Best for:** When you want to avoid evicting items that were recently cached with a long TTL. Makes intuitive sense but in practice volatile-lru usually performs better.

---

## 3. Comparison Table

| Policy | Eviction pool | Selection criterion | Best for |
|--------|-------------|--------------------|---------| 
| `allkeys-lru` | All keys | Least recently used | General web app cache (default choice) |
| `allkeys-lfu` | All keys | Least frequently used | Hot/cold access patterns |
| `allkeys-random` | All keys | Random | Uniform access, rarely useful |
| `volatile-lru` | Keys with TTL | Least recently used | Mix of permanent + cached data |
| `volatile-lfu` | Keys with TTL | Least frequently used | Same as volatile-lru + skewed access |
| `volatile-ttl` | Keys with TTL | Soonest to expire | Rarely the right choice |
| `noeviction` | N/A | Returns error on write when full | When you must never lose data (use with caution) |

---

## 4. Configuring maxmemory and Policy

```bash
# In redis.conf
maxmemory 4gb                      # max memory Redis will use
maxmemory-policy allkeys-lru       # eviction policy when full

# Or set at runtime (takes effect immediately, not persistent across restart)
redis-cli CONFIG SET maxmemory 4gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### How much memory to allocate

Redis uses memory for:
- Your data (keys + values)
- Internal data structures overhead (~50–100 bytes per key)
- Replication buffer (if running with replicas)
- AOF buffer (if persistence is on)

A common formula:

```
maxmemory = (total server RAM × 0.75) - replication_buffer

For a server with 16GB RAM, no replicas:
  maxmemory = 16GB × 0.75 = 12GB
  (leave 4GB for OS, Redis overhead, and system processes)
```

If using Redis Cluster (multiple shards), the above applies per shard.

---

## 5. How Redis Implements LRU — Not Exact

Redis does not implement a true LRU (which would require tracking access time for every key). Instead it uses **approximate LRU**: when eviction is needed, Redis samples N random keys and evicts the one that was least recently accessed among the sample.

```
maxmemory-samples 5   # default: sample 5 keys, pick LRU among them
maxmemory-samples 10  # better approximation, slightly more CPU
```

With `maxmemory-samples = 5`, Redis picks 5 random keys and evicts the least recently used among those 5. This is not perfect LRU but is statistically close enough and much cheaper to implement.

For LFU, Redis tracks access frequency per key using a probabilistic counter that decays over time, so old hot keys do not stay "hot" forever.

---

## 6. Detecting Eviction Problems

Evictions themselves are not always a problem — if evictions are happening on rarely-accessed keys, your hit ratio stays high and the cache is working well.

Evictions become a problem when frequently accessed keys are being evicted — which causes cache misses and DB load to increase.

### Key metrics to monitor

```bash
# Redis INFO stats
redis-cli INFO stats | grep evicted_keys
# evicted_keys: total keys evicted since start

redis-cli INFO stats | grep keyspace_hits
redis-cli INFO stats | grep keyspace_misses
# Compute hit ratio: hits / (hits + misses)

redis-cli INFO memory | grep used_memory_human
# Current memory usage
```

### Alert thresholds

| Metric | Alert when |
|--------|-----------|
| `evicted_keys` rate | Rising rapidly — memory is undersized |
| Hit ratio | Dropping while evictions are increasing — evicting hot keys |
| `used_memory` / `maxmemory` | > 90% — close to eviction threshold |

### What to do when evictions are causing misses

1. **Increase `maxmemory`** — add more RAM or move Redis to a bigger instance
2. **Switch to allkeys-lfu** if your workload has skewed access — LFU keeps hot keys better than LRU
3. **Audit key sizes** — find keys consuming disproportionate memory:
   ```bash
   redis-cli --bigkeys  # scans for largest keys by type
   redis-cli MEMORY USAGE key_name  # bytes used by a specific key
   ```
4. **Remove unnecessary keys** — are you caching data that is never read? Check hit count per key.
5. **Reduce value sizes** — compress values before storing, or cache less data per key

---

## 7. Memory Optimization Tips

**Use hashes for small objects instead of separate string keys:**

```python
# Less efficient: one string key per field
redis.set("user:42:name", "Alice")
redis.set("user:42:email", "alice@example.com")
redis.set("user:42:city", "Bangalore")
# 3 keys × (~50 bytes key overhead each) = ~150 bytes overhead

# More efficient: one hash
redis.hset("user:42", mapping={"name": "Alice", "email": "alice@...", "city": "Bangalore"})
# 1 key × 50 bytes overhead = ~50 bytes overhead
# Redis optimizes small hashes into a compact encoding (ziplist)
```

**Compress large values:**

```python
import zlib
import json

def cache_set_compressed(key: str, data: dict, ttl: int):
    json_bytes = json.dumps(data).encode()
    compressed = zlib.compress(json_bytes)
    redis.setex(key, ttl, compressed)

def cache_get_compressed(key: str):
    compressed = redis.get(key)
    if not compressed:
        return None
    return json.loads(zlib.decompress(compressed).decode())
```

For large JSON objects (user activity history, complex nested data), compression can reduce memory usage by 60–80%.

---

## Summary

- When Redis is full, the eviction policy determines which keys to delete
- `allkeys-lru`: evict least recently used from all keys — best general-purpose default
- `allkeys-lfu`: evict least frequently used — better for hot/cold access patterns
- `volatile-lru/lfu`: only evict keys with TTL — protects critical permanent keys
- `noeviction`: returns error when full — almost always wrong for a cache
- Monitor hit ratio and eviction rate together — rising evictions + falling hit ratio = memory is too small or wrong policy
- Optimize memory: use hashes for small objects, compress large values, audit key sizes

---

## ⚠️ Common Mistakes

- Using `noeviction` policy for a cache — causes application errors when memory fills up instead of graceful eviction
- Not setting `maxmemory` at all — Redis uses all available server RAM, starving the OS and other processes
- Not monitoring eviction rate — evictions silently hurt hit ratio without visible errors
- Using LRU when the workload has heavy skew — LFU would maintain better hit ratios for viral/trending content

---

> Next: [Lesson 4.6 — Redis Deep Dive](./lesson-4.6-redis-deep-dive.md)