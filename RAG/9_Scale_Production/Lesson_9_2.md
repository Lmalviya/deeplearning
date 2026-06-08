# Lesson 9.2 — Rate Limiting, Backpressure, and Graceful Degradation

---

## Why You Need All Three

These three concepts are often confused but address distinct problems:

**Rate limiting** controls *how many requests* a client can make in a time window. It protects your system from being overwhelmed by any single user or tenant.

**Backpressure** controls *how requests flow* through the pipeline when a downstream component is saturated. Instead of queuing infinitely or dropping requests silently, backpressure propagates the congestion signal upstream and lets the system make intelligent decisions.

**Graceful degradation** defines *what the system does* when it cannot provide full service. Instead of failing completely, it provides reduced-quality service — answering from cache, skipping re-ranking, using a faster LLM — while maintaining availability.

A RAG system at production scale needs all three working together.

---

## Rate Limiting

### Why RAG Needs Rate Limiting

Without rate limits, a single client can:
- Run automated scripts that hammer your API, exhausting LLM API quotas for all users.
- Accidentally loop queries (a bug in client code) and overwhelm the system.
- Conduct cost attacks that run up your OpenAI bill.

### Token Bucket Rate Limiter

The token bucket algorithm is the most practical rate limiting approach for APIs. Each client gets a "bucket" of tokens that refills at a constant rate. Each request consumes tokens. When the bucket is empty, requests are rejected.

```python
import redis.asyncio as aioredis
import time
from fastapi import Request, HTTPException

class TokenBucketRateLimiter:
    """
    Redis-backed token bucket rate limiter.
    Distributed — works correctly across multiple API pod instances.
    """
    
    def __init__(
        self,
        redis_url: str,
        default_rate: int = 60,          # tokens per minute
        default_burst: int = 20,         # max burst size
        window_seconds: int = 60
    ):
        self.redis = aioredis.from_url(redis_url)
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.window = window_seconds
    
    async def check_and_consume(
        self,
        client_id: str,
        tokens_required: int = 1,
        rate_override: int = None,
        burst_override: int = None
    ) -> dict:
        """
        Check if client has capacity and consume tokens if so.
        Returns: {allowed: bool, remaining: int, retry_after: int}
        """
        rate = rate_override or self.default_rate
        burst = burst_override or self.default_burst
        
        key = f"rate_limit:{client_id}"
        now = time.time()
        
        # Lua script for atomic token bucket operation
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local burst = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local tokens_required = tonumber(ARGV[4])
        local window = tonumber(ARGV[5])
        
        -- Get current state
        local data = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(data[1]) or burst
        local last_refill = tonumber(data[2]) or now
        
        -- Refill tokens based on time elapsed
        local elapsed = now - last_refill
        local refill = elapsed * (rate / window)
        tokens = math.min(burst, tokens + refill)
        
        -- Check if request can proceed
        if tokens >= tokens_required then
            tokens = tokens - tokens_required
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, window * 2)
            return {1, math.floor(tokens), 0}
        else
            -- Calculate when enough tokens will be available
            local wait_time = math.ceil((tokens_required - tokens) * window / rate)
            return {0, math.floor(tokens), wait_time}
        end
        """
        
        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            key,
            rate,
            burst,
            now,
            tokens_required,
            self.window
        )
        
        allowed, remaining, retry_after = result
        
        return {
            "allowed": bool(allowed),
            "remaining": int(remaining),
            "retry_after": int(retry_after)
        }


# Tiered rate limits by user plan
RATE_LIMITS_BY_PLAN = {
    "free": {"rate": 10, "burst": 5},       # 10 req/min
    "standard": {"rate": 60, "burst": 20},  # 60 req/min
    "enterprise": {"rate": 500, "burst": 100}  # 500 req/min
}

# FastAPI middleware
async def rate_limit_middleware(request: Request, call_next):
    user_plan = request.state.user.plan if hasattr(request.state, 'user') else "free"
    client_id = request.state.user.user_id if hasattr(request.state, 'user') else request.client.host
    
    limits = RATE_LIMITS_BY_PLAN.get(user_plan, RATE_LIMITS_BY_PLAN["free"])
    
    result = await rate_limiter.check_and_consume(
        client_id=client_id,
        rate_override=limits["rate"],
        burst_override=limits["burst"]
    )
    
    if not result["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "retry_after": result["retry_after"],
                "message": f"Rate limit exceeded. Retry after {result['retry_after']} seconds."
            },
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + result["retry_after"]),
                "Retry-After": str(result["retry_after"])
            }
        )
    
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(result["remaining"])
    return response
```

### Cost-Aware Rate Limiting

For RAG systems, simple request counting is insufficient. A query with a 10,000-token context costs 10× more than one with 1,000 tokens. Implement token-cost-weighted rate limiting:

```python
async def cost_aware_rate_limit(
    client_id: str,
    query: str,
    context_tokens_estimate: int,
    rate_limiter: TokenBucketRateLimiter
) -> dict:
    """
    Consume rate limit tokens proportional to estimated LLM cost.
    """
    # Estimate token cost (1 token = 1 rate limit unit, normalized)
    # A 2000-token query costs 20 units; a 200-token query costs 2 units
    cost_units = max(1, context_tokens_estimate // 100)
    
    return await rate_limiter.check_and_consume(
        client_id=client_id,
        tokens_required=cost_units
    )
```

---

## Backpressure

### The Problem Without Backpressure

Without backpressure, when the LLM API is slow (high load), requests queue up in memory. Each API pod accumulates hundreds of in-flight requests. Memory usage grows. Eventually, pods OOM crash — exactly when you need them most.

Backpressure prevents this by making slowness visible and actionable at the source, rather than letting it cascade silently.

### Request Queue with Backpressure

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

@dataclass
class QueuedRequest:
    query: str
    user_context: dict
    future: asyncio.Future
    enqueued_at: float = field(default_factory=lambda: time.time())
    priority: int = 1  # Higher = more important

class BackpressureQueue:
    """
    Priority queue with backpressure.
    Rejects new requests when queue is full (fast fail instead of slow failure).
    """
    
    def __init__(
        self,
        max_size: int = 100,
        max_wait_seconds: float = 5.0,
        workers: int = 10
    ):
        self.queue = asyncio.PriorityQueue(maxsize=max_size)
        self.max_wait = max_wait_seconds
        self.workers = workers
        self._processing = False
    
    async def submit(
        self,
        query: str,
        user_context: dict,
        priority: int = 1
    ) -> Any:
        """
        Submit a query for processing.
        Raises immediately if queue is full (backpressure applied to caller).
        """
        if self.queue.full():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_overloaded",
                    "message": "System is at capacity. Please retry in a moment.",
                    "retry_after": 2
                }
            )
        
        future = asyncio.Future()
        request = QueuedRequest(
            query=query,
            user_context=user_context,
            future=future,
            priority=priority
        )
        
        # Use negative priority for min-heap (higher priority = lower number)
        await self.queue.put((-priority, time.time(), request))
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=self.max_wait)
            return result
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "request_timeout",
                    "message": "Request timed out waiting for processing."
                }
            )
    
    async def start_workers(self, process_fn):
        """Start N worker coroutines that process queued requests."""
        self._processing = True
        workers = [
            asyncio.create_task(self._worker(process_fn))
            for _ in range(self.workers)
        ]
        return workers
    
    async def _worker(self, process_fn):
        """Worker that continuously processes queued requests."""
        while self._processing:
            try:
                _, _, request = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                # Check if request has already timed out waiting in queue
                wait_time = time.time() - request.enqueued_at
                if wait_time > self.max_wait:
                    request.future.set_exception(
                        Exception("Request expired in queue")
                    )
                    continue
                
                # Process the request
                try:
                    result = await process_fn(request.query, request.user_context)
                    request.future.set_result(result)
                except Exception as e:
                    request.future.set_exception(e)
                
            except asyncio.TimeoutError:
                continue  # No requests in queue, try again
```

### Circuit Breaker Pattern

When a downstream service (LLM API, Qdrant) is failing, fail fast instead of continuing to send requests that will timeout.

```python
from enum import Enum
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Failing fast — not sending requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit breaker for external service calls.
    Opens when failure rate exceeds threshold.
    Automatically tests recovery after reset_timeout.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,     # failures before opening
        success_threshold: int = 2,     # successes to close from half-open
        reset_timeout: float = 30.0,    # seconds before trying half-open
        window_seconds: float = 60.0    # sliding window for failure counting
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.reset_timeout = reset_timeout
        self.window = window_seconds
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        self._lock = asyncio.Lock()
    
    async def call(self, fn, *args, fallback=None, **kwargs):
        """
        Execute fn with circuit breaker protection.
        Uses fallback if circuit is open.
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_state_change > self.reset_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    if fallback:
                        return await fallback(*args, **kwargs)
                    raise HTTPException(
                        status_code=503,
                        detail={"error": "service_unavailable", "retry_after": 10}
                    )
        
        try:
            result = await fn(*args, **kwargs)
            
            async with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        print("Circuit breaker: CLOSED (service recovered)")
                else:
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
        
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.last_state_change = time.time()
                    print(f"Circuit breaker: OPEN (after {self.failure_count} failures)")
            
            raise


# Instantiate circuit breakers for external services
llm_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    reset_timeout=30.0
)

qdrant_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    reset_timeout=10.0
)
```

---

## Graceful Degradation

When the system is overloaded or components are failing, degrade gracefully — provide reduced-quality service rather than no service.

### Degradation Levels

```python
from enum import Enum

class ServiceLevel(Enum):
    FULL = "full"           # All features enabled
    DEGRADED_1 = "deg_1"   # Skip re-ranking (faster)
    DEGRADED_2 = "deg_2"   # Cache-only + skip re-ranking
    DEGRADED_3 = "deg_3"   # Pre-computed answers only (emergency)
    OFFLINE = "offline"    # Return maintenance message

class AdaptiveServiceLevel:
    """
    Automatically adjusts service level based on system health.
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self._current_level = ServiceLevel.FULL
    
    async def get_current_level(self) -> ServiceLevel:
        """Check current service level from Redis (shared across pods)."""
        level_str = await self.redis.get("service_level")
        if level_str:
            return ServiceLevel(level_str.decode())
        return ServiceLevel.FULL
    
    async def update_level(self, new_level: ServiceLevel, reason: str):
        """Set new service level (affects all pods immediately)."""
        await self.redis.set("service_level", new_level.value)
        await self.redis.set("service_level_reason", reason)
        print(f"Service level changed to {new_level.value}: {reason}")
    
    async def auto_adjust_from_metrics(self, metrics: dict):
        """Automatically adjust service level based on current metrics."""
        
        p95_latency = metrics.get("p95_latency_ms", 0)
        error_rate = metrics.get("error_rate", 0)
        qdrant_healthy = metrics.get("qdrant_healthy", True)
        llm_healthy = metrics.get("llm_healthy", True)
        
        if not qdrant_healthy or not llm_healthy:
            await self.update_level(ServiceLevel.DEGRADED_3, "Core service unavailable")
        elif error_rate > 0.10:
            await self.update_level(ServiceLevel.DEGRADED_2, f"High error rate: {error_rate:.1%}")
        elif p95_latency > 5000:
            await self.update_level(ServiceLevel.DEGRADED_1, f"High latency: {p95_latency:.0f}ms")
        else:
            current = await self.get_current_level()
            if current != ServiceLevel.FULL:
                await self.update_level(ServiceLevel.FULL, "System recovered")


async def answer_with_degradation(
    query: str,
    user_context: dict,
    service_level: ServiceLevel,
    retriever,
    reranker,
    llm_client,
    exact_cache
) -> dict:
    """
    Answer query at the appropriate service level.
    """
    
    if service_level == ServiceLevel.OFFLINE:
        return {
            "answer": "The system is temporarily unavailable for maintenance. Please try again in a few minutes.",
            "service_level": "offline"
        }
    
    if service_level == ServiceLevel.DEGRADED_3:
        # Cache only — no new LLM calls
        cached = await exact_cache.get(query, {}, 10)
        if cached:
            return {**cached, "service_level": "degraded_3", "from_cache": True}
        
        return {
            "answer": "I'm currently unable to answer new questions. Please try again shortly.",
            "service_level": "degraded_3",
            "from_cache": False
        }
    
    if service_level == ServiceLevel.DEGRADED_2:
        # Cache first, then basic retrieval without re-ranking
        cached = await exact_cache.get(query, user_context, 10)
        if cached:
            return {**cached, "service_level": "degraded_2_cached"}
        
        # Basic retrieval, no re-ranking, faster LLM
        chunks = await retriever.retrieve(query, k=5)  # Fewer chunks
        context = format_context(chunks)
        answer = await generate_with_faster_model(query, context, llm_client)
        return {"answer": answer, "service_level": "degraded_2"}
    
    if service_level == ServiceLevel.DEGRADED_1:
        # Full retrieval but skip cross-encoder re-ranking
        chunks = await retriever.retrieve_without_reranking(query)
        context = format_context(chunks)
        answer = await llm_client.generate(query, context)
        return {"answer": answer, "service_level": "degraded_1"}
    
    # Full service
    chunks = await retriever.retrieve(query)
    reranked = reranker.rerank(query, chunks)
    context = format_context(reranked)
    answer = await llm_client.generate(query, context)
    return {"answer": answer, "service_level": "full"}
```

### Communicating Degradation to Users

Users should know when they are getting reduced service:

```python
SERVICE_LEVEL_USER_MESSAGES = {
    ServiceLevel.FULL: None,
    ServiceLevel.DEGRADED_1: None,  # Transparent — user does not notice
    ServiceLevel.DEGRADED_2: "⚡ Some features are temporarily limited. Answers may be less detailed.",
    ServiceLevel.DEGRADED_3: "⚠️ Operating in limited mode. Using recent cached answers.",
    ServiceLevel.OFFLINE: "🔴 System maintenance in progress. Please check back shortly."
}
```

---

## The Complete Flow Under Load

```
Request arrives
      │
      ▼
[Rate Limiter]
  Reject if over rate limit → 429 Too Many Requests
      │
      ▼
[Backpressure Queue]
  Reject if queue full → 503 Service Unavailable
      │
      ▼
[Service Level Check]
  Get current degradation level from Redis
      │
      ▼
[Circuit Breakers]
  Check: Is Qdrant up? Is LLM API up?
      │
      ▼
[Appropriate Service Level Handler]
  FULL → Full pipeline
  DEGRADED_1 → No re-ranking
  DEGRADED_2 → Cache + basic retrieval
  DEGRADED_3 → Cache only
      │
      ▼
[Response with service level header]
```

---

## Summary

- Rate limiting protects from client-side abuse and runaway costs. Use token bucket with per-plan tiered limits. Implement cost-aware rate limiting (token-weighted) for RAG systems where query cost varies significantly.
- Backpressure prevents queue buildup from cascading into OOM crashes. Use bounded queues with fast rejection when full. Priority queues ensure high-value requests get served during load shedding.
- Circuit breakers fail fast when downstream services are failing. Three states: closed (normal), open (failing fast), half-open (testing recovery). Prevents timeout storms.
- Graceful degradation provides reduced-quality service rather than no service. Four levels: full → skip re-ranking → cache-first → cache-only. Automatic adjustment based on system health metrics.
- Service level is stored in Redis so all pods pick up changes atomically within one request cycle.

---

## What's Next

Lesson 9.3 covers async indexing pipelines — queues, workers, retry logic, and how to design a robust ingestion system that handles failures without losing documents.