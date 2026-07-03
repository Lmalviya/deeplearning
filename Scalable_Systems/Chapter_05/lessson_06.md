# Lesson 5.6 — Consumer Lag and Backpressure

> **Chapter 5 — Async Processing and Message Queues**
> Previous: [Lesson 5.5 — Idempotency](./lesson-5.5-idempotency.md) | Next: [Lesson 5.7 — Event-Driven Architecture](./lesson-5.7-event-driven-architecture.md)

---

## What this lesson covers

- What consumer lag is and how to measure it
- The causes of growing consumer lag
- How to scale consumers to reduce lag
- Backpressure — protecting downstream services from overload
- Lag alerting and when lag is acceptable vs when it is an incident

---

## 1. What Consumer Lag Is

Consumer lag is how far behind a consumer is relative to the latest message in the queue.

```
Kafka Topic: "orders" — Partition 0

Latest offset (most recent message): 10,000
Consumer group "fulfillment" current offset: 9,500

Consumer lag = 10,000 - 9,500 = 500 messages behind
```

In SQS terms, lag is the "approximate number of messages not visible" or "messages in queue."

### Why lag matters

```
Lag = 0:      Consumer is keeping up — processing in real time
Lag = 500:    Consumer is slightly behind — probably fine
Lag = 50,000: Consumer is 50× behind — users are waiting 50× longer
              for their order confirmations, notifications, etc.
Lag = growing: Consumer cannot keep up — lag will grow until
               the queue is full or consumer catches up
```

Lag translates directly to delay for your users. If the welcome email worker has a lag of 10,000 messages and processes 100 messages/minute, users wait 100 minutes for their welcome email.

---

## 2. Causes of Growing Consumer Lag

### Cause 1 — Producer rate > consumer rate

```
Producer: 1,000 messages/second (traffic spike)
Consumer: 800 messages/second (max throughput)
Lag growth: 200 messages/second

After 10 minutes: lag = 200 × 600 = 120,000 messages
```

Even a temporary spike creates lag that takes time to drain after the spike ends.

### Cause 2 — Consumer is slow (each message takes too long)

```
1 consumer, processing time: 500ms per message
Consumer throughput: 2 messages/second

Producer: 10 messages/second
Lag growth: 8 messages/second

Fix: add more consumers OR speed up message processing
```

### Cause 3 — Consumer failure / crash

All consumers in a group crash. Messages queue up with no one to process them. Lag grows at the rate of production until consumers restart.

### Cause 4 — Downstream service is slow or unavailable

```
Consumer calls email API to send emails
Email API is slow (5 seconds per call instead of 100ms)
Consumer throughput drops 50×
Lag grows rapidly
```

---

## 3. How to Reduce Lag — Scaling Consumers

The most direct solution: add more consumer instances.

```
Before:
  1 consumer × 2 messages/second = 2 msg/sec throughput
  Producer rate: 10 msg/sec
  Lag growing: 8 msg/sec

After (5 consumers):
  5 consumers × 2 messages/second = 10 msg/sec throughput
  Producer rate: 10 msg/sec
  Lag stable: 0 msg/sec growth (starts draining previous lag)

After (10 consumers):
  10 consumers × 2 messages/second = 20 msg/sec throughput
  Lag draining: 10 msg/sec (catches up to real-time twice as fast)
```

### Kafka: partition count limits consumer parallelism

```
Topic: 4 partitions
Consumer Group: 4 consumers → maximum parallelism (one consumer per partition)

Add 5th consumer: idle, no partition to assign
→ To add more consumers, must increase partition count first

Increasing partition count:
  kafka-topics --alter --topic orders --partitions 8
  Now: 8 consumers can work in parallel
```

**Important:** Increasing partition count in Kafka is a one-time operation — you can add partitions but never remove them (it would break key-based ordering guarantees). Plan partition count generously upfront.

### SQS: consumer count is not bounded

SQS does not have partition limits. You can run as many consumers as needed — SQS distributes messages among all of them.

```python
# AWS Lambda auto-scales consumers based on queue depth
# Configure Lambda event source mapping with batch size
sqs_event_source = {
    "EventSourceArn": queue_arn,
    "BatchSize": 10,
    "FunctionResponseTypes": ["ReportBatchItemFailures"],
    "ScalingConfig": {
        "MaximumConcurrency": 100  # up to 100 concurrent Lambda instances
    }
}
```

---

## 4. Backpressure — Protecting Downstream Services

Scaling consumers increases throughput — which increases load on the services consumers call. If the email API can handle 100 requests/second and you run 200 consumers each sending 1 email/second, the email API is overloaded.

**Backpressure** is the mechanism by which a downstream service signals "I am overwhelmed, slow down."

### Implementing backpressure in consumers

**Method 1 — Rate limiting consumers**

```python
import asyncio
from asyncio import Semaphore

# Limit concurrent processing across all consumers in this process
semaphore = Semaphore(10)  # at most 10 concurrent operations

async def process_message(message):
    async with semaphore:  # blocks if 10 are already running
        await send_email(message['to'], message['template'])
```

**Method 2 — Exponential backoff on downstream failure**

```python
import time

def process_with_backoff(message, max_retries=5):
    for attempt in range(max_retries):
        try:
            email_api.send(message['to'], message['template'])
            return  # success
        except EmailAPIRateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            # Exponential: 1s, 2s, 4s, 8s, 16s...
            time.sleep(wait_time)
        except EmailAPIUnavailableError:
            # Service is down — re-raise to trigger retry via queue
            raise
    raise MaxRetriesExceeded(f"Failed after {max_retries} attempts")
```

**Method 3 — Circuit breaker**

Stop calling a service that is failing rather than hammering it with retries:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = "closed"  # closed = normal, open = blocking calls

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"  # try one request
            else:
                raise CircuitOpenError("Circuit is open, not calling downstream")

        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            self.state = "closed"
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise

email_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

def process_message(message):
    email_circuit_breaker.call(email_api.send, message['to'], message['template'])
```

---

## 5. Monitoring Consumer Lag

### Kafka lag monitoring

```bash
# Check consumer group lag via kafka-consumer-groups tool
kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --group fulfillment-service \
  --describe

# Output:
GROUP               TOPIC   PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
fulfillment-service orders  0          9500            10000           500
fulfillment-service orders  1          8900            9000            100
fulfillment-service orders  2          10500           10500           0
```

### SQS lag monitoring (via CloudWatch)

```
Metrics:
  ApproximateNumberOfMessagesNotVisible  ← messages being processed
  ApproximateNumberOfMessagesVisible     ← messages waiting (this is the lag)
  ApproximateAgeOfOldestMessage          ← how old is the oldest unprocessed message
```

### Alert thresholds

| Metric | Warning | Critical |
|--------|---------|---------|
| Consumer lag (messages) | > 1,000 | > 10,000 |
| Consumer lag (time) | > 1 minute | > 10 minutes |
| Oldest message age | > 2× normal processing time | > 10× normal processing time |
| Lag growing rate | Positive (lag increasing) | Lag doubling every 5 minutes |

### The lag vs time tradeoff

Not all lag is equal. Context matters:

```
Email welcome worker lag: 5,000 messages (20 minute delay)
  → Users wait 20 minutes for their welcome email
  → Acceptable? Maybe. (Depends on your product expectations)

Payment confirmation worker lag: 50 messages (30 second delay)
  → Users wait 30 seconds to see their payment confirmed
  → Probably NOT acceptable.
```

Set alert thresholds appropriate to the business impact of each queue, not a one-size-fits-all number.

---

## Summary

- Consumer lag = latest offset - consumer's current offset (Kafka) or queue depth (SQS)
- Lag grows when producer rate > consumer rate, consumer is slow, consumers crash, or downstream is slow
- Fix growing lag: add more consumers (bounded by partition count in Kafka), speed up per-message processing
- Backpressure protects downstream services: semaphores to limit concurrency, exponential backoff on rate limit errors, circuit breakers to stop calling failing services
- Monitor lag in time units (oldest message age), not just message count — 10,000 messages at 10ms each is 100 seconds; at 10 seconds each it's 28 hours
- Set per-queue alert thresholds based on business impact — payment lag has a tighter SLA than welcome email lag

---

## ⚠️ Common Mistakes

- Not monitoring consumer lag at all — lag grows for hours, users complain, engineers are surprised
- Running fewer consumers than partitions in Kafka — some partitions have no consumer, those messages wait
- Scaling consumers without checking downstream capacity — 100 consumers overwhelming the email API they all call
- No circuit breaker on downstream calls — a failing email service causes all consumers to block/retry, wasting resources and growing lag
- Treating all lag as equal — 10 minute welcome email delay is different from 10 minute payment confirmation delay

---

> Next: [Lesson 5.7 — Event-Driven Architecture](./lesson-5.7-event-driven-architecture.md)