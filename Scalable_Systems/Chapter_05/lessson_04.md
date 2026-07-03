# Lesson 5.4 — RabbitMQ and SQS

> **Chapter 5 — Async Processing and Message Queues**
> Previous: [Lesson 5.3 — Kafka Deep Dive](./lesson-5.3-kafka.md) | Next: [Lesson 5.5 — Idempotency](./lesson-5.5-idempotency.md)

---

## What this lesson covers

- RabbitMQ — exchanges, routing, and when to use it over Kafka
- AWS SQS — the simplest managed queue and its specific features
- Visibility timeout, long polling, and FIFO queues in SQS
- Dead letter queue configuration for both
- The decision matrix: Kafka vs RabbitMQ vs SQS

---

## 1. RabbitMQ — The Traditional Message Broker

RabbitMQ is a general-purpose message broker implementing the AMQP protocol. It routes messages through **exchanges** to **queues**. Unlike Kafka, RabbitMQ deletes messages after they are acknowledged — it is a true queue, not a log.

### Core concepts

```
Producer → Exchange → (routing logic) → Queue → Consumer
```

**Exchange:** Receives messages from producers and routes them to queues based on routing rules.

**Queue:** Holds messages until consumed. Unlike Kafka partitions, queues are not inherently ordered for parallel consumption.

**Binding:** A rule connecting an exchange to a queue (with an optional routing key).

### Exchange Types

**Direct exchange — route by exact routing key**

```python
# Producer
channel.basic_publish(
    exchange='notifications',
    routing_key='email',     # exact match
    body=json.dumps(message)
)

# Consumer binds to 'email' routing key
channel.queue_bind(queue='email_workers', exchange='notifications', routing_key='email')
# Only messages with routing_key='email' reach this queue
```

**Topic exchange — route by pattern**

```python
# Producer publishes with hierarchical routing key
channel.basic_publish(exchange='events', routing_key='user.42.signup', body=...)

# Consumer 1: all user events
channel.queue_bind(queue='user_events', exchange='events', routing_key='user.#')

# Consumer 2: all signup events for any entity
channel.queue_bind(queue='signup_events', exchange='events', routing_key='*.*.signup')
```

**Fanout exchange — broadcast to all queues**

```python
# Every queue bound to this exchange receives every message
channel.exchange_declare(exchange='notifications_broadcast', exchange_type='fanout')
# Used for cache invalidation broadcast, real-time updates to multiple services
```

### When to choose RabbitMQ

- **Complex routing logic** — different message types need to go to different queues based on content or routing key. Kafka has no routing; all consumers see all messages.
- **Task queues with competing consumers** — simple work distribution across multiple workers.
- **Lower complexity than Kafka** — RabbitMQ is simpler to operate for smaller-scale use cases.
- **Protocol support** — RabbitMQ supports AMQP, STOMP, MQTT — useful for IoT or cross-platform messaging.

**RabbitMQ limitations:**
- Messages are deleted after consumption — no replay
- Throughput lower than Kafka at very high volume
- Harder to scale horizontally than Kafka
- Queue depth can cause performance issues (very large queues slow down RabbitMQ)

---

## 2. AWS SQS — The Simplest Managed Queue

Amazon SQS is the easiest queue to get started with. It is fully managed — no servers to provision, no cluster to maintain. Two types: Standard and FIFO.

### SQS Standard Queue

- **At-least-once delivery:** Messages may be delivered more than once
- **Best-effort ordering:** Messages may be delivered out of order
- **Throughput:** Unlimited (scales automatically)

```python
import boto3

sqs = boto3.client('sqs')

# Send a message
sqs.send_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/123456/email-jobs',
    MessageBody=json.dumps({
        'message_id': str(uuid.uuid4()),
        'to': 'alice@example.com',
        'template': 'welcome'
    }),
    MessageGroupId='emails'
)

# Receive messages (poll)
response = sqs.receive_message(
    QueueUrl=queue_url,
    MaxNumberOfMessages=10,       # up to 10 at once
    WaitTimeSeconds=20,           # long polling — wait up to 20s for messages
    VisibilityTimeout=60          # 60 seconds to process before re-delivery
)

for message in response.get('Messages', []):
    body = json.loads(message['Body'])
    process(body)
    # Delete after successful processing
    sqs.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=message['ReceiptHandle']
    )
```

### Visibility Timeout — Crucial to Get Right

When SQS delivers a message, it becomes invisible to other consumers for the `VisibilityTimeout` duration. If the consumer does not delete it within that time, it becomes visible again.

```
VisibilityTimeout: 30 seconds
Job takes: 25 seconds normally, but sometimes 45 seconds under load

Normal case: consumer deletes in 25s → fine
Slow case: consumer takes 45s → message becomes visible at 30s
         → another consumer picks it up → DUPLICATE PROCESSING

Fix: Set VisibilityTimeout to max_expected_processing_time × 1.5
     Or extend the timeout programmatically during long processing:

sqs.change_message_visibility(
    QueueUrl=queue_url,
    ReceiptHandle=receipt_handle,
    VisibilityTimeout=60  # extend by 60 more seconds
)
```

### Long Polling vs Short Polling

```python
# Short polling (default, bad): returns immediately even if no messages
# Creates unnecessary API calls and charges when queue is empty
response = sqs.receive_message(QueueUrl=queue_url, WaitTimeSeconds=0)

# Long polling (correct): waits up to 20 seconds for messages to arrive
# Reduces empty responses, lower cost, lower latency
response = sqs.receive_message(QueueUrl=queue_url, WaitTimeSeconds=20)
```

Always use long polling. Short polling wastes API calls and costs money at scale.

### SQS FIFO Queue

- **Exactly-once processing:** SQS deduplicates messages within a 5-minute window using `MessageDeduplicationId`
- **Strict ordering:** Messages in the same `MessageGroupId` are delivered in order
- **Throughput:** Up to 3,000 messages/second (standard) or 300 messages/second (per API call)

```python
sqs.send_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/123456/orders.fifo',
    MessageBody=json.dumps(order),
    MessageGroupId=f"user_{user_id}",          # ordering group
    MessageDeduplicationId=f"order_{order_id}" # deduplication key
)
```

**When to use FIFO:** When order matters (payment events for a user must be processed in order) and you need deduplication built in.

**FIFO limitation:** The throughput cap (3,000 msg/sec) is not suitable for very high volume.

### SQS Dead Letter Queue Configuration

```python
# Create DLQ
dlq = sqs.create_queue(QueueName='email-jobs-dlq')
dlq_arn = sqs.get_queue_attributes(
    QueueUrl=dlq['QueueUrl'], AttributeNames=['QueueArn']
)['Attributes']['QueueArn']

# Create main queue with DLQ policy
main_queue = sqs.create_queue(
    QueueName='email-jobs',
    Attributes={
        'RedrivePolicy': json.dumps({
            'deadLetterTargetArn': dlq_arn,
            'maxReceiveCount': '5'  # after 5 failures → DLQ
        })
    }
)
```

---

## 3. The Decision Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHOOSE YOUR QUEUE                            │
│                                                                 │
│  Need message replay or multiple consumers per event?           │
│  ├── YES → Kafka                                                │
│  └── NO ↓                                                      │
│                                                                 │
│  Need complex routing (topic patterns, fanout, direct)?         │
│  ├── YES → RabbitMQ                                             │
│  └── NO ↓                                                      │
│                                                                 │
│  On AWS and want zero infrastructure management?                │
│  ├── YES → SQS (Standard or FIFO)                              │
│  └── NO → RabbitMQ (simpler than Kafka for basic task queues)  │
└─────────────────────────────────────────────────────────────────┘
```

| | Kafka | RabbitMQ | SQS Standard | SQS FIFO |
|---|---|---|---|---|
| Message retention after consume | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Multiple consumers (same message) | ✅ Consumer groups | ✅ Fanout exchange | ❌ No (one consumer) | ❌ No |
| Complex routing | ❌ No | ✅ Yes | ❌ No | ❌ No |
| Ordering guarantee | ✅ Per partition | ❌ Best effort | ❌ Best effort | ✅ Per group |
| Deduplication | ❌ Manual | ❌ Manual | ❌ Manual | ✅ Built in (5 min) |
| Max throughput | 1M+ msg/sec | ~100K msg/sec | Unlimited | 3K msg/sec |
| Infrastructure | Self-managed | Self-managed | Fully managed | Fully managed |
| Operational complexity | High | Medium | Low | Low |

---

## Summary

- RabbitMQ routes messages through exchanges (direct, topic, fanout) to queues — best for complex routing and moderate-volume task queues
- SQS Standard: easiest managed queue, at-least-once, best-effort ordering, unlimited throughput
- SQS FIFO: ordered and deduplicated, capped at 3K msg/sec, best for payment flows and ordered processing
- Always use long polling with SQS — short polling wastes API calls
- Set visibility timeout to 1.5× your maximum expected processing time, or extend it during processing
- Configure DLQ on every queue — without it, failed messages loop or disappear silently
- Decision: Kafka for replay/multi-consumer, RabbitMQ for routing, SQS for simplicity

---

> Next: [Lesson 5.5 — Idempotency](./lesson-5.5-idempotency.md)