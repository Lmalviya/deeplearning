# Lesson 5.3 — Kafka Deep Dive

> **Chapter 5 — Async Processing and Message Queues**
> Previous: [Lesson 5.2 — Message Queue Fundamentals](./lesson-5.2-message-queue-fundamentals.md) | Next: [Lesson 5.4 — RabbitMQ and SQS](./lesson-5.4-rabbitmq-sqs.md)

---

## What this lesson covers

- Kafka's core concepts: topics, partitions, offsets, consumer groups
- Why Kafka is a log, not a queue — and why this matters
- When Kafka is the right choice vs simpler queues
- The hot partition problem and how to fix it
- Kafka's retention model and how it enables replayability

---

## 1. Kafka is a Distributed Commit Log, Not a Queue

This is the most important thing to understand about Kafka. Traditional message queues delete messages after consumption. Kafka keeps messages for a configurable retention period (default: 7 days) regardless of whether they were consumed.

```
Traditional Queue (SQS, RabbitMQ):
  Producer → [Queue] → Consumer reads → Message DELETED

Kafka:
  Producer → [Topic/Partition] → Consumer reads → Message STAYS
                                → Consumer 2 reads → Message STAYS
                                → Consumer 3 reads from beginning → Message STAYS
                                (until retention period expires)
```

**Why this matters:**
- Multiple independent consumers can each read the full stream at their own pace
- If a consumer has a bug and processes messages wrong, it can re-read from a previous offset and reprocess
- A new service can be added and read historical data from whenever Kafka has retained

---

## 2. Core Concepts

### Topics and Partitions

A **topic** is a named stream of messages (like a table in a database). A topic is split into **partitions** — ordered, immutable sequences of messages.

```
Topic: "orders"
  Partition 0: [msg1, msg3, msg5, msg7, ...]
  Partition 1: [msg2, msg4, msg6, msg8, ...]
  Partition 2: [msg9, msg10, msg11, ...]

Each message in a partition has an offset (its position):
  Partition 0: offset 0=msg1, offset 1=msg3, offset 2=msg5 ...
```

**Why partitions exist:** A single partition is processed by one consumer at a time (for ordering). Multiple partitions allow multiple consumers to work in parallel. More partitions = more parallelism.

### Offsets

An **offset** is a sequential integer that identifies a message's position within a partition. Consumers track their offset — how far they have read.

```
Partition 0:
  offset 0: {"order_id": "ord_1", ...}
  offset 1: {"order_id": "ord_3", ...}
  offset 2: {"order_id": "ord_5", ...}
  offset 3: {"order_id": "ord_7", ...}

Consumer group "shipping" is at offset 2 on partition 0
  → has processed ord_1 and ord_3
  → will next consume offset 2 (ord_5)

Consumer crashes, restarts
  → reads committed offset from Kafka: offset 2
  → resumes from ord_5 — no messages lost
```

Consumers **commit** their offset after processing. On restart, they resume from the committed offset.

### Consumer Groups

A **consumer group** is a set of consumers that together consume a topic. Kafka ensures each partition is consumed by exactly one consumer in the group at a time.

```
Topic: "orders" (3 partitions)
Consumer Group: "shipping-service" (3 consumers)

Partition 0 → Consumer A (exclusively)
Partition 1 → Consumer B (exclusively)
Partition 2 → Consumer C (exclusively)

→ 3 consumers process partitions in parallel
→ All messages are processed, none are duplicated within the group
```

**Scaling consumers:** Adding consumers up to the number of partitions increases throughput. Adding more consumers than partitions is wasteful — extra consumers are idle.

```
3 partitions, 4 consumers in group:
  Partition 0 → Consumer A
  Partition 1 → Consumer B
  Partition 2 → Consumer C
  Consumer D: idle (no partition to assign)
```

**Multiple consumer groups:** Two independent services can each have their own consumer group, both reading the full topic independently:

```
Topic: "user_events"
Consumer Group "analytics": reads all events for analytics
Consumer Group "notifications": reads all events for notifications
Consumer Group "ml-pipeline": reads all events for ML training

All three read independently — no coordination needed
```

This is Kafka's superpower for event-driven architectures.

---

## 3. Producers and Partition Assignment

When a producer sends a message, Kafka must decide which partition to put it in:

```python
# No key: round-robin across partitions (even distribution)
producer.send("orders", value=order_data)

# With key: consistent hashing (same key always → same partition)
producer.send("orders", key=user_id, value=order_data)
```

**Why partition keys matter:**

With a key, all messages for the same key go to the same partition, in order. This guarantees ordering per key:

```
key=user_42: all messages for user 42 → partition 0, in insertion order
key=user_99: all messages for user 99 → partition 1, in insertion order

Within partition 0: user 42's events are ordered
Across partitions: no global ordering guarantee
```

If you need to process all events for a user in order, use user_id as the partition key.

---

## 4. The Hot Partition Problem

If your partition key is not evenly distributed, one partition receives disproportionate traffic — a hot partition.

```
Partition key: user_id
Normal users: 100 messages/day
Celebrity user (user_id=1): 10,000,000 messages/day

All messages for user_id=1 → partition 0
Partition 0 consumer: overwhelmed, falls behind
Other partitions: nearly idle

Result: consumer lag builds on partition 0
        overall throughput limited by the hottest partition
```

### Solutions

**Option 1 — Add randomness to hot key:**
```python
# For the celebrity user, spread across sub-partitions
if is_hot_user(user_id):
    sub_partition = random.randint(0, 9)
    key = f"{user_id}_{sub_partition}"
else:
    key = str(user_id)
```

**Option 2 — Use a different partition key:** If user_id creates hotspots, try a more evenly distributed key (e.g. a hash of user_id + timestamp bucket).

**Option 3 — Increase partition count:** More partitions means hotspots are diluted.

---

## 5. Retention and Replayability

Kafka retains messages for a configurable duration, regardless of consumption:

```bash
# Topic configuration
retention.ms = 604800000      # 7 days (default)
retention.bytes = 1073741824  # OR: 1GB per partition
```

**What retention enables:**

**Reprocessing:** If your consumer had a bug and processed messages incorrectly, reset the consumer offset to the beginning and reprocess all messages with the fixed consumer.

```python
# Reset consumer offset to beginning (reprocess all retained messages)
consumer.seek_to_beginning()

# Or reset to a specific timestamp
target_time = datetime(2024, 1, 10, 0, 0, 0)
offsets = consumer.offsets_for_times({partition: target_time.timestamp() * 1000})
consumer.seek(partition, offsets[partition].offset)
```

**New service onboarding:** When you add a new service that needs historical data, it reads from offset 0 and processes all retained history before catching up to the present.

**Audit trail:** Kafka's log is an immutable history of events. Events cannot be modified or deleted (before retention expires). This makes it a natural audit log.

---

## 6. When to Use Kafka vs Simpler Queues

| Use Kafka when | Use SQS/RabbitMQ when |
|---------------|----------------------|
| Multiple independent services need to consume the same events | One consumer per message (task queue) |
| You need message replay / reprocessing history | No need for replay |
| High throughput (100K+ messages/second) | Moderate volume |
| Event sourcing / event-driven architecture | Simple background task processing |
| You want to build a real-time data pipeline | Occasional background jobs |
| Ordering within a key is required | No ordering required |

---

## Summary

- Kafka is a distributed commit log — messages are retained after consumption, not deleted
- Topics have partitions — each partition is an ordered, immutable sequence of messages
- Offsets track consumer position — restart resumes from committed offset, no message loss
- Consumer groups: each partition assigned to one consumer — parallelism bounded by partition count
- Partition keys guarantee ordering per key — all events for a user go to the same partition
- Hot partition problem: popular keys overload one partition — fix with sub-partitioning or different key strategy
- Retention enables replayability: new services can catch up, bugs can be reprocessed
- Use Kafka for high-volume, multi-consumer, event-sourcing scenarios. Use SQS/RabbitMQ for simple task queues.

---

> Next: [Lesson 5.4 — RabbitMQ and SQS](./lesson-5.4-rabbitmq-sqs.md)