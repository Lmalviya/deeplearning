# Lesson 5.7 — Event-Driven Architecture

> **Chapter 5 — Async Processing and Message Queues**
> Previous: [Lesson 5.6 — Consumer Lag and Backpressure](./lesson-5.6-consumer-lag.md) | Next: [Chapter 6 — The Delivery Layer](../chapter-6/lesson-6.1-object-storage.md)

---

## What this lesson covers

- Events vs commands vs queries — the three message types
- Event sourcing — storing state as a sequence of events
- CQRS — separating read and write models
- The benefits and real costs of event-driven design
- When event-driven is worth the complexity and when it is not

---

## 1. Three Types of Messages

Not all messages are the same. Understanding the type of message shapes how you design the system.

### Commands — "Do this"

A command tells a service to perform an action. It is directed at a specific service. It expects the service to do something.

```json
{
  "type": "SendWelcomeEmail",
  "to": "alice@example.com",
  "name": "Alice Chen"
}
```

A command is an **instruction**. The sender knows who will process it. If no one processes it, that is an error.

### Events — "This happened"

An event announces that something occurred. It is not directed at any specific receiver. The publisher does not know or care who is listening.

```json
{
  "type": "UserCreated",
  "user_id": 42,
  "email": "alice@example.com",
  "name": "Alice Chen",
  "timestamp": "2024-01-15T08:30:00Z"
}
```

An event is a **fact**. It happened in the past. Multiple services can react to it independently.

### Queries — "Tell me this"

A query asks for data. It is synchronous (request-response) and does not change state.

```
GET /api/users/42
→ Returns user data
```

### Why the distinction matters

| | Command | Event | Query |
|---|---|---|---|
| Direction | Directed (to specific service) | Broadcast (no specific receiver) | Direct (request-response) |
| Coupling | Tight (sender knows receiver) | Loose (sender doesn't know receivers) | Tight |
| Receivers | One | Many | One |
| Past/Future | Future instruction | Past fact | Present state |
| On failure | Error (must be handled) | Missed (receiver must catch up) | Error |

---

## 2. Event-Driven Architecture — The Core Pattern

In an event-driven system, services communicate by publishing and consuming events. No service calls another service directly for notifications or side effects.

### Tightly coupled (command-based)

```
User Service → HTTP POST /email-service/send-welcome-email
User Service → HTTP POST /analytics-service/track-signup
User Service → HTTP POST /recommendation-service/seed-recommendations
User Service → HTTP POST /notification-service/send-push

User Service knows about and calls 4 other services.
Adding a 5th service that needs to react to signups requires
modifying User Service.
```

### Loosely coupled (event-driven)

```
User Service:
  1. Create user in DB
  2. Publish "UserCreated" event to Kafka
  3. Return response — done

Email Service (consumer):       receives "UserCreated" → sends welcome email
Analytics Service (consumer):   receives "UserCreated" → tracks acquisition
Recommendation Service:         receives "UserCreated" → seeds recommendations
Notification Service:           receives "UserCreated" → sends push notification

User Service knows about none of these.
Adding a new service that reacts to signups: just add a new consumer.
User Service is not modified.
```

---

## 3. Event Sourcing — State as a Sequence of Events

Traditional databases store the **current state**: "Alice's balance is $500."

Event sourcing stores the **history of events** that led to the current state: "Alice deposited $1000, then withdrew $300, then withdrew $200."

```
Traditional (current state):
  accounts table:
  | id | owner | balance |
  | 42 | Alice |    500  |

Event sourcing (event log):
  events table:
  | id | account_id | type       | amount | timestamp           |
  |  1 |         42 | deposited  |   1000 | 2024-01-10 09:00:00 |
  |  2 |         42 | withdrew   |    300 | 2024-01-12 14:00:00 |
  |  3 |         42 | withdrew   |    200 | 2024-01-15 11:00:00 |

Current balance = sum of all events = 1000 - 300 - 200 = 500 ✅
```

To get the current state, you "replay" the events from the beginning (or from a snapshot).

### Benefits of event sourcing

**Complete audit trail:** Every change is recorded permanently. "What was Alice's balance on January 11th?" is trivially answerable (replay events up to Jan 11).

**Temporal queries:** State at any point in time is queryable.

**Event replay:** If business logic changes (bug fix, rule change), you can replay all historical events through the new logic to produce correct current state.

**Debugging:** You know exactly how a system arrived at its current state — replay the events that led there.

### Costs of event sourcing

**Query complexity:** Getting current state requires replaying events. Use **snapshots** to avoid replaying from the beginning on every query:

```python
def get_account_balance(account_id: int) -> int:
    # Find the most recent snapshot
    snapshot = db.query("""
        SELECT balance, last_event_id FROM account_snapshots
        WHERE account_id = %s
        ORDER BY last_event_id DESC LIMIT 1
    """, account_id)

    if snapshot:
        # Replay only events since the snapshot
        events_since = db.query("""
            SELECT type, amount FROM events
            WHERE account_id = %s AND id > %s
            ORDER BY id ASC
        """, account_id, snapshot['last_event_id'])
        balance = snapshot['balance']
    else:
        # No snapshot — replay from beginning
        events_since = db.query("SELECT type, amount FROM events WHERE account_id = %s ORDER BY id", account_id)
        balance = 0

    for event in events_since:
        if event['type'] == 'deposited':
            balance += event['amount']
        elif event['type'] == 'withdrew':
            balance -= event['amount']

    return balance
```

**Schema evolution:** Events are immutable — you cannot change old events. If the event schema changes, you must handle multiple schema versions in your consumers.

**High event volume:** Systems with frequent state changes generate enormous event logs. Storage and replay performance must be managed.

---

## 4. CQRS — Command Query Responsibility Segregation

CQRS separates the data model used for writes (commands) from the model used for reads (queries).

```
Traditional (one model for everything):
  API → DB (write) + DB (read)
  One DB does both — optimizing for one often hurts the other

CQRS (separate models):
  Write side: Command → Write Model → Event Store / Command DB
  Read side:  Query  → Read Model  → Read DB (optimized for reads)

  Events from write side propagate to update the read model
```

### Concrete example: E-commerce order system

```
Write model (normalized, ACID-consistent):
  orders: {id, user_id, status, created_at}
  order_items: {order_id, product_id, qty, price}
  payments: {order_id, amount, status}
  
  Optimized for: writes, consistency, complex updates

Read model (denormalized, query-optimized):
  order_summary: {
    order_id, user_name, user_email, 
    items: [{name, qty, price}],
    total, status, payment_status,
    created_at
  }
  
  Optimized for: "show me order details" query without JOINs
```

```
Flow:
  1. User places order → write to command DB (normalized)
  2. "OrderPlaced" event published
  3. Read model updater consumes event → updates order_summary table
  4. User views order → read from read model (no JOINs needed)
```

### When CQRS is worth it

- **Very different read vs write access patterns** — write model is normalized for integrity; read model is denormalized for performance
- **Read/write throughput mismatch** — 1,000× more reads than writes; separate read and write stores can be scaled independently
- **Complex reads** — the read model is built specifically for the queries you need, not general-purpose

### When CQRS is NOT worth it

- Simple CRUD applications where read and write patterns are similar
- Small teams — CQRS adds significant complexity (two models, sync between them, eventual consistency)
- When strong consistency is required — CQRS makes reads eventually consistent (read model may lag behind write model)

---

## 5. The Real Costs of Event-Driven Architecture

Event-driven is often presented as purely beneficial. The costs are real and significant.

| Benefit | Real cost |
|---------|----------|
| Loose coupling | Debugging is harder — a bug may span 5 services connected by events |
| Independent scaling | Each service must be operated independently — more infrastructure |
| Easy to add consumers | Hard to know all the things that will happen when you publish an event |
| Resilience via async | Failures are silent — events fail to process, users affected hours later |
| Replay capability | Replay can cause unintended side effects (emails sent twice, charges duplicated) |

### The "what happens when I publish this event?" problem

In a tightly coupled system, you can read the code and know exactly what will happen. In event-driven systems:

```
You publish: "OrderPlaced" event

What happens?
  Email service: sends confirmation email
  Analytics: records order
  Inventory: decrements stock
  Recommendation: updates "users who bought X also bought..."
  Fraud detection: checks for suspicious patterns
  Loyalty points: awards points
  Warehouse: creates pick list
  Supplier: triggers reorder if stock low
  ...

Every team that subscribed without telling you is now a consumer.
If you change the event schema, they all break.
```

This is why event-driven systems need a **schema registry** (Confluent Schema Registry for Kafka) — a central place to define and version event schemas, enforced at publish and consume time.

### The right scale for event-driven

Event-driven architecture pays off at a certain organizational scale:

```
2-person startup:
  Event-driven is overkill — direct function calls are fine
  Operational overhead far outweighs the loose coupling benefit

5-team company (5–10 services):
  Event-driven starts paying off
  Teams can deploy independently without coordinating

50-team company (50+ services):
  Event-driven is almost mandatory
  Direct coupling between 50 services would be unmanageable
```

---

## ✅ Chapter 5 Complete

Chapter 5 has covered the full async layer:

- **5.1** Why async — the case for deferring work, the cost of sync blocking, three async patterns
- **5.2** Queue fundamentals — producers, consumers, brokers, queues vs topics, acknowledgement, delivery guarantees, DLQ
- **5.3** Kafka — distributed log, topics/partitions/offsets/consumer groups, hot partitions, retention and replay
- **5.4** RabbitMQ and SQS — exchanges and routing, SQS visibility timeout and long polling, FIFO, decision matrix
- **5.5** Idempotency — the idempotency key pattern, DB-based and Redis-based implementation, payment APIs, natural idempotency
- **5.6** Consumer lag — causes, scaling consumers, backpressure (semaphores, backoff, circuit breakers), monitoring
- **5.7** Event-driven architecture — events vs commands, event sourcing, CQRS, the real costs and organizational fit

---

> Next: [Chapter 6 — The Delivery Layer](../chapter-6/lesson-6.1-object-storage.md)