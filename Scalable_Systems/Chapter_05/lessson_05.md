# Lesson 5.5 — Idempotency — The Hardest Queue Problem

> **Chapter 5 — Async Processing and Message Queues**
> Previous: [Lesson 5.4 — RabbitMQ and SQS](./lesson-5.4-rabbitmq-sqs.md) | Next: [Lesson 5.6 — Consumer Lag and Backpressure](./lesson-5.6-consumer-lag.md)

---

## What this lesson covers

- What idempotency means and why at-least-once delivery requires it
- The idempotency key pattern — the standard solution
- How to implement idempotency at the database level
- Idempotency for external API calls (payments, email)
- Natural idempotency — operations that are already safe to repeat

---

## 1. The Problem — At-Least-Once Means Duplicates Will Happen

As established in Lesson 5.2, the standard delivery guarantee is at-least-once. This means:

```
Normal flow:
  Broker → Consumer: message (visibility hidden)
  Consumer: processes successfully
  Consumer → Broker: ACK
  Broker: deletes message ✅

But this can also happen:
  Broker → Consumer: message (visibility hidden)
  Consumer: processes successfully (payment charged, email sent)
  Network glitch: ACK is lost
  Broker: timeout expires, re-delivers message
  Consumer: processes AGAIN (payment charged TWICE, email sent TWICE) ❌
```

Duplicate processing is not a bug in your queue configuration. It is a fundamental property of distributed systems. Your consumers must be designed to handle it.

**Idempotency:** An operation is idempotent if applying it multiple times has the same effect as applying it once.

```
Idempotent:     SET user.name = "Alice"   ← run 100 times, same result
Not idempotent: increment user.balance by 100  ← run 100 times, balance grows by 10,000
```

---

## 2. The Idempotency Key Pattern

The standard solution: **include a unique identifier in every message and record which identifiers have been processed.**

```json
{
  "idempotency_key": "send_email_user_42_welcome_2024-01-15",
  "event_type": "user.created",
  "user_id": 42,
  "email": "alice@example.com"
}
```

Before processing, check if this idempotency key has been seen before. If yes, skip. If no, process and record.

### Implementation with database

```python
def process_welcome_email(message: dict):
    idempotency_key = message['idempotency_key']

    # Check if already processed
    already_processed = db.query(
        "SELECT id FROM processed_messages WHERE idempotency_key = %s",
        idempotency_key
    )
    if already_processed:
        print(f"Duplicate message {idempotency_key}, skipping")
        return  # safe to return — work was already done

    # Process the message
    email_service.send_welcome_email(
        to=message['email'],
        name=message['name']
    )

    # Record as processed (within a transaction to avoid race conditions)
    db.execute(
        "INSERT INTO processed_messages (idempotency_key, processed_at) VALUES (%s, NOW())",
        idempotency_key
    )
```

```sql
-- The table that tracks processed messages
CREATE TABLE processed_messages (
    idempotency_key VARCHAR(255) PRIMARY KEY,
    processed_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Clean up old records (optional, after 7 days they cannot be re-delivered from queue anyway)
CREATE INDEX idx_pm_processed_at ON processed_messages(processed_at);
```

### Race condition in the check-then-process pattern

Two consumers receive the same duplicate message simultaneously:

```
Consumer A: SELECT → not found → processes → INSERT
Consumer B: SELECT → not found (A hasn't inserted yet) → processes → INSERT (UNIQUE violation)
```

Fix: use a unique constraint on `idempotency_key` and catch the duplicate key error:

```python
try:
    db.execute(
        "INSERT INTO processed_messages (idempotency_key) VALUES (%s)",
        idempotency_key
    )
except UniqueConstraintViolation:
    print(f"Race condition: {idempotency_key} processed by another consumer")
    return  # Other consumer won — this is fine, work was done

# If INSERT succeeded, we "own" this message — proceed
email_service.send_welcome_email(...)
```

The unique constraint makes the check-and-record atomic.

### Implementation with Redis

For high-volume scenarios where DB writes for every message add overhead:

```python
def process_message_idempotent(message: dict):
    key = f"processed:{message['idempotency_key']}"

    # SET with NX (only if not exists) and EX (expire after 7 days)
    already_processing = not redis.set(key, "1", nx=True, ex=604800)

    if already_processing:
        return  # already processed or in progress

    # Process message
    do_the_work(message)
```

Redis `SET NX` is atomic — only one consumer can set the key and "win" the right to process.

---

## 3. Idempotency for Payment Charges

Payments are the highest-stakes case for idempotency. Charging a customer twice is a serious bug.

Most payment providers (Stripe, Razorpay) have built-in idempotency key support:

```python
import stripe

def charge_customer(user_id: int, amount_cents: int, order_id: str):
    try:
        charge = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency="inr",
            customer=get_stripe_customer_id(user_id),
            idempotency_key=f"order_{order_id}_charge"  # Stripe-level idempotency
            # If this is called twice with the same key:
            # Stripe returns the SAME PaymentIntent, does NOT charge twice
        )
        return charge
    except stripe.error.IdempotencyError:
        # Called with same key but different parameters — programming error
        raise
```

**Always use idempotency keys when calling payment APIs.** The key should be derived from the business entity (order ID, not a random UUID), so the same logical operation always produces the same key.

---

## 4. Idempotency for Email Sending

Sending an email is not natively idempotent — if you call the email API twice, the user gets two emails. You must implement idempotency yourself.

```python
def send_welcome_email_idempotent(user_id: int, email: str):
    idempotency_key = f"welcome_email_user_{user_id}"

    # Check if email was already sent
    sent = db.query(
        "SELECT sent_at FROM email_log WHERE idempotency_key = %s",
        idempotency_key
    )
    if sent:
        return  # already sent

    # Send the email
    email_provider.send(to=email, template="welcome")

    # Record that it was sent
    db.execute(
        "INSERT INTO email_log (idempotency_key, sent_at) VALUES (%s, NOW())",
        idempotency_key
    )
```

---

## 5. Natural Idempotency — Operations That Are Already Safe

Some operations are naturally idempotent — you do not need an idempotency key because repeating them has no additional effect:

```python
# Naturally idempotent:

# SET operations (not increment)
db.execute("UPDATE users SET name = 'Alice' WHERE id = 42")
# Run 10 times → name is 'Alice' (same result)

# Upsert (INSERT ... ON CONFLICT DO UPDATE)
db.execute("""
    INSERT INTO user_preferences (user_id, theme) VALUES (%s, %s)
    ON CONFLICT (user_id) DO UPDATE SET theme = EXCLUDED.theme
""", user_id, 'dark')
# Run 10 times → preference is 'dark' (safe)

# DELETE with WHERE clause
db.execute("DELETE FROM sessions WHERE token = %s", token)
# Run 10 times → session is deleted (subsequent runs are no-ops)

# NOT naturally idempotent:

redis.incr("page_views:post:42")      # increments every time
db.execute("INSERT INTO events ...")  # inserts duplicates
db.execute("UPDATE account SET balance = balance + 100 ...")  # adds 100 every time
```

When designing consumer logic, **prefer naturally idempotent operations.** Use upserts instead of inserts. Use SET instead of INCREMENT where possible.

---

## 6. Choosing the Idempotency Key

The idempotency key must be:

**Deterministic:** The same logical operation always produces the same key. Do not use random UUIDs that change on retry.

```python
# Wrong: random UUID changes on every retry
idempotency_key = str(uuid.uuid4())  # ← different on each retry, no deduplication

# Right: derived from the business entity
idempotency_key = f"order_{order_id}_charge"  # ← same on retry
idempotency_key = f"welcome_email_user_{user_id}"  # ← same on retry
idempotency_key = f"invoice_{invoice_id}_pdf"  # ← same on retry
```

**Unique per logical operation:** Different operations must have different keys.

```python
# Wrong: same key for different emails to the same user
idempotency_key = f"email_user_{user_id}"  # ← collision between welcome and password reset emails

# Right: include the operation type
idempotency_key = f"welcome_email_user_{user_id}"
idempotency_key = f"password_reset_email_user_{user_id}_2024-01-15"  # include time for repeatable ops
```

**Scoped to a reasonable window:** The key only needs to be unique within the message retention window (e.g. 7 days for Kafka). Old idempotency records can be purged.

---

## Summary

- At-least-once delivery guarantees duplicates will happen — consumers must be idempotent
- Idempotency key pattern: include a unique key in every message, record processed keys, skip duplicates
- Use a unique constraint on the idempotency key table to handle concurrent duplicate processing safely
- Payment APIs (Stripe, Razorpay) support idempotency keys natively — always use them
- Email sending is not naturally idempotent — implement your own deduplication with an email log
- Prefer naturally idempotent operations: SET over increment, upsert over insert
- Derive idempotency keys from business entities (order ID, user ID) — never use random UUIDs that change on retry

---

## ⚠️ Common Mistakes

- Using a random UUID as the idempotency key — it changes on every retry, providing no protection
- No idempotency key at all — assuming "our queue won't deliver duplicates" — it will, eventually
- Idempotency check without a unique constraint — race condition allows two consumers to both process the same message
- Not cleaning up old idempotency records — table grows unbounded, slows down idempotency checks
- Checking idempotency after the side effect — check BEFORE calling the payment API or sending the email

---

> Next: [Lesson 5.6 — Consumer Lag and Backpressure](./lesson-5.6-consumer-lag.md)