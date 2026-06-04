# Lesson 5.6 — Conversational RAG: Memory, History Compression, and Session Context

---

## The Multi-Turn Challenge

Single-turn RAG is stateless: each query is independent, retrieval runs fresh, context is assembled from scratch. This is simple and predictable.

Conversational RAG adds a dimension that fundamentally changes how retrieval and generation must work: history. A conversation has memory. Each turn builds on the previous ones. The user may refer back to earlier topics, ask follow-up questions, change direction, or make requests that are only meaningful in the context of what came before.

The challenges this introduces:

**Query dependence:** "Can I apply for that?" — "that" refers to something mentioned three turns ago. Without resolving this reference, retrieval retrieves nothing useful.

**Context accumulation:** After 20 turns, the conversation history plus retrieved context easily exceeds the LLM's context window. You cannot include everything.

**Topic drift:** The user may start asking about leave policy, drift to compensation, then return to leave policy. The retrieval system needs to track what topic is currently active.

**Redundant retrieval:** Turn 3 and turn 8 might ask about essentially the same thing. Retrieving fresh context for both is wasteful and sometimes confusing (retrieved chunks may be slightly different, causing the LLM to give slightly different answers to the same underlying question).

**Memory vs. retrieval:** Some things should be remembered from earlier in the conversation (the user told you their department) rather than retrieved from documents. Others must always be retrieved fresh (current policy status). These are different types of "memory" and must be handled differently.

---

## The Three Types of Memory in Conversational RAG

Before designing a solution, distinguish what must be remembered and how:

**1. Conversation context memory:**
What has been discussed, what the user has told you, what references exist in the conversation ("that contract", "the policy I mentioned"). Needed for query resolution. Lives in the conversation history.

**2. Retrieved document memory:**
Chunks retrieved in previous turns that are still relevant. Avoids redundant retrieval and provides consistency across turns on the same topic.

**3. User/session state memory:**
Persistent facts about the user that apply across the whole session: their department, their role, their stated preferences. Informs retrieval filtering and response personalization.

Each type requires a different management strategy.

---

## Managing Conversation History

### The Raw History Approach (Small Scale)

For short conversations, include the full history in every LLM call:

```python
class ConversationalRAGSession:
    def __init__(self, retriever, llm_client, max_history_tokens: int = 4000):
        self.retriever = retriever
        self.llm = llm_client
        self.history = []  # List of {role, content} dicts
        self.max_history_tokens = max_history_tokens
        self.session_state = {}  # Persistent user/session facts
    
    async def turn(self, user_message: str) -> str:
        # Step 1: Resolve references using conversation context
        standalone_query = await self._resolve_query(user_message)
        
        # Step 2: Retrieve context for the resolved query
        retrieved_chunks = await self.retriever.retrieve(standalone_query)
        context = format_context(retrieved_chunks)
        
        # Step 3: Build the full message list for this turn
        messages = self._build_messages(user_message, context)
        
        # Step 4: Generate response
        response = await self.llm.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=800,
            temperature=0.1
        )
        
        assistant_message = response.choices[0].message.content
        
        # Step 5: Update history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        
        # Step 6: Extract and store any session state updates
        await self._update_session_state(user_message, assistant_message)
        
        return assistant_message
    
    def _build_messages(self, current_user_message: str, context: str) -> list[dict]:
        """Build the message list for the LLM call, respecting token budget."""
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        
        # System prompt
        system = {
            "role": "system",
            "content": self._build_system_prompt()
        }
        
        # Current user message with retrieved context
        current_message = {
            "role": "user",
            "content": f"[Context from documents]\n{context}\n\n[Question]\n{current_user_message}"
        }
        
        # Trim history to fit within token budget
        available_tokens = self.max_history_tokens
        trimmed_history = []
        
        # Walk history from most recent backward
        for msg in reversed(self.history):
            msg_tokens = len(enc.encode(msg["content"]))
            if available_tokens - msg_tokens > 0:
                trimmed_history.insert(0, msg)
                available_tokens -= msg_tokens
            else:
                break
        
        return [system] + trimmed_history + [current_message]
    
    def _build_system_prompt(self) -> str:
        base_prompt = "You are a helpful assistant answering questions from company documents."
        
        # Inject session state into system prompt
        if self.session_state:
            state_text = "\n".join([f"- {k}: {v}" for k, v in self.session_state.items()])
            base_prompt += f"\n\nKnown user context:\n{state_text}"
        
        return base_prompt
```

### The History Problem: Context Window Exhaustion

A 20-turn conversation at 200 tokens per turn uses 4,000 tokens just for history. Add retrieved context (2,000–5,000 tokens), system prompt (500 tokens), and current message — you are at 7,000–10,000 tokens before generation. For a 16K window model like gpt-4o-mini, this is tight. For longer conversations, it breaks.

The solution is history compression: summarize or compress older history while keeping recent turns verbatim.

---

## History Compression Strategies

### Strategy 1 — Sliding Window

Keep the last N turns verbatim and discard everything older.

```python
def sliding_window_history(
    history: list[dict],
    window_size: int = 6  # Keep last 6 turns (3 exchanges)
) -> list[dict]:
    """Keep only the most recent turns."""
    return history[-window_size:]
```

Simple but lossy. Topics discussed more than `window_size` turns ago are completely forgotten. If the user refers back to turn 2 in turn 15, the reference cannot be resolved.

### Strategy 2 — Progressive Summarization

Summarize older turns into a compact summary, keep recent turns verbatim.

```python
async def progressive_summarize_history(
    history: list[dict],
    llm_client,
    keep_recent_turns: int = 4,
    max_summary_tokens: int = 500
) -> list[dict]:
    """
    Compress old history into a summary, keep recent history verbatim.
    """
    if len(history) <= keep_recent_turns * 2:
        return history  # Not enough history to compress
    
    # Split: history to compress vs history to keep
    compress_history = history[:-keep_recent_turns * 2]
    keep_history = history[-keep_recent_turns * 2:]
    
    # Format the history to compress
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:200]}"
        for msg in compress_history
    ])
    
    # Generate summary
    summary_prompt = f"""Summarize this conversation history concisely.
Preserve: key facts shared by the user, decisions made, topics covered, 
entities mentioned (people, documents, policies).
Discard: pleasantries, repeated information, procedural back-and-forth.

Conversation to summarize:
{history_text}

Write a compact summary (max {max_summary_tokens // 4} words):"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=max_summary_tokens,
        temperature=0.0
    )
    
    summary = response.choices[0].message.content
    
    # Return: summary as a special message + recent verbatim history
    summary_message = {
        "role": "system",
        "content": f"[Earlier conversation summary]\n{summary}"
    }
    
    return [summary_message] + keep_history
```

### Strategy 3 — Entity-Centric Memory

Instead of summarizing the conversation chronologically, extract and maintain a structured memory of entities and facts mentioned.

```python
async def update_entity_memory(
    user_message: str,
    assistant_response: str,
    existing_memory: dict,
    llm_client
) -> dict:
    """
    Extract new facts from this turn and update the entity memory.
    """
    
    prompt = f"""Update the entity memory with new information from this conversation turn.

Existing memory:
{json.dumps(existing_memory, indent=2) if existing_memory else "Empty"}

New turn:
USER: {user_message}
ASSISTANT: {assistant_response}

Extract any new facts about:
- User (their role, department, situation, preferences)
- Documents mentioned (titles, IDs, topics)
- Decisions made or actions to take
- Key entities referenced (people, policies, contracts)

Return updated memory as JSON. Merge with existing, don't replace unless correcting a fact."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=400,
        temperature=0.0
    )
    
    import json
    updates = json.loads(response.choices[0].message.content)
    
    # Deep merge with existing memory
    merged = {**existing_memory}
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    
    return merged
```

Entity-centric memory is more compact than conversation history and survives across multiple compression cycles. It captures the semantic content of the conversation rather than its conversational form.

**Combined approach (recommended for production):**

```
Conversation memory:
├── Entity memory (structured, all turns) — always included
├── Recent verbatim history (last 4-6 turns) — always included  
└── Compressed summary (older turns) — included if needed
```

---

## Conversational Query Resolution

The most critical piece of conversational RAG: before retrieval, resolve queries that contain references to conversation context.

This was introduced in Lesson 3.4, but here we go deeper with session state integration.

```python
async def resolve_conversational_query(
    current_query: str,
    history: list[dict],
    entity_memory: dict,
    session_state: dict,
    llm_client
) -> dict:
    """
    Convert a context-dependent query into a standalone retrieval query.
    Uses both conversation history and structured entity memory.
    """
    
    # Quick check: does the query need resolution?
    resolution_signals = [
        "that", "it", "this", "those", "they", "them",
        "the same", "what about", "and also", "how about",
        "the one", "mentioned", "above"
    ]
    
    query_lower = current_query.lower()
    needs_resolution = any(signal in query_lower for signal in resolution_signals)
    
    if not needs_resolution and len(current_query.split()) > 5:
        # Self-contained query — return as-is with optional session filter
        return {
            "standalone_query": current_query,
            "resolved": False,
            "metadata_filter": build_filter_from_session(session_state)
        }
    
    # Build context for resolution
    recent_history = history[-6:]  # Last 3 exchanges
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:300]}"
        for msg in recent_history
    ])
    
    memory_text = format_entity_memory(entity_memory)
    
    prompt = f"""Resolve this query into a standalone retrieval query.

Entity memory (facts known about this user's context):
{memory_text}

Recent conversation:
{history_text}

Current query: "{current_query}"

Rewrite as a complete, self-contained search query that does not rely on 
conversation context. Include all relevant entities, topics, and constraints.

Return JSON:
{{
    "standalone_query": "the resolved, standalone query",
    "key_entities": ["list of entities this query is about"],
    "filters": {{
        "document_type": "type if implied or null",
        "specific_document": "document name/id if implied or null"
    }},
    "resolution_notes": "brief explanation of what was resolved"
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=300,
        temperature=0.0
    )
    
    import json
    result = json.loads(response.choices[0].message.content)
    result["resolved"] = True
    result["original_query"] = current_query
    
    # Build metadata filter from resolved entities + session state
    result["metadata_filter"] = build_filter_from_resolution(result, session_state)
    
    return result
```

---

## Retrieved Context Caching

In long conversations, the user may ask about the same topic multiple times. Retrieving fresh context each time is redundant and can cause inconsistency (slightly different chunks each time, leading to slightly different answers).

Cache retrieved contexts by topic within a session:

```python
class SessionRetrievalCache:
    def __init__(self, embedding_model, similarity_threshold: float = 0.85):
        self.cache = []  # List of {query_embedding, context, metadata, turn_number}
        self.embedder = embedding_model
        self.threshold = similarity_threshold
    
    async def get_or_retrieve(
        self,
        query: str,
        retriever,
        current_turn: int,
        max_cache_age_turns: int = 5
    ) -> tuple[list[dict], bool]:
        """
        Return cached context if a similar query was recently answered,
        otherwise retrieve fresh context.
        Returns (chunks, from_cache).
        """
        query_embedding = await self.embedder.embed(query)
        
        # Check cache for similar recent query
        for cached in reversed(self.cache):
            # Only use cache from recent turns
            if current_turn - cached["turn_number"] > max_cache_age_turns:
                continue
            
            similarity = cosine_similarity(query_embedding, cached["query_embedding"])
            
            if similarity >= self.threshold:
                return cached["chunks"], True
        
        # Cache miss — retrieve fresh
        chunks = await retriever.retrieve(query)
        
        self.cache.append({
            "query": query,
            "query_embedding": query_embedding,
            "chunks": chunks,
            "turn_number": current_turn
        })
        
        return chunks, False
    
    def invalidate(self, doc_ids: list[str] = None):
        """
        Invalidate cache entries. Call when documents are updated.
        """
        if doc_ids is None:
            self.cache.clear()
        else:
            self.cache = [
                c for c in self.cache
                if not any(
                    chunk["metadata"].get("doc_id") in doc_ids
                    for chunk in c["chunks"]
                )
            ]
```

---

## The Complete Conversational RAG Session

Putting all components together:

```python
class ProductionConversationalRAG:
    def __init__(self, retriever, llm_client, embedding_model):
        self.retriever = retriever
        self.llm = llm_client
        self.embedder = embedding_model
        
        # Session state
        self.history = []
        self.entity_memory = {}
        self.session_state = {}
        self.turn_number = 0
        
        # Retrieval cache
        self.cache = SessionRetrievalCache(embedding_model)
    
    async def turn(self, user_message: str) -> dict:
        self.turn_number += 1
        
        # Step 1: Resolve conversational references
        resolution = await resolve_conversational_query(
            current_query=user_message,
            history=self.history,
            entity_memory=self.entity_memory,
            session_state=self.session_state,
            llm_client=self.llm
        )
        
        standalone_query = resolution["standalone_query"]
        metadata_filter = resolution.get("metadata_filter")
        
        # Step 2: Retrieve (with caching)
        chunks, from_cache = await self.cache.get_or_retrieve(
            query=standalone_query,
            retriever=self.retriever,
            current_turn=self.turn_number
        )
        
        # Apply metadata filter to cached results if needed
        if metadata_filter and from_cache:
            chunks = [c for c in chunks if matches_filter(c, metadata_filter)]
        
        # Step 3: Compress history if needed
        compressed_history = await self._get_compressed_history()
        
        # Step 4: Build context
        context = format_context(chunks)
        
        # Step 5: Build messages
        messages = self._build_messages(
            user_message=user_message,
            context=context,
            compressed_history=compressed_history
        )
        
        # Step 6: Generate
        response = await self.llm.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=800,
            temperature=0.1
        )
        
        assistant_message = response.choices[0].message.content
        
        # Step 7: Update history and memory
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        
        self.entity_memory = await update_entity_memory(
            user_message=user_message,
            assistant_response=assistant_message,
            existing_memory=self.entity_memory,
            llm_client=self.llm
        )
        
        return {
            "answer": assistant_message,
            "resolved_query": standalone_query,
            "from_cache": from_cache,
            "turn": self.turn_number
        }
    
    async def _get_compressed_history(self) -> list[dict]:
        """Return history suitable for the context window."""
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        
        max_history_tokens = 3000
        
        total_tokens = sum(len(enc.encode(m["content"])) for m in self.history)
        
        if total_tokens <= max_history_tokens:
            return self.history
        
        # Compress
        return await progressive_summarize_history(
            history=self.history,
            llm_client=self.llm,
            keep_recent_turns=3
        )
    
    def _build_messages(
        self,
        user_message: str,
        context: str,
        compressed_history: list[dict]
    ) -> list[dict]:
        
        system_content = "You are a helpful assistant for company documentation."
        
        if self.entity_memory:
            memory_text = "\n".join([f"- {k}: {v}" for k, v in self.entity_memory.items() if v])
            system_content += f"\n\nConversation context:\n{memory_text}"
        
        messages = [{"role": "system", "content": system_content}]
        messages.extend(compressed_history)
        
        # Current turn with context
        messages.append({
            "role": "user",
            "content": f"[Document context]\n{context}\n\n[Question]\n{user_message}"
        })
        
        return messages
    
    def get_session_summary(self) -> dict:
        """Return a summary of the session for debugging or handoff."""
        return {
            "turn_count": self.turn_number,
            "entity_memory": self.entity_memory,
            "session_state": self.session_state,
            "history_length": len(self.history),
            "cache_size": len(self.cache.cache)
        }
```

---

## Handling Topic Switches

When a user abruptly changes topic mid-conversation, the cached context from the previous topic may be irrelevant or misleading.

```python
async def detect_topic_switch(
    new_message: str,
    recent_history: list[dict],
    embedding_model
) -> bool:
    """
    Detect if the user has switched to a significantly different topic.
    """
    if len(recent_history) < 2:
        return False
    
    # Embed recent conversation and new message
    recent_text = " ".join([m["content"] for m in recent_history[-4:]])
    
    recent_embedding = await embedding_model.embed(recent_text)
    new_embedding = await embedding_model.embed(new_message)
    
    similarity = cosine_similarity(recent_embedding, new_embedding)
    
    return similarity < 0.4  # Low similarity = topic switch


async def handle_topic_switch(
    session: ProductionConversationalRAG,
    new_message: str
) -> None:
    """
    When a topic switch is detected, invalidate the retrieval cache
    so fresh context is retrieved for the new topic.
    """
    is_switch = await detect_topic_switch(
        new_message=new_message,
        recent_history=session.history,
        embedding_model=session.embedder
    )
    
    if is_switch:
        session.cache.invalidate()  # Clear cache for fresh retrieval
```

---

## Common Conversational RAG Mistakes

**Mistake 1: Not resolving references before retrieval.**
"What does it say about that?" → retrieval query is "what does it say about that" → retrieves nothing useful. Always resolve before retrieval.

**Mistake 2: Including too much history.**
Including 30 turns of raw history in every LLM call wastes tokens, increases latency, and hurts generation quality. Compress aggressively.

**Mistake 3: Forgetting the entity memory.**
The user told you their department in turn 1. By turn 15, the raw history may have been compressed away. Without entity memory, you have lost that critical context.

**Mistake 4: Never invalidating the retrieval cache.**
If a document is updated between turns, the cache holds stale content. Always invalidate when relevant documents change.

**Mistake 5: Treating every turn as independent.**
Single-turn retrieval always retrieves top-K by similarity. In conversational context, you may want to prioritize documents discussed earlier in the conversation. Session-aware retrieval — boosting previously cited sources — provides more consistent responses.

---

## Summary

- Conversational RAG has three types of memory: conversation context (history), retrieved document memory (session cache), and user/session state (persistent facts). Each requires different management.
- History compression is necessary for long conversations. Progressive summarization (compress old, keep recent verbatim) plus entity-centric memory provides a good balance.
- Query resolution must happen before retrieval in every turn. Without it, references like "that", "it", "the same policy" retrieve garbage.
- Session retrieval caching avoids redundant retrieval for similar questions within a conversation and improves answer consistency.
- Topic switch detection and cache invalidation keep the system responsive when conversations change direction.
- Entity memory survives compression cycles by extracting structured facts rather than relying on verbatim history.
- Build a session summary method for debugging — you need visibility into what the session "knows" when things go wrong.

---

## What's Next

Part 5 is complete. Part 6 begins with Lesson 6.1 — evaluation philosophy: the difference between offline and online evaluation, component-level versus end-to-end measurement, and how to design an evaluation strategy that actually tells you whether your system is improving.