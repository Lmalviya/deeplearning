# Lesson 8.7 — Serving LLMs: Self-Hosted (vLLM, Ollama) vs. API, Latency vs. Cost

---

## The LLM Serving Decision

In a RAG pipeline, the LLM is the most expensive component — both in compute cost and in latency. The decision between using a managed API (OpenAI, Anthropic, Cohere) and self-hosting an open-source model (Llama, Mistral, Phi) is one of the highest-impact architectural decisions you will make.

This lesson covers the serving options in technical depth, how to evaluate model quality for your use case, and how to make the right choice for different scenarios.

---

## Managed API Options

### OpenAI

The most capable and widely used. GPT-4o is the current flagship for RAG generation.

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="sk-...",
    max_retries=3,
    timeout=30.0
)

async def generate(query: str, context: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=800,
        temperature=0.1
    )
    return response.choices[0].message.content
```

**Pricing (approximate, 2024):**
- GPT-4o: $5/M input tokens, $15/M output tokens
- GPT-4o-mini: $0.15/M input tokens, $0.60/M output tokens
- GPT-4o Batch API: 50% discount on above prices

**Rate limits:** Default limits can be insufficient at high QPS. Request limit increases from OpenAI for production workloads. Use the Batch API for non-real-time indexing tasks (summarization, classification).

**Model pinning:** OpenAI updates models. Pin to a specific version when behavioral consistency is critical:

```python
model="gpt-4o-2024-05-13"  # Pinned version, not "gpt-4o" which gets updated
```

### Anthropic Claude

Strong on instruction following and long contexts. Claude 3.5 Sonnet is the primary RAG model.

```python
import anthropic

client = anthropic.AsyncAnthropic(api_key="sk-ant-...")

async def generate(query: str, context: str) -> str:
    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return message.content[0].text
```

Claude 3.5 Sonnet is competitive with GPT-4o on most RAG tasks, with stronger performance on long-context tasks (200K token window). Anthropic has EU data residency options for GDPR compliance.

### Cohere

Cohere's Command R+ is purpose-built for RAG. It has native grounding, built-in citation, and a retrieval-augmented generation mode.

```python
import cohere

co = cohere.Client(api_key="...")

def generate_with_cohere_rag(query: str, documents: list[dict]) -> dict:
    """
    Cohere's native RAG mode — handles grounding and citations automatically.
    """
    response = co.chat(
        message=query,
        model="command-r-plus",
        documents=[
            {
                "id": doc["chunk_id"],
                "title": doc["metadata"]["doc_title"],
                "snippet": doc["text"]
            }
            for doc in documents
        ]
    )
    
    return {
        "answer": response.text,
        "citations": response.citations,   # Automatic source attribution
        "grounded": True
    }
```

Cohere's `documents` parameter handles the context injection, grounding, and citation generation natively — less prompt engineering required.

---

## Self-Hosted: vLLM

vLLM is the production-grade LLM inference server. It provides the same OpenAI-compatible API but runs locally.

### Why vLLM Over Alternatives

**PagedAttention:** vLLM's key innovation. Traditional LLM inference pre-allocates a fixed KV cache per sequence. PagedAttention manages the KV cache in fixed-size pages (like OS virtual memory), enabling much more efficient memory usage. Result: 2-4× higher throughput vs. naive serving at the same hardware cost.

**Continuous batching:** vLLM does not wait for a batch to be complete before starting processing. It processes tokens continuously, dynamically adding and removing sequences from batches. This dramatically reduces latency for concurrent requests compared to static batching.

**Supported models:** Llama 3 (8B, 70B), Mistral, Phi-3, Gemma, Falcon, Yi, and most Hugging Face transformers.

### vLLM Deployment

```python
# Launch vLLM server (command line)
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Meta-Llama-3-70B-Instruct \
#   --tensor-parallel-size 4 \     # Across 4 GPUs
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.90 \
#   --max-num-seqs 256 \            # Max concurrent sequences
#   --port 8080

# Client code — identical to OpenAI client (drop-in replacement)
from openai import AsyncOpenAI

vllm_client = AsyncOpenAI(
    base_url="http://vllm-server:8080/v1",
    api_key="not-needed"  # vLLM doesn't require API key by default
)

async def generate_with_vllm(query: str, context: str) -> str:
    response = await vllm_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=800,
        temperature=0.1
    )
    return response.choices[0].message.content
```

### vLLM GPU Requirements

```
Model               Parameters  GPU Memory  Hardware
──────────────────────────────────────────────────
Llama 3 8B          8B          ~16GB       1× A10G or 2× T4
Llama 3 70B         70B         ~140GB      4× A100 or 8× A10G
Mistral 7B          7B          ~14GB       1× A10G
Phi-3 Mini          3.8B        ~8GB        1× A10G (efficient)
Phi-3 Medium        14B         ~28GB       2× A10G
Mixtral 8×7B        47B*        ~90GB       4× A10G (*MoE architecture)
```

*Mixtral uses Mixture of Experts — 47B parameters total but only ~13B active per token, making it faster than its parameter count suggests.

### Quantization: Reducing Memory Requirements

Quantization reduces model precision to use less GPU memory, at the cost of small accuracy degradation.

```python
# 4-bit quantization using bitsandbytes (via vLLM)
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Meta-Llama-3-70B-Instruct \
#   --quantization awq \          # AWQ 4-bit quantization
#   --tensor-parallel-size 2 \   # Now fits on 2× A100 instead of 4
#   --gpu-memory-utilization 0.90

# Quantization options:
# awq:  4-bit, ~2× memory reduction, <1% quality loss for RAG tasks
# gptq: 4-bit, similar to AWQ
# fp8:  8-bit, <0.5× memory reduction, negligible quality loss
```

With AWQ 4-bit quantization, Llama 3 70B fits on 2× A100 80GB instead of 4× — halving the hardware cost with minimal quality impact for RAG applications.

---

## Self-Hosted: Ollama

Ollama is designed for simplicity rather than maximum performance. It runs models locally with one command, ideal for development and low-volume production.

```bash
# Install and run
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3:70b
ollama serve

# Ollama exposes an OpenAI-compatible API
```

```python
# Using Ollama with OpenAI client
from openai import AsyncOpenAI

ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = await ollama_client.chat.completions.create(
    model="llama3:70b",
    messages=[...]
)
```

**Ollama vs. vLLM:**

| | Ollama | vLLM |
|---|---|---|
| **Setup complexity** | Very low (one command) | Medium (GPU config, model download) |
| **Throughput** | Low-moderate | High |
| **Production-ready** | Small scale | Yes |
| **Quantization** | Built-in (auto) | Explicit configuration |
| **Best for** | Development, demos, internal tools | Production RAG serving |

---

## Streaming Responses

For user-facing RAG, streaming is critical for perceived responsiveness. Users see the first tokens within 500-800ms instead of waiting 2-3 seconds for the full response.

```python
async def stream_rag_response(
    query: str,
    context: str,
    llm_client,
    websocket  # WebSocket connection to the user
):
    """
    Stream tokens to the user as they are generated.
    """
    
    stream = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=800,
        temperature=0.1,
        stream=True  # Enable streaming
    )
    
    full_response = []
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response.append(token)
            
            # Send token to user immediately
            await websocket.send_json({
                "type": "token",
                "content": token
            })
    
    # Send completion signal with full response for logging
    complete_response = "".join(full_response)
    await websocket.send_json({
        "type": "complete",
        "content": complete_response
    })
    
    # Log the complete response for tracing
    await log_trace(query, context, complete_response)
```

**Streaming with FastAPI and Server-Sent Events:**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    
    async def generate_stream():
        # Retrieval (non-streaming)
        chunks = await retrieve(request.query)
        context = format_context(chunks)
        
        # Generation (streaming)
        stream = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[...],
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
        
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )
```

---

## Latency Benchmarks by Serving Option

These are approximate figures for a typical RAG query (2,000 token context, 300 token output):

```
Time to first token (TTFT):
  OpenAI GPT-4o:              300-700ms  (network + model)
  Anthropic Claude 3.5 Sonnet: 400-800ms
  vLLM Llama 3 70B (4× A100):  80-150ms  (no network hop)
  vLLM Llama 3 8B (1× A10G):   30-60ms

Time to completion (TTC):
  OpenAI GPT-4o:              1.5-3.0s
  Anthropic Claude 3.5 Sonnet: 2.0-4.0s
  vLLM Llama 3 70B (4× A100):  1.0-2.0s
  vLLM Llama 3 8B (1× A10G):   0.5-1.2s

Throughput (tokens/second per GPU):
  vLLM Llama 3 8B (1× A10G):   ~3,000 tokens/sec
  vLLM Llama 3 70B (4× A100):  ~1,500 tokens/sec
```

---

## Model Quality for RAG: What Matters

For RAG-specific tasks, models are evaluated on:

1. **Instruction following:** Does the model stay grounded in the context, respect the "only use provided context" instruction?
2. **Citation accuracy:** Does it correctly attribute claims to sources?
3. **IDK calibration:** Does it say IDK when the context is insufficient?
4. **Factual faithfulness:** Does it accurately represent what the context says?

On these dimensions, Llama 3 70B is competitive with GPT-4o for most domain-specific RAG applications. The gap is most noticeable in:
- Complex multi-hop reasoning across many documents.
- Edge cases in instruction following (subtle contradictions in context).
- Code generation quality.

For support chatbots, internal knowledge bases, and document Q&A: Llama 3 70B is often good enough. For legal, medical, or financial precision: GPT-4o or Claude 3.5 Sonnet.

---

## Making the Decision

```
Is data residency a hard requirement?
  YES → Self-hosted (vLLM). No choice.
  NO → continue

What is daily query volume?
  < 50K/day → OpenAI/Anthropic API (cheaper including ops time)
  > 100K/day → Self-hosted (cheaper at scale)

What is latency requirement?
  TTFT < 500ms → Self-hosted (no network hop)
  No hard requirement → Either

Is GPT-4 class quality required?
  YES (complex reasoning, nuanced compliance) → OpenAI/Anthropic
  NO (domain Q&A, support) → Llama 3 70B likely sufficient

Do you have GPU infrastructure?
  NO → Managed API or managed GPU hosting (Lambda Labs, RunPod)
  YES → vLLM

Recommendation summary:
  - Startup/MVP: OpenAI API (gpt-4o-mini for cost, gpt-4o for quality)
  - Growing: OpenAI API until 50K+ queries/day, then evaluate self-hosted
  - Enterprise with data requirements: vLLM + Llama 3 70B AWQ
  - Hybrid: OpenAI for interactive, vLLM Batch for offline/indexing tasks
```

---

## Summary

- Three managed API options: OpenAI (GPT-4o flagship, gpt-4o-mini for cost), Anthropic (Claude 3.5 Sonnet, strong long-context), Cohere (Command R+ with native RAG grounding).
- vLLM is the production standard for self-hosted LLM serving. PagedAttention and continuous batching give 2-4× throughput improvement over naive serving.
- AWQ 4-bit quantization halves GPU memory requirements with < 1% quality loss for RAG tasks. Llama 3 70B goes from 4× A100 to 2× A100.
- Ollama is for development and small-scale production. vLLM is for production-grade serving.
- Streaming responses (SSE or WebSocket) are critical for user-perceived latency — first token at 500ms feels faster than full response at 2 seconds.
- Llama 3 70B is competitive with GPT-4o for domain-specific RAG Q&A. The gap is meaningful for complex multi-hop reasoning and precision-critical domains.
- Decision factors: data residency (hard requirement → self-hosted), volume (> 100K/day → self-hosted cheaper), latency (TTFT < 500ms → self-hosted), quality requirements.

---

## What's Next

Lesson 8.8 covers CI/CD for RAG systems — index versioning, model versioning, regression testing pipelines, and how to deploy RAG changes safely.