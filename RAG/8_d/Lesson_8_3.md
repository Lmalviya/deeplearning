# Lesson 8.3 — Containerizing a RAG System with Docker: Service Decomposition and Compose Setup

---

## Why Containers for RAG

A production RAG system has multiple components that need to run together, be independently deployable, and be reproducible across development, staging, and production environments. Without containers, dependency management becomes a nightmare — the embedding server needs Python 3.11 with specific CUDA libraries, the API server needs a different set of dependencies, and the indexing worker needs yet another set.

Docker solves this by packaging each service with all its dependencies into a portable, isolated unit. Docker Compose orchestrates multiple containers as a single application.

This lesson walks through decomposing a RAG system into services, writing Dockerfiles for each, and wiring them together with Docker Compose.

---

## Service Decomposition

A production RAG system naturally decomposes into these services:

```
rag-system/
├── services/
│   ├── api/              # FastAPI query endpoint (user-facing)
│   ├── indexing-worker/  # Async indexing pipeline
│   ├── embedding-server/ # GPU embedding serving
│   ├── reranker/         # Cross-encoder re-ranking
│   └── scheduler/        # Cron jobs (reconciliation, health checks)
├── infrastructure/
│   ├── qdrant/           # Vector database config
│   ├── redis/            # Cache config
│   └── postgres/         # Metadata/registry DB init scripts
└── docker-compose.yml
```

Each service is independently scalable: you can run 10 API containers but only 1 embedding server (which batches internally). You can run 5 indexing workers without touching the API layer.

---

## Dockerfile: API Service

```dockerfile
# services/api/Dockerfile
FROM python:3.11-slim

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies (separate layer for better caching)
# requirements.txt changes less frequently than source code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Non-root user for security
RUN useradd -m -u 1000 raguser
USER raguser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop"]
```

```
# services/api/requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.30.0
qdrant-client==1.9.0
openai==1.35.0
redis==5.0.0
pydantic==2.7.0
sentence-transformers==3.0.0
tiktoken==0.7.0
```

**Multi-stage build for production (reduces image size):**

```dockerfile
# services/api/Dockerfile.prod
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image (minimal)
FROM python:3.11-slim AS runtime

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

RUN useradd -m -u 1000 raguser
USER raguser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Multi-stage builds reduce the final image from ~1.5GB to ~400MB by excluding build tools.

---

## Dockerfile: Embedding Server (GPU)

The embedding server requires CUDA. This is the most complex Dockerfile.

```dockerfile
# services/embedding-server/Dockerfile
# CUDA base image — must match your GPU driver version
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# PyTorch with CUDA support (large download — layer first for caching)
RUN pip install --no-cache-dir \
    torch==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download embedding model at build time (bakes model into image)
# Alternative: mount model from external volume at runtime
ARG MODEL_NAME=BAAI/bge-large-en-v1.5
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

COPY . .

RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

EXPOSE 8001

# GPU health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "embedding_server:app", "--host", "0.0.0.0", "--port", "8001"]
```

**Trade-off:** Baking the model into the image (as above) creates a large image (~3-6GB) but eliminates model download time at container startup. For teams with good container registries, this is preferred. Alternative: mount the model from a shared volume.

---

## Dockerfile: Indexing Worker

```dockerfile
# services/indexing-worker/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \    # For PDF processing (pdfplumber)
    tesseract-ocr \    # For OCR fallback
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 raguser
USER raguser

# No port exposed — this is a background worker, not a server
# Reads from SQS/Redis queue and processes documents

CMD ["python", "-m", "celery", "-A", "worker", "worker", \
     "--loglevel=info", "--concurrency=4", "--queues=indexing,urgent"]
```

---

## Docker Compose: Development Setup

```yaml
# docker-compose.yml (development)
version: "3.9"

services:
  # ─── Infrastructure ───────────────────────────────────────
  
  qdrant:
    image: qdrant/qdrant:v1.9.0
    ports:
      - "6333:6333"   # HTTP API
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      retries: 5
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: rag_metadata
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U raguser -d rag_metadata"]
      interval: 10s
      retries: 5
    restart: unless-stopped

  # ─── Application Services ─────────────────────────────────

  embedding-server:
    build:
      context: ./services/embedding-server
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      MODEL_NAME: BAAI/bge-large-en-v1.5
      BATCH_SIZE: 64
      MAX_WAIT_MS: 10
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - qdrant
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      start_period: 60s  # Model loading takes time
    restart: unless-stopped

  reranker:
    build:
      context: ./services/reranker
      dockerfile: Dockerfile
    ports:
      - "8002:8002"
    environment:
      MODEL_NAME: cross-encoder/ms-marco-MiniLM-L-6-v2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      start_period: 45s
    restart: unless-stopped

  api:
    build:
      context: ./services/api
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      QDRANT_URL: http://qdrant:6333
      REDIS_URL: redis://redis:6379/0
      POSTGRES_URL: postgresql://raguser:${POSTGRES_PASSWORD}@postgres:5432/rag_metadata
      EMBEDDING_SERVER_URL: http://embedding-server:8001
      RERANKER_URL: http://reranker:8002
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      LOG_LEVEL: INFO
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
      embedding-server:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
    restart: unless-stopped

  indexing-worker:
    build:
      context: ./services/indexing-worker
      dockerfile: Dockerfile
    environment:
      QDRANT_URL: http://qdrant:6333
      REDIS_URL: redis://redis:6379/0
      POSTGRES_URL: postgresql://raguser:${POSTGRES_PASSWORD}@postgres:5432/rag_metadata
      EMBEDDING_SERVER_URL: http://embedding-server:8001
      S3_BUCKET: ${S3_BUCKET}
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
    depends_on:
      embedding-server:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    # Scale workers independently
    deploy:
      replicas: 3
    restart: unless-stopped

volumes:
  qdrant_storage:
  redis_data:
  postgres_data:
```

---

## Environment Variables and Secrets

Never bake secrets into images or Docker Compose files. Use `.env` files for development, and secret management services for production.

```bash
# .env (development only — never commit this)
POSTGRES_PASSWORD=dev_password_only
OPENAI_API_KEY=sk-...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET=rag-documents-dev
```

```bash
# .env.example (commit this — no secrets)
POSTGRES_PASSWORD=change_me
OPENAI_API_KEY=sk-your-key-here
AWS_ACCESS_KEY_ID=your-key-id
AWS_SECRET_ACCESS_KEY=your-secret
S3_BUCKET=your-bucket-name
```

For production, use AWS Secrets Manager, HashiCorp Vault, or Kubernetes Secrets rather than environment variables.

---

## Health Checks and Dependencies

Every service should expose a `/health` endpoint that checks its own health and the health of its critical dependencies:

```python
# services/api/health.py
from fastapi import APIRouter
from qdrant_client import QdrantClient
import redis.asyncio as aioredis

router = APIRouter()

@router.get("/health")
async def health_check():
    checks = {}
    
    # Check Qdrant connectivity
    try:
        qdrant = QdrantClient(url=QDRANT_URL)
        collections = qdrant.get_collections()
        checks["qdrant"] = "healthy"
    except Exception as e:
        checks["qdrant"] = f"unhealthy: {str(e)[:100]}"
    
    # Check Redis connectivity
    try:
        r = aioredis.from_url(REDIS_URL)
        await r.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)[:100]}"
    
    # Check embedding server
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{EMBEDDING_SERVER_URL}/health", timeout=2.0)
            checks["embedding_server"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except Exception as e:
        checks["embedding_server"] = f"unhealthy: {str(e)[:100]}"
    
    all_healthy = all(v == "healthy" for v in checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks
    }
```

The `depends_on` with `condition: service_healthy` in Docker Compose ensures that the API container does not start until all its dependencies pass their health checks.

---

## Useful Docker Commands

```bash
# Build all services
docker compose build

# Start all services (detached)
docker compose up -d

# View logs from a specific service
docker compose logs -f api

# View logs from all services
docker compose logs -f

# Scale the indexing worker
docker compose up -d --scale indexing-worker=5

# Restart a single service (after code change)
docker compose restart api

# Shell into a running container
docker compose exec api bash

# Run a one-off command
docker compose run --rm api python manage.py migrate

# Stop and remove all containers (keep volumes)
docker compose down

# Stop and remove everything including volumes (WARNING: data loss)
docker compose down -v

# Check resource usage
docker stats
```

---

## Common Pitfalls

**Pitfall 1: Connecting containers by localhost.**
Inside a container, `localhost` refers to the container itself, not the host or other containers. Use service names defined in docker-compose.yml: `http://qdrant:6333`, not `http://localhost:6333`.

**Pitfall 2: Missing health checks and `depends_on` conditions.**
Without `condition: service_healthy`, a service can start before its dependencies are ready, causing startup failures. Always define health checks and `depends_on` with `condition`.

**Pitfall 3: Storing secrets in images.**
Any secret in a Dockerfile layer is retrievable from the image. Use environment variables passed at runtime, not baked into the image.

**Pitfall 4: Running as root inside containers.**
By default, containers run as root. Create a non-root user and switch to it. This limits the blast radius if a container is compromised.

**Pitfall 5: Not limiting resources.**
Without CPU/memory limits, one misbehaving container can starve others. Set `--cpus` and `--memory` limits in production.

```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '0.5'
      memory: 1G
```

---

## Summary

- Decompose the RAG system into independently deployable services: API, embedding server, re-ranker, indexing workers, and infrastructure (Qdrant, Redis, PostgreSQL).
- Each service gets its own Dockerfile. Use multi-stage builds to minimize image size.
- GPU services (embedding, re-ranking) require CUDA base images and GPU reservation in Docker Compose.
- Use `depends_on` with `condition: service_healthy` and `/health` endpoints to enforce startup order.
- Never bake secrets into images. Use `.env` files for development, secret managers for production.
- Service names in docker-compose.yml are the hostnames — use `http://qdrant:6333`, not `localhost`.

---

## What's Next

Lesson 8.4 covers AWS deployment — EC2 vs. ECS vs. Lambda for RAG components, S3 for document storage, and production-ready architecture patterns on AWS.