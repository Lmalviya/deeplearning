# Lesson 8.4 — AWS Deployment: EC2 vs. ECS vs. Lambda, S3 for Documents, Architecture Patterns

---

## AWS Service Choices for RAG Components

A RAG system on AWS uses a combination of compute, storage, and managed services. The key architectural decisions:

- Where does the RAG API server run? (EC2, ECS, Lambda)
- Where does the vector database run? (EC2, ECS, managed Qdrant Cloud)
- Where does the embedding model run? (EC2 with GPU, ECS with GPU, SageMaker)
- Where are documents stored? (S3)
- How does the indexing pipeline work? (SQS + ECS workers, or Lambda)

---

## Compute Options: EC2 vs. ECS vs. Lambda

### EC2: Direct Virtual Machines

Run containers or processes directly on EC2 instances. Most flexible, most operational overhead.

**When to use EC2:**
- GPU workloads (embedding server, LLM serving) — EC2 GPU instances (g5, p4, p3 families) with full NVIDIA driver access.
- Long-running processes with predictable load.
- When you need fine-grained control over instance type, networking, and storage.
- Vector database hosting — Qdrant runs well on high-memory EC2 instances (r5, r6i families).

**GPU instance recommendations for RAG:**
```
Embedding server (bge-large-en-v1.5):
  g5.xlarge — 1× A10G GPU, 24GB VRAM, $1.006/hr
  Sufficient for 200 embeddings/second

Re-ranking (MiniLM L-12):
  g5.xlarge — shared with embedding server feasible
  Or g4dn.xlarge — 1× T4 GPU, $0.526/hr

LLM serving (Llama 3 70B):
  g5.48xlarge — 8× A10G, 192GB VRAM total, $16.29/hr
  p4d.24xlarge — 8× A100, 320GB VRAM total, $32.77/hr

Vector DB (Qdrant, 10M vectors):
  r6i.2xlarge — 64GB RAM, no GPU needed, $0.504/hr
```

### ECS (Elastic Container Service): Managed Container Orchestration

Run Docker containers on managed infrastructure. Less overhead than EC2, more control than Lambda.

**ECS Fargate:** Serverless compute for containers. No instance management. AWS provisions, manages, and scales the underlying infrastructure.

**ECS EC2:** Run ECS tasks on EC2 instances you manage. Necessary for GPU workloads (Fargate does not support GPUs).

```python
# ECS Task Definition for RAG API (Fargate)
{
    "family": "rag-api",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "2048",      # 2 vCPU
    "memory": "4096",   # 4GB RAM
    "containerDefinitions": [
        {
            "name": "rag-api",
            "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/rag-api:latest",
            "portMappings": [{"containerPort": 8000}],
            "environment": [
                {"name": "QDRANT_URL", "value": "http://qdrant.internal:6333"},
                {"name": "LOG_LEVEL", "value": "INFO"}
            ],
            "secrets": [
                {
                    "name": "OPENAI_API_KEY",
                    "valueFrom": "arn:aws:secretsmanager:us-east-1:account:secret:rag/openai-key"
                }
            ],
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                "interval": 30,
                "timeout": 10,
                "retries": 3
            },
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/rag-api",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

**When to use ECS Fargate:**
- Stateless API servers.
- Indexing workers (variable load, scale to zero when idle).
- Services where you want automatic scaling without managing EC2 instances.

**When to use ECS EC2 (not Fargate):**
- GPU workloads (embedding server, LLM serving).
- When you need specific instance types Fargate does not support.
- Cost optimization at high sustained load (EC2 instances cheaper than Fargate at high utilization).

### Lambda: Serverless Functions

Run code in response to events with no server management. Scale to zero when idle.

**When Lambda makes sense for RAG:**
- Document upload triggers (S3 event → Lambda → SQS queue for indexing).
- Lightweight pre/post-processing steps.
- Webhook handlers (GitHub push → Lambda → trigger indexing).
- Low-traffic RAG endpoints where cold start latency is acceptable (> 1s to first response).

**When Lambda does NOT make sense for RAG:**
- High-traffic query endpoints (cold starts, concurrency limits).
- GPU workloads (Lambda has no GPU support).
- Long-running processes (15-minute max execution time).
- Large container images (the embedding model alone is several GB, and Lambda has container image size limits that interact poorly with this).

**Lambda for document upload triggering:**

```python
# lambda/document_upload_trigger.py
import boto3
import json

sqs = boto3.client('sqs')
INDEXING_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/account/rag-indexing"

def handler(event, context):
    """
    Triggered by S3 PutObject events.
    Queues the new document for indexing.
    """
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        size = record['s3']['object']['size']
        
        # Skip non-document files
        if not key.lower().endswith(('.pdf', '.docx', '.html', '.md', '.txt')):
            return
        
        # Queue for indexing
        message = {
            "source_path": f"s3://{bucket}/{key}",
            "operation": "upsert",
            "document_metadata": {
                "source": "s3_upload",
                "file_size": size
            }
        }
        
        sqs.send_message(
            QueueUrl=INDEXING_QUEUE_URL,
            MessageBody=json.dumps(message),
            MessageAttributes={
                "Priority": {
                    "DataType": "String",
                    "StringValue": "normal"
                }
            }
        )
        
        print(f"Queued for indexing: s3://{bucket}/{key}")
```

---

## S3 for Document Storage

S3 is the standard document store for RAG systems on AWS. Documents uploaded to S3, processed by the indexing pipeline, with chunks stored in Qdrant.

### S3 Bucket Structure

```
rag-documents-{environment}/
├── raw/                    # Original uploaded documents
│   ├── hr/
│   │   └── policies/
│   │       └── 2024/
│   │           └── employee_handbook_v3.pdf
│   └── legal/
│       └── contracts/
├── processed/              # Pre-processed/converted versions
│   └── hr/policies/2024/
│       └── employee_handbook_v3.txt
├── extracted/              # Extracted figures, tables
│   └── hr/policies/2024/
│       ├── employee_handbook_v3_table_1.json
│       └── employee_handbook_v3_figure_1.png
└── audit/                  # Immutable audit logs
    └── 2024/06/
        └── queries.jsonl
```

### S3 Bucket Configuration

```python
import boto3

s3 = boto3.client('s3')

# Create bucket with appropriate settings
def setup_document_bucket(bucket_name: str, region: str):
    
    # Create bucket
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": region}
    )
    
    # Versioning (important for tracking document changes)
    s3.put_bucket_versioning(
        Bucket=bucket_name,
        VersioningConfiguration={"Status": "Enabled"}
    )
    
    # Server-side encryption (SSE-S3 by default, SSE-KMS for sensitive data)
    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            "Rules": [{
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "aws:kms",
                    "KMSMasterKeyID": "your-kms-key-id"
                }
            }]
        }
    )
    
    # Lifecycle policy: move older documents to cheaper storage tiers
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket_name,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "archive-old-raw-docs",
                    "Status": "Enabled",
                    "Prefix": "raw/",
                    "Transitions": [
                        {
                            "Days": 90,
                            "StorageClass": "STANDARD_IA"  # Infrequent access after 90 days
                        },
                        {
                            "Days": 365,
                            "StorageClass": "GLACIER_IR"  # Archive after 1 year
                        }
                    ]
                }
            ]
        }
    )
    
    # Block all public access (documents should never be public)
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True
        }
    )
    
    # S3 event notification → SQS for automatic indexing trigger
    s3.put_bucket_notification_configuration(
        Bucket=bucket_name,
        NotificationConfiguration={
            "QueueConfigurations": [
                {
                    "QueueArn": "arn:aws:sqs:us-east-1:account:rag-indexing",
                    "Events": ["s3:ObjectCreated:*"],
                    "Filter": {
                        "Key": {
                            "FilterRules": [{"Name": "prefix", "Value": "raw/"}]
                        }
                    }
                }
            ]
        }
    )
```

---

## Complete AWS Architecture Pattern

```
                         [Users]
                            │
                   [Application Load Balancer]
                            │
                    ┌───────┴───────┐
                    │               │
              [ECS Fargate]   [CloudFront]
              [RAG API]       [Static assets]
             (auto-scaled)
                    │
          ┌─────────┼──────────┐
          │         │          │
    [EC2 GPU]  [Qdrant on   [ElastiCache
    [Embed +    EC2 r6i]     Redis]
     Rerank]
          │
    [ECS Fargate]  [SQS Queue]  [S3 Bucket]
    [Indexing        │              │
     Workers]    [Lambda]       [Documents]
                  [Trigger]
          │
    [RDS PostgreSQL]
    [Metadata/Registry]
          │
    [CloudWatch]
    [Monitoring]
```

### Key AWS Networking Configuration

Everything lives inside a VPC with private subnets. Only the ALB is public.

```python
# Terraform-style configuration (pseudocode)

vpc = {
    "cidr": "10.0.0.0/16",
    "subnets": {
        "public": ["10.0.1.0/24", "10.0.2.0/24"],     # ALB only
        "private": ["10.0.10.0/24", "10.0.11.0/24"],  # Application services
        "data": ["10.0.20.0/24", "10.0.21.0/24"]      # Databases
    }
}

security_groups = {
    "alb": {
        "ingress": [{"port": 443, "source": "0.0.0.0/0"}]
    },
    "rag_api": {
        "ingress": [{"port": 8000, "source": "sg-alb"}]
    },
    "qdrant": {
        "ingress": [
            {"port": 6333, "source": "sg-rag-api"},
            {"port": 6333, "source": "sg-indexing-worker"},
            {"port": 6334, "source": "sg-rag-api"}
        ]
    },
    "embedding_server": {
        "ingress": [
            {"port": 8001, "source": "sg-rag-api"},
            {"port": 8001, "source": "sg-indexing-worker"}
        ]
    },
    "redis": {
        "ingress": [{"port": 6379, "source": "sg-rag-api"}]
    },
    "rds": {
        "ingress": [
            {"port": 5432, "source": "sg-rag-api"},
            {"port": 5432, "source": "sg-indexing-worker"}
        ]
    }
}
```

### IAM Roles

Each service gets the minimum required permissions:

```json
// ECS Task Role for RAG API
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": "arn:aws:s3:::rag-documents-prod/*"
        },
        {
            "Effect": "Allow",
            "Action": ["secretsmanager:GetSecretValue"],
            "Resource": "arn:aws:secretsmanager:us-east-1:account:secret:rag/*"
        },
        {
            "Effect": "Allow",
            "Action": ["cloudwatch:PutMetricData"],
            "Resource": "*"
        }
    ]
}

// ECS Task Role for Indexing Worker (more permissions)
{
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": ["arn:aws:s3:::rag-documents-prod/*", "arn:aws:s3:::rag-documents-prod"]
        },
        {
            "Effect": "Allow",
            "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"],
            "Resource": "arn:aws:sqs:us-east-1:account:rag-indexing"
        },
        {
            "Effect": "Allow",
            "Action": ["textract:AnalyzeDocument"],
            "Resource": "*"
        }
    ]
}
```

---

## Auto Scaling Configuration

```python
# ECS Auto Scaling for RAG API

autoscaling = {
    "min_capacity": 2,    # Always at least 2 for high availability
    "max_capacity": 20,   # Scale up to 20 instances at peak
    
    "scale_out_policy": {
        "metric": "ECSServiceAverageCPUUtilization",
        "target": 70,      # Scale out when CPU > 70%
        "scale_out_cooldown": 60,  # seconds
    },
    
    "scale_in_policy": {
        "metric": "ECSServiceAverageCPUUtilization",
        "target": 70,
        "scale_in_cooldown": 300,  # 5 min before scaling in (avoid flapping)
    },
    
    # ALB-based scaling: scale based on request count per target
    "request_count_policy": {
        "metric": "RequestCountPerTarget",
        "target": 1000,    # 1000 requests/minute per instance
    }
}
```

---

## Cost Optimization Tips

**Reserved Instances / Savings Plans for stable load:**
The RAG API and Qdrant nodes run 24/7. EC2 Reserved Instances or Compute Savings Plans reduce costs by 40-60% vs. On-Demand.

**Spot Instances for indexing workers:**
Indexing workers can tolerate interruption — if the spot instance is reclaimed, the SQS message returns to the queue and another worker picks it up. Spot Instances cost 70-90% less than On-Demand.

```python
# ECS capacity provider with Spot instances for indexing workers
capacity_provider = {
    "type": "FARGATE_SPOT",  # Spot pricing for Fargate
    "weight": 4,              # 80% Spot
    "base": 0
}
fallback_provider = {
    "type": "FARGATE",        # On-Demand fallback
    "weight": 1,              # 20% On-Demand (for stability)
    "base": 1                 # Always keep at least 1 On-Demand
}
```

**S3 Intelligent-Tiering:**
Enable S3 Intelligent-Tiering for document storage. AWS automatically moves infrequently accessed documents to cheaper storage tiers, reducing storage costs by 20-40% without manual lifecycle rule management.

---

## Summary

- EC2 GPU instances for embedding server and LLM serving — Fargate does not support GPUs.
- ECS Fargate for stateless API servers and indexing workers — no instance management, easy auto-scaling.
- Lambda for event triggers (S3 upload → SQS queue) — not for the main query path.
- S3 for all document storage with versioning, encryption, and lifecycle policies.
- VPC with private subnets for all application services — only ALB is public-facing.
- IAM roles with minimum required permissions per service — not a single shared role.
- Spot Instances for indexing workers; Reserved Instances for always-on services.

---

## What's Next

Lesson 8.5 covers Kubernetes for RAG — pods, Horizontal Pod Autoscalers, resource limits, and rolling deployments for production-grade container orchestration beyond Docker Compose.