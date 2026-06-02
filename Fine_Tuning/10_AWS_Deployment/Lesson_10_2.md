# Lesson 10.2 — Deploying a Fine-Tuned Model to a SageMaker Endpoint

> **The interview question this answers:** "Walk me through how you would deploy your fine-tuned model on AWS SageMaker."

---

## Three Deployment Paths

When deploying an LLM on SageMaker, you choose between three approaches depending on your setup:

| Approach | When to use | Complexity |
|---|---|---|
| **HuggingFace TGI DLC** | Model is on HF Hub or S3 in HF format, standard serving | Low |
| **Custom container with vLLM** | Need vLLM's PagedAttention, maximum throughput | Medium |
| **Bring-your-own model + pre-built PyTorch DLC** | Custom inference logic in Python, standard model | Low |

All three follow the same structural pattern: upload model to S3 → define SageMaker Model → configure endpoint → deploy.

---

## Path 1: HuggingFace TGI DLC (Recommended for Most Cases)

SageMaker provides a native HuggingFace DLC (Deep Learning Container) running TGI. This is the path of least resistance for any HuggingFace-format model.

### Step 1: Save Model Artifacts to S3

```python
import boto3
import tarfile
import os

def upload_model_to_s3(local_model_dir: str, bucket: str, prefix: str) -> str:
    """
    Package the model directory as a tar.gz and upload to S3.
    SageMaker expects model artifacts as a tar.gz.
    """
    
    # Create tar.gz of the model directory
    tar_path = "/tmp/model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_model_dir, arcname=".")
    
    # Upload to S3
    s3_client = boto3.client("s3")
    s3_key = f"{prefix}/model.tar.gz"
    s3_client.upload_file(tar_path, bucket, s3_key)
    
    s3_uri = f"s3://{bucket}/{s3_key}"
    print(f"Model uploaded to: {s3_uri}")
    return s3_uri

# Usage
s3_model_uri = upload_model_to_s3(
    local_model_dir="./llama3-8b-final",
    bucket="my-ml-bucket",
    prefix="models/llama3-8b-v1"
)
```

Alternatively, if you push your model to HuggingFace Hub, you can specify the Hub model ID directly (no S3 upload needed for TGI).

### Step 2: Deploy via HuggingFace SageMaker SDK

```python
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

# SageMaker session and role
sess = sagemaker.Session()
role = sagemaker.get_execution_role()  # IAM role SageMaker uses

# TGI DLC image URI (use latest version)
# Find current versions at: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
tgi_image_uri = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu22.04"
)

# Option A: Model from S3
model = HuggingFaceModel(
    model_data=s3_model_uri,          # S3 URI of tar.gz artifact
    role=role,
    image_uri=tgi_image_uri,
    env={
        # TGI server configuration
        "HF_MODEL_ID": "/opt/ml/model",  # Path inside container (from tar.gz)
        "MAX_INPUT_LENGTH": "4096",
        "MAX_TOTAL_TOKENS": "8192",
        "MAX_BATCH_PREFILL_TOKENS": "8192",
        "QUANTIZE": "awq",               # Enable AWQ quantization in TGI
        "HF_MODEL_QUANTIZE": "awq",
        "NUM_SHARD": "1",               # Number of GPUs (tensor parallelism)
        "SM_NUM_GPUS": "1",
    }
)

# Option B: Model directly from HuggingFace Hub
model = HuggingFaceModel(
    role=role,
    image_uri=tgi_image_uri,
    env={
        "HF_MODEL_ID": "meta-llama/Meta-Llama-3-8B-Instruct",
        "HF_TOKEN": "<your-hf-token>",  # Store in Secrets Manager in production
        "MAX_INPUT_LENGTH": "4096",
        "MAX_TOTAL_TOKENS": "8192",
        "NUM_SHARD": "1",
    }
)

# Deploy to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",      # 1× A10G 24GB — fits 8B AWQ INT4
    endpoint_name="llama3-8b-endpoint",
    # Optional: container startup timeout (TGI takes time to load)
    container_startup_health_check_timeout=600,
)

print(f"Endpoint deployed: {predictor.endpoint_name}")
```

### Step 3: Invoke the Endpoint

```python
import json

# Direct boto3 invocation
sagemaker_runtime = boto3.client("sagemaker-runtime")

payload = {
    "inputs": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "return_full_text": False,
    }
}

response = sagemaker_runtime.invoke_endpoint(
    EndpointName="llama3-8b-endpoint",
    ContentType="application/json",
    Body=json.dumps(payload)
)

result = json.loads(response["Body"].read().decode())
print(result[0]["generated_text"])
```

---

## Path 2: Custom vLLM Container

For maximum throughput with PagedAttention and continuous batching, deploy vLLM in a custom container.

### Step 1: Build the vLLM Inference Container

```dockerfile
# Dockerfile for vLLM SageMaker inference container
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip

# Install vLLM and SageMaker inference toolkit
RUN pip install vllm==0.4.2 sagemaker-inference boto3

# SageMaker expects inference code at /opt/ml/code/
COPY inference.py /opt/ml/code/inference.py
WORKDIR /opt/ml/code

# SageMaker inference server entry point
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM inference.py

EXPOSE 8080
```

```python
# inference.py — the SageMaker inference handler
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import json
import os

# SageMaker-required functions
def model_fn(model_dir):
    """Load model when container starts."""
    engine_args = AsyncEngineArgs(
        model=model_dir,
        tensor_parallel_size=int(os.environ.get("NUM_GPUS", "1")),
        quantization=os.environ.get("QUANTIZATION", None),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.90")),
    )
    
    llm = LLM(**vars(engine_args))
    return llm

def input_fn(request_body, content_type="application/json"):
    """Parse incoming request."""
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Run inference."""
    prompts = input_data.get("inputs", [])
    if isinstance(prompts, str):
        prompts = [prompts]
    
    params = input_data.get("parameters", {})
    sampling_params = SamplingParams(
        max_tokens=params.get("max_new_tokens", 512),
        temperature=params.get("temperature", 0.7),
        top_p=params.get("top_p", 0.9),
    )
    
    outputs = model.generate(prompts, sampling_params)
    
    return [
        {"generated_text": output.outputs[0].text}
        for output in outputs
    ]

def output_fn(prediction, accept="application/json"):
    """Format output."""
    return json.dumps(prediction), accept
```

```bash
# Build and push to ECR
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
IMAGE_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/vllm-inference:v1"

aws ecr get-login-password | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
docker build -t vllm-inference:v1 .
docker tag vllm-inference:v1 ${IMAGE_URI}
docker push ${IMAGE_URI}
```

### Step 2: Deploy with Custom Container

```python
from sagemaker.model import Model

model = Model(
    image_uri=IMAGE_URI,
    model_data=s3_model_uri,
    role=role,
    env={
        "NUM_GPUS": "4",
        "QUANTIZATION": "awq",
        "MAX_MODEL_LEN": "8192",
        "GPU_MEM_UTIL": "0.90",
    }
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.12xlarge",    # 4× A10G — fits 70B AWQ INT4
    endpoint_name="llama3-70b-vllm",
    container_startup_health_check_timeout=900,  # vLLM + 70B needs more startup time
)
```

---

## Endpoint Configuration: Instance Type Selection Guide

```
7B model, BF16 (14 GB):
→ ml.g5.2xlarge (1× A10G 24GB) — fits with room for KV cache

7B model, AWQ INT4 (3.5 GB):
→ ml.g5.xlarge (1× A10G 24GB) — generous KV cache headroom
→ ml.inf2.xlarge (Inferentia2, 32GB) — cheaper, ~30% cost savings

13B model, BF16 (26 GB):
→ ml.g5.12xlarge (4× A10G 96GB) — tensor parallel across 4 GPUs, or
→ single A100 (p4d) if available

70B model, AWQ INT4 (35 GB):
→ ml.g5.12xlarge (4× A10G 96GB) — fits with room for KV cache
→ ml.g5.48xlarge (8× A10G) for higher throughput

70B model, BF16 (140 GB):
→ ml.p4d.24xlarge (8× A100 40GB = 320GB) — tensor parallel 8-way
→ ml.p4de.24xlarge (8× A100 80GB = 640GB) — generous KV cache

High throughput, cost-sensitive:
→ ml.inf2.8xlarge (AWS Inferentia2) — significantly cheaper per request
   at comparable throughput for supported models
```

---

## Deleting Endpoints (Critical for Cost Management)

SageMaker endpoints are billed per second of uptime — even when receiving zero traffic. Always delete endpoints that are not in active use.

```python
# Delete endpoint when no longer needed
import boto3

sm_client = boto3.client("sagemaker")

sm_client.delete_endpoint(EndpointName="llama3-8b-endpoint")

# Also delete the endpoint configuration if not needed
sm_client.delete_endpoint_config(EndpointConfigName="llama3-8b-endpoint-config")

print("Endpoint deleted. Billing stopped.")
```

> **Interview note:** "How do you deploy a fine-tuned LLM on AWS?" The complete answer covers: (1) save model artifacts as tar.gz to S3 (or push to HF Hub), (2) use SageMaker's HuggingFace TGI DLC or a custom vLLM container image from ECR, (3) create a SageMaker Model pointing to the S3 artifact and container image, (4) deploy to a real-time endpoint with the appropriate GPU instance (ml.g5.2xlarge for 7B BF16, ml.g5.12xlarge for 70B INT4), (5) set container startup timeout appropriately — large models need 10–15 minutes to load. Always delete endpoints when not in use.

---

## Summary

- Three deployment paths: HuggingFace TGI DLC (easiest, from S3 or HF Hub), custom vLLM container (maximum throughput via PagedAttention), custom PyTorch DLC (flexible custom inference code).
- All paths follow: model artifact to S3 → SageMaker Model object → EndpointConfig → Endpoint.
- TGI DLC supports AWQ quantization natively via `QUANTIZE=awq` environment variable — no custom code needed.
- vLLM custom container gives PagedAttention and continuous batching for high-concurrency production workloads.
- Instance selection is memory-constrained: model weights + KV cache must fit. 7B BF16=14GB → g5.2xlarge; 70B AWQ INT4=35GB → g5.12xlarge.
- SageMaker endpoints are billed per uptime second — delete endpoints not in active use.

---
