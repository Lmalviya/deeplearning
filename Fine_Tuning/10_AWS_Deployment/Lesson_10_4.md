# Lesson 10.4 — Monitoring and Observability: CloudWatch, Model Monitor, and LLM-Specific Metrics

---

## Why Monitoring Is Not Optional

Deploying a model and assuming it works correctly is a common mistake. In production, three things go wrong silently:

1. **Performance degrades over time** — input distributions shift, the model's accuracy drops, but nobody notices because no one is measuring it.
2. **The model behaves differently than in evaluation** — real traffic is different from your eval set. Edge cases accumulate.
3. **Costs spiral** — an endpoint left running at peak capacity over a weekend that nobody needed.

Monitoring is the difference between a production system and a research experiment with an HTTP endpoint.

---

## Layer 1: Infrastructure Metrics (CloudWatch)

SageMaker automatically emits these metrics to CloudWatch for every endpoint. No configuration required.

**The core endpoint metrics:**

| Metric | Unit | What it means | Alert threshold |
|---|---|---|---|
| `Invocations` | Count | Total inference requests per minute | — |
| `InvocationErrors` | Count | Total failed requests | > 1% of Invocations |
| `Invocations4XXErrors` | Count | Client errors (bad request format, auth) | Sudden spike → client issue |
| `Invocations5XXErrors` | Count | Server errors (OOM, model crash) | Any → immediate investigation |
| `ModelLatency` | Microseconds | Time inside the model container (compute) | P99 > 30s for LLMs |
| `OverheadLatency` | Microseconds | SageMaker infrastructure overhead | Should be < 100ms |
| `MemoryUtilization` | Percent | CPU/system RAM usage | — |
| `GPUUtilization` | Percent | GPU compute utilization | < 20% → over-provisioned |
| `GPUMemoryUtilization` | Percent | GPU VRAM usage | > 90% → risk of OOM |

**Setting up CloudWatch alarms:**

```python
import boto3

cloudwatch = boto3.client("cloudwatch")

# Alarm: high error rate
cloudwatch.put_metric_alarm(
    AlarmName="LLMEndpointHighErrorRate",
    MetricName="Invocations5XXErrors",
    Namespace="AWS/SageMaker",
    Dimensions=[
        {"Name": "EndpointName", "Value": "llama3-8b-endpoint"},
        {"Name": "VariantName", "Value": "AllTraffic"}
    ],
    Statistic="Sum",
    Period=300,               # 5-minute evaluation period
    EvaluationPeriods=2,      # 2 consecutive periods above threshold
    Threshold=5.0,            # More than 5 errors in 5 minutes
    ComparisonOperator="GreaterThanOrEqualToThreshold",
    AlarmActions=["arn:aws:sns:us-east-1:123456:MLAlerts"],  # SNS notification
    TreatMissingData="notBreaching"
)

# Alarm: high P99 latency  
cloudwatch.put_metric_alarm(
    AlarmName="LLMEndpointHighLatency",
    MetricName="ModelLatency",
    Namespace="AWS/SageMaker",
    Dimensions=[
        {"Name": "EndpointName", "Value": "llama3-8b-endpoint"},
        {"Name": "VariantName", "Value": "AllTraffic"}
    ],
    ExtendedStatistic="p99",
    Period=300,
    EvaluationPeriods=3,
    Threshold=30000000,       # 30 seconds in microseconds
    ComparisonOperator="GreaterThanOrEqualToThreshold",
    AlarmActions=["arn:aws:sns:us-east-1:123456:MLAlerts"],
)
```

**Monitoring GPU utilization:** Low GPU utilization (< 30%) consistently indicates you are over-provisioned. Consider a smaller instance or handling more concurrent requests (check if batching is working). Very high GPU memory utilization (> 90%) is a risk — the next long-context request may trigger an OOM error.

---

## Layer 2: SageMaker Model Monitor — Data and Model Drift

Model Monitor helps you detect when the production environment changes in ways that would affect your model's quality.

### Data Quality Monitor

Detects statistical changes in the input data distribution — the model is receiving inputs that are different from what it was trained on.

```python
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

# Step 1: Enable data capture on the endpoint (captures % of requests to S3)
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=20,             # Capture 20% of requests
    destination_s3_uri="s3://my-bucket/data-capture/",
    capture_options=["REQUEST", "RESPONSE"],
)

# Step 2: Compute a baseline from your training/validation data
monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

monitor.suggest_baseline(
    baseline_dataset="s3://my-bucket/validation-data/baseline.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri="s3://my-bucket/monitoring/baseline/",
    wait=True,
)

# Step 3: Schedule monitoring job to run regularly
monitor.create_monitoring_schedule(
    monitor_schedule_name="llm-data-monitor",
    endpoint_input=predictor.endpoint_name,
    output_s3_uri="s3://my-bucket/monitoring/reports/",
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_constraints(),
    schedule_cron_expression="cron(0 * ? * * *)",  # Hourly
)
```

**What data quality monitor detects for LLMs:**
- Input prompt length distribution shift (users suddenly sending much longer prompts → token budget issues)
- Input feature distribution changes (new topic domains, new languages)
- System prompt length changes if captured

### Model Quality Monitor

Detects degradation in the model's prediction quality over time — requires ground truth labels.

For LLMs, this is complex because there is no ground truth "label" for a free-text response. Practical approaches:
- For classification tasks (sentiment, intent): collect human labels and compare
- For constrained outputs (JSON, structured extraction): check format compliance rate automatically
- For Q&A: use a separate evaluation LLM to score sampled outputs (LLM-as-judge pattern)

---

## Layer 3: LLM-Specific Production Metrics

Standard ML monitoring was designed for classification/regression models. For LLMs, you need additional custom metrics.

### Response Quality Monitoring

```python
import boto3
import random
import asyncio
from openai import AsyncOpenAI

cloudwatch = boto3.client("cloudwatch")

async def monitor_response_quality(
    endpoint_name: str,
    sample_rate: float = 0.05,  # Sample 5% of responses
):
    """
    Periodically sample endpoint responses and score them with an LLM judge.
    Emit custom CloudWatch metrics.
    """
    
    # This runs as a background Lambda function triggered by EventBridge
    # Collects recent request/response pairs from data capture S3 bucket
    
    sample_pairs = get_captured_samples(endpoint_name, sample_rate)
    
    judge_client = AsyncOpenAI()
    
    for prompt, response in sample_pairs:
        score = await judge_response_quality(judge_client, prompt, response)
        
        # Emit custom metric to CloudWatch
        cloudwatch.put_metric_data(
            Namespace="LLMMonitoring",
            MetricData=[{
                "MetricName": "ResponseQualityScore",
                "Dimensions": [
                    {"Name": "EndpointName", "Value": endpoint_name}
                ],
                "Value": score,
                "Unit": "None"
            }]
        )

async def judge_response_quality(client, prompt: str, response: str) -> float:
    """Score a response on a 0-10 scale using GPT-4o-mini as judge."""
    
    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Rate this AI response on a scale of 1-10 for quality.
            
Question: {prompt[:500]}
Response: {response[:1000]}

Score (1-10, no explanation):"""
        }],
        max_tokens=5,
        temperature=0.0,
    )
    
    try:
        return float(result.choices[0].message.content.strip())
    except ValueError:
        return 5.0  # Default if parsing fails
```

### Custom LLM Metrics to Track

| Metric | How to measure | What it signals |
|---|---|---|
| Response length distribution | Parse response, count tokens | Unexpected length → format issue |
| Refusal rate | Count responses containing refusal phrases | Drift in safety behavior |
| Response latency P50/P95/P99 | CloudWatch ModelLatency percentiles | User experience degradation |
| Token throughput (tokens/sec) | Count output tokens / latency | Serving efficiency |
| Format compliance rate | Parse structured output (JSON, markdown) | Instruction following degradation |
| Hallucination rate (sampled) | Fact-check sampled responses with retrieval | Knowledge accuracy |
| Cost per 1000 requests | Endpoint cost / request count | Budget tracking |

```python
def emit_llm_metrics(endpoint_name: str, response_data: dict):
    """Emit LLM-specific metrics to CloudWatch."""
    
    cloudwatch = boto3.client("cloudwatch")
    
    metrics = [
        {
            "MetricName": "OutputTokenCount",
            "Value": response_data["output_token_count"],
            "Unit": "Count"
        },
        {
            "MetricName": "InputTokenCount",
            "Value": response_data["input_token_count"],
            "Unit": "Count"
        },
        {
            "MetricName": "TokensPerSecond",
            "Value": response_data["output_token_count"] / response_data["latency_seconds"],
            "Unit": "Count/Second"
        },
        {
            "MetricName": "IsRefusal",
            "Value": 1.0 if response_data["is_refusal"] else 0.0,
            "Unit": "None"
        }
    ]
    
    cloudwatch.put_metric_data(
        Namespace="LLMMonitoring",
        MetricData=[
            {**m, "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}]}
            for m in metrics
        ]
    )
```

---

## Layer 4: Cost Monitoring

GPU endpoints are expensive. Without active cost monitoring, costs can spiral quickly.

```python
# Set up a budget alert in AWS Budgets
import boto3

budgets_client = boto3.client("budgets")

budgets_client.create_budget(
    AccountId="123456789012",
    Budget={
        "BudgetName": "LLM-Serving-Monthly",
        "BudgetLimit": {"Amount": "5000", "Unit": "USD"},
        "BudgetType": "COST",
        "TimeUnit": "MONTHLY",
        "CostFilters": {
            "Service": ["Amazon SageMaker"],
            "TagKeyValue": ["user:Project$LLMProduction"]  # Tag your endpoints!
        }
    },
    NotificationsWithSubscribers=[{
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 80,          # Alert at 80% of budget
            "ThresholdType": "PERCENTAGE",
        },
        "Subscribers": [{
            "SubscriptionType": "EMAIL",
            "Address": "ml-team@company.com"
        }]
    }]
)
```

**Cost calculation for LLM endpoints:**
```
Hourly cost (ml.g5.2xlarge) = $1.21/hr
Requests per hour (at 10 req/min) = 600 requests/hr
Cost per request = $1.21 / 600 = $0.002 per request
Cost per 1000 requests = $2.02

At 10,000 requests/day:
  - Daily cost: $20.20
  - Monthly cost: ~$606

With auto-scaling (off-peak scale-in to 0 instances after scale-to-zero):
  - 8 hours full load, 16 hours idle (if you scale to zero): 
  - Daily cost: 8 × $1.21 = $9.68 → monthly ~$290
```

**The "forgotten endpoint" problem:** A single ml.p4d.24xlarge endpoint running idle for a month costs $32.8/hr × 720 hours = **$23,616**. Tag all endpoints with owner and project; set CloudWatch alarm if an endpoint has zero invocations for 4+ hours.

---

## CloudWatch Dashboard: The Single-Pane View

```python
import json

dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "title": "Request Volume",
                "metrics": [
                    ["AWS/SageMaker", "Invocations", "EndpointName", "llama3-8b-endpoint"]
                ],
                "period": 300, "stat": "Sum", "view": "timeSeries"
            }
        },
        {
            "type": "metric",
            "properties": {
                "title": "Latency P50/P95/P99",
                "metrics": [
                    ["AWS/SageMaker", "ModelLatency", "EndpointName", "llama3-8b-endpoint",
                     {"stat": "p50", "label": "P50"}],
                    ["...", {"stat": "p95", "label": "P95"}],
                    ["...", {"stat": "p99", "label": "P99"}]
                ],
                "period": 300, "view": "timeSeries"
            }
        },
        {
            "type": "metric",
            "properties": {
                "title": "Error Rate",
                "metrics": [
                    ["AWS/SageMaker", "Invocations5XXErrors", "EndpointName", "llama3-8b-endpoint"],
                    ["AWS/SageMaker", "Invocations4XXErrors", "..."]
                ],
                "period": 300, "stat": "Sum", "view": "timeSeries"
            }
        },
        {
            "type": "metric",
            "properties": {
                "title": "GPU Utilization",
                "metrics": [
                    ["AWS/SageMaker", "GPUUtilization", "EndpointName", "llama3-8b-endpoint"],
                    ["AWS/SageMaker", "GPUMemoryUtilization", "..."]
                ],
                "period": 60, "stat": "Average", "view": "timeSeries"
            }
        }
    ]
}

cloudwatch.put_dashboard(
    DashboardName="LLM-Production",
    DashboardBody=json.dumps(dashboard_body)
)
```

> **Interview note:** "How would you monitor a deployed LLM endpoint?" Three layers: (1) Infrastructure via CloudWatch — invocations, error rates (5XX), ModelLatency P99, GPU utilization. Alert on any 5XX spike and sustained P99 > 30s. (2) SageMaker Model Monitor for data drift — enable data capture, set a baseline from validation data, schedule hourly monitoring jobs. (3) LLM-specific quality metrics — sample 5% of responses, score with a judge model, emit custom CloudWatch metrics. Track refusal rate, response length distribution, and format compliance. Set a budget alert to catch runaway costs from forgotten endpoints.

---

## Summary

- **Infrastructure metrics** (CloudWatch, automatic): `Invocations`, `Invocations5XXErrors`, `ModelLatency`, `GPUMemoryUtilization`. The minimum viable monitoring set.
- **Critical alarms:** 5XX error rate > 1%, P99 ModelLatency > 30 seconds, GPUMemoryUtilization > 90% (OOM risk).
- **SageMaker Model Monitor:** Enable data capture on the endpoint (sample 10–20%), compute a statistical baseline from validation data, schedule hourly monitoring jobs. Detects input distribution drift.
- **LLM-specific metrics** require custom implementation: response quality (LLM-as-judge on sampled outputs), refusal rate, output token length distribution, format compliance rate, tokens-per-second throughput.
- **Cost monitoring:** Tag every endpoint, set AWS Budgets alerts at 80% of monthly budget, and alert on zero-invocation endpoints (the "forgotten endpoint" problem).
- Build a CloudWatch dashboard with all four layers visible: request volume, latency percentiles, error rate, GPU utilization. This is the single-pane view your on-call team needs.

---
