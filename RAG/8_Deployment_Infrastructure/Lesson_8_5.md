# Lesson 8.5 — Kubernetes for RAG: Pods, HPA, Resource Limits, and Rolling Deployments

---

## Why Kubernetes for RAG

Docker Compose is sufficient for development and small-scale production. Kubernetes (K8s) becomes the right tool when you need:

- **Horizontal scaling:** Automatically add more API pods during peak traffic, remove them when traffic drops.
- **Self-healing:** When a pod crashes, Kubernetes automatically restarts it. When a node fails, pods are rescheduled on healthy nodes.
- **Rolling deployments:** Deploy a new version of the API without downtime — gradually shift traffic from old pods to new pods.
- **Resource management:** Enforce CPU and memory limits so one misbehaving service cannot starve others on the same node.
- **Multi-environment parity:** The same Kubernetes manifests work in development (Minikube), staging, and production (EKS, GKE, AKS).

This lesson covers the Kubernetes primitives that matter most for RAG systems and how to configure them.

---

## Core Kubernetes Concepts Applied to RAG

### Deployment: The RAG API

A Deployment manages a set of identical pods. It ensures the desired number of replicas are running and handles rollouts.

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-production
  labels:
    app: rag-api
    version: "1.2.0"
spec:
  replicas: 3   # Start with 3; HPA will adjust
  
  selector:
    matchLabels:
      app: rag-api
  
  # Rolling update strategy: zero-downtime deployments
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2          # Allow 2 extra pods during update
      maxUnavailable: 0    # Never reduce below desired count during update
  
  template:
    metadata:
      labels:
        app: rag-api
        version: "1.2.0"
    spec:
      # Spread pods across availability zones for resilience
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: rag-api
      
      containers:
        - name: rag-api
          image: your-registry/rag-api:1.2.0
          
          ports:
            - containerPort: 8000
          
          # Resource limits and requests
          resources:
            requests:
              cpu: "500m"      # 0.5 CPU cores guaranteed
              memory: "1Gi"   # 1GB RAM guaranteed
            limits:
              cpu: "2000m"     # Max 2 CPU cores
              memory: "3Gi"   # Max 3GB RAM — OOM kill if exceeded
          
          # Environment variables (non-sensitive)
          env:
            - name: QDRANT_URL
              value: "http://qdrant-service:6333"
            - name: EMBEDDING_SERVER_URL
              value: "http://embedding-service:8001"
            - name: LOG_LEVEL
              value: "INFO"
          
          # Secrets from Kubernetes Secret objects (not hardcoded)
          envFrom:
            - secretRef:
                name: rag-api-secrets
          
          # Startup probe: don't send traffic until the model is loaded
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            failureThreshold: 30   # Allow 30 × 10s = 5 minutes for startup
            periodSeconds: 10
          
          # Liveness probe: restart if the service becomes unresponsive
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          
          # Readiness probe: only send traffic when ready
          readinessProbe:
            httpGet:
              path: /health/ready    # More thorough check than /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 2
      
      # Graceful shutdown: allow in-flight requests to complete
      terminationGracePeriodSeconds: 60
```

**The difference between the three probes:**
- `startupProbe`: runs until it succeeds. Kubernetes does not check liveness/readiness until startup succeeds. Use for slow-starting applications (model loading).
- `livenessProbe`: if this fails, Kubernetes kills and restarts the pod. Use for detecting deadlocks or unrecoverable states.
- `readinessProbe`: if this fails, the pod is removed from the Service's load balancer. Use when the pod is alive but temporarily unable to handle requests (e.g., reconnecting to Qdrant).

---

### StatefulSet: Qdrant Vector Database

Databases need stable network identities and persistent storage. StatefulSets provide this.

```yaml
# k8s/qdrant-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: rag-production
spec:
  serviceName: "qdrant-headless"
  replicas: 3   # 3-node Qdrant cluster
  
  selector:
    matchLabels:
      app: qdrant
  
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:v1.9.0
          
          ports:
            - containerPort: 6333  # HTTP
            - containerPort: 6334  # gRPC
            - containerPort: 6335  # P2P (cluster communication)
          
          resources:
            requests:
              cpu: "2000m"
              memory: "16Gi"
            limits:
              cpu: "4000m"
              memory: "32Gi"
          
          env:
            - name: QDRANT__CLUSTER__ENABLED
              value: "true"
            - name: QDRANT__CLUSTER__P2P__PORT
              value: "6335"
          
          volumeMounts:
            - name: qdrant-storage
              mountPath: /qdrant/storage
          
          livenessProbe:
            httpGet:
              path: /healthz
              port: 6333
            initialDelaySeconds: 30
            periodSeconds: 30
  
  # Persistent volume claim template — each pod gets its own PVC
  volumeClaimTemplates:
    - metadata:
        name: qdrant-storage
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: "gp3"        # AWS EBS gp3
        resources:
          requests:
            storage: 100Gi             # 100GB per node
```

---

### Horizontal Pod Autoscaler: Scaling the API

HPA automatically adjusts the number of API pods based on load metrics.

```yaml
# k8s/api-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  
  minReplicas: 2   # Never scale below 2 (high availability)
  maxReplicas: 20  # Never scale above 20
  
  metrics:
    # Scale on CPU utilization
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70    # Scale out when avg CPU > 70%
    
    # Scale on memory utilization
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 75
    
    # Scale on custom metric: request latency (requires Prometheus adapter)
    - type: External
      external:
        metric:
          name: rag_api_p95_latency_ms
          selector:
            matchLabels:
              deployment: rag-api
        target:
          type: AverageValue
          averageValue: "2500"   # Scale out when p95 > 2.5s
  
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # Wait 60s before scaling up again
      policies:
        - type: Pods
          value: 4                      # Add at most 4 pods at a time
          periodSeconds: 60
    
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down (avoid flapping)
      policies:
        - type: Pods
          value: 2                     # Remove at most 2 pods at a time
          periodSeconds: 60
```

---

### GPU Workload: Embedding Server

GPU pods require special node selectors and resource requests.

```yaml
# k8s/embedding-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-server
  namespace: rag-production
spec:
  replicas: 2   # 2 GPU nodes for redundancy
  
  selector:
    matchLabels:
      app: embedding-server
  
  template:
    metadata:
      labels:
        app: embedding-server
    spec:
      # Only schedule on GPU nodes
      nodeSelector:
        accelerator: nvidia-a10g
      
      # Tolerate the GPU taint (GPU nodes are typically tainted)
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      
      containers:
        - name: embedding-server
          image: your-registry/embedding-server:1.0.0
          
          resources:
            limits:
              nvidia.com/gpu: 1     # Request exactly 1 GPU
              cpu: "4000m"
              memory: "16Gi"
            requests:
              nvidia.com/gpu: 1
              cpu: "2000m"
              memory: "8Gi"
          
          startupProbe:
            httpGet:
              path: /health
              port: 8001
            failureThreshold: 60   # Up to 10 minutes for model loading
            periodSeconds: 10
          
          livenessProbe:
            httpGet:
              path: /health
              port: 8001
            periodSeconds: 30
```

---

### ConfigMap and Secret: Configuration Management

```yaml
# k8s/rag-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-production
data:
  QDRANT_URL: "http://qdrant-service:6333"
  REDIS_URL: "redis://redis-service:6379/0"
  EMBEDDING_SERVER_URL: "http://embedding-service:8001"
  RERANKER_URL: "http://reranker-service:8002"
  LOG_LEVEL: "INFO"
  MAX_RETRIEVED_CHUNKS: "50"
  RERANK_TOP_K: "10"
  CONTEXT_BUDGET_TOKENS: "12000"

---
# k8s/rag-secret.yaml
# Never commit actual secrets — use External Secrets Operator or Sealed Secrets
apiVersion: v1
kind: Secret
metadata:
  name: rag-api-secrets
  namespace: rag-production
type: Opaque
data:
  # These should be managed by External Secrets Operator pulling from AWS Secrets Manager
  # Not hardcoded here
  OPENAI_API_KEY: <base64-encoded-value-from-secret-manager>
  POSTGRES_PASSWORD: <base64-encoded-value-from-secret-manager>
```

**Using External Secrets Operator (ESO) for production secrets:**

```yaml
# k8s/external-secret.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: rag-api-secrets
  namespace: rag-production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
    kind: ClusterSecretStore
  target:
    name: rag-api-secrets     # Creates a K8s Secret with this name
  data:
    - secretKey: OPENAI_API_KEY
      remoteRef:
        key: rag/production/api-keys
        property: openai_api_key
    - secretKey: POSTGRES_PASSWORD
      remoteRef:
        key: rag/production/database
        property: password
```

ESO pulls secrets from AWS Secrets Manager and creates Kubernetes Secrets automatically. No secrets in git.

---

### Rolling Deployment: Zero-Downtime Updates

The `RollingUpdate` strategy (configured in the Deployment) handles zero-downtime deployments automatically. But you need to ensure readiness probes work correctly for traffic to shift smoothly.

```bash
# Deploy a new version
kubectl set image deployment/rag-api \
  rag-api=your-registry/rag-api:1.3.0 \
  -n rag-production

# Watch the rollout
kubectl rollout status deployment/rag-api -n rag-production

# If something goes wrong, rollback immediately
kubectl rollout undo deployment/rag-api -n rag-production

# See rollout history
kubectl rollout history deployment/rag-api -n rag-production
```

**The rollout process:**
1. Kubernetes starts new pods with the new image (maxSurge allows this while staying at replicas).
2. New pods pass readiness probes and are added to the Service.
3. Old pods begin receiving no new traffic (readiness probe signals this).
4. Old pods are terminated after `terminationGracePeriodSeconds` (in-flight requests complete).
5. Repeat until all pods are updated.

With `maxUnavailable: 0`, there is never a moment with fewer than `replicas` ready pods serving traffic.

---

### Resource Management and Namespace Isolation

Use namespaces to isolate environments and set resource quotas:

```yaml
# k8s/namespace-with-quota.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-production

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: rag-production-quota
  namespace: rag-production
spec:
  hard:
    pods: "50"                    # Max 50 pods in this namespace
    requests.cpu: "40"            # Total CPU requests: 40 cores
    requests.memory: "160Gi"      # Total memory requests: 160GB
    limits.cpu: "80"
    limits.memory: "320Gi"
    requests.nvidia.com/gpu: "4"  # Max 4 GPUs

---
# Prevent runaway resource usage per container
apiVersion: v1
kind: LimitRange
metadata:
  name: rag-default-limits
  namespace: rag-production
spec:
  limits:
    - type: Container
      default:
        cpu: "500m"
        memory: "512Mi"
      defaultRequest:
        cpu: "100m"
        memory: "128Mi"
      max:
        cpu: "8"
        memory: "64Gi"
```

---

## Kubernetes vs. ECS: When to Choose Each

| | ECS | Kubernetes |
|---|---|---|
| **Operational complexity** | Low | High |
| **Learning curve** | Moderate | Steep |
| **AWS integration** | Deep native | Via add-ons |
| **Multi-cloud** | AWS only | Portable |
| **Custom scheduling** | Limited | Powerful |
| **Ecosystem** | Limited | Vast |
| **Best for** | AWS-only shops, smaller teams | Large teams, multi-cloud, complex scheduling |

For teams already using AWS and with < 10 engineers, ECS Fargate is usually the better choice. Kubernetes becomes worthwhile when you need multi-cloud portability, complex scheduling (GPU nodes, spot handling), or a large engineering team that already knows it.

---

## Summary

- Use Deployments for stateless services (RAG API, indexing workers). Use StatefulSets for stateful services with persistent storage (Qdrant).
- Three probe types serve different purposes: startupProbe for slow-starting services, livenessProbe for crash detection, readinessProbe for traffic gating.
- HPA scales pods based on CPU, memory, or custom metrics. Configure scale-down stabilization to prevent flapping.
- GPU workloads require node selectors, tolerations, and `nvidia.com/gpu` resource requests.
- Never store secrets in ConfigMaps or hardcoded in manifests. Use External Secrets Operator pulling from AWS Secrets Manager or Vault.
- Rolling updates with `maxUnavailable: 0` ensure zero-downtime deployments. Correct readiness probes are required for this to work.
- Kubernetes adds significant operational overhead. Prefer ECS for AWS-only shops with smaller teams.

---

## What's Next

Lesson 8.6 covers scaling the retrieval layer — read replicas, sharding, query caching, and the architecture patterns for handling millions of queries per day.