# Kubernetes Deployment Guide

## Overview

This guide explains how to deploy the Code Base Support Agent to a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl configured to access your cluster
- Docker registry access (Docker Hub, GCR, ECR, etc.)
- GitLab access token

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t your-registry/codebase-agent:latest .

# Push to your registry
docker push your-registry/codebase-agent:latest
```

### 2. Configure Secrets

Edit `k8s-deployment.yaml` and update the following:

```yaml
# In the Secret section:
stringData:
  GITLAB_TOKEN: "your-actual-gitlab-token"

# In the ConfigMap section:
data:
  GITLAB_REPO_URL: "https://gitlab.com/your-username/your-repo.git"

# In the Deployment section:
spec:
  template:
    spec:
      containers:
      - name: codebase-agent
        image: your-registry/codebase-agent:latest  # Update this
```

### 3. Deploy to Kubernetes

```bash
# Apply the deployment
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -n codebase-agent
kubectl logs -n codebase-agent -l app=codebase-agent
```

### 4. Access the Application

#### Option A: Port Forward (Development)
```bash
kubectl port-forward -n codebase-agent svc/codebase-agent 7860:80
# Access at http://localhost:7860
```

#### Option B: Ingress (Production)
Update the Ingress section in `k8s-deployment.yaml`:
```yaml
spec:
  rules:
  - host: codebase-agent.your-domain.com
```

Then apply and access via your domain.

## Configuration

### Environment Variables

All configuration is done via the ConfigMap and Secret:

| Variable | Description | Default |
|----------|-------------|---------|
| `GITLAB_TOKEN` | GitLab access token (Secret) | - |
| `GITLAB_REPO_URL` | Repository URL | - |
| `GITLAB_BRANCH` | Branch to clone | `main` |
| `MODEL` | LLM model name | `llama3.2` |
| `USE_OPENAI_EMBEDDINGS` | Use OpenAI for embeddings | `false` |
| `CHUNK_SIZE` | Text chunk size | `2000` |
| `RETRIEVER_K` | Number of results to retrieve | `25` |
| `FORCE_REFRESH_DB` | Force rebuild vector DB | `false` |
| `SERVER_PORT` | Gradio server port | `7860` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Persistent Storage

The deployment uses a PersistentVolumeClaim for:
- `/app/knowledge-base` - Cloned Git repository
- `/app/vector_db` - Vector database

Default size: 10Gi (adjust in `k8s-deployment.yaml` if needed)

## Scaling

The deployment is configured for single replica by default:

```yaml
spec:
  replicas: 1
```

**Note:** For Gradio applications with stateful vector databases, scaling beyond 1 replica requires shared storage or database externalization.

## Resource Limits

Default resource allocation:
- Requests: 2Gi memory, 1 CPU
- Limits: 4Gi memory, 2 CPUs

Adjust based on your workload:
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

## Monitoring

### Health Checks

- **Liveness Probe**: Checks if the application is running
- **Readiness Probe**: Checks if the application is ready to serve traffic

Both probes use the Gradio root endpoint (`/`).

### Logs

```bash
# View logs
kubectl logs -n codebase-agent -l app=codebase-agent -f

# View logs from specific pod
kubectl logs -n codebase-agent <pod-name> -f
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod to see events
kubectl describe pod -n codebase-agent <pod-name>

# Check logs
kubectl logs -n codebase-agent <pod-name>
```

### Common Issues

1. **Git Clone Fails**
   - Verify `GITLAB_TOKEN` in Secret
   - Check repository URL and branch name
   - Ensure network access to GitLab

2. **Out of Memory**
   - Increase memory limits in deployment
   - Reduce `CHUNK_SIZE` or `RETRIEVER_K`
   - Check vector DB size

3. **Slow Initialization**
   - First startup takes longer (cloning repo, building DB)
   - Set `FORCE_REFRESH_DB=false` after first run
   - Consider using init containers for repo cloning

## Local Development

Test locally before deploying:

```bash
# Set environment variables
export GITLAB_TOKEN="your-token"
export GITLAB_REPO_URL="your-repo-url"
export OPEN_BROWSER="true"

# Run locally
python main.py
```

## Security Considerations

1. **Secrets Management**
   - Use Kubernetes Secrets for sensitive data
   - Consider external secret management (e.g., Vault, AWS Secrets Manager)

2. **Network Policies**
   - Restrict ingress/egress traffic
   - Use service mesh for mTLS

3. **RBAC**
   - Create service accounts with minimal permissions
   - Restrict access to namespaces

## Updating

```bash
# Build new image
docker build -t your-registry/codebase-agent:v2 .
docker push your-registry/codebase-agent:v2

# Update deployment
kubectl set image deployment/codebase-agent codebase-agent=your-registry/codebase-agent:v2 -n codebase-agent

# Or apply updated YAML
kubectl apply -f k8s-deployment.yaml
```

## Cleanup

```bash
# Delete all resources
kubectl delete namespace codebase-agent
```
