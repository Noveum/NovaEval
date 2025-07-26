# NovaEval API Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the NovaEval API.

## Prerequisites

- Kubernetes cluster (minikube, kind, or cloud provider)
- `kubectl` configured to connect to your cluster
- Docker image built locally: `novaeval-api:working`

## Quick Start

1. **Build the Docker image** (if not already done):
   ```bash
   docker build -f Dockerfile.api -t novaeval-api:working .
   ```

2. **Create API keys secret**:
   ```bash
   cd kubernetes
   ./create-secret.sh
   ```

3. **Deploy the application**:
   ```bash
   kubectl apply -f api-deployment.yaml
   ```

4. **Verify deployment**:
   ```bash
   kubectl get pods -l app=novaeval-api
   kubectl get services -l app=novaeval-api
   ```

5. **Test the API**:
   ```bash
   # Port forward to access locally
   kubectl port-forward svc/novaeval-api-service 8000:80

   # Test health endpoint
   curl http://localhost:8000/health
   ```

## Configuration

### Environment Variables

The following environment variables are configured via ConfigMap and Secret:

| Variable | Source | Description | Default |
|----------|--------|-------------|---------|
| `LOG_LEVEL` | ConfigMap | Logging level | `INFO` |
| `HOST` | Deployment | Server host | `0.0.0.0` |
| `PORT` | Deployment | Server port | `8000` |
| `MAX_CONCURRENT_EVALUATIONS` | ConfigMap | Max concurrent evaluations | `5` |
| `EVALUATION_TIMEOUT_SECONDS` | ConfigMap | Evaluation timeout | `3600` |
| `RESULT_CACHE_TTL_SECONDS` | ConfigMap | Result cache TTL | `7200` |
| `OPENAI_API_KEY` | Secret | OpenAI API key | `optional` |
| `ANTHROPIC_API_KEY` | Secret | Anthropic API key | `optional` |
| `AZURE_OPENAI_API_KEY` | Secret | Azure OpenAI API key | `optional` |
| `AZURE_OPENAI_ENDPOINT` | Secret | Azure OpenAI endpoint | `optional` |
| `GOOGLE_API_KEY` | Secret | Google API key | `optional` |

### Resource Limits

- **Requests**: 256Mi memory, 200m CPU
- **Limits**: 1Gi memory, 500m CPU

### Health Checks

- **Readiness**: `/health` endpoint, 5s delay, 10s interval
- **Liveness**: `/health` endpoint, 30s delay, 30s interval
- **Startup**: `/health` endpoint, 10s delay, 5s interval, max 60s

## Services

### Internal Service (ClusterIP)
- **Name**: `novaeval-api-service`
- **Port**: 80 → 8000
- **Type**: ClusterIP (internal only)

### External Service (LoadBalancer)
- **Name**: `novaeval-api-external`
- **Port**: 8000 → 8000
- **Type**: LoadBalancer (external access)

## Scaling

To scale the deployment:

```bash
kubectl scale deployment novaeval-api --replicas=5
```

## Troubleshooting

### Check Pod Status
```bash
kubectl get pods -l app=novaeval-api
kubectl describe pod -l app=novaeval-api
```

### View Logs
```bash
kubectl logs -l app=novaeval-api -f
```

### Check Service Discovery
```bash
kubectl get endpoints novaeval-api-service
```

### Access API Documentation
```bash
# Port forward
kubectl port-forward svc/novaeval-api-service 8000:80

# Open in browser
open http://localhost:8000/docs
```

## Security Notes

- All API keys are stored in Kubernetes Secrets
- Containers run as non-root user (`novaeval:1000`)
- Resource limits prevent resource exhaustion
- Health checks ensure only healthy pods receive traffic

## Files

- `api-deployment.yaml` - Complete deployment manifests
- `create-secret.sh` - Helper script to create API key secrets
- `deployment.yaml` - Original CLI deployment (legacy)
- `README.md` - This documentation
