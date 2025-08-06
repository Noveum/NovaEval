# NovaEval API

A FastAPI-based HTTP API for the NovaEval evaluation framework, providing access to models, datasets, scorers, and evaluation orchestration.

## Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -e ".[api]"
   ```

2. **Run the API server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -f Dockerfile.api -t novaeval-api:latest .
   ```

2. **Create environment file**:
   ```bash
   # Create .env file with API keys (more secure than exposing in command line)
   echo "OPENAI_API_KEY=your-openai-key" > .env
   echo "ANTHROPIC_API_KEY=your-anthropic-key" >> .env
   echo "GOOGLE_API_KEY=your-google-key" >> .env
   echo "AZURE_OPENAI_API_KEY=your-azure-key" >> .env
   echo "AZURE_OPENAI_ENDPOINT=your-azure-endpoint" >> .env

   # Set appropriate permissions
   chmod 600 .env
   ```

   > **Security Note**: Using `--env-file` prevents API keys from appearing in process listings and shell history. For production deployments, consider using Docker secrets or [Kubernetes secrets](../kubernetes/README.md) for even better security.

3. **Run the container**:
   ```bash
   # Run with env file (more secure)
   docker run -d -p 8000:8000 \
     --env-file .env \
     --name novaeval-api \
     novaeval-api:latest
   ```

4. **Test the deployment**:
   ```bash
   curl http://localhost:8000/health
   ```

### Kubernetes Deployment

See `../kubernetes/README.md` for complete Kubernetes deployment instructions.

## API Endpoints

### Health & Status

```bash
# Health check
curl http://localhost:8000/health

# API info
curl http://localhost:8000/

# API ping
curl http://localhost:8000/api/v1/ping
```

### Component Discovery

```bash
# List all components
curl http://localhost:8000/api/v1/components/

# List models
curl http://localhost:8000/api/v1/components/models

# List datasets
curl http://localhost:8000/api/v1/components/datasets

# List scorers
curl http://localhost:8000/api/v1/components/scorers

# Get component details
curl http://localhost:8000/api/v1/components/models/openai
```

### Model Operations

```bash
# Get model info
curl http://localhost:8000/api/v1/models/openai/info

# Single prediction
curl -X POST http://localhost:8000/api/v1/models/openai/predict \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "provider": "openai",
      "identifier": "gpt-3.5-turbo",
      "temperature": 0.7
    },
    "prompt": "What is machine learning?"
  }'

# Batch prediction
curl -X POST http://localhost:8000/api/v1/models/openai/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "provider": "openai",
      "identifier": "gpt-3.5-turbo",
      "temperature": 0.7
    },
    "prompts": [
      "What is AI?",
      "What is ML?"
    ]
  }'

# Instantiate model with custom config
curl -X POST http://localhost:8000/api/v1/models/instantiate \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "provider": "openai",
      "identifier": "gpt-4",
      "temperature": 0.9,
      "max_tokens": 1000
    }
  }'
```

### Dataset Operations

```bash
# Get dataset info
curl http://localhost:8000/api/v1/datasets/mmlu/info

# Load dataset
curl -X POST http://localhost:8000/api/v1/datasets/mmlu/load \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "name": "mmlu",
      "split": "test",
      "limit": 10
    }
  }'

# Query dataset with pagination
curl -X POST http://localhost:8000/api/v1/datasets/mmlu/query \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "name": "mmlu",
      "split": "test"
    },
    "limit": 5,
    "offset": 10
  }'

# Sample dataset
curl -X POST http://localhost:8000/api/v1/datasets/mmlu/sample \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "name": "mmlu",
      "split": "test"
    },
    "count": 3,
    "method": "random"
  }'
```

### Scorer Operations

```bash
# Get scorer info
curl http://localhost:8000/api/v1/scorers/accuracy/info

# Score single prediction
curl -X POST http://localhost:8000/api/v1/scorers/accuracy/score \
  -H "Content-Type: application/json" \
  -d '{
    "scorer_config": {
      "name": "accuracy"
    },
    "predicted": "Paris",
    "ground_truth": "Paris",
    "context": "What is the capital of France?"
  }'

# Batch scoring
curl -X POST http://localhost:8000/api/v1/scorers/accuracy/score/batch \
  -H "Content-Type: application/json" \
  -d '{
    "scorer_config": {
      "name": "accuracy"
    },
    "predictions": [
      {
        "predicted": "Paris",
        "ground_truth": "Paris",
        "context": "Capital of France?"
      },
      {
        "predicted": "London",
        "ground_truth": "London",
        "context": "Capital of UK?"
      }
    ]
  }'

# Get scorer statistics
curl http://localhost:8000/api/v1/scorers/accuracy/stats
```

### Evaluation Orchestration

```bash
# Submit evaluation job (JSON config)
curl -X POST http://localhost:8000/api/v1/evaluations/submit \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_evaluation",
    "description": "Test evaluation job",
    "models": [{
      "provider": "openai",
      "identifier": "gpt-3.5-turbo",
      "temperature": 0.7
    }],
    "datasets": [{
      "name": "mmlu",
      "split": "test",
      "limit": 5
    }],
    "scorers": [{
      "name": "accuracy"
    }]
  }'

# Submit evaluation job (YAML file upload)
curl -X POST http://localhost:8000/api/v1/evaluations/submit \
  -F "config_file=@evaluation_config.yaml"

# Get evaluation status
curl http://localhost:8000/api/v1/evaluations/{task_id}/status

# Get evaluation results
curl http://localhost:8000/api/v1/evaluations/{task_id}/result

# List all evaluations
curl http://localhost:8000/api/v1/evaluations/

# Cancel evaluation
curl -X DELETE http://localhost:8000/api/v1/evaluations/{task_id}

# Get task manager stats
curl http://localhost:8000/api/v1/evaluations/stats
```

## Python Client Examples

### Basic Usage

```python
import requests
import json

# Base configuration
API_BASE = "http://localhost:8000"
headers = {"Content-Type": "application/json"}

# Health check
response = requests.get(f"{API_BASE}/health")
print(response.json())
```

### Model Prediction

```python
# Single prediction
model_config = {
    "provider": "openai",
    "identifier": "gpt-3.5-turbo",
    "temperature": 0.7
}

prediction_data = {
    "config": model_config,
    "prompt": "Explain quantum computing in simple terms."
}

response = requests.post(
    f"{API_BASE}/api/v1/models/openai/predict",
    headers=headers,
    json=prediction_data
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Model: {result['inference_details']['identifier']}")
```

### Batch Prediction

```python
# Batch prediction
batch_data = {
    "config": model_config,
    "prompts": [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is deep learning?"
    ]
}

response = requests.post(
    f"{API_BASE}/api/v1/models/openai/predict/batch",
    headers=headers,
    json=batch_data
)

results = response.json()
for i, pred in enumerate(results['predictions']):
    print(f"Q{i+1}: {batch_data['prompts'][i]}")
    print(f"A{i+1}: {pred['prediction']}\n")
```

### Dataset Loading

```python
# Load dataset
dataset_config = {
    "name": "mmlu",
    "split": "test",
    "limit": 10
}

load_data = {"config": dataset_config}

response = requests.post(
    f"{API_BASE}/api/v1/datasets/mmlu/load",
    headers=headers,
    json=load_data
)

dataset = response.json()
print(f"Loaded {len(dataset['records'])} records")
for record in dataset['records'][:3]:
    print(f"Q: {record.get('question', 'N/A')}")
    print(f"A: {record.get('answer', 'N/A')}\n")
```

### Full Evaluation Pipeline

```python
# Submit evaluation job
evaluation_config = {
    "name": "python_client_evaluation",
    "description": "Evaluation submitted via Python client",
    "models": [{
        "provider": "openai",
        "identifier": "gpt-3.5-turbo",
        "temperature": 0.7
    }],
    "datasets": [{
        "name": "mmlu",
        "split": "test",
        "limit": 5
    }],
    "scorers": [{
        "name": "accuracy"
    }]
}

# Submit job
response = requests.post(
    f"{API_BASE}/api/v1/evaluations/submit",
    headers=headers,
    json=evaluation_config
)

task_info = response.json()
task_id = task_info['task_id']
print(f"Evaluation submitted: {task_id}")

# Poll for completion
import time

while True:
    status_response = requests.get(f"{API_BASE}/api/v1/evaluations/{task_id}/status")
    status = status_response.json()

    print(f"Status: {status['status']}")

    if status['status'] in ['completed', 'failed', 'cancelled']:
        break

    time.sleep(5)

# Get results
if status['status'] == 'completed':
    result_response = requests.get(f"{API_BASE}/api/v1/evaluations/{task_id}/result")
    results = result_response.json()

    print("Evaluation Results:")
    print(f"Total samples: {results['summary']['total_samples']}")
    print(f"Average score: {results['summary']['average_score']:.3f}")
```

### Error Handling

```python
def make_api_request(url, method='GET', **kwargs):
    """Helper function with error handling"""
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()  # Raises HTTPError for bad status codes
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {response.status_code}: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return None

# Usage with error handling
result = make_api_request(
    f"{API_BASE}/api/v1/models/openai/predict",
    method='POST',
    headers=headers,
    json=prediction_data
)

if result:
    print(f"Success: {result['prediction']}")
else:
    print("Request failed")
```

## Configuration

### Environment Variables

Set these environment variables for API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="your-azure-endpoint"
export GOOGLE_API_KEY="your-google-key"
```

### Configuration Options

The API can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `MAX_CONCURRENT_EVALUATIONS` | `5` | Max concurrent evaluations |
| `EVALUATION_TIMEOUT_SECONDS` | `3600` | Evaluation timeout |
| `RESULT_CACHE_TTL_SECONDS` | `7200` | Result cache TTL |

## Monitoring & Observability

### Structured Logging

The API uses structured JSON logging for better observability:

```bash
# View logs in production
docker logs novaeval-api

# In Kubernetes
kubectl logs -l app=novaeval-api -f
```

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "service": "novaeval-api",
  "version": "1.0.0"
}
```

## Development

### Local Development Setup

```bash
# Clone repository
git clone <repository-url>
cd NovaEval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[api]"

# Run in development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Install test dependencies
pip install -e ".[api,test]"

# Run tests
pytest tests/api/ -v

# Run with coverage
pytest tests/api/ -v --cov=app --cov-report=term-missing
```

## Troubleshooting

### Common Issues

1. **API Key Missing**:
   ```
   Error: Model API key not configured
   Solution: Set appropriate environment variables
   ```

2. **Import Errors**:
   ```
   Error: ModuleNotFoundError: No module named 'app'
   Solution: Install package with pip install -e ".[api]"
   ```

3. **Port Already in Use**:
   ```
   Error: Address already in use
   Solution: Use different port with --port 8001
   ```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload
```

### Health Check Failures

If health checks fail in Kubernetes:

```bash
# Check pod logs
kubectl logs -l app=novaeval-api

# Check service endpoints
kubectl get endpoints novaeval-api-service

# Test health endpoint directly
kubectl port-forward svc/novaeval-api-service 8000:80
curl http://localhost:8000/health
```
