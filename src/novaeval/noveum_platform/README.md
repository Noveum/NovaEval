# Noveum Platform API Client

A comprehensive Python client for the Noveum Platform API, providing easy-to-use methods for traces, datasets, and scorer results with full type safety and error handling.

## Features

- **27 API Methods**: Complete coverage of Traces, Datasets, and Scorer Results APIs
- **Unified Interface**: Single `NoveumClient` class for all operations
- **Type Safety**: Comprehensive Pydantic models for request/response validation
- **Error Handling**: Custom exceptions for different error types
- **Authentication**: Bearer token authentication with environment variable support
- **Connection Pooling**: Efficient HTTP requests using `requests.Session`
- **Logging**: Integrated with NovaEval's logging system

## Quick Start

```python
from novaeval.noveum_platform import NoveumClient

# Initialize client (reads NOVEUM_API_KEY from environment)
client = NoveumClient(organization_id="your-org-id")

# Traces
traces = client.query_traces(project="my-project", size=20)

# Datasets
dataset = client.create_dataset(name="My Dataset", dataset_type="custom")
client.add_dataset_items("my-dataset", "1.0.0", items=[...])

# Scorer Results
client.create_scorer_result({
    "datasetSlug": "my-dataset",
    "itemId": "001",
    "scorerId": "quality-score",
    "score": 0.95
})
```

## API Methods

### Trace Methods (7)
- `ingest_trace(trace: Dict[str, Any])` - Ingest single trace
- `ingest_traces(traces: List[Dict[str, Any]])` - Ingest multiple traces
- `query_traces(organization_id, from_, size, start_time, end_time, project, environment, status, user_id, session_id, tags, sort, search_term, include_spans)` - Query traces with filters and pagination
- `get_trace(trace_id: str)` - Get specific trace by ID
- `get_trace_spans(trace_id: str)` - Get spans for a trace
- `get_directory_tree()` - Get organization directory structure
- `get_connection_status()` - Check API connection status

### Dataset Methods (14)
- `create_dataset(name, slug, description, visibility, dataset_type, environment, schema_version, tags, custom_attributes)` - Create new dataset
- `list_datasets(limit, offset, visibility, organizationSlug, includeVersions)` - List datasets with filters and pagination
- `get_dataset(slug: str)` - Get specific dataset
- `update_dataset(slug, name, description, visibility, dataset_type, environment, schema_version, tags, custom_attributes)` - Update dataset metadata
- `delete_dataset(slug: str)` - Delete dataset
- `list_dataset_versions(dataset_slug: str)` - List dataset versions
- `create_dataset_version(dataset_slug: str, version_data: Dict[str, Any])` - Create version
- `get_dataset_version(dataset_slug: str, version: str)` - Get specific version
- `publish_dataset_version(dataset_slug: str, version: str)` - Publish version
- `list_dataset_items(dataset_slug, version, limit, offset)` - List dataset items
- `add_dataset_items(dataset_slug: str, version: str, items: List[Dict[str, Any]])` - Add items
- `delete_all_dataset_items(dataset_slug: str, version: Optional[str])` - Delete all items
- `get_dataset_item(dataset_slug: str, item_key: str)` - Get specific item
- `delete_dataset_item(dataset_slug: str, item_id: str)` - Delete specific item

### Scorer Results Methods (6)
- `list_scorer_results(organizationSlug, datasetSlug, itemId, scorerId, limit, offset)` - List scorer results with filters
- `create_scorer_result(result_data: Dict[str, Any])` - Create single result
- `create_scorer_results_batch(results: List[Dict[str, Any]])` - Batch create results
- `get_scorer_result(dataset_slug: str, item_id: str, scorer_id: str)` - Get specific result
- `update_scorer_result(dataset_slug: str, item_id: str, scorer_id: str, result_data: Dict[str, Any])` - Update result
- `delete_scorer_result(dataset_slug: str, item_id: str, scorer_id: str)` - Delete result

## Configuration

Set environment variables:
```bash
export NOVEUM_API_KEY="your-api-key"
export NOVEUM_ORGANIZATION_ID="your-org-id"  # Optional
```

Or pass directly to client:
```python
client = NoveumClient(
    api_key="your-api-key",
    organization_id="your-org-id",
    base_url="https://noveum.ai",  # Optional
    timeout=30.0  # Optional
)
```

## Pydantic Models

The client includes comprehensive Pydantic models for type safety:

### Dataset Models
- `DatasetCreateRequest` - Dataset creation validation
- `DatasetUpdateRequest` - Dataset update validation
- `DatasetVersionCreateRequest` - Version creation validation
- `DatasetItemsCreateRequest` - Items creation validation
- `DatasetItem` - Individual item validation
- `DatasetsQueryParams` - Dataset listing parameters
- `DatasetItemsQueryParams` - Dataset items listing parameters

### Scorer Results Models
- `ScorerResultCreateRequest` - Scorer result creation validation
- `ScorerResultUpdateRequest` - Scorer result update validation
- `ScorerResultsBatchRequest` - Batch results validation
- `ScorerResultsQueryParams` - Scorer results listing parameters

### Trace Models
- `TracesQueryParams` - Trace query parameters

## Error Handling

The client raises specific exceptions for different error types:

```python
from novaeval.noveum_platform import NoveumClient, NotFoundError, AuthenticationError

try:
    dataset = client.get_dataset("non-existent-slug")
except NotFoundError as e:
    print(f"Dataset not found: {e.message}")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
```

Available exceptions:
- `NoveumAPIError` - Base exception class
- `AuthenticationError` - Invalid API key
- `ValidationError` - Invalid request format
- `ForbiddenError` - Access denied
- `NotFoundError` - Resource not found
- `ConflictError` - Resource conflict
- `RateLimitError` - Rate limit exceeded
- `ServerError` - Server error

## Examples

### Working with Datasets

```python
# Create a dataset
dataset = client.create_dataset(
    name="My Evaluation Dataset",
    description="Dataset for model evaluation",
    dataset_type="custom",
    visibility="org"
)

# Add items to the dataset
items = [
    {
        "item_key": "q1",
        "item_type": "question_answer",
        "content": {
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        "metadata": {"difficulty": "easy"}
    }
]
client.add_dataset_items(dataset["slug"], "1.0.0", items)

# List dataset items
items_response = client.list_dataset_items(dataset["slug"], limit=10)
```

### Working with Scorer Results

```python
# Create a scorer result
result = client.create_scorer_result({
    "datasetSlug": "my-dataset",
    "itemId": "q1",
    "scorerId": "accuracy-scorer",
    "score": 0.95,
    "metadata": {"confidence": 0.9},
    "details": {"correct_answer": True}
})

# Batch create results
results = [
    {
        "datasetSlug": "my-dataset",
        "itemId": "q1",
        "scorerId": "accuracy-scorer",
        "score": 0.95
    },
    {
        "datasetSlug": "my-dataset", 
        "itemId": "q2",
        "scorerId": "accuracy-scorer",
        "score": 0.87
    }
]
client.create_scorer_results_batch(results)

# Query results
results = client.list_scorer_results(
    organizationSlug="my-org",
    datasetSlug="my-dataset",
    limit=100
)
```

### Working with Traces

```python
# Ingest traces
trace_data = {
    "trace_id": "trace-123",
    "name": "Model Inference",
    "start_time": "2024-01-15T10:30:00Z",
    "end_time": "2024-01-15T10:30:05Z",
    "status": "success",
    "project": "my-project"
}
client.ingest_trace(trace_data)

# Query traces
traces = client.query_traces(
    project="my-project",
    start_time="2024-01-01T00:00:00Z",
    end_time="2024-12-31T23:59:59Z",
    size=50
)
```

## Requirements

- Python 3.9+
- requests
- pydantic
- python-dotenv

## Installation

```bash
pip install -e .
```

## Known Issues

### 🚨 API Issues (For API Developer)

These are functional problems with the API endpoints that need to be fixed on the server side:

#### 1. **Connection Status Endpoint - Routing Issue**
- **Endpoint**: `GET /api/v1/traces/connection-status`
- **Problem**: Returns 404 with error "Trace with ID 'connection-status' not found"
- **Root Cause**: The API incorrectly interprets the endpoint as a trace ID lookup instead of executing the connection status check
- **Impact**: Cannot verify API connectivity
- **Fix Needed**: Fix routing to properly handle the connection-status endpoint

#### 2. **Dataset Version Publishing - Authentication Issue**
- **Endpoint**: `POST /api/v1/datasets/{datasetSlug}/versions/{version}/publish`
- **Problem**: Returns 401 Unauthorized even with valid API key and existing dataset versions
- **Tested Versions**: 0.0.1, 1.0.0 (both fail)
- **Root Cause**: Either endpoint is deprecated, requires elevated permissions, or has authentication issues
- **Impact**: Cannot publish dataset versions, blocking dataset item access
- **Fix Needed**: Either fix authentication or provide alternative publishing method

#### 3. **Dataset Item Access - Dependency Issue**
- **Endpoint**: `GET /api/v1/datasets/{datasetSlug}/items/{itemKey}`
- **Problem**: Returns 404 Not Found even with valid dataset slugs and item keys
- **Root Cause**: Dataset items are only accessible after publishing, but publishing is broken (see issue #2)
- **Impact**: Cannot retrieve individual dataset items
- **Fix Needed**: Either fix publishing or make items accessible without publishing

### 📚 Documentation Issues (For Documentation Developer)

These are missing or incorrect specifications in the OpenAPI documentation that make the API difficult to use:

#### 1. **Trace Ingestion - Missing Required Fields**
- **Endpoints**: `POST /api/v1/traces/single`, `POST /api/v1/traces`
- **Problem**: OpenAPI spec doesn't document required request body schema
- **Missing Fields**:
  - `duration_ms` (number) - Required at trace level
  - `span_count` (number) - Required at trace level  
  - `sdk` (object) - Required at trace level
  - `spans` (array) - Required, each span needs `duration_ms`
- **Impact**: Developers get 400 ValidationError without knowing what fields are required
- **Fix Needed**: Add complete request body schema to OpenAPI spec

#### 2. **Dataset Item Deletion - Missing Request Body Schema**
- **Endpoint**: `DELETE /api/v1/datasets/{datasetSlug}/items`
- **Problem**: OpenAPI spec doesn't document that request body is required
- **Missing Schema**: `{"itemIds": ["id1", "id2", ...]}` array in request body
- **Impact**: Developers send DELETE without body and get 400 ValidationError
- **Fix Needed**: Add request body schema to OpenAPI spec

#### 3. **Scorer Results - Missing Query Parameters**
- **Endpoints**: All scorer result endpoints
- **Problem**: OpenAPI spec doesn't document required `organizationSlug` query parameter
- **Missing Parameter**: `organizationSlug` (string) - Required for all scorer operations
- **Impact**: Developers get 400 ValidationError without knowing the parameter is required
- **Fix Needed**: Add `organizationSlug` query parameter to all scorer result endpoints in OpenAPI spec

### ✅ Client-Side Fixes Applied

These issues were resolved by updating the client implementation:

- **`ingest_traces()`** - Fixed to handle both wrapped and unwrapped trace formats
- **`delete_all_dataset_items()`** - Fixed to require and send proper item IDs
- **All Scorer Result methods** - Fixed to include required `organizationSlug` parameter


## License

See the main project license.