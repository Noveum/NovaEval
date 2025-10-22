# Noveum Platform API Client

A comprehensive Python client for the Noveum Platform API, providing easy-to-use methods for traces, datasets, and scorer results with full type safety and error handling.

## Features

- **26 API Methods**: Complete coverage of Traces, Datasets, and Scorer Results APIs
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
client = NoveumClient()

# Traces
traces = client.query_traces(project="my-project", size=20)

# Datasets
dataset = client.create_dataset(name="My Dataset", dataset_type="custom")
client.add_dataset_items("my-dataset", "1.0.0", items=[...])

```

## API Methods

### Trace Methods (6)
- `ingest_trace(trace: Dict[str, Any])` - Ingest single trace
- `ingest_traces(traces: List[Dict[str, Any]])` - Ingest multiple traces
- `query_traces(from_, size, start_time, end_time, project, environment, status, user_id, session_id, tags, sort, search_term, include_spans)` - Query traces with filters and pagination
- `get_trace(trace_id: str)` - Get specific trace by ID
- `get_trace_spans(trace_id: str)` - Get spans for a trace
- `get_connection_status()` - Check API connection status

### Dataset Methods (14)
- `create_dataset(name, slug, description, visibility, dataset_type, environment, schema_version, tags, custom_attributes)` - Create new dataset
- `list_datasets(limit, offset, visibility, includeVersions)` - List datasets with filters and pagination
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


## Configuration

Set environment variables:
```bash
export NOVEUM_API_KEY="your-api-key"
```


## Examples

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

## Complete API Demo

For a comprehensive demonstration of all 26 API methods with real examples, see the [Noveum Platform API Demo Notebook](../../examples/noveum_platform_api_demo.ipynb). This interactive notebook covers:

- All trace methods (6) with filtering, pagination, and search examples
- All dataset methods (14) with version management and item operations
- All scorer results methods (6) with batch operations
- Real-world usage patterns and best practices

