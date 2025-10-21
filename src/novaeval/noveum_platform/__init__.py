"""
Noveum Platform API Client.

This package provides a Python client for interacting with the Noveum Platform
API. It includes authentication, request/response handling, and comprehensive
error handling for traces, datasets, and scorer results.

Example:
    from novaeval.noveum_platform import NoveumClient

    client = NoveumClient(api_key="your-api-key")

    # Traces
    traces = client.query_traces(project="my-project", size=10)

    # Datasets
    dataset = client.create_dataset(name="My Dataset")
    client.add_dataset_items("my-dataset", "1.0.0", items=[...])

    # Scorer Results
    client.create_scorer_result({
        "datasetSlug": "my-dataset",
        "itemId": "001",
        "scorerId": "quality-score",
        "score": 0.95
    })
"""

from .client import NoveumClient
from .exceptions import (
    AuthenticationError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    NoveumAPIError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from .models import (
    DatasetCreateRequest,
    DatasetItemsCreateRequest,
    DatasetItemsQueryParams,
    DatasetsQueryParams,
    DatasetUpdateRequest,
    DatasetVersionCreateRequest,
    ScorerResultCreateRequest,
    ScorerResultsBatchRequest,
    ScorerResultsQueryParams,
    ScorerResultUpdateRequest,
    TracesQueryParams,
)

# Environment loading is handled in client.py or must be performed by the importer
# before importing this module to avoid side effects and duplication.

__all__ = [
    # Specific exceptions
    "AuthenticationError",
    "ConflictError",
    "DatasetCreateRequest",
    "DatasetItemsCreateRequest",
    "DatasetItemsQueryParams",
    "DatasetUpdateRequest",
    "DatasetVersionCreateRequest",
    "DatasetsQueryParams",
    "ForbiddenError",
    "NotFoundError",
    # Base exception
    "NoveumAPIError",
    # Main client class
    "NoveumClient",
    "RateLimitError",
    "ScorerResultCreateRequest",
    "ScorerResultUpdateRequest",
    "ScorerResultsBatchRequest",
    "ScorerResultsQueryParams",
    "ServerError",
    # Models
    "TracesQueryParams",
    "ValidationError",
]

__version__ = "0.1.0"
