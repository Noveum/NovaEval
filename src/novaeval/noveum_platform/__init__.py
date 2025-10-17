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

from dotenv import load_dotenv

# Load environment variables at package level
load_dotenv()

from .client import NoveumClient
from .exceptions import (
    AuthenticationError,
    ConflictError,
    ForbiddenError,
    NoveumAPIError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from .models import (
    DatasetCreateRequest,
    DatasetItemsCreateRequest,
    DatasetItemsQueryParams,
    DatasetUpdateRequest,
    DatasetVersionCreateRequest,
    DatasetsQueryParams,
    ScorerResultCreateRequest,
    ScorerResultUpdateRequest,
    ScorerResultsBatchRequest,
    ScorerResultsQueryParams,
    TracesQueryParams,
)

__all__ = [
    # Main client class
    "NoveumClient",
    
    # Base exception
    "NoveumAPIError",
    
    # Specific exceptions
    "AuthenticationError",
    "ValidationError", 
    "ForbiddenError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "ServerError",
    
    # Models
    "TracesQueryParams",
    "DatasetsQueryParams",
    "DatasetItemsQueryParams",
    "ScorerResultsQueryParams",
    "DatasetCreateRequest",
    "DatasetUpdateRequest",
    "DatasetVersionCreateRequest",
    "DatasetItemsCreateRequest",
    "ScorerResultCreateRequest",
    "ScorerResultUpdateRequest",
    "ScorerResultsBatchRequest",
]

__version__ = "0.1.0"
