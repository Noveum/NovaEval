"""
Noveum Platform API Client.

This package provides a Python client for interacting with the Noveum Platform
API. It includes authentication, request/response handling, and comprehensive
error handling for traces, datasets, and scorer results.

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
from .noveum_datasets_api import DatasetsAPI
from .noveum_scorer_results_api import ScorerResultsAPI
from .noveum_traces_api import TracesAPI

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
    # API Classes
    "DatasetsAPI",
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
    "ScorerResultsAPI",
    "ScorerResultsBatchRequest",
    "ScorerResultsQueryParams",
    "ServerError",
    "TracesAPI",
    # Models
    "TracesQueryParams",
    "ValidationError",
]

__version__ = "0.1.0"
