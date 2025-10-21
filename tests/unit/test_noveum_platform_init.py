"""
Unit tests for novaeval.noveum_platform.__init__.py module.

Tests the noveum_platform module's import functionality and __all__ exports.
"""

import pytest


@pytest.mark.unit
def test_imports():
    """Test that all classes can be imported from the noveum_platform module."""
    from novaeval.noveum_platform import (
        NoveumClient,
        AuthenticationError,
        ConflictError,
        ForbiddenError,
        NotFoundError,
        NoveumAPIError,
        RateLimitError,
        ServerError,
        ValidationError,
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

    # Verify all classes are imported correctly
    assert NoveumClient is not None
    assert AuthenticationError is not None
    assert ConflictError is not None
    assert ForbiddenError is not None
    assert NotFoundError is not None
    assert NoveumAPIError is not None
    assert RateLimitError is not None
    assert ServerError is not None
    assert ValidationError is not None
    assert DatasetCreateRequest is not None
    assert DatasetItemsCreateRequest is not None
    assert DatasetItemsQueryParams is not None
    assert DatasetsQueryParams is not None
    assert DatasetUpdateRequest is not None
    assert DatasetVersionCreateRequest is not None
    assert ScorerResultCreateRequest is not None
    assert ScorerResultsBatchRequest is not None
    assert ScorerResultsQueryParams is not None
    assert ScorerResultUpdateRequest is not None
    assert TracesQueryParams is not None


@pytest.mark.unit
def test_all_exports():
    """Test that __all__ contains the expected exports."""
    from novaeval.noveum_platform import __all__

    expected_exports = [
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
        "NoveumAPIError",
        "NoveumClient",
        "RateLimitError",
        "ScorerResultCreateRequest",
        "ScorerResultUpdateRequest",
        "ScorerResultsBatchRequest",
        "ScorerResultsQueryParams",
        "ServerError",
        "TracesQueryParams",
        "ValidationError",
    ]

    # Check that all expected exports are present
    for export in expected_exports:
        assert export in __all__, f"Expected export '{export}' not found in __all__"

    # Check that __all__ has the expected length
    assert len(__all__) == len(expected_exports)


@pytest.mark.unit
def test_star_import():
    """Test that star import works correctly."""
    import novaeval.noveum_platform as noveum_platform_module

    # Test that we can access all exported items
    for item_name in noveum_platform_module.__all__:
        assert hasattr(noveum_platform_module, item_name)
        item = getattr(noveum_platform_module, item_name)
        assert item is not None


@pytest.mark.unit
def test_direct_imports():
    """Test that classes can be imported directly from submodules."""
    # Test client import
    from novaeval.noveum_platform.client import NoveumClient
    assert NoveumClient is not None

    # Test exceptions import
    from novaeval.noveum_platform.exceptions import (
        AuthenticationError,
        ConflictError,
        ForbiddenError,
        NotFoundError,
        NoveumAPIError,
        RateLimitError,
        ServerError,
        ValidationError,
    )
    assert AuthenticationError is not None
    assert ConflictError is not None
    assert ForbiddenError is not None
    assert NotFoundError is not None
    assert NoveumAPIError is not None
    assert RateLimitError is not None
    assert ServerError is not None
    assert ValidationError is not None

    # Test models import
    from novaeval.noveum_platform.models import (
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
    assert DatasetCreateRequest is not None
    assert DatasetItemsCreateRequest is not None
    assert DatasetItemsQueryParams is not None
    assert DatasetsQueryParams is not None
    assert DatasetUpdateRequest is not None
    assert DatasetVersionCreateRequest is not None
    assert ScorerResultCreateRequest is not None
    assert ScorerResultsBatchRequest is not None
    assert ScorerResultsQueryParams is not None
    assert ScorerResultUpdateRequest is not None
    assert TracesQueryParams is not None


@pytest.mark.unit
def test_version_attribute():
    """Test that the module has a version attribute."""
    from novaeval.noveum_platform import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


@pytest.mark.unit
def test_import_without_load_dotenv():
    """Test that the module can be imported without calling load_dotenv() first."""
    # This test ensures that removing load_dotenv() from __init__.py doesn't break imports
    # The module should still be importable, but environment variables won't be loaded
    # unless load_dotenv() is called explicitly or client.py is imported
    
    # Import the module without calling load_dotenv() first
    from novaeval.noveum_platform import NoveumClient
    
    # The import should succeed
    assert NoveumClient is not None
    
    # Note: This test doesn't verify that environment variables are loaded,
    # just that the import doesn't fail due to the removal of load_dotenv()
