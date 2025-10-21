"""
Unit tests for Noveum Platform API Pydantic models.
"""

import pytest
from pydantic import ValidationError

from novaeval.noveum_platform.models import (
    DatasetCreateRequest,
    DatasetItem,
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


class TestTracesQueryParams:
    """Test cases for TracesQueryParams model."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default values."""
        params = TracesQueryParams()

        assert params.organization_id is None
        assert params.from_ is None
        assert params.size == 20
        assert params.start_time is None
        assert params.end_time is None
        assert params.project is None
        assert params.environment is None
        assert params.status is None
        assert params.user_id is None
        assert params.session_id is None
        assert params.tags is None
        assert params.sort == "start_time:desc"
        assert params.search_term is None
        assert params.include_spans is False

    @pytest.mark.unit
    def test_init_with_values(self):
        """Test initialization with custom values."""
        params = TracesQueryParams(
            organization_id="org-123",
            from_=10,
            size=50,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-12-31T23:59:59Z",
            project="test-project",
            environment="production",
            status="success",
            user_id="user-123",
            session_id="session-456",
            tags=["tag1", "tag2"],
            sort="end_time:asc",
            search_term="test query",
            include_spans=True,
        )

        assert params.organization_id == "org-123"
        assert params.from_ == 10
        assert params.size == 50
        assert params.start_time == "2024-01-01T00:00:00Z"
        assert params.end_time == "2024-12-31T23:59:59Z"
        assert params.project == "test-project"
        assert params.environment == "production"
        assert params.status == "success"
        assert params.user_id == "user-123"
        assert params.session_id == "session-456"
        assert params.tags == ["tag1", "tag2"]
        assert params.sort == "end_time:asc"
        assert params.search_term == "test query"
        assert params.include_spans is True

    @pytest.mark.unit
    def test_size_validation_min(self):
        """Test size validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            TracesQueryParams(size=0)

        assert "greater than or equal to 1" in str(exc_info.value)

    @pytest.mark.unit
    def test_size_validation_max(self):
        """Test size validation maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            TracesQueryParams(size=101)

        assert "less than or equal to 100" in str(exc_info.value)

    @pytest.mark.unit
    def test_from_validation_min(self):
        """Test from_ validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            TracesQueryParams(from_=-1)

        assert "greater than or equal to 0" in str(exc_info.value)

    @pytest.mark.unit
    def test_sort_validation(self):
        """Test sort field validation with valid values."""
        valid_sorts = [
            "start_time:asc",
            "start_time:desc",
            "end_time:asc",
            "end_time:desc",
            "duration_ms:asc",
            "duration_ms:desc",
        ]

        for sort_value in valid_sorts:
            params = TracesQueryParams(sort=sort_value)
            assert params.sort == sort_value

    @pytest.mark.unit
    def test_sort_validation_invalid(self):
        """Test sort field validation with invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            TracesQueryParams(sort="invalid_sort")

        assert "Input should be" in str(exc_info.value)

    @pytest.mark.unit
    def test_to_query_params(self):
        """Test to_query_params method."""
        params = TracesQueryParams(
            organization_id="org-123",
            from_=10,
            size=50,
            project="test-project",
            tags=["tag1", "tag2"],
            include_spans=True,
        )

        result = params.to_query_params()

        expected = {
            "organization_id": "org-123",
            "from": 10,  # Note: alias conversion
            "size": 50,
            "project": "test-project",
            "tags": ["tag1", "tag2"],
            "include_spans": True,
            "sort": "start_time:desc",  # Default value is included
        }

        assert result == expected

    @pytest.mark.unit
    def test_to_query_params_excludes_none(self):
        """Test to_query_params excludes None values."""
        params = TracesQueryParams(organization_id="org-123")
        result = params.to_query_params()

        assert "organization_id" in result
        assert "from" not in result
        assert "start_time" not in result


class TestDatasetCreateRequest:
    """Test cases for DatasetCreateRequest model."""

    @pytest.mark.unit
    def test_init_required_fields(self):
        """Test initialization with only required fields."""
        request = DatasetCreateRequest(name="Test Dataset")

        assert request.name == "Test Dataset"
        assert request.slug is None
        assert request.description is None
        assert request.visibility == "org"
        assert request.dataset_type == "custom"
        assert request.environment is None
        assert request.schema_version is None
        assert request.tags is None
        assert request.custom_attributes is None

    @pytest.mark.unit
    def test_init_all_fields(self):
        """Test initialization with all fields."""
        request = DatasetCreateRequest(
            name="Test Dataset",
            slug="test-dataset",
            description="A test dataset",
            visibility="private",
            dataset_type="agent",
            environment="production",
            schema_version="1.0",
            tags=["test", "evaluation"],
            custom_attributes={"key": "value"},
        )

        assert request.name == "Test Dataset"
        assert request.slug == "test-dataset"
        assert request.description == "A test dataset"
        assert request.visibility == "private"
        assert request.dataset_type == "agent"
        assert request.environment == "production"
        assert request.schema_version == "1.0"
        assert request.tags == ["test", "evaluation"]
        assert request.custom_attributes == {"key": "value"}

    @pytest.mark.unit
    def test_name_validation_min_length(self):
        """Test name validation minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetCreateRequest(name="")

        assert "at least 1 character" in str(exc_info.value)

    @pytest.mark.unit
    def test_visibility_validation(self):
        """Test visibility field validation."""
        valid_visibilities = ["public", "org", "private"]

        for visibility in valid_visibilities:
            request = DatasetCreateRequest(name="Test", visibility=visibility)
            assert request.visibility == visibility

    @pytest.mark.unit
    def test_visibility_validation_invalid(self):
        """Test visibility field validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetCreateRequest(name="Test", visibility="invalid")

        assert "Input should be" in str(exc_info.value)

    @pytest.mark.unit
    def test_dataset_type_validation(self):
        """Test dataset_type field validation."""
        valid_types = ["agent", "conversational", "g-eval", "custom"]

        for dataset_type in valid_types:
            request = DatasetCreateRequest(name="Test", dataset_type=dataset_type)
            assert request.dataset_type == dataset_type

    @pytest.mark.unit
    def test_dataset_type_validation_invalid(self):
        """Test dataset_type field validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetCreateRequest(name="Test", dataset_type="invalid")

        assert "Input should be" in str(exc_info.value)


class TestDatasetUpdateRequest:
    """Test cases for DatasetUpdateRequest model."""

    @pytest.mark.unit
    def test_init_all_optional(self):
        """Test initialization with all fields optional."""
        request = DatasetUpdateRequest()

        assert request.name is None
        assert request.description is None
        assert request.visibility is None
        assert request.dataset_type is None
        assert request.environment is None
        assert request.schema_version is None
        assert request.tags is None
        assert request.custom_attributes is None

    @pytest.mark.unit
    def test_init_with_values(self):
        """Test initialization with some values."""
        request = DatasetUpdateRequest(
            name="Updated Dataset",
            description="Updated description",
            visibility="public",
        )

        assert request.name == "Updated Dataset"
        assert request.description == "Updated description"
        assert request.visibility == "public"
        assert request.dataset_type is None

    @pytest.mark.unit
    def test_name_validation_min_length(self):
        """Test name validation minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetUpdateRequest(name="")

        assert "at least 1 character" in str(exc_info.value)


class TestDatasetsQueryParams:
    """Test cases for DatasetsQueryParams model."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default values."""
        params = DatasetsQueryParams()

        assert params.limit == 20
        assert params.offset == 0
        assert params.visibility is None
        assert params.organizationSlug is None
        assert params.includeVersions is False

    @pytest.mark.unit
    def test_init_with_values(self):
        """Test initialization with custom values."""
        params = DatasetsQueryParams(
            limit=100,
            offset=50,
            visibility="public",
            organizationSlug="test-org",
            includeVersions=True,
        )

        assert params.limit == 100
        assert params.offset == 50
        assert params.visibility == "public"
        assert params.organizationSlug == "test-org"
        assert params.includeVersions is True

    @pytest.mark.unit
    def test_limit_validation_min(self):
        """Test limit validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetsQueryParams(limit=0)

        assert "greater than or equal to 1" in str(exc_info.value)

    @pytest.mark.unit
    def test_limit_validation_max(self):
        """Test limit validation maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetsQueryParams(limit=1001)

        assert "less than or equal to 1000" in str(exc_info.value)

    @pytest.mark.unit
    def test_offset_validation_min(self):
        """Test offset validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetsQueryParams(offset=-1)

        assert "greater than or equal to 0" in str(exc_info.value)

    @pytest.mark.unit
    def test_to_query_params(self):
        """Test to_query_params method."""
        params = DatasetsQueryParams(
            limit=50, offset=10, visibility="org", includeVersions=True
        )

        result = params.to_query_params()

        expected = {
            "limit": 50,
            "offset": 10,
            "visibility": "org",
            "includeVersions": True,
        }

        assert result == expected

    @pytest.mark.unit
    def test_to_query_params_excludes_none(self):
        """Test to_query_params excludes None values."""
        params = DatasetsQueryParams(limit=20)
        result = params.to_query_params()

        assert "limit" in result
        assert "visibility" not in result
        assert "organizationSlug" not in result


class TestDatasetVersionCreateRequest:
    """Test cases for DatasetVersionCreateRequest model."""

    @pytest.mark.unit
    def test_init_required_fields(self):
        """Test initialization with only required fields."""
        request = DatasetVersionCreateRequest(version="1.0.0")

        assert request.version == "1.0.0"
        assert request.description is None
        assert request.metadata is None

    @pytest.mark.unit
    def test_init_all_fields(self):
        """Test initialization with all fields."""
        request = DatasetVersionCreateRequest(
            version="2.0.0",
            description="Major update",
            metadata={"changelog": "Added new features"},
        )

        assert request.version == "2.0.0"
        assert request.description == "Major update"
        assert request.metadata == {"changelog": "Added new features"}

    @pytest.mark.unit
    def test_version_validation_min_length(self):
        """Test version validation minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetVersionCreateRequest(version="")

        assert "at least 1 character" in str(exc_info.value)


class TestDatasetItem:
    """Test cases for DatasetItem model."""

    @pytest.mark.unit
    def test_init_required_fields(self):
        """Test initialization with only required fields."""
        item = DatasetItem(
            item_key="item-1",
            item_type="question_answer",
            content={"question": "What is AI?", "answer": "Artificial Intelligence"},
        )

        assert item.item_key == "item-1"
        assert item.item_type == "question_answer"
        assert item.content == {
            "question": "What is AI?",
            "answer": "Artificial Intelligence",
        }
        assert item.metadata is None
        assert item.trace_id is None
        assert item.span_id is None

    @pytest.mark.unit
    def test_init_all_fields(self):
        """Test initialization with all fields."""
        item = DatasetItem(
            item_key="item-1",
            item_type="question_answer",
            content={"question": "What is AI?", "answer": "Artificial Intelligence"},
            metadata={"difficulty": "easy"},
            trace_id="trace-123",
            span_id="span-456",
        )

        assert item.item_key == "item-1"
        assert item.item_type == "question_answer"
        assert item.content == {
            "question": "What is AI?",
            "answer": "Artificial Intelligence",
        }
        assert item.metadata == {"difficulty": "easy"}
        assert item.trace_id == "trace-123"
        assert item.span_id == "span-456"

    @pytest.mark.unit
    def test_item_key_validation_min_length(self):
        """Test item_key validation minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetItem(item_key="", item_type="test", content={})

        assert "at least 1 character" in str(exc_info.value)

    @pytest.mark.unit
    def test_item_type_validation_min_length(self):
        """Test item_type validation minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetItem(item_key="test", item_type="", content={})

        assert "at least 1 character" in str(exc_info.value)


class TestDatasetItemsCreateRequest:
    """Test cases for DatasetItemsCreateRequest model."""

    @pytest.mark.unit
    def test_init_required_fields(self):
        """Test initialization with required fields."""
        items = [
            DatasetItem(item_key="item-1", item_type="test", content={}),
            DatasetItem(item_key="item-2", item_type="test", content={}),
        ]
        request = DatasetItemsCreateRequest(items=items)

        assert len(request.items) == 2
        items_list = list(request.items)
        assert items_list[0].item_key == "item-1"
        assert items_list[1].item_key == "item-2"

    @pytest.mark.unit
    def test_items_validation_min_items(self):
        """Test items validation minimum items."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetItemsCreateRequest(items=[])

        assert "at least 1 item" in str(exc_info.value)


class TestDatasetItemsQueryParams:
    """Test cases for DatasetItemsQueryParams model."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default values."""
        params = DatasetItemsQueryParams()

        assert params.version is None
        assert params.limit == 50
        assert params.offset == 0
        assert params.item_type is None
        assert params.search is None
        assert params.sort_by is None
        assert params.sort_order == "asc"

    @pytest.mark.unit
    def test_init_with_values(self):
        """Test initialization with custom values."""
        params = DatasetItemsQueryParams(
            version="1.0.0",
            limit=25,
            offset=10,
            item_type="conversation",
            search="test query",
            sort_by="created_at",
            sort_order="desc",
        )

        assert params.version == "1.0.0"
        assert params.limit == 25
        assert params.offset == 10
        assert params.item_type == "conversation"
        assert params.search == "test query"
        assert params.sort_by == "created_at"
        assert params.sort_order == "desc"

    @pytest.mark.unit
    def test_limit_validation_min(self):
        """Test limit validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetItemsQueryParams(limit=0)

        assert "greater than or equal to 1" in str(exc_info.value)

    @pytest.mark.unit
    def test_limit_validation_max(self):
        """Test limit validation maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetItemsQueryParams(limit=1001)

        assert "less than or equal to 1000" in str(exc_info.value)

    @pytest.mark.unit
    def test_offset_validation_min(self):
        """Test offset validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetItemsQueryParams(offset=-1)

        assert "greater than or equal to 0" in str(exc_info.value)

    @pytest.mark.unit
    def test_to_query_params(self):
        """Test to_query_params method."""
        params = DatasetItemsQueryParams(
            version="1.0.0",
            limit=25,
            offset=10,
            item_type="conversation",
            search="test",
            sort_by="created_at",
            sort_order="desc",
        )

        result = params.to_query_params()

        expected = {
            "version": "1.0.0",
            "limit": 25,
            "offset": 10,
            "item_type": "conversation",
            "search": "test",
            "sort_by": "created_at",
            "sort_order": "desc",
        }

        assert result == expected

    @pytest.mark.unit
    def test_item_type_validation_max_length(self):
        """Test item_type validation maximum length."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetItemsQueryParams(item_type="x" * 101)

        assert "at most 100 characters" in str(exc_info.value)

    @pytest.mark.unit
    def test_search_validation_max_length(self):
        """Test search validation maximum length."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetItemsQueryParams(search="x" * 257)

        assert "at most 256 characters" in str(exc_info.value)

    @pytest.mark.unit
    def test_sort_by_validation_pattern(self):
        """Test sort_by validation pattern."""
        # Valid patterns
        valid_sorts = ["created_at", "item_key", "scorer.accuracy", "scorer.confidence"]
        for sort_val in valid_sorts:
            params = DatasetItemsQueryParams(sort_by=sort_val)
            assert params.sort_by == sort_val

        # Invalid patterns
        with pytest.raises(ValidationError) as exc_info:
            DatasetItemsQueryParams(sort_by="invalid-sort")

        assert "String should match pattern" in str(exc_info.value)

    @pytest.mark.unit
    def test_sort_order_validation(self):
        """Test sort_order validation."""
        valid_orders = ["asc", "desc"]
        for order in valid_orders:
            params = DatasetItemsQueryParams(sort_order=order)
            assert params.sort_order == order

        with pytest.raises(ValidationError) as exc_info:
            DatasetItemsQueryParams(sort_order="invalid")

        assert "Input should be" in str(exc_info.value)


class TestScorerResultCreateRequest:
    """Test cases for ScorerResultCreateRequest model."""

    @pytest.mark.unit
    def test_init_required_fields(self):
        """Test initialization with only required fields."""
        request = ScorerResultCreateRequest(
            datasetSlug="test-dataset", itemId="item-1", scorerId="accuracy-scorer"
        )

        assert request.datasetSlug == "test-dataset"
        assert request.itemId == "item-1"
        assert request.scorerId == "accuracy-scorer"
        assert request.score is None
        assert request.metadata is None
        assert request.details is None

    @pytest.mark.unit
    def test_init_all_fields(self):
        """Test initialization with all fields."""
        request = ScorerResultCreateRequest(
            datasetSlug="test-dataset",
            itemId="item-1",
            scorerId="accuracy-scorer",
            score=0.95,
            metadata={"confidence": 0.9},
            details={"correct_answer": True},
        )

        assert request.datasetSlug == "test-dataset"
        assert request.itemId == "item-1"
        assert request.scorerId == "accuracy-scorer"
        assert request.score == 0.95
        assert request.metadata == {"confidence": 0.9}
        assert request.details == {"correct_answer": True}


class TestScorerResultUpdateRequest:
    """Test cases for ScorerResultUpdateRequest model."""

    @pytest.mark.unit
    def test_init_all_optional(self):
        """Test initialization with all fields optional."""
        request = ScorerResultUpdateRequest()

        assert request.score is None
        assert request.metadata is None
        assert request.details is None

    @pytest.mark.unit
    def test_init_with_values(self):
        """Test initialization with some values."""
        request = ScorerResultUpdateRequest(score=0.87, metadata={"updated": True})

        assert request.score == 0.87
        assert request.metadata == {"updated": True}
        assert request.details is None


class TestScorerResultsQueryParams:
    """Test cases for ScorerResultsQueryParams model."""

    @pytest.mark.unit
    def test_init_required_fields(self):
        """Test initialization with required organizationSlug."""
        params = ScorerResultsQueryParams(organizationSlug="test-org")

        assert params.organizationSlug == "test-org"
        assert params.datasetSlug is None
        assert params.itemId is None
        assert params.scorerId is None
        assert params.limit == 100
        assert params.offset == 0

    @pytest.mark.unit
    def test_init_all_fields(self):
        """Test initialization with all fields."""
        params = ScorerResultsQueryParams(
            organizationSlug="test-org",
            datasetSlug="test-dataset",
            itemId="item-1",
            scorerId="accuracy-scorer",
            limit=50,
            offset=25,
        )

        assert params.organizationSlug == "test-org"
        assert params.datasetSlug == "test-dataset"
        assert params.itemId == "item-1"
        assert params.scorerId == "accuracy-scorer"
        assert params.limit == 50
        assert params.offset == 25

    @pytest.mark.unit
    def test_organization_slug_required(self):
        """Test organizationSlug is required."""
        with pytest.raises(ValidationError) as exc_info:
            ScorerResultsQueryParams()

        assert "Field required" in str(exc_info.value)

    @pytest.mark.unit
    def test_limit_validation_min(self):
        """Test limit validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            ScorerResultsQueryParams(organizationSlug="test", limit=0)

        assert "greater than or equal to 1" in str(exc_info.value)

    @pytest.mark.unit
    def test_limit_validation_max(self):
        """Test limit validation maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            ScorerResultsQueryParams(organizationSlug="test", limit=1001)

        assert "less than or equal to 1000" in str(exc_info.value)

    @pytest.mark.unit
    def test_offset_validation_min(self):
        """Test offset validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            ScorerResultsQueryParams(organizationSlug="test", offset=-1)

        assert "greater than or equal to 0" in str(exc_info.value)

    @pytest.mark.unit
    def test_to_query_params(self):
        """Test to_query_params method."""
        params = ScorerResultsQueryParams(
            organizationSlug="test-org", datasetSlug="test-dataset", limit=50, offset=10
        )

        result = params.to_query_params()

        expected = {
            "organizationSlug": "test-org",
            "datasetSlug": "test-dataset",
            "limit": 50,
            "offset": 10,
        }

        assert result == expected


class TestScorerResultsBatchRequest:
    """Test cases for ScorerResultsBatchRequest model."""

    @pytest.mark.unit
    def test_init_required_fields(self):
        """Test initialization with required fields."""
        results = [
            ScorerResultCreateRequest(
                datasetSlug="test-dataset", itemId="item-1", scorerId="scorer-1"
            ),
            ScorerResultCreateRequest(
                datasetSlug="test-dataset", itemId="item-2", scorerId="scorer-1"
            ),
        ]
        request = ScorerResultsBatchRequest(results=results)

        assert len(request.results) == 2
        results_list = list(request.results)
        assert results_list[0].itemId == "item-1"
        assert results_list[1].itemId == "item-2"

    @pytest.mark.unit
    def test_results_validation_min_items(self):
        """Test results validation minimum items."""
        with pytest.raises(ValidationError) as exc_info:
            ScorerResultsBatchRequest(results=[])

        assert "at least 1 item" in str(exc_info.value)
