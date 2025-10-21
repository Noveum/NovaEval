# Datasets API Changes

This document outlines the breaking changes made to the datasets API methods in `src/novaeval/noveum_platform/` to align with the new OpenAPI specification.

## Overview

The datasets API has been redesigned around automatic versioning with `current_release` and `next_release` concepts. This eliminates the need for manual version management in most cases and simplifies the API surface.

## Breaking Changes

### 1. Dataset Items Methods

#### `add_dataset_items()` - **BREAKING CHANGE**
```python
# OLD
add_dataset_items(dataset_slug: str, version: str, items: list[dict[str, Any]])

# NEW
add_dataset_items(dataset_slug: str, items: list[dict[str, Any]])
```
- **Removed**: `version` parameter
- **Behavior**: Items are automatically added to the `next_release` version
- **Migration**: Remove the `version` parameter from all calls

#### `list_dataset_items()` - **ENHANCED**
```python
# OLD
list_dataset_items(dataset_slug: str, version: Optional[str] = None, 
                   limit: Optional[int] = None, offset: Optional[int] = None)

# NEW
list_dataset_items(dataset_slug: str, version: Optional[str] = None,
                   limit: Optional[int] = 50, offset: Optional[int] = 0,
                   item_type: Optional[str] = None, search: Optional[str] = None,
                   sort_by: Optional[str] = None, 
                   sort_order: Optional[Literal["asc", "desc"]] = "asc")
```
- **Added**: `item_type`, `search`, `sort_by`, `sort_order` parameters
- **Changed**: Default `limit` from `None` to `50`, `offset` from `None` to `0`
- **Migration**: No breaking changes, but new parameters available

#### `delete_all_dataset_items()` - **BREAKING CHANGE**
```python
# OLD
delete_all_dataset_items(dataset_slug: str, version: Optional[str] = None, 
                        item_ids: Optional[list[str]] = None)

# NEW
delete_all_dataset_items(dataset_slug: str, item_ids: list[str])
```
- **Removed**: `version` parameter
- **Changed**: `item_ids` is now required (not Optional)
- **Migration**: Remove `version` parameter and ensure `item_ids` is always provided

### 2. Version Management Methods

#### `publish_dataset_version()` - **BREAKING CHANGE**
```python
# OLD
publish_dataset_version(dataset_slug: str, version: str)

# NEW
publish_dataset_version(dataset_slug: str)
```
- **Removed**: `version` parameter
- **Behavior**: Automatically publishes the `next_release` and increments version
- **Migration**: Remove the `version` parameter from all calls

#### `list_dataset_versions()` - **ENHANCED**
```python
# OLD
list_dataset_versions(dataset_slug: str)

# NEW
list_dataset_versions(dataset_slug: str, limit: Optional[int] = 50, 
                     offset: Optional[int] = 0)
```
- **Added**: `limit` and `offset` parameters for pagination
- **Migration**: No breaking changes, but pagination now available

### 3. New Methods

#### `get_dataset_versions_diff()` - **NEW**
```python
get_dataset_versions_diff(dataset_slug: str) -> dict[str, Any]
```
- **Purpose**: Get detailed diff between `current_release` and `next_release`
- **Endpoint**: `GET /api/v1/datasets/{datasetSlug}/versions/diff`
- **Use Case**: Review changes before publishing

## Updated Pydantic Models

### `DatasetItemsQueryParams`
```python
# NEW FIELDS ADDED:
item_type: Optional[str] = Field(None, max_length=100, description="Filter by item type")
search: Optional[str] = Field(None, max_length=256, description="Search term for filtering items")
sort_by: Optional[str] = Field(None, pattern=r"^(scorer\.[a-zA-Z0-9_-]{1,64}|[a-zA-Z_][a-zA-Z0-9_]{0,63})$", description="Field to sort by")
sort_order: Optional[Literal["asc", "desc"]] = Field("asc", description="Sort order")

# UPDATED DEFAULTS:
limit: Optional[int] = Field(50, ge=1, le=1000, description="Number of items to return")
offset: Optional[int] = Field(0, ge=0, description="Number of items to skip")
```

### `DatasetItemsCreateRequest`
```python
# REMOVED:
version: str = Field(..., description="Dataset version")

# KEPT:
items: list[DatasetItem] = Field(..., min_length=1, description="Items to add")
```

## Migration Guide

### 1. Adding Items to Datasets
```python
# OLD
client.add_dataset_items("my-dataset", "1.0.0", items)

# NEW
client.add_dataset_items("my-dataset", items)
```

### 2. Publishing Versions
```python
# OLD
client.publish_dataset_version("my-dataset", "1.0.0")

# NEW
client.publish_dataset_version("my-dataset")
```

### 3. Deleting Items
```python
# OLD
client.delete_all_dataset_items("my-dataset", version="1.0.0", item_ids=["id1", "id2"])

# NEW
client.delete_all_dataset_items("my-dataset", ["id1", "id2"])
```

### 4. Enhanced Item Filtering
```python
# NEW CAPABILITIES
client.list_dataset_items(
    "my-dataset",
    item_type="conversation",
    search="user query",
    sort_by="created_at",
    sort_order="desc"
)
```

## API Behavior Changes

1. **Automatic Versioning**: Items are automatically added to `next_release` version
2. **Simplified Publishing**: No need to specify version when publishing
3. **Enhanced Filtering**: New search and sorting capabilities for dataset items
4. **Pagination**: All list methods now support proper pagination
5. **Version Diff**: New capability to review changes before publishing

## Files Modified

- `src/novaeval/noveum_platform/client.py` - Updated all dataset methods
- `src/novaeval/noveum_platform/models.py` - Updated Pydantic models

## Testing

All changes have been tested and validated against the new OpenAPI specification. The models validate correctly and all linter checks pass.
