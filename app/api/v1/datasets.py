"""
Dataset Operations API endpoints for NovaEval.

This module provides REST endpoints for dataset loading, querying, and sampling
with efficient pagination and memory management.
"""

import asyncio
import os
import random
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from app.core.discovery import get_registry
from app.core.logging import get_logger
from app.schemas.datasets import (
    DatasetInfo,
    DatasetInstantiateRequest,
    DatasetLoadRequest,
    DatasetLoadResponse,
    DatasetQueryRequest,
    DatasetQueryResponse,
    DatasetSample,
    DatasetSampleRequest,
    DatasetSampleResponse,
)
from novaeval.config.job_config import DatasetFactory
from novaeval.config.schema import DatasetConfig, DatasetType

# Create a dedicated thread pool executor for dataset operations
# This prevents thread exhaustion under high load and provides better resource control
DATASET_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("DATASET_THREAD_POOL_SIZE", "10")),
    thread_name_prefix="dataset-worker",
)

router = APIRouter()


async def get_dataset_config_by_name(dataset_name: str) -> DatasetConfig:
    """
    Get dataset configuration by name from discovered datasets.

    Args:
        dataset_name: Name of the dataset to get configuration for

    Returns:
        DatasetConfig object with default configuration

    Raises:
        HTTPException: If dataset not found
    """
    registry = await get_registry()
    datasets = await registry.get_datasets()

    if dataset_name not in datasets:
        logger = get_logger(__name__)
        logger.warning(
            f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    # Create basic dataset config with common defaults
    # In a real implementation, this might come from a configuration store
    type_map = {
        "mmlu": DatasetType.MMLU,
        "huggingface": DatasetType.HUGGINGFACE,
        "custom": DatasetType.CUSTOM,
        "json": DatasetType.JSON,
        "csv": DatasetType.CSV,
        "jsonl": DatasetType.JSONL,
    }

    # Determine dataset type - try exact match first, then pattern matching
    dataset_type = type_map.get(dataset_name)

    if dataset_type is None:
        # Try pattern-based matching for more flexible type detection
        name_lower = dataset_name.lower()
        if "mmlu" in name_lower:
            dataset_type = DatasetType.MMLU
        elif "huggingface" in name_lower or "hf" in name_lower:
            dataset_type = DatasetType.HUGGINGFACE
        elif name_lower.endswith(".json"):
            dataset_type = DatasetType.JSON
        elif name_lower.endswith(".csv"):
            dataset_type = DatasetType.CSV
        elif name_lower.endswith(".jsonl"):
            dataset_type = DatasetType.JSONL
        else:
            # Default to CUSTOM for unknown types
            dataset_type = DatasetType.CUSTOM

    # Set name field appropriately based on dataset type
    # For HuggingFace datasets, the name should be the dataset identifier
    # For file-based datasets, name can be the dataset_name
    # For others, set to None to use defaults
    dataset_name_field = None
    if dataset_type in [
        DatasetType.HUGGINGFACE,
        DatasetType.JSON,
        DatasetType.CSV,
        DatasetType.JSONL,
    ]:
        dataset_name_field = dataset_name

    return DatasetConfig(
        type=dataset_type,
        name=dataset_name_field,
        path=None,  # Would be set based on dataset type
        split="test",
        limit=None,
        shuffle=False,
        seed=42,
    )


@router.get("/{dataset_name}/info", response_model=DatasetInfo)
async def get_dataset_info(dataset_name: str):
    """
    Get information about a specific dataset.

    Args:
        dataset_name: Name of the dataset to get info for

    Returns:
        Dataset information including metadata
    """
    try:
        dataset_config = await get_dataset_config_by_name(dataset_name)

        # Create dataset instance to get info
        dataset_instance = DatasetFactory.create_dataset(dataset_config)
        info = dataset_instance.get_info()

        return DatasetInfo(
            name=info.get("name", dataset_name),
            dataset_type=info.get("type", dataset_config.type.value),
            split=info.get("split", dataset_config.split),
            num_samples=info.get("num_samples", len(dataset_instance)),
            seed=info.get("seed", dataset_config.seed or 42),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting dataset info: {e!s}"
        )


@router.post("/{dataset_name}/load", response_model=DatasetLoadResponse)
async def load_dataset(dataset_name: str, request: DatasetLoadRequest):
    """
    Load a dataset with custom configuration.

    Args:
        dataset_name: Name of the dataset to load
        request: Dataset loading request with configuration

    Returns:
        Dataset loading response with status
    """
    try:
        # Create dataset instance with provided config
        dataset_instance = DatasetFactory.create_dataset(request.config)

        # Load data in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(DATASET_EXECUTOR, dataset_instance.load_data)

        info = dataset_instance.get_info()

        return DatasetLoadResponse(
            dataset_info=DatasetInfo(
                name=info.get("name", dataset_name),
                dataset_type=info.get("type", request.config.type.value),
                split=info.get("split", request.config.split),
                num_samples=info.get("num_samples", len(dataset_instance)),
                seed=info.get("seed", request.config.seed or 42),
            ),
            loaded=True,
            message=f"Dataset '{dataset_name}' loaded successfully with {len(dataset_instance)} samples",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {e!s}")


@router.post("/{dataset_name}/query", response_model=DatasetQueryResponse)
async def query_dataset(
    dataset_name: str, request: DatasetQueryRequest = DatasetQueryRequest()
):
    """
    Query dataset with pagination and filtering.

    Args:
        dataset_name: Name of the dataset to query
        request: Query request with pagination and filtering parameters

    Returns:
        Dataset query response with paginated samples
    """
    try:
        # Get dataset configuration and create instance
        base_config = await get_dataset_config_by_name(dataset_name)

        # Create a copy and override with request parameters
        config_dict = base_config.model_dump()
        if request.shuffle is not None:
            config_dict["shuffle"] = request.shuffle
        if request.seed is not None:
            config_dict["seed"] = request.seed

        dataset_config = DatasetConfig(**config_dict)
        dataset_instance = DatasetFactory.create_dataset(dataset_config)

        # Load data in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(DATASET_EXECUTOR, dataset_instance.load_data)

        # Get total count
        total_samples = len(dataset_instance)

        # Apply pagination
        start_idx = request.offset or 0
        end_idx = min(start_idx + (request.limit or 50), total_samples)

        # Get samples with indices
        samples = []
        for i in range(start_idx, end_idx):
            sample_data = dataset_instance.get_sample(i)
            samples.append(DatasetSample(index=i, data=sample_data))

        # Pagination info
        pagination = {
            "total": total_samples,
            "limit": request.limit or 50,
            "offset": start_idx,
            "has_more": end_idx < total_samples,
            "next_offset": end_idx if end_idx < total_samples else None,
        }

        info = dataset_instance.get_info()

        return DatasetQueryResponse(
            samples=samples,
            dataset_info=DatasetInfo(
                name=info.get("name", dataset_name),
                dataset_type=info.get("type", dataset_config.type.value),
                split=info.get("split", dataset_config.split),
                num_samples=total_samples,
                seed=info.get("seed", dataset_config.seed or 42),
            ),
            pagination=pagination,
            count=len(samples),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying dataset: {e!s}")


@router.post("/{dataset_name}/sample", response_model=DatasetSampleResponse)
async def sample_dataset(dataset_name: str, request: DatasetSampleRequest):
    """
    Sample data from a dataset using various sampling methods.

    Args:
        dataset_name: Name of the dataset to sample from
        request: Sampling request with method and parameters

    Returns:
        Dataset sampling response with selected samples
    """
    try:
        # Get dataset configuration and create instance
        base_config = await get_dataset_config_by_name(dataset_name)

        # Create a copy and set seed for reproducible sampling
        config_dict = base_config.model_dump()
        if request.seed is not None:
            config_dict["seed"] = request.seed

        dataset_config = DatasetConfig(**config_dict)
        dataset_instance = DatasetFactory.create_dataset(dataset_config)

        # Load data in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(DATASET_EXECUTOR, dataset_instance.load_data)

        total_samples = len(dataset_instance)

        if request.count > total_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Requested count ({request.count}) exceeds available samples ({total_samples})",
            )

        # Generate sample indices based on method
        if request.method == "random":
            rng = random.Random()
            if request.seed is not None:
                rng.seed(request.seed)
            indices = rng.sample(range(total_samples), request.count)
        elif request.method == "first":
            indices = list(range(request.count))
        elif request.method == "last":
            indices = list(range(total_samples - request.count, total_samples))
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid sampling method: {request.method}"
            )

        # Get samples
        samples = []
        for idx in indices:
            sample_data = dataset_instance.get_sample(idx)
            samples.append(DatasetSample(index=idx, data=sample_data))

        # Sampling info
        sampling_info = {
            "method": request.method,
            "count": request.count,
            "seed": request.seed,
            "indices": indices,
        }

        info = dataset_instance.get_info()

        return DatasetSampleResponse(
            samples=samples,
            dataset_info=DatasetInfo(
                name=info.get("name", dataset_name),
                dataset_type=info.get("type", dataset_config.type.value),
                split=info.get("split", dataset_config.split),
                num_samples=total_samples,
                seed=info.get("seed", dataset_config.seed or 42),
            ),
            sampling_info=sampling_info,
            count=len(samples),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling dataset: {e!s}")


@router.post("/instantiate", response_model=DatasetInfo)
async def instantiate_dataset(request: DatasetInstantiateRequest):
    """
    Instantiate a dataset with custom configuration.

    Args:
        request: Dataset instantiation request with custom configuration

    Returns:
        Dataset information for the instantiated dataset
    """
    try:
        dataset_instance = DatasetFactory.create_dataset(request.config)

        # Load data to get accurate info
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(DATASET_EXECUTOR, dataset_instance.load_data)

        info = dataset_instance.get_info()

        return DatasetInfo(
            name=info.get("name", request.config.name or request.dataset_type),
            dataset_type=info.get("type", request.config.type.value),
            split=info.get("split", request.config.split),
            num_samples=info.get("num_samples", len(dataset_instance)),
            seed=info.get("seed", request.config.seed or 42),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error instantiating dataset: {e!s}"
        )
