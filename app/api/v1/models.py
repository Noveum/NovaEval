"""
Model Operations API endpoints for NovaEval.

This module provides REST endpoints for model prediction operations
including single and batch predictions with proper error handling.
"""

import asyncio
import time

from fastapi import APIRouter, HTTPException

from app.core.discovery import get_registry
from app.schemas.models import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfo,
    ModelInstantiateRequest,
    PredictRequest,
    PredictResponse,
)
from novaeval.config.job_config import ModelFactory
from novaeval.config.schema import ModelConfig, ModelProvider

router = APIRouter()


class ModelOperationError(Exception):
    """Custom exception for model operation errors."""

    pass


async def get_model_config_by_name(model_name: str) -> ModelConfig:
    """
    Get model configuration by name from discovered models.

    Args:
        model_name: Name of the model to get configuration for

    Returns:
        ModelConfig object with default configuration

    Raises:
        HTTPException: If model not found
    """
    registry = await get_registry()
    models = await registry.get_models()

    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {list(models.keys())}",
        )

    # Create basic model config with common defaults
    # In a real implementation, this might come from a configuration store
    provider_map = {
        "openai": ModelProvider.OPENAI,
        "anthropic": ModelProvider.ANTHROPIC,
        "azure_openai": ModelProvider.AZURE_OPENAI,
        "gemini": ModelProvider.GOOGLE_VERTEX,
    }

    if model_name not in provider_map:
        raise HTTPException(
            status_code=400,
            detail=f"Provider mapping not found for model '{model_name}'",
        )

    return ModelConfig(
        provider=provider_map[model_name],
        model_name=model_name,
        temperature=0.0,
        max_tokens=1000,
        timeout=60,
        retry_attempts=3,
    )


@router.get("/{model_name}/info", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model to get info for

    Returns:
        Model information including metadata (without instantiating the model)
    """
    try:
        model_config = await get_model_config_by_name(model_name)

        # Return static information without instantiating the model
        return ModelInfo(
            name=model_name,
            identifier=model_config.model_name,
            provider=model_config.provider.value,
            total_requests=0,  # No requests yet since we're not instantiating
            total_tokens=0,
            total_cost=0.0,
            errors=[],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {e!s}")


@router.post("/{model_name}/predict", response_model=PredictResponse)
async def predict(model_name: str, request: PredictRequest):
    """
    Generate a single prediction using the specified model.

    Args:
        model_name: Name of the model to use for prediction
        request: Prediction request with prompt and parameters

    Returns:
        Prediction response with generated text and metadata
    """
    start_time = time.time()

    try:
        # Get model configuration and create model instance
        model_config = await get_model_config_by_name(model_name)

        # Override config with request parameters if provided
        if request.max_tokens is not None:
            model_config.max_tokens = request.max_tokens
        if request.temperature is not None:
            model_config.temperature = request.temperature

        model_instance = ModelFactory.create_model(model_config)

        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            None,
            lambda: model_instance.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop,
                **request.additional_params,
            ),
        )

        processing_time = time.time() - start_time

        return PredictResponse(
            prediction=prediction,
            inference_details=model_instance.get_info(),
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e!s}")


@router.post("/{model_name}/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(model_name: str, request: BatchPredictRequest):
    """
    Generate batch predictions using the specified model.

    Args:
        model_name: Name of the model to use for predictions
        request: Batch prediction request with prompts and parameters

    Returns:
        Batch prediction response with generated texts and metadata
    """
    start_time = time.time()

    try:
        # Get model configuration and create model instance
        model_config = await get_model_config_by_name(model_name)

        # Override config with request parameters if provided
        if request.max_tokens is not None:
            model_config.max_tokens = request.max_tokens
        if request.temperature is not None:
            model_config.temperature = request.temperature

        model_instance = ModelFactory.create_model(model_config)

        # Run batch prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            None,
            lambda: model_instance.generate_batch(
                prompts=request.prompts,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop,
                **request.additional_params,
            ),
        )

        processing_time = time.time() - start_time

        return BatchPredictResponse(
            predictions=predictions,
            inference_details=model_instance.get_info(),
            processing_time=processing_time,
            count=len(predictions),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during batch prediction: {e!s}"
        )


@router.post("/instantiate", response_model=ModelInfo)
async def instantiate_model(request: ModelInstantiateRequest):
    """
    Instantiate a model with custom configuration.

    Args:
        request: Model instantiation request with custom configuration

    Returns:
        Model information for the instantiated model
    """
    try:
        model_instance = ModelFactory.create_model(request.config)
        info = model_instance.get_info()

        return ModelInfo(
            name=info.get("name", request.config.model_name),
            identifier=info.get("model_name", request.config.model_name),
            provider=request.config.provider.value,
            total_requests=info.get("total_requests", 0),
            total_tokens=info.get("total_tokens", 0),
            total_cost=info.get("total_cost", 0.0),
            errors=info.get("errors", []),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error instantiating model: {e!s}")


# Note: Streaming support could be added here in the future
# @router.post("/{model_name}/predict/stream")
# async def predict_stream(model_name: str, request: PredictRequest):
#     """Stream prediction results for long-running predictions."""
#     # Implementation would depend on model streaming capabilities
