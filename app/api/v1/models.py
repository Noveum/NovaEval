"""
Model Operations API endpoints for NovaEval.

This module provides REST endpoints for model prediction operations
including single and batch predictions with proper error handling.
"""

import asyncio
import time

from fastapi import APIRouter, HTTPException

from app.core.discovery import get_registry
from app.core.logging import get_logger
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
        logger = get_logger(__name__)
        logger.warning(
            f"Model '{model_name}' not found. Available: {list(models.keys())}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found",
        )

    # Get provider from the model metadata dynamically
    # This allows any registered model to work, not just hardcoded ones
    model_metadata = models[model_name]
    provider = _get_provider_from_metadata(model_name, model_metadata)

    return ModelConfig(
        provider=provider,
        model_name=model_name,
        temperature=0.0,
        max_tokens=1000,
        timeout=60,
        retry_attempts=3,
    )


def _get_provider_from_metadata(model_name: str, metadata) -> ModelProvider:
    """
    Dynamically determine the provider from the model metadata.

    Args:
        model_name: Name of the model
        metadata: ComponentMetadata object from the registry

    Returns:
        ModelProvider enum value
    """
    # First try to infer from the module path
    module_path = metadata.module_path.lower()

    if "openai" in module_path:
        return ModelProvider.OPENAI
    elif "anthropic" in module_path:
        return ModelProvider.ANTHROPIC
    elif "azure" in module_path:
        return ModelProvider.AZURE_OPENAI
    elif "gemini" in module_path or "google" in module_path:
        return ModelProvider.GOOGLE_VERTEX

    # Fallback to entry point name mapping for known models
    name_to_provider = {
        "openai": ModelProvider.OPENAI,
        "anthropic": ModelProvider.ANTHROPIC,
        "azure_openai": ModelProvider.AZURE_OPENAI,
        "gemini": ModelProvider.GOOGLE_VERTEX,
    }

    if model_name in name_to_provider:
        return name_to_provider[model_name]

    # Final fallback: try to infer from model name patterns
    model_name_lower = model_name.lower()
    if "openai" in model_name_lower:
        return ModelProvider.OPENAI
    elif "anthropic" in model_name_lower or "claude" in model_name_lower:
        return ModelProvider.ANTHROPIC
    elif "azure" in model_name_lower:
        return ModelProvider.AZURE_OPENAI
    elif "gemini" in model_name_lower or "google" in model_name_lower:
        return ModelProvider.GOOGLE_VERTEX
    else:
        # Default to OpenAI for backward compatibility
        return ModelProvider.OPENAI


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
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
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
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error instantiating model: {e!s}")


# Note: Streaming support could be added here in the future
# @router.post("/{model_name}/predict/stream")
# async def predict_stream(model_name: str, request: PredictRequest):
#     """Stream prediction results for long-running predictions."""
#     # Implementation would depend on model streaming capabilities
