"""
Scorer Operations API endpoints for NovaEval.

This module provides REST endpoints for scoring operations including
single and batch scoring with proper handling of model-dependent scorers.
"""

import asyncio
import time

from fastapi import APIRouter, HTTPException

from app.core.discovery import get_registry
from app.core.logging import get_logger
from app.schemas.scorers import (
    BatchScoringRequest,
    BatchScoringResponse,
    ScorerInfo,
    ScorerInstantiateRequest,
    ScorerStatsResponse,
    ScoringRequest,
    ScoringResponse,
)
from novaeval.config.job_config import ModelFactory, ScorerFactory
from novaeval.config.schema import ModelConfig, ModelProvider, ScorerConfig, ScorerType

router = APIRouter()


# Model-dependent scorer types that require a model instance
MODEL_DEPENDENT_SCORERS = {
    ScorerType.G_EVAL,
    ScorerType.RAG_ANSWER_RELEVANCY,
    ScorerType.RAG_FAITHFULNESS,
    ScorerType.RAGAS,
    ScorerType.CONVERSATIONAL_METRICS,
}


async def get_scorer_config_by_name(scorer_name: str) -> ScorerConfig:
    """
    Get scorer configuration by name from discovered scorers.

    Args:
        scorer_name: Name of the scorer to get configuration for

    Returns:
        ScorerConfig object with default configuration

    Raises:
        HTTPException: If scorer not found
    """
    registry = await get_registry()
    scorers = await registry.get_scorers()

    if scorer_name not in scorers:
        logger = get_logger(__name__)
        logger.warning(
            f"Scorer '{scorer_name}' not found. Available: {list(scorers.keys())}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Scorer '{scorer_name}' not found",
        )

    # Get scorer type from the scorer metadata dynamically
    # This allows any registered scorer to work, not just hardcoded ones
    scorer_metadata = scorers[scorer_name]
    scorer_type = _get_scorer_type_from_metadata(scorer_name, scorer_metadata)

    return ScorerConfig(type=scorer_type, name=scorer_name, threshold=0.7, weight=1.0)


def _get_scorer_type_from_metadata(scorer_name: str, metadata) -> ScorerType:
    """
    Dynamically determine the scorer type from the scorer metadata.

    Args:
        scorer_name: Name of the scorer
        metadata: ComponentMetadata object from the registry

    Returns:
        ScorerType enum value
    """
    # First try to infer from the module path and class name
    module_path = metadata.module_path.lower()
    class_name = metadata.class_name.lower()

    # Infer type from module path
    if "accuracy" in module_path:
        return ScorerType.ACCURACY
    elif "g_eval" in module_path or "geval" in class_name:
        return ScorerType.G_EVAL
    elif "conversational" in module_path:
        return ScorerType.CONVERSATIONAL_METRICS
    elif "rag" in module_path:
        # For RAG scorers, look at the specific class name
        if "answer" in class_name and "relevancy" in class_name:
            return ScorerType.RAG_ANSWER_RELEVANCY
        elif "faithfulness" in class_name:
            return ScorerType.RAG_FAITHFULNESS
        elif "precision" in class_name:
            return ScorerType.RAG_CONTEXTUAL_PRECISION
        elif "recall" in class_name:
            return ScorerType.RAG_CONTEXTUAL_RECALL
        elif "ragas" in class_name:
            return ScorerType.RAGAS
        else:
            # Default RAG scorer type
            return ScorerType.RAG_ANSWER_RELEVANCY
    elif "panel" in module_path or "judge" in module_path:
        return ScorerType.CUSTOM

    # Fallback to entry point name mapping for known scorers
    name_to_type = {
        "accuracy": ScorerType.ACCURACY,
        "exact_match": ScorerType.ACCURACY,
        "f1": ScorerType.ACCURACY,
        "g_eval": ScorerType.G_EVAL,
        "conversational": ScorerType.CONVERSATIONAL_METRICS,
        "answer_relevancy": ScorerType.RAG_ANSWER_RELEVANCY,
        "faithfulness": ScorerType.RAG_FAITHFULNESS,
        "contextual_precision": ScorerType.RAG_CONTEXTUAL_PRECISION,
        "contextual_recall": ScorerType.RAG_CONTEXTUAL_RECALL,
        "ragas": ScorerType.RAGAS,
        "panel_judge": ScorerType.CUSTOM,
    }

    if scorer_name in name_to_type:
        return name_to_type[scorer_name]

    # Final fallback: try to infer from scorer name patterns
    scorer_name_lower = scorer_name.lower()
    if (
        "accuracy" in scorer_name_lower
        or "exact" in scorer_name_lower
        or "f1" in scorer_name_lower
    ):
        return ScorerType.ACCURACY
    elif "g_eval" in scorer_name_lower or "geval" in scorer_name_lower:
        return ScorerType.G_EVAL
    elif "conversational" in scorer_name_lower:
        return ScorerType.CONVERSATIONAL_METRICS
    elif "relevancy" in scorer_name_lower:
        return ScorerType.RAG_ANSWER_RELEVANCY
    elif "faithfulness" in scorer_name_lower:
        return ScorerType.RAG_FAITHFULNESS
    elif "precision" in scorer_name_lower:
        return ScorerType.RAG_CONTEXTUAL_PRECISION
    elif "recall" in scorer_name_lower:
        return ScorerType.RAG_CONTEXTUAL_RECALL
    elif "ragas" in scorer_name_lower:
        return ScorerType.RAGAS
    elif "panel" in scorer_name_lower or "judge" in scorer_name_lower:
        return ScorerType.CUSTOM
    else:
        # Default to accuracy for unknown scorers
        return ScorerType.ACCURACY


async def create_default_model_for_scorer():
    """Create a default model for model-dependent scorers."""
    from app.core.config import get_settings

    settings = get_settings()

    model_config = ModelConfig(
        provider=ModelProvider(settings.default_model_provider),
        model_name=settings.default_model_name,
        temperature=settings.default_model_temperature,
        max_tokens=settings.default_model_max_tokens,
    )
    return ModelFactory.create_model(model_config)


@router.get("/{scorer_name}/info", response_model=ScorerInfo)
async def get_scorer_info(scorer_name: str):
    """
    Get information about a specific scorer.

    Args:
        scorer_name: Name of the scorer to get info for

    Returns:
        Scorer information including metadata
    """
    try:
        scorer_config = await get_scorer_config_by_name(scorer_name)

        # Check if scorer requires a model
        requires_model = scorer_config.type in MODEL_DEPENDENT_SCORERS

        # Create scorer instance to get info
        if requires_model:
            model = await create_default_model_for_scorer()
            scorer_instance = ScorerFactory.create_scorer(scorer_config, model)
        else:
            scorer_instance = ScorerFactory.create_scorer(scorer_config, None)

        info = scorer_instance.get_info()

        return ScorerInfo(
            name=info.get("name", scorer_name),
            description=info.get("description", f"{scorer_name} scorer"),
            type=info.get("type", scorer_config.type.value),
            total_scores=info.get("total_scores", 0),
            average_score=info.get("average_score", 0.0),
            requires_model=requires_model,
            config=info.get("config", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting scorer info: {e!s}")


@router.post("/{scorer_name}/score", response_model=ScoringResponse)
async def score_single(scorer_name: str, request: ScoringRequest):
    """
    Perform single scoring operation.

    Args:
        scorer_name: Name of the scorer to use
        request: Scoring request with prediction and ground truth

    Returns:
        Scoring response with score and metadata
    """
    start_time = time.time()

    try:
        # Get scorer configuration and check if model is required
        scorer_config = await get_scorer_config_by_name(scorer_name)
        requires_model = scorer_config.type in MODEL_DEPENDENT_SCORERS

        # Create scorer instance
        if requires_model:
            model = await create_default_model_for_scorer()
            scorer_instance = ScorerFactory.create_scorer(scorer_config, model)
        else:
            scorer_instance = ScorerFactory.create_scorer(scorer_config, None)

        # Perform scoring in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        score_result = await loop.run_in_executor(
            None,
            lambda: scorer_instance.score(
                prediction=request.prediction,
                ground_truth=request.ground_truth,
                context=request.context,
            ),
        )

        processing_time = time.time() - start_time

        # Handle different score result types
        if isinstance(score_result, dict):
            score_value = score_result.get("score")
            if score_value is None:
                if len(score_result) == 1:
                    score_value = next(iter(score_result.values()))
                else:
                    raise ValueError(
                        "Ambiguous score result: multiple values without 'score' key"
                    )
            reasoning = score_result.get("reasoning", "Score calculated successfully")
            metadata = {
                k: v for k, v in score_result.items() if k not in ["score", "reasoning"]
            }
        else:
            score_value = float(score_result)
            reasoning = "Score calculated successfully"
            metadata = {}

        # Check if score passes threshold
        passed = score_value >= scorer_config.threshold

        return ScoringResponse(
            score=score_value,
            passed=passed,
            reasoning=reasoning,
            metadata=metadata,
            scorer_info=scorer_instance.get_info(),
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during scoring: {e!s}")


@router.post("/{scorer_name}/score/batch", response_model=BatchScoringResponse)
async def score_batch(scorer_name: str, request: BatchScoringRequest):
    """
    Perform batch scoring operations.

    Args:
        scorer_name: Name of the scorer to use
        request: Batch scoring request with predictions and ground truths

    Returns:
        Batch scoring response with scores and statistics
    """
    start_time = time.time()

    try:
        # Validate input lengths match
        if len(request.predictions) != len(request.ground_truths):
            raise HTTPException(
                status_code=400,
                detail="Number of predictions must match number of ground truths",
            )

        # Get scorer configuration and check if model is required
        scorer_config = await get_scorer_config_by_name(scorer_name)
        requires_model = scorer_config.type in MODEL_DEPENDENT_SCORERS

        # Create scorer instance
        if requires_model:
            model = await create_default_model_for_scorer()
            scorer_instance = ScorerFactory.create_scorer(scorer_config, model)
        else:
            scorer_instance = ScorerFactory.create_scorer(scorer_config, None)

        # Perform batch scoring in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        score_results = await loop.run_in_executor(
            None,
            lambda: scorer_instance.score_batch(
                predictions=request.predictions,
                ground_truths=request.ground_truths,
                contexts=request.contexts,
            ),
        )

        processing_time = time.time() - start_time

        # Process results
        scores = []
        passed = []
        reasonings = []
        metadata = []

        for result in score_results:
            if isinstance(result, dict):
                score_value = result.get("score")
                if score_value is None:
                    if len(result) == 1:
                        score_value = next(iter(result.values()))
                    else:
                        raise ValueError(
                            "Ambiguous score result: multiple values without 'score' key"
                        )
                reasoning = result.get("reasoning", "Score calculated successfully")
                meta = {
                    k: v for k, v in result.items() if k not in ["score", "reasoning"]
                }
            else:
                score_value = float(result)
                reasoning = "Score calculated successfully"
                meta = {}

            scores.append(score_value)
            passed.append(score_value >= scorer_config.threshold)
            reasonings.append(reasoning)
            metadata.append(meta)

        # Calculate statistics
        statistics = {
            "mean": sum(scores) / len(scores) if scores else 0,
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "pass_rate": sum(passed) / len(passed) if passed else 0,
            "threshold": scorer_config.threshold,
        }

        return BatchScoringResponse(
            scores=scores,
            passed=passed,
            reasonings=reasonings,
            metadata=metadata,
            scorer_info=scorer_instance.get_info(),
            processing_time=processing_time,
            count=len(scores),
            statistics=statistics,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during batch scoring: {e!s}"
        )


@router.get("/{scorer_name}/stats", response_model=ScorerStatsResponse)
async def get_scorer_stats(scorer_name: str):
    """
    Get detailed statistics for a scorer.

    Args:
        scorer_name: Name of the scorer to get stats for

    Returns:
        Scorer statistics response
    """
    try:
        scorer_config = await get_scorer_config_by_name(scorer_name)
        requires_model = scorer_config.type in MODEL_DEPENDENT_SCORERS

        # Create scorer instance
        if requires_model:
            model = await create_default_model_for_scorer()
            scorer_instance = ScorerFactory.create_scorer(scorer_config, model)
        else:
            scorer_instance = ScorerFactory.create_scorer(scorer_config, None)

        info = scorer_instance.get_info()
        stats = scorer_instance.get_stats()

        return ScorerStatsResponse(
            scorer_info=ScorerInfo(
                name=info.get("name", scorer_name),
                description=info.get("description", f"{scorer_name} scorer"),
                type=info.get("type", scorer_config.type.value),
                total_scores=info.get("total_scores", 0),
                average_score=info.get("average_score", 0.0),
                requires_model=requires_model,
                config=info.get("config", {}),
            ),
            statistics=stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting scorer stats: {e!s}"
        )


@router.post("/instantiate", response_model=ScorerInfo)
async def instantiate_scorer(request: ScorerInstantiateRequest):
    """
    Instantiate a scorer with custom configuration.

    Args:
        request: Scorer instantiation request with custom configuration

    Returns:
        Scorer information for the instantiated scorer
    """
    try:
        requires_model = request.scorer_config.type in MODEL_DEPENDENT_SCORERS

        # Create scorer instance
        if requires_model:
            if request.inference_settings:
                # Use provided model config
                model_config = ModelConfig(**request.inference_settings)
                model = ModelFactory.create_model(model_config)
            else:
                # Use default model
                model = await create_default_model_for_scorer()
            scorer_instance = ScorerFactory.create_scorer(request.scorer_config, model)
        else:
            scorer_instance = ScorerFactory.create_scorer(request.scorer_config, None)

        info = scorer_instance.get_info()

        return ScorerInfo(
            name=info.get("name", request.scorer_config.name or "custom_scorer"),
            description=info.get(
                "description", f"{request.scorer_config.type.value} scorer"
            ),
            type=info.get("type", request.scorer_config.type.value),
            total_scores=info.get("total_scores", 0),
            average_score=info.get("average_score", 0.0),
            requires_model=requires_model,
            config=info.get("config", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error instantiating scorer: {e!s}"
        )
