"""
Component Discovery API endpoints for NovaEval.

This module provides REST endpoints for discovering and inspecting
available models, datasets, and scorers in the NovaEval system.
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.discovery import get_registry


# Response schemas
class ComponentSummary(BaseModel):
    """Summary information for a component."""

    name: str
    component_type: str
    class_name: str
    description: Optional[str] = None


class ComponentDetail(BaseModel):
    """Detailed information for a component."""

    name: str
    entry_point: str
    component_type: str
    class_name: str
    module_path: str
    description: Optional[str] = None
    config_schema: dict[str, Any]
    parameters: dict[str, Any]


class ComponentListResponse(BaseModel):
    """Response for component list endpoints."""

    count: int
    components: list[ComponentSummary]


class ComponentTypesResponse(BaseModel):
    """Response for component types endpoint."""

    types: list[str]
    counts: dict[str, int]


# Create router
router = APIRouter()


async def _list_components(
    component_type: str, reload: bool = False
) -> ComponentListResponse:
    """Generic function to list components of a given type."""
    registry = await get_registry()

    # Map component types to registry methods
    getters = {
        "model": registry.get_models,
        "dataset": registry.get_datasets,
        "scorer": registry.get_scorers,
    }

    components_dict = await getters[component_type](reload=reload)

    components = [
        ComponentSummary(
            name=name,
            component_type=metadata.component_type,
            class_name=metadata.class_name,
            description=metadata.description,
        )
        for name, metadata in components_dict.items()
    ]

    return ComponentListResponse(count=len(components), components=components)


async def _get_component_details(
    component_type: str, name: str, reload: bool = False
) -> ComponentDetail:
    """Generic function to get details for a specific component."""
    registry = await get_registry()
    metadata = await registry.get_component(component_type, name, reload=reload)

    if not metadata:
        raise HTTPException(
            status_code=404, detail=f"{component_type.title()} '{name}' not found"
        )

    return ComponentDetail(
        name=metadata.name,
        entry_point=metadata.entry_point,
        component_type=metadata.component_type,
        class_name=metadata.class_name,
        module_path=metadata.module_path,
        description=metadata.description,
        config_schema=metadata.config_schema,
        parameters=metadata.parameters,
    )


@router.get("/", response_model=ComponentTypesResponse, summary="List component types")
async def list_component_types():
    """
    Get overview of all component types and their counts.

    Returns:
        ComponentTypesResponse: Summary of available component types
    """
    registry = await get_registry()

    models = await registry.get_models()
    datasets = await registry.get_datasets()
    scorers = await registry.get_scorers()

    return ComponentTypesResponse(
        types=["model", "dataset", "scorer"],
        counts={"model": len(models), "dataset": len(datasets), "scorer": len(scorers)},
    )


@router.get(
    "/models", response_model=ComponentListResponse, summary="List available models"
)
async def list_models(
    reload: bool = Query(False, description="Reload components from entry points")
):
    """
    Get all available models.

    Args:
        reload: Whether to reload components from entry points

    Returns:
        ComponentListResponse: List of available models
    """
    return await _list_components("model", reload)


@router.get(
    "/datasets", response_model=ComponentListResponse, summary="List available datasets"
)
async def list_datasets(
    reload: bool = Query(False, description="Reload components from entry points")
):
    """
    Get all available datasets.

    Args:
        reload: Whether to reload components from entry points

    Returns:
        ComponentListResponse: List of available datasets
    """
    return await _list_components("dataset", reload)


@router.get(
    "/scorers", response_model=ComponentListResponse, summary="List available scorers"
)
async def list_scorers(
    reload: bool = Query(False, description="Reload components from entry points")
):
    """
    Get all available scorers.

    Args:
        reload: Whether to reload components from entry points

    Returns:
        ComponentListResponse: List of available scorers
    """
    return await _list_components("scorer", reload)


@router.get(
    "/models/{name}", response_model=ComponentDetail, summary="Get model details"
)
async def get_model_details(
    name: str,
    reload: bool = Query(False, description="Reload components from entry points"),
):
    """
    Get detailed information about a specific model.

    Args:
        name: Model name
        reload: Whether to reload components from entry points

    Returns:
        ComponentDetail: Detailed model information

    Raises:
        HTTPException: If model is not found
    """
    return await _get_component_details("model", name, reload)


@router.get(
    "/datasets/{name}", response_model=ComponentDetail, summary="Get dataset details"
)
async def get_dataset_details(
    name: str,
    reload: bool = Query(False, description="Reload components from entry points"),
):
    """
    Get detailed information about a specific dataset.

    Args:
        name: Dataset name
        reload: Whether to reload components from entry points

    Returns:
        ComponentDetail: Detailed dataset information

    Raises:
        HTTPException: If dataset is not found
    """
    return await _get_component_details("dataset", name, reload)


@router.get(
    "/scorers/{name}", response_model=ComponentDetail, summary="Get scorer details"
)
async def get_scorer_details(
    name: str,
    reload: bool = Query(False, description="Reload components from entry points"),
):
    """
    Get detailed information about a specific scorer.

    Args:
        name: Scorer name
        reload: Whether to reload components from entry points

    Returns:
        ComponentDetail: Detailed scorer information

    Raises:
        HTTPException: If scorer is not found
    """
    return await _get_component_details("scorer", name, reload)


@router.post("/reload", summary="Reload all components")
async def reload_components():
    """
    Reload all components from entry points.

    This endpoint forces a reload of all component registries,
    useful when new components have been installed or entry
    points have been modified.

    Returns:
        Dict: Summary of reloaded components
    """
    registry = await get_registry()
    await registry.reload_all()

    models = await registry.get_models()
    datasets = await registry.get_datasets()
    scorers = await registry.get_scorers()

    return {
        "message": "Components reloaded successfully",
        "counts": {
            "models": len(models),
            "datasets": len(datasets),
            "scorers": len(scorers),
        },
        "components": {
            "models": list(models.keys()),
            "datasets": list(datasets.keys()),
            "scorers": list(scorers.keys()),
        },
    }
