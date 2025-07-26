"""
Component discovery system for NovaEval API.

This module provides auto-discovery of models, datasets, and scorers
using entry points and schema extraction from Pydantic models.
"""

import inspect
import logging
from importlib.metadata import entry_points
from typing import Any, Optional

from pydantic import BaseModel

from novaeval.config.schema import DatasetConfig, ModelConfig, ScorerConfig

logger = logging.getLogger(__name__)


class ComponentMetadata(BaseModel):
    """Metadata for a discovered component."""

    name: str
    entry_point: str
    component_type: str
    class_name: str
    module_path: str
    description: Optional[str] = None
    config_schema: dict[str, Any]
    parameters: dict[str, Any] = {}


class ComponentRegistry:
    """
    Registry for auto-discovering and managing NovaEval components.

    This class uses entry points to discover available models, datasets,
    and scorers, and extracts their configuration schemas using Pydantic
    model introspection.
    """

    def __init__(self):
        """Initialize the component registry."""
        self._models_cache: Optional[dict[str, ComponentMetadata]] = None
        self._datasets_cache: Optional[dict[str, ComponentMetadata]] = None
        self._scorers_cache: Optional[dict[str, ComponentMetadata]] = None

    async def get_models(self, reload: bool = False) -> dict[str, ComponentMetadata]:
        """
        Get all available models.

        Args:
            reload: Whether to reload from entry points

        Returns:
            Dictionary of model name to metadata
        """
        if self._models_cache is None or reload:
            self._models_cache = await self._discover_components(
                "novaeval.models", ModelConfig, "model"
            )
        return self._models_cache

    async def get_datasets(self, reload: bool = False) -> dict[str, ComponentMetadata]:
        """
        Get all available datasets.

        Args:
            reload: Whether to reload from entry points

        Returns:
            Dictionary of dataset name to metadata
        """
        if self._datasets_cache is None or reload:
            self._datasets_cache = await self._discover_components(
                "novaeval.datasets", DatasetConfig, "dataset"
            )
        return self._datasets_cache

    async def get_scorers(self, reload: bool = False) -> dict[str, ComponentMetadata]:
        """
        Get all available scorers.

        Args:
            reload: Whether to reload from entry points

        Returns:
            Dictionary of scorer name to metadata
        """
        if self._scorers_cache is None or reload:
            self._scorers_cache = await self._discover_components(
                "novaeval.scorers", ScorerConfig, "scorer"
            )
        return self._scorers_cache

    async def get_component(
        self, component_type: str, name: str, reload: bool = False
    ) -> Optional[ComponentMetadata]:
        """
        Get a specific component by type and name.

        Args:
            component_type: Type of component (model, dataset, scorer)
            name: Component name
            reload: Whether to reload from entry points

        Returns:
            Component metadata if found, None otherwise
        """
        if component_type == "model":
            components = await self.get_models(reload)
        elif component_type == "dataset":
            components = await self.get_datasets(reload)
        elif component_type == "scorer":
            components = await self.get_scorers(reload)
        else:
            return None

        return components.get(name)

    async def reload_all(self) -> None:
        """Reload all component caches from entry points."""
        self._models_cache = None
        self._datasets_cache = None
        self._scorers_cache = None

        # Pre-load all caches
        await self.get_models(reload=True)
        await self.get_datasets(reload=True)
        await self.get_scorers(reload=True)

    async def _discover_components(
        self, group: str, config_class: type[BaseModel], component_type: str
    ) -> dict[str, ComponentMetadata]:
        """
        Discover components for a specific entry point group.

        Args:
            group: Entry point group name
            config_class: Pydantic config class for schema extraction
            component_type: Component type string

        Returns:
            Dictionary of component name to metadata
        """
        components = {}

        try:
            # Get entry points for the group
            eps = entry_points().select(group=group)

            for ep in eps:
                try:
                    # Load the component class
                    component_class = ep.load()

                    # Extract metadata
                    metadata = ComponentMetadata(
                        name=ep.name,
                        entry_point=f"{ep.module}:{ep.attr}",
                        component_type=component_type,
                        class_name=component_class.__name__,
                        module_path=component_class.__module__,
                        description=self._extract_description(component_class),
                        config_schema=self._extract_config_schema(config_class),
                        parameters=self._extract_parameters(component_class),
                    )

                    components[ep.name] = metadata

                except Exception as e:
                    # Log error but continue with other components
                    logger.warning(f"Failed to load component {ep.name}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to discover components for group {group}: {e}")

        return components

    def _extract_description(self, component_class: type) -> Optional[str]:
        """
        Extract description from component class docstring.

        Args:
            component_class: Component class

        Returns:
            First line of docstring or None
        """
        if component_class.__doc__:
            # Get first line of docstring, stripped
            lines = component_class.__doc__.strip().split("\n")
            return lines[0].strip() if lines else None
        return None

    def _extract_config_schema(self, config_class: type[BaseModel]) -> dict[str, Any]:
        """
        Extract JSON schema from Pydantic config class.

        Args:
            config_class: Pydantic config class

        Returns:
            JSON schema dictionary
        """
        try:
            return config_class.model_json_schema()
        except Exception as e:
            logger.warning(f"Failed to extract schema from {config_class}: {e}")
            return {}

    def _extract_parameters(self, component_class: type) -> dict[str, Any]:
        """
        Extract constructor parameters from component class.

        Args:
            component_class: Component class

        Returns:
            Dictionary of parameter information
        """
        try:
            signature = inspect.signature(component_class.__init__)
            parameters = {}

            for name, param in signature.parameters.items():
                if name == "self":
                    continue

                param_info = {
                    "name": name,
                    "annotation": (
                        str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else None
                    ),
                    "default": (
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None
                    ),
                    "kind": param.kind.name,
                }
                parameters[name] = param_info

            return parameters
        except Exception as e:
            logger.warning(f"Failed to extract parameters from {component_class}: {e}")
            return {}


# Global registry instance
_registry = ComponentRegistry()


async def get_registry() -> ComponentRegistry:
    """
    Get the global component registry instance.

    Returns:
        ComponentRegistry instance
    """
    return _registry
