"""
Unit tests for novaeval.agents.__init__.py module.

Tests the agent module's import functionality and __all__ exports.
"""

import pytest


@pytest.mark.unit
def test_imports():
    """Test that all classes can be imported from the agents module."""
    from novaeval.agents import (
        AgentData,
        ToolCall,
        ToolResult,
        ToolSchema,
    )

    # Verify all classes are imported correctly
    assert AgentData is not None
    assert ToolCall is not None
    assert ToolResult is not None
    assert ToolSchema is not None


@pytest.mark.unit
def test_all_exports():
    """Test that __all__ contains the expected exports."""
    from novaeval.agents import __all__

    expected_exports = [
        "AgentData",
        "ToolCall",
        "ToolResult",
        "ToolSchema",
    ]

    # Verify all expected items are in __all__
    assert set(__all__) == set(expected_exports)
    assert len(__all__) == len(expected_exports)


@pytest.mark.unit
def test_star_import():
    """Test that star import works correctly."""
    import novaeval.agents as agents_module

    # Test that we can access all exported items
    for item_name in agents_module.__all__:
        assert hasattr(agents_module, item_name)
        item = getattr(agents_module, item_name)
        assert item is not None


@pytest.mark.unit
def test_direct_imports():
    """Test that classes can be imported directly from submodules."""
    from novaeval.agents.agent_data import AgentData, ToolCall, ToolResult, ToolSchema
    from novaeval.datasets.agent_dataset import AgentDataset

    # Verify direct imports work
    assert AgentData is not None
    assert AgentDataset is not None
    assert ToolCall is not None
    assert ToolResult is not None
    assert ToolSchema is not None


@pytest.mark.unit
def test_class_types():
    """Test that imported classes are of the correct type."""
    from novaeval.agents import (
        AgentData,
        ToolCall,
        ToolResult,
        ToolSchema,
    )

    # All should be classes (type)
    assert isinstance(AgentData, type)
    assert isinstance(ToolCall, type)
    assert isinstance(ToolResult, type)
    assert isinstance(ToolSchema, type)


@pytest.mark.unit
def test_instantiation():
    """Test that classes can be instantiated."""
    from novaeval.agents import (
        AgentData,
        ToolCall,
        ToolResult,
        ToolSchema,
    )

    # Test basic instantiation without errors

    agent_data = AgentData()
    assert agent_data is not None

    tool_schema = ToolSchema(name="test", description="test desc")
    assert tool_schema is not None

    tool_call = ToolCall(tool_name="test", parameters={}, call_id="test")
    assert tool_call is not None

    tool_result = ToolResult(call_id="test", result="result", success=True)
    assert tool_result is not None
