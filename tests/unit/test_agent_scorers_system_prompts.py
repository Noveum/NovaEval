"""
Unit tests for novaeval.agents.agent_scorers_system_prompts module.

Tests all system prompt constants used by agent scorers.
"""

import pytest

from novaeval.scorers.agent_scorers_system_prompts import (
    CONTEXT_RELEVANCY_PROMPT,
    PARAMETER_CORRECTNESS_PROMPT,
    ROLE_ADHERENCE_PROMPT,
    TASK_PROGRESSION_PROMPT,
    TOOL_CORRECTNESS_PROMPT,
    TOOL_RELEVANCY_PROMPT,
)


@pytest.mark.unit
def test_tool_relevancy_prompt_exists():
    """Test that TOOL_RELEVANCY_PROMPT is defined and not empty."""
    assert TOOL_RELEVANCY_PROMPT is not None
    assert isinstance(TOOL_RELEVANCY_PROMPT, str)
    assert len(TOOL_RELEVANCY_PROMPT.strip()) > 0


@pytest.mark.unit
def test_tool_correctness_prompt_exists():
    """Test that TOOL_CORRECTNESS_PROMPT is defined and not empty."""
    assert TOOL_CORRECTNESS_PROMPT is not None
    assert isinstance(TOOL_CORRECTNESS_PROMPT, str)
    assert len(TOOL_CORRECTNESS_PROMPT.strip()) > 0


@pytest.mark.unit
def test_parameter_correctness_prompt_exists():
    """Test that PARAMETER_CORRECTNESS_PROMPT is defined and not empty."""
    assert PARAMETER_CORRECTNESS_PROMPT is not None
    assert isinstance(PARAMETER_CORRECTNESS_PROMPT, str)
    assert len(PARAMETER_CORRECTNESS_PROMPT.strip()) > 0


@pytest.mark.unit
def test_task_progression_prompt_exists():
    """Test that TASK_PROGRESSION_PROMPT is defined and not empty."""
    assert TASK_PROGRESSION_PROMPT is not None
    assert isinstance(TASK_PROGRESSION_PROMPT, str)
    assert len(TASK_PROGRESSION_PROMPT.strip()) > 0


@pytest.mark.unit
def test_context_relevancy_prompt_exists():
    """Test that CONTEXT_RELEVANCY_PROMPT is defined and not empty."""
    assert CONTEXT_RELEVANCY_PROMPT is not None
    assert isinstance(CONTEXT_RELEVANCY_PROMPT, str)
    assert len(CONTEXT_RELEVANCY_PROMPT.strip()) > 0


@pytest.mark.unit
def test_role_adherence_prompt_exists():
    """Test that ROLE_ADHERENCE_PROMPT is defined and not empty."""
    assert ROLE_ADHERENCE_PROMPT is not None
    assert isinstance(ROLE_ADHERENCE_PROMPT, str)
    assert len(ROLE_ADHERENCE_PROMPT.strip()) > 0


@pytest.mark.unit
def test_all_prompts_have_format_placeholders():
    """Test that all prompts contain expected format placeholders."""

    # Tool relevancy should have tools_available and tool_calls
    assert "{tools_available}" in TOOL_RELEVANCY_PROMPT
    assert "{tool_calls}" in TOOL_RELEVANCY_PROMPT

    # Tool correctness should have expected_tool_call and tool_calls
    assert "{expected_tool_call}" in TOOL_CORRECTNESS_PROMPT
    assert "{tool_calls}" in TOOL_CORRECTNESS_PROMPT

    # Parameter correctness should have tool_calls_with_parameters and tool_call_results
    assert "{tool_calls_with_parameters}" in PARAMETER_CORRECTNESS_PROMPT
    assert "{tool_call_results}" in PARAMETER_CORRECTNESS_PROMPT

    # Task progression should have agent info fields
    assert "{agent_role}" in TASK_PROGRESSION_PROMPT
    assert "{agent_task}" in TASK_PROGRESSION_PROMPT
    assert "{system_prompt}" in TASK_PROGRESSION_PROMPT
    assert "{agent_response}" in TASK_PROGRESSION_PROMPT

    # Context relevancy should have agent info fields
    assert "{agent_role}" in CONTEXT_RELEVANCY_PROMPT
    assert "{agent_task}" in CONTEXT_RELEVANCY_PROMPT
    assert "{agent_response}" in CONTEXT_RELEVANCY_PROMPT

    # Role adherence should have agent info and tool calls
    assert "{agent_role}" in ROLE_ADHERENCE_PROMPT
    assert "{agent_task}" in ROLE_ADHERENCE_PROMPT
    assert "{agent_response}" in ROLE_ADHERENCE_PROMPT
    assert "{tool_calls}" in ROLE_ADHERENCE_PROMPT


@pytest.mark.unit
def test_prompts_contain_json_response_format():
    """Test that all prompts specify JSON response format."""
    prompts = [
        TOOL_RELEVANCY_PROMPT,
        TOOL_CORRECTNESS_PROMPT,
        PARAMETER_CORRECTNESS_PROMPT,
        TASK_PROGRESSION_PROMPT,
        CONTEXT_RELEVANCY_PROMPT,
        ROLE_ADHERENCE_PROMPT,
    ]

    for prompt in prompts:
        # Each prompt should mention JSON format
        assert "JSON" in prompt or "json" in prompt
        # Each prompt should have score and reasoning fields in example
        assert "score" in prompt
        assert "reasoning" in prompt


@pytest.mark.unit
def test_prompts_contain_evaluation_criteria():
    """Test that all prompts contain evaluation criteria and scoring ranges."""

    # Tool relevancy: 1-10 scale
    assert "1-10" in TOOL_RELEVANCY_PROMPT
    assert "1-3:" in TOOL_RELEVANCY_PROMPT
    assert "9-10:" in TOOL_RELEVANCY_PROMPT

    # Tool correctness: 1-10 scale
    assert "1-10" in TOOL_CORRECTNESS_PROMPT
    assert "1-3:" in TOOL_CORRECTNESS_PROMPT
    assert "9-10:" in TOOL_CORRECTNESS_PROMPT

    # Parameter correctness: 1-10 scale
    assert "1-10" in PARAMETER_CORRECTNESS_PROMPT
    assert "1-3:" in PARAMETER_CORRECTNESS_PROMPT
    assert "9-10:" in PARAMETER_CORRECTNESS_PROMPT

    # Task progression: 1-5 scale
    assert "1-5" in TASK_PROGRESSION_PROMPT
    assert "1:" in TASK_PROGRESSION_PROMPT
    assert "5:" in TASK_PROGRESSION_PROMPT

    # Context relevancy: 1-10 scale
    assert "1-10" in CONTEXT_RELEVANCY_PROMPT
    assert "1-3:" in CONTEXT_RELEVANCY_PROMPT
    assert "9-10:" in CONTEXT_RELEVANCY_PROMPT

    # Role adherence: 1-10 scale
    assert "1-10" in ROLE_ADHERENCE_PROMPT
    assert "1-3:" in ROLE_ADHERENCE_PROMPT
    assert "9-10:" in ROLE_ADHERENCE_PROMPT


@pytest.mark.unit
def test_prompts_have_clear_instructions():
    """Test that all prompts have clear instruction sections."""
    prompts = [
        TOOL_RELEVANCY_PROMPT,
        TOOL_CORRECTNESS_PROMPT,
        PARAMETER_CORRECTNESS_PROMPT,
        TASK_PROGRESSION_PROMPT,
        CONTEXT_RELEVANCY_PROMPT,
        ROLE_ADHERENCE_PROMPT,
    ]

    for prompt in prompts:
        # Each prompt should have an instructions section
        assert "Instructions" in prompt or "Evaluate" in prompt
        # Each prompt should specify response format
        assert "Format your response" in prompt or "Your response should be" in prompt


@pytest.mark.unit
def test_prompt_formatting_safety():
    """Test that prompts can be safely formatted with sample data."""

    # Test tool relevancy prompt formatting
    try:
        formatted = TOOL_RELEVANCY_PROMPT.format(
            tools_available="[sample tools]", tool_calls="[sample calls]"
        )
        assert (
            len(formatted) >= len(TOOL_RELEVANCY_PROMPT) - 50
        )  # Allow for placeholder replacement
    except KeyError:
        pytest.fail("TOOL_RELEVANCY_PROMPT missing required format keys")

    # Test tool correctness prompt formatting
    try:
        formatted = TOOL_CORRECTNESS_PROMPT.format(
            expected_tool_call="sample expected call", tool_calls="[sample calls]"
        )
        assert len(formatted) >= len(TOOL_CORRECTNESS_PROMPT) - 50
    except KeyError:
        pytest.fail("TOOL_CORRECTNESS_PROMPT missing required format keys")

    # Test parameter correctness prompt formatting
    try:
        formatted = PARAMETER_CORRECTNESS_PROMPT.format(
            tool_calls_with_parameters="sample calls with params",
            tool_call_results="sample results",
        )
        assert len(formatted) >= len(PARAMETER_CORRECTNESS_PROMPT) - 50
    except KeyError:
        pytest.fail("PARAMETER_CORRECTNESS_PROMPT missing required format keys")

    # Test task progression prompt formatting
    try:
        formatted = TASK_PROGRESSION_PROMPT.format(
            agent_role="test role",
            agent_task="test task",
            system_prompt="test prompt",
            agent_response="test response",
        )
        assert len(formatted) >= len(TASK_PROGRESSION_PROMPT) - 50
    except KeyError:
        pytest.fail("TASK_PROGRESSION_PROMPT missing required format keys")

    # Test context relevancy prompt formatting
    try:
        formatted = CONTEXT_RELEVANCY_PROMPT.format(
            agent_role="test role",
            agent_task="test task",
            agent_response="test response",
        )
        assert len(formatted) >= len(CONTEXT_RELEVANCY_PROMPT) - 50
    except KeyError:
        pytest.fail("CONTEXT_RELEVANCY_PROMPT missing required format keys")

    # Test role adherence prompt formatting
    try:
        formatted = ROLE_ADHERENCE_PROMPT.format(
            agent_role="test role",
            agent_task="test task",
            agent_response="test response",
            tool_calls="[sample calls]",
        )
        assert len(formatted) >= len(ROLE_ADHERENCE_PROMPT) - 50
    except KeyError:
        pytest.fail("ROLE_ADHERENCE_PROMPT missing required format keys")


@pytest.mark.unit
def test_prompts_contain_examples():
    """Test that prompts contain example JSON responses."""
    prompts = [
        TOOL_RELEVANCY_PROMPT,
        TOOL_CORRECTNESS_PROMPT,
        PARAMETER_CORRECTNESS_PROMPT,
        TASK_PROGRESSION_PROMPT,
        CONTEXT_RELEVANCY_PROMPT,
        ROLE_ADHERENCE_PROMPT,
    ]

    for prompt in prompts:
        # Each prompt should contain an example with curly braces (JSON)
        assert "{" in prompt and "}" in prompt
        # Each prompt should have example score and reasoning
        example_section = prompt[prompt.find("{") : prompt.rfind("}") + 1]
        assert "score" in example_section
        assert "reasoning" in example_section


@pytest.mark.unit
def test_all_prompts_are_different():
    """Test that all prompts are unique and not duplicated."""
    prompts = [
        TOOL_RELEVANCY_PROMPT,
        TOOL_CORRECTNESS_PROMPT,
        PARAMETER_CORRECTNESS_PROMPT,
        TASK_PROGRESSION_PROMPT,
        CONTEXT_RELEVANCY_PROMPT,
        ROLE_ADHERENCE_PROMPT,
    ]

    # Check that all prompts are different
    for i, prompt1 in enumerate(prompts):
        for j, prompt2 in enumerate(prompts):
            if i != j:
                assert prompt1 != prompt2, f"Prompts {i} and {j} are identical"


@pytest.mark.unit
def test_prompt_lengths():
    """Test that prompts are reasonably sized (not too short or too long)."""
    prompts = [
        TOOL_RELEVANCY_PROMPT,
        TOOL_CORRECTNESS_PROMPT,
        PARAMETER_CORRECTNESS_PROMPT,
        TASK_PROGRESSION_PROMPT,
        CONTEXT_RELEVANCY_PROMPT,
        ROLE_ADHERENCE_PROMPT,
    ]

    for prompt in prompts:
        # Prompts should be substantial but not excessive
        assert (
            500 < len(prompt) < 5000
        ), f"Prompt length {len(prompt)} is outside reasonable range"
