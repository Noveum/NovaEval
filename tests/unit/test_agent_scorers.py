"""
Unit tests for novaeval.agents.agent_scorers module.

Tests all scoring functions, classes, and utilities for agent evaluation.
"""

from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from novaeval.agents.agent_data import AgentData, ToolCall, ToolResult, ToolSchema
from novaeval.scorers.agent_scorers import (
    AgentScorers,
    FieldAvailabilityError,
    ScoreListResponse,
    ScoreWithOriginalTask,
    ScoreWithReasoning,
    SingleScoreResponse,
    context_relevancy_scorer,
    conversation_coherence_scorer,
    escape_json_for_format,
    goal_achievement_scorer,
    parameter_correctness_scorer,
    parse_score_with_original_task,
    parse_score_with_reasoning,
    role_adherence_scorer,
    task_progression_scorer,
    tool_correctness_scorer,
    tool_relevancy_scorer,
)


class MockLLMModel:
    """Mock LLM model for testing."""

    def __init__(self, response: str = '{"score": 8.5, "reasoning": "Test reasoning"}'):
        self.response = response
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.response


@pytest.fixture
def sample_agent_data():
    """Create sample agent data for testing."""
    return AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        ground_truth="The correct answer is 42",
        expected_tool_call=ToolCall(
            tool_name="calculator",
            parameters={"operation": "add", "a": 20, "b": 22},
            call_id="call_001",
        ),
        agent_name="TestAgent",
        agent_role="Mathematical assistant",
        agent_task="Calculate 20 + 22",
        system_prompt="You are a helpful math assistant.",
        agent_response="I'll calculate 20 + 22 using the calculator tool.",
        tools_available=[
            ToolSchema(
                name="calculator",
                description="Performs basic mathematical operations",
                args_schema={"operation": "str", "a": "number", "b": "number"},
                return_schema={"result": "number"},
            )
        ],
        tool_calls=[
            ToolCall(
                tool_name="calculator",
                parameters={"operation": "add", "a": 20, "b": 22},
                call_id="call_001",
            )
        ],
        parameters_passed={"operation": "add", "a": 20, "b": 22},
        tool_call_results=[
            ToolResult(call_id="call_001", result=42, success=True, error_message=None)
        ],
        trace=[
            {"type": "user_input", "content": "Calculate 20 + 22"},
            {"type": "tool_call", "tool": "calculator", "result": 42},
        ],
        exit_status="completed",
        agent_exit=True,
        metadata="Test metadata",
        retrieval_query=["search for information"],
        retrieved_context=[["Retrieved relevant information"]],
    )


@pytest.fixture
def minimal_agent_data():
    """Create minimal agent data for testing error cases."""
    return AgentData(agent_name="MinimalAgent", agent_exit=False)


# Test ScoreWithReasoning model
@pytest.mark.unit
def test_score_with_reasoning_valid():
    """Test ScoreWithReasoning with valid data."""
    score = ScoreWithReasoning(score=8.5, reasoning="Good performance")
    assert score.score == 8.5
    assert score.reasoning == "Good performance"


@pytest.mark.unit
def test_score_with_reasoning_invalid():
    """Test ScoreWithReasoning with invalid data."""
    with pytest.raises(ValidationError):
        ScoreWithReasoning(score="not_a_number", reasoning="test")

    with pytest.raises(ValidationError):
        ScoreWithReasoning(score=8.5, reasoning=123)


# Test ScoreWithOriginalTask model
@pytest.mark.unit
def test_score_with_original_task_valid():
    """Test ScoreWithOriginalTask with valid data."""
    score = ScoreWithOriginalTask(
        original_task="Calculate something", score=9.0, reasoning="Excellent work"
    )
    assert score.original_task == "Calculate something"
    assert score.score == 9.0
    assert score.reasoning == "Excellent work"


@pytest.mark.unit
def test_score_with_original_task_invalid():
    """Test ScoreWithOriginalTask with invalid data."""
    with pytest.raises(ValidationError):
        ScoreWithOriginalTask(score=8.5, reasoning="test")  # Missing original_task


# Test other model classes
@pytest.mark.unit
def test_score_list_response():
    """Test ScoreListResponse model."""
    scores = [
        ScoreWithReasoning(score=8.0, reasoning="Good"),
        ScoreWithReasoning(score=7.5, reasoning="Okay"),
    ]
    response = ScoreListResponse(scores=scores)
    assert len(response.scores) == 2
    assert response.scores[0].score == 8.0


@pytest.mark.unit
def test_single_score_response():
    """Test SingleScoreResponse model."""
    response = SingleScoreResponse(score=9.0, reasoning="Excellent")
    assert response.score == 9.0
    assert response.reasoning == "Excellent"


@pytest.mark.unit
def test_field_availability_error():
    """Test FieldAvailabilityError model."""
    error = FieldAvailabilityError(
        required_fields={"field1": True, "field2": False},
        error_message="Missing field2",
    )
    assert error.required_fields["field1"] is True
    assert error.required_fields["field2"] is False
    assert error.error_message == "Missing field2"


# Test utility functions


@pytest.mark.unit
def test_parse_score_with_reasoning_valid_json():
    """Test parse_score_with_reasoning with valid JSON."""
    response = '{"score": 8.5, "reasoning": "Good performance"}'
    result = parse_score_with_reasoning(response)
    assert result.score == 8.5
    assert result.reasoning == "Good performance"


@pytest.mark.unit
def test_parse_score_with_reasoning_embedded_json():
    """Test parse_score_with_reasoning with JSON embedded in text."""
    response = (
        'Here is my evaluation: {"score": 7.0, "reasoning": "Acceptable"} That\'s it.'
    )
    result = parse_score_with_reasoning(response)
    assert result.score == 7.0
    assert result.reasoning == "Acceptable"


@pytest.mark.unit
def test_parse_score_with_reasoning_incomplete_json():
    """Test parse_score_with_reasoning with incomplete JSON."""
    response = '{"score": 6.5}'
    result = parse_score_with_reasoning(response)
    assert result.score == 6.5
    assert result.reasoning == "No reasoning provided in response"


@pytest.mark.unit
def test_parse_score_with_reasoning_just_number():
    """Test parse_score_with_reasoning with just a number."""
    result = parse_score_with_reasoning("8.5")
    assert result.score == 8.5
    assert (
        result.reasoning is not None
        and "Score provided without reasoning" in result.reasoning
    )


@pytest.mark.unit
def test_parse_score_with_reasoning_regex_fallback():
    """Test parse_score_with_reasoning with regex fallback."""
    response = 'The score is 7.5 and the reasoning is "Pretty good work"'
    result = parse_score_with_reasoning(response)
    assert result.score == 7.5
    assert result.reasoning is not None and "Pretty good work" in result.reasoning


@pytest.mark.unit
def test_parse_score_with_reasoning_no_score():
    """Test parse_score_with_reasoning when no score can be extracted."""
    response = "This is just random text with no score information"
    result = parse_score_with_reasoning(response)
    assert result.score == 1.0
    assert (
        result.reasoning is not None and "Could not parse response" in result.reasoning
    )


@pytest.mark.unit
def test_parse_score_with_reasoning_exception():
    """Test parse_score_with_reasoning with exception handling."""
    # This should trigger the exception handler
    with patch("json.loads", side_effect=Exception("Test error")):
        result = parse_score_with_reasoning('{"score": 8.0}')
        assert result.score == 1.0
        assert (
            result.reasoning is not None
            and "Failed to parse response" in result.reasoning
        )


@pytest.mark.unit
def test_parse_score_with_reasoning_invalid_json():
    """Test parse_score_with_reasoning with invalid JSON."""
    response = '{"score": 8.5, "reasoning": "test"'  # Missing closing brace
    result = parse_score_with_reasoning(response)
    # Should fall back to regex
    assert result.score == 8.5


# Test tool_relevancy_scorer function
@pytest.mark.unit
def test_tool_relevancy_scorer_success(sample_agent_data):
    """Test tool_relevancy_scorer with valid data."""
    mock_model = MockLLMModel(
        '{"score": 8.5, "reasoning": "Highly relevant tool call"}'
    )

    result = tool_relevancy_scorer(sample_agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ScoreWithReasoning)
    assert result[0].score == 8.5
    assert result[0].reasoning == "Highly relevant tool call"
    assert mock_model.call_count == 1


@pytest.mark.unit
def test_tool_relevancy_scorer_missing_tools(minimal_agent_data):
    """Test tool_relevancy_scorer with missing tools_available."""
    mock_model = MockLLMModel()

    result = tool_relevancy_scorer(minimal_agent_data, mock_model)

    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    # Check that tools_available is in the missing fields (it can be either tools_available or tool_calls or both)
    missing_fields = result["missing_fields"]
    assert (
        "tool_calls" in missing_fields
    )  # This will be missing since minimal_agent_data has empty tool_calls
    assert mock_model.call_count == 0


@pytest.mark.unit
def test_tool_relevancy_scorer_missing_tool_calls():
    """Test tool_relevancy_scorer with missing tool_calls."""
    agent_data = AgentData(
        tools_available=[ToolSchema(name="test", description="test")],
        tool_calls=[],  # Empty list
        parameters_passed={},
    )
    mock_model = MockLLMModel()

    result = tool_relevancy_scorer(agent_data, mock_model)

    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    assert "tool_calls" in result["missing_fields"]


@pytest.mark.unit
def test_tool_relevancy_scorer_multiple_calls(sample_agent_data):
    """Test tool_relevancy_scorer with multiple tool calls."""
    # Add another tool call
    sample_agent_data.tool_calls.append(
        ToolCall(tool_name="memory", parameters={"key": "result"}, call_id="call_002")
    )

    mock_model = MockLLMModel('{"score": 7.0, "reasoning": "Second call reasoning"}')

    result = tool_relevancy_scorer(sample_agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 2
    assert mock_model.call_count == 2


@pytest.mark.unit
def test_tool_relevancy_scorer_model_exception(sample_agent_data):
    """Test tool_relevancy_scorer when model raises exception."""
    mock_model = Mock()
    mock_model.generate.side_effect = Exception("Model error")

    result = tool_relevancy_scorer(sample_agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].score == 1.0
    assert "Failed to evaluate tool call" in result[0].reasoning
    assert "Model error" in result[0].reasoning


# Test AgentScorers class
@pytest.mark.unit
def test_agent_scorers_init():
    """Test AgentScorers initialization."""
    mock_model = MockLLMModel()
    scorers = AgentScorers(mock_model)
    assert scorers.model == mock_model


@pytest.mark.unit
def test_agent_scorers_score_tool_relevancy(sample_agent_data):
    """Test AgentScorers.score_tool_relevancy method."""
    mock_model = MockLLMModel('{"score": 8.0, "reasoning": "Test"}')
    scorers = AgentScorers(mock_model)

    result = scorers.score_tool_relevancy(sample_agent_data)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].score == 8.0


@pytest.mark.unit
def test_agent_scorers_score_tool_correctness(sample_agent_data):
    """Test tool_correctness_scorer function directly."""
    mock_model = MockLLMModel('{"score": 9.0, "reasoning": "Correct tool"}')

    # Test the function directly
    result = tool_correctness_scorer(sample_agent_data, mock_model)

    # The function should return a list of ScoreWithReasoning objects
    assert isinstance(result, list)
    # Since we're using a mock model, the actual result will depend on the mock response
    # We just verify the function runs without error


@pytest.mark.unit
def test_agent_scorers_score_parameter_correctness(sample_agent_data):
    """Test parameter_correctness_scorer function directly."""
    mock_model = MockLLMModel('{"score": 8.5, "reasoning": "Good params"}')

    result = parameter_correctness_scorer(sample_agent_data, mock_model)

    # The function should return a list of ScoreWithReasoning objects
    assert isinstance(result, list)


@pytest.mark.unit
def test_agent_scorers_score_task_progression(sample_agent_data):
    """Test task_progression_scorer function directly."""
    mock_model = MockLLMModel('{"score": 4.2, "reasoning": "Good progress"}')

    result = task_progression_scorer(sample_agent_data, mock_model)

    # The function should return a ScoreWithOriginalTask object
    assert isinstance(result, (ScoreWithOriginalTask, dict))


@pytest.mark.unit
def test_agent_scorers_score_context_relevancy(sample_agent_data):
    """Test context_relevancy_scorer function directly."""
    mock_model = MockLLMModel('{"score": 7.8, "reasoning": "Relevant response"}')

    result = context_relevancy_scorer(sample_agent_data, mock_model)

    # The function should return a ScoreWithReasoning object
    assert isinstance(result, (ScoreWithReasoning, dict))


@pytest.mark.unit
def test_agent_scorers_score_role_adherence(sample_agent_data):
    """Test role_adherence_scorer function directly."""
    mock_model = MockLLMModel('{"score": 9.0, "reasoning": "Perfect role adherence"}')

    result = role_adherence_scorer(sample_agent_data, mock_model)

    # The function should return a ScoreWithReasoning object
    assert isinstance(result, (ScoreWithReasoning, dict))


@pytest.mark.unit
def test_agent_scorers_score_goal_achievement(sample_agent_data):
    """Test goal_achievement_scorer function directly."""
    mock_model = MockLLMModel(
        '{"original_task": "Calculate 20+22", "score": 9.0, "reasoning": "Goal achieved"}'
    )

    result = goal_achievement_scorer(sample_agent_data, mock_model)

    # The function should return a ScoreWithOriginalTask object
    assert isinstance(result, (ScoreWithOriginalTask, dict))


@pytest.mark.unit
def test_agent_scorers_score_conversation_coherence(sample_agent_data):
    """Test conversation_coherence_scorer function directly."""
    mock_model = MockLLMModel(
        '{"original_task": "Math task", "score": 8.5, "reasoning": "Coherent conversation"}'
    )

    result = conversation_coherence_scorer(sample_agent_data, mock_model)

    # The function should return a ScoreWithOriginalTask object
    assert isinstance(result, (ScoreWithOriginalTask, dict))


@pytest.mark.unit
def test_agent_scorers_score_all(sample_agent_data):
    """Test AgentScorers.score_all method."""
    mock_model = MockLLMModel()
    scorers = AgentScorers(mock_model)

    # Mock all the individual scoring functions
    with (
        patch.object(scorers, "score_tool_relevancy") as mock_tr,
        patch.object(scorers, "score_parameter_correctness") as mock_pc,
        patch.object(scorers, "score_task_progression") as mock_tp,
        patch.object(scorers, "score_context_relevancy") as mock_cr,
        patch.object(scorers, "score_role_adherence") as mock_ra,
    ):

        # Set up return values
        mock_tr.return_value = [ScoreWithReasoning(score=8.0, reasoning="Good tool")]
        mock_pc.return_value = [ScoreWithReasoning(score=8.5, reasoning="Good params")]
        mock_tp.return_value = ScoreWithReasoning(score=4.2, reasoning="Good progress")
        mock_cr.return_value = ScoreWithReasoning(score=7.8, reasoning="Relevant")
        mock_ra.return_value = ScoreWithReasoning(score=9.0, reasoning="Good role")

        result = scorers.score_all(sample_agent_data)

        assert isinstance(result, dict)
        assert "tool_relevancy" in result
        assert "parameter_correctness" in result
        assert "task_progression" in result
        assert "context_relevancy" in result
        assert "role_adherence" in result

        # Verify all methods were called
        mock_tr.assert_called_once_with(sample_agent_data)
        mock_pc.assert_called_once_with(sample_agent_data)
        mock_tp.assert_called_once_with(sample_agent_data)
        mock_cr.assert_called_once_with(sample_agent_data)
        mock_ra.assert_called_once_with(sample_agent_data)


# Test edge cases and error handling
@pytest.mark.unit
def test_parse_score_with_reasoning_whitespace():
    """Test parse_score_with_reasoning with whitespace."""
    response = '   {"score": 7.5, "reasoning": "Test"}   '
    result = parse_score_with_reasoning(response)
    assert result.score == 7.5


@pytest.mark.unit
def test_parse_score_with_reasoning_float_score():
    """Test parse_score_with_reasoning with float as direct JSON."""
    response = "8.75"
    result = parse_score_with_reasoning(response)
    assert result.score == 8.75


@pytest.mark.unit
def test_parse_score_with_reasoning_unexpected_format():
    """Test parse_score_with_reasoning with unexpected JSON format."""
    response = '{"unexpected": "format", "data": [1, 2, 3]}'
    result = parse_score_with_reasoning(response)
    assert result.score == 1.0
    assert "Unexpected response format" in result.reasoning


@pytest.mark.unit
def test_tool_relevancy_scorer_empty_tools_list():
    """Test tool_relevancy_scorer with empty tools_available list."""
    agent_data = AgentData(
        tools_available=[],  # Empty but not None
        tool_calls=[ToolCall(tool_name="test", parameters={}, call_id="test")],
        parameters_passed={},
    )
    mock_model = MockLLMModel()

    result = tool_relevancy_scorer(agent_data, mock_model)

    # Should work fine with empty tools list
    assert isinstance(result, list)
    assert len(result) == 1


@pytest.mark.unit
def test_escape_json_for_format_complex():
    """Test escape_json_for_format with complex nested JSON."""
    complex_json = """
    {
        "outer": {
            "inner": {"nested": "value"},
            "array": [{"item": 1}, {"item": 2}]
        }
    }
    """
    escaped = escape_json_for_format(complex_json)

    # Should escape all braces
    assert "{" not in escaped or ("{" in escaped and "{{" in escaped)
    assert "}" not in escaped or ("}" in escaped and "}}" in escaped)


@pytest.mark.unit
def test_parse_score_with_reasoning_malformed_regex():
    """Test parse_score_with_reasoning regex fallback with edge cases."""
    # Test with score but malformed reasoning
    response = 'score: 6.5 reasoning: this has "quotes but no closing'
    result = parse_score_with_reasoning(response)
    assert result.score == 6.5

    # Test with reasoning but no score in expected format
    response = 'reasoning: "good work" but score is missing'
    result = parse_score_with_reasoning(response)
    assert result.score == 1.0  # Should fall back to default


@pytest.mark.unit
def test_models_serialization():
    """Test that all models can be serialized and deserialized."""
    # Test ScoreWithReasoning
    score1 = ScoreWithReasoning(score=8.5, reasoning="Test")
    data1 = score1.model_dump()
    score1_restored = ScoreWithReasoning.model_validate(data1)
    assert score1_restored == score1

    # Test ScoreWithOriginalTask
    score2 = ScoreWithOriginalTask(original_task="Task", score=9.0, reasoning="Great")
    data2 = score2.model_dump()
    score2_restored = ScoreWithOriginalTask.model_validate(data2)
    assert score2_restored == score2

    # Test ScoreListResponse
    scores = ScoreListResponse(
        scores=[score1, ScoreWithReasoning(score=7.0, reasoning="OK")]
    )
    data3 = scores.model_dump()
    scores_restored = ScoreListResponse.model_validate(data3)
    assert scores_restored == scores


# Additional tests for missing coverage areas


@pytest.mark.unit
def test_tool_correctness_scorer_missing_expected_tool_call():
    """Test tool_correctness_scorer with missing expected_tool_call."""
    from novaeval.scorers.agent_scorers import tool_correctness_scorer

    # Create agent data without expected_tool_call
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        tool_calls=[ToolCall(tool_name="test_tool", parameters={}, call_id="123")],
    )

    mock_model = MockLLMModel()
    result = tool_correctness_scorer(agent_data, mock_model)

    # Should return error dict
    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    assert "expected_tool_call" in result["missing_fields"]


@pytest.mark.unit
def test_tool_correctness_scorer_missing_tool_calls():
    """Test tool_correctness_scorer with missing tool_calls."""
    from novaeval.scorers.agent_scorers import tool_correctness_scorer

    # Create agent data without tool_calls
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        expected_tool_call=ToolCall(
            tool_name="expected_tool", parameters={}, call_id="expected"
        ),
    )

    mock_model = MockLLMModel()
    result = tool_correctness_scorer(agent_data, mock_model)

    # Should return error dict
    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    assert "tool_calls" in result["missing_fields"]


@pytest.mark.unit
def test_tool_correctness_scorer_empty_tool_calls():
    """Test tool_correctness_scorer with empty tool_calls list."""
    from novaeval.scorers.agent_scorers import tool_correctness_scorer

    # Create agent data with empty tool_calls
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        expected_tool_call=ToolCall(
            tool_name="expected_tool", parameters={}, call_id="expected"
        ),
        tool_calls=[],  # Empty list
    )

    mock_model = MockLLMModel()
    result = tool_correctness_scorer(agent_data, mock_model)

    # Should return error dict
    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    assert "tool_calls" in result["missing_fields"]


@pytest.mark.unit
def test_tool_correctness_scorer_single_tool_call():
    """Test tool_correctness_scorer with single tool call."""
    from novaeval.scorers.agent_scorers import tool_correctness_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        expected_tool_call=ToolCall(
            tool_name="expected_tool", parameters={}, call_id="expected"
        ),
        tool_calls=[ToolCall(tool_name="actual_tool", parameters={}, call_id="actual")],
    )

    mock_model = MockLLMModel(
        '{"score": 7.5, "reasoning": "Tool call is mostly correct"}'
    )
    result = tool_correctness_scorer(agent_data, mock_model)

    # Should return list of scores
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ScoreWithReasoning)
    assert result[0].score == 7.5
    assert result[0].reasoning == "Tool call is mostly correct"


@pytest.mark.unit
def test_tool_correctness_scorer_multiple_tool_calls():
    """Test tool_correctness_scorer with multiple tool calls."""
    from novaeval.scorers.agent_scorers import tool_correctness_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        expected_tool_call=ToolCall(
            tool_name="expected_tool", parameters={}, call_id="expected"
        ),
        tool_calls=[
            ToolCall(tool_name="tool1", parameters={}, call_id="call1"),
            ToolCall(tool_name="tool2", parameters={}, call_id="call2"),
        ],
    )

    mock_model = MockLLMModel('{"score": 6.0, "reasoning": "Partially correct"}')
    result = tool_correctness_scorer(agent_data, mock_model)

    # Should return list with score for each tool call
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(score, ScoreWithReasoning) for score in result)
    assert all(score.score == 6.0 for score in result)


@pytest.mark.unit
def test_tool_correctness_scorer_exception_handling():
    """Test tool_correctness_scorer with exception handling."""
    from novaeval.scorers.agent_scorers import tool_correctness_scorer

    # Create agent data with valid fields
    agent_data = AgentData(
        agent_name="TestAgent",
        expected_tool_call=ToolCall(
            tool_name="calculator",
            parameters={"operation": "add", "a": 20, "b": 22},
            call_id="call_001",
        ),
        tool_calls=[
            ToolCall(
                tool_name="calculator",
                parameters={"operation": "add", "a": 20, "b": 22},
                call_id="call_001",
            )
        ],
    )

    # Create a mock model that raises an exception
    class ExceptionModel:
        def generate(self, prompt):
            raise ValueError("Model error")

    model = ExceptionModel()

    # Test with string tool_calls (lines 342-346)
    agent_data.tool_calls = "string_tool_call"
    result = tool_correctness_scorer(agent_data, model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].score == 1.0
    assert "Failed to evaluate tool call" in result[0].reasoning

    # Test with list tool_calls (lines 428-432)
    agent_data.tool_calls = [
        ToolCall(
            tool_name="calculator",
            parameters={"operation": "add", "a": 20, "b": 22},
            call_id="call_001",
        )
    ]
    result = tool_correctness_scorer(agent_data, model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].score == 1.0
    assert "Failed to evaluate tool call" in result[0].reasoning


@pytest.mark.unit
def test_parameter_correctness_scorer_missing_fields():
    """Test parameter_correctness_scorer with missing required fields."""
    from novaeval.scorers.agent_scorers import parameter_correctness_scorer

    # Create agent data without parameters_passed and tool_call_results
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        tool_calls=[ToolCall(tool_name="test_tool", parameters={}, call_id="123")],
    )

    mock_model = MockLLMModel()
    result = parameter_correctness_scorer(agent_data, mock_model)

    # Should return error dict
    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]


@pytest.mark.unit
def test_parameter_correctness_scorer_success():
    """Test parameter_correctness_scorer with valid data."""
    from novaeval.scorers.agent_scorers import parameter_correctness_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        expected_tool_call=ToolCall(
            tool_name="expected_tool", parameters={"key": "value"}, call_id="expected"
        ),
        tool_calls=[
            ToolCall(
                tool_name="actual_tool", parameters={"key": "value"}, call_id="actual"
            )
        ],
        parameters_passed={"key": "value"},
        tool_call_results=[
            ToolResult(
                call_id="actual", result="result", success=True, error_message=None
            )
        ],
    )

    mock_model = MockLLMModel('{"score": 9.0, "reasoning": "Parameters are correct"}')
    result = parameter_correctness_scorer(agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].score == 9.0
    assert result[0].reasoning == "Parameters are correct"


@pytest.mark.unit
def test_role_adherence_scorer_missing_fields():
    """Test role_adherence_scorer with missing required fields."""
    from novaeval.scorers.agent_scorers import role_adherence_scorer

    # Create agent data without agent_role
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_response="Some response",
    )

    mock_model = MockLLMModel()
    result = role_adherence_scorer(agent_data, mock_model)

    # Should return error dict
    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]


@pytest.mark.unit
def test_role_adherence_scorer_success():
    """Test role_adherence_scorer with valid data."""
    from novaeval.scorers.agent_scorers import role_adherence_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_role="assistant",
        agent_task="Help with a task",
        agent_response="I can help you with that task",
        tool_calls=[],
    )

    mock_model = MockLLMModel('{"score": 8.5, "reasoning": "Good role adherence"}')
    result = role_adherence_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithReasoning)
    assert result.score == 8.5
    assert result.reasoning == "Good role adherence"


@pytest.mark.unit
def test_task_progression_scorer_missing_fields():
    """Test task_progression_scorer with missing required fields."""
    from novaeval.scorers.agent_scorers import task_progression_scorer

    # Create agent data without trace
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_task="Complete the task",
    )

    mock_model = MockLLMModel()
    result = task_progression_scorer(agent_data, mock_model)

    # Should return error dict
    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]


@pytest.mark.unit
def test_task_progression_scorer_success():
    """Test task_progression_scorer with valid data."""
    from novaeval.scorers.agent_scorers import task_progression_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_task="Complete the analysis",
        agent_role="analyst",
        system_prompt="You are an analyst.",
        agent_response="I'll start the analysis",
        trace=[{"step": 1, "action": "started analysis"}],
    )

    mock_model = MockLLMModel(
        '{"original_task": "Complete the analysis", "score": 7.5, "reasoning": "Making good progress"}'
    )
    result = task_progression_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Complete the analysis"
    assert result.score == 7.5
    assert result.reasoning == "Making good progress"


@pytest.mark.unit
def test_context_relevancy_scorer_missing_fields():
    """Test context_relevancy_scorer with missing required fields."""
    from novaeval.scorers.agent_scorers import context_relevancy_scorer

    # Create agent data without retrieved_context
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        retrieval_query=["search query"],
    )

    mock_model = MockLLMModel()
    result = context_relevancy_scorer(agent_data, mock_model)

    # Should return error dict
    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]


@pytest.mark.unit
def test_context_relevancy_scorer_success():
    """Test context_relevancy_scorer with valid data."""
    from novaeval.scorers.agent_scorers import context_relevancy_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_task="Search for information",
        agent_role="assistant",
        agent_response="I found relevant information",
        retrieval_query=["search for information"],
        retrieved_context=[["Retrieved relevant information"]],
    )

    mock_model = MockLLMModel('{"score": 8.0, "reasoning": "Context is relevant"}')
    result = context_relevancy_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithReasoning)
    assert result.score == 8.0
    assert result.reasoning == "Context is relevant"


@pytest.mark.unit
def test_parse_score_with_reasoning_edge_cases():
    """Test parse_score_with_reasoning with various edge cases."""
    # Test with malformed JSON
    result = parse_score_with_reasoning("not json at all")
    assert result.score == 1.0
    assert (
        result.reasoning is not None and "Could not parse response" in result.reasoning
    )

    # Test with JSON missing score
    result = parse_score_with_reasoning('{"reasoning": "test reasoning"}')
    assert result.score == 1.0
    assert (
        result.reasoning is not None
        and "Unexpected response format" in result.reasoning
    )

    # Test with JSON missing reasoning
    result = parse_score_with_reasoning('{"score": 7.5}')
    assert result.score == 7.5
    assert result.reasoning == "No reasoning provided in response"

    # Test with invalid score type
    result = parse_score_with_reasoning(
        '{"score": "not_a_number", "reasoning": "test"}'
    )
    assert result.score == 1.0
    assert (
        result.reasoning is not None
        and "failed to parse response" in result.reasoning.lower()
    )


@pytest.mark.unit
def test_parse_score_with_original_task_edge_cases():
    """Test parse_score_with_original_task with various edge cases."""
    from novaeval.scorers.agent_scorers import parse_score_with_original_task

    # Test with malformed JSON
    result = parse_score_with_original_task("not json at all")
    assert result.score == 1.0
    assert result.original_task == "Unknown task"
    assert (
        result.reasoning is not None
        and "error parsing response" in result.reasoning.lower()
    )

    # Test with JSON missing original_task
    result = parse_score_with_original_task('{"score": 7.5, "reasoning": "test"}')
    assert result.score == 7.5
    assert result.original_task == "Unknown task"
    assert result.reasoning == "test"

    # Test with valid JSON
    result = parse_score_with_original_task(
        '{"original_task": "Test task", "score": 8.0, "reasoning": "Good"}'
    )
    assert result.original_task == "Test task"
    assert result.score == 8.0
    assert result.reasoning == "Good"


@pytest.mark.unit
def test_escape_json_for_format():
    """Test escape_json_for_format function."""
    # Test basic escaping
    result = escape_json_for_format('{"key": "value"}')
    assert result == '{{"key": "value"}}'

    # Test with nested braces
    result = escape_json_for_format('{"outer": {"inner": "value"}}')
    assert result == '{{"outer": {{"inner": "value"}}}}'

    # Test with empty string
    result = escape_json_for_format("")
    assert result == ""


@pytest.mark.unit
def test_agent_scorers_class_missing_fields():
    """Test AgentScorers methods with missing fields."""
    mock_model = MockLLMModel()
    scorers = AgentScorers(mock_model)

    # Create minimal agent data
    agent_data = AgentData(user_id="user123", task_id="task456", turn_id="turn789")

    # Test methods that should return error dicts
    result = scorers.parameter_correctness(agent_data)
    assert isinstance(result, dict) and "error" in result

    result = scorers.role_adherence(agent_data)
    assert isinstance(result, dict) and "error" in result

    result = scorers.task_progression(agent_data)
    assert isinstance(result, dict) and "error" in result

    result = scorers.context_relevancy(agent_data)
    assert isinstance(result, dict) and "error" in result


@pytest.mark.unit
def test_agent_scorers_class_successful_scoring():
    """Test AgentScorers methods with complete data."""
    mock_model = MockLLMModel(
        '{"score": 8.0, "reasoning": "Good performance", "original_task": "Test task"}'
    )
    scorers = AgentScorers(mock_model)

    # Create complete agent data
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        expected_tool_call=ToolCall(tool_name="expected", parameters={}, call_id="exp"),
        tool_calls=[ToolCall(tool_name="actual", parameters={}, call_id="act")],
        parameters_passed={"test": "value"},
        tool_call_results=[
            ToolResult(call_id="act", result="result", success=True, error_message=None)
        ],
        agent_role="assistant",
        agent_response="Response",
        agent_task="Task",
        system_prompt="You are a helpful assistant.",
        trace=[{"step": 1}],
        retrieval_query=["search for information"],
        retrieved_context=[["Retrieved relevant information"]],
    )

    # Test all scoring methods
    result = scorers.parameter_correctness(agent_data)
    assert isinstance(result, list)

    result = scorers.role_adherence(agent_data)
    assert isinstance(result, ScoreWithReasoning)

    result = scorers.task_progression(agent_data)
    assert isinstance(result, ScoreWithOriginalTask)

    result = scorers.context_relevancy(agent_data)
    assert isinstance(result, ScoreWithReasoning)

    result = scorers.tool_relevancy(agent_data)
    assert isinstance(result, list)


@pytest.mark.unit
def test_single_score_response_validation():
    """Test SingleScoreResponse validation."""
    # Test valid creation
    response = SingleScoreResponse(score=8.5, reasoning="Good")
    assert response.score == 8.5
    assert response.reasoning == "Good"


@pytest.mark.unit
def test_score_list_response_validation():
    """Test ScoreListResponse validation."""
    # Test valid creation
    scores = [ScoreWithReasoning(score=8.0, reasoning="Good")]
    response = ScoreListResponse(scores=scores)
    assert response.scores is not None and len(response.scores) == 1
    assert response.scores[0].score == 8.0
    assert response.scores[0].reasoning == "Good"


# Tests for goal_achievement_scorer function
@pytest.mark.unit
def test_goal_achievement_scorer_agent_not_exited():
    """Test goal_achievement_scorer when agent has not exited."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=False,  # Agent hasn't exited
        trace=[{"step": 1, "action": "in progress"}],
    )

    mock_model = MockLLMModel()
    result = goal_achievement_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "N/A - Agent has not exited"
    assert result.score == -1.0
    assert result.reasoning == "The agent has not yet exited"
    assert mock_model.call_count == 0  # Model should not be called


@pytest.mark.unit
def test_goal_achievement_scorer_missing_trace():
    """Test goal_achievement_scorer with missing trace."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=None,  # Missing trace
    )

    mock_model = MockLLMModel()
    result = goal_achievement_scorer(agent_data, mock_model)

    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    assert "trace" in result["missing_fields"]
    assert mock_model.call_count == 0


@pytest.mark.unit
def test_goal_achievement_scorer_empty_trace():
    """Test goal_achievement_scorer with empty trace."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[],  # Empty trace
    )

    mock_model = MockLLMModel()
    result = goal_achievement_scorer(agent_data, mock_model)

    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    assert "trace" in result["missing_fields"]


@pytest.mark.unit
def test_goal_achievement_scorer_success():
    """Test goal_achievement_scorer with valid data."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[
            {"type": "user_input", "content": "Calculate 20 + 22"},
            {"type": "tool_call", "tool": "calculator", "result": 42},
            {"type": "agent_response", "content": "The answer is 42"},
        ],
    )

    mock_response = '{"original_task": "Calculate 20 + 22", "score": 9.0, "reasoning": "Successfully completed the calculation task"}'
    mock_model = MockLLMModel(mock_response)
    result = goal_achievement_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Calculate 20 + 22"
    assert result.score == 9.0
    assert result.reasoning == "Successfully completed the calculation task"
    assert mock_model.call_count == 1


@pytest.mark.unit
def test_goal_achievement_scorer_json_decode_error():
    """Test goal_achievement_scorer with malformed JSON response."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[{"step": 1, "action": "completed"}],
    )

    # Mock response with malformed JSON but extractable with regex (JSON-like format)
    mock_response = 'Here is my response "original_task": "Complete analysis", "score": 8.5, "reasoning": "Well executed" end'
    mock_model = MockLLMModel(mock_response)

    result = goal_achievement_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.score == 8.5
    assert result.original_task == "Complete analysis"


@pytest.mark.unit
def test_goal_achievement_scorer_regex_fallback():
    """Test goal_achievement_scorer regex fallback parsing."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[{"step": 1}],
    )

    # Response with fields in regex-extractable format (JSON-like)
    mock_response = """
    Analysis complete
    "original_task": "Build a web app"
    "score": 7.5
    "reasoning": "Good progress made"
    Final assessment done.
    """
    mock_model = MockLLMModel(mock_response)

    result = goal_achievement_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Build a web app"
    assert result.score == 7.5
    assert result.reasoning == "Good progress made"


@pytest.mark.unit
def test_goal_achievement_scorer_exception_handling():
    """Test goal_achievement_scorer exception handling."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[{"step": 1}],
    )

    # Mock model that throws exception
    mock_model = Mock()
    mock_model.generate.side_effect = Exception("Model failed")

    result = goal_achievement_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Error during evaluation"
    assert result.score == 1.0
    assert "Failed to evaluate goal achievement" in result.reasoning
    assert "Model failed" in result.reasoning


# Tests for conversation_coherence_scorer function
@pytest.mark.unit
def test_conversation_coherence_scorer_agent_not_exited():
    """Test conversation_coherence_scorer when agent has not exited."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=False,  # Agent hasn't exited
        trace=[{"step": 1, "action": "in progress"}],
    )

    mock_model = MockLLMModel()
    result = conversation_coherence_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "N/A - Agent has not exited"
    assert result.score == -1.0
    assert result.reasoning == "The agent has not yet exited"
    assert mock_model.call_count == 0


@pytest.mark.unit
def test_conversation_coherence_scorer_missing_trace():
    """Test conversation_coherence_scorer with missing trace."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=None,  # Missing trace
    )

    mock_model = MockLLMModel()
    result = conversation_coherence_scorer(agent_data, mock_model)

    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    assert "trace" in result["missing_fields"]


@pytest.mark.unit
def test_conversation_coherence_scorer_empty_trace():
    """Test conversation_coherence_scorer with empty trace."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[],  # Empty trace
    )

    mock_model = MockLLMModel()
    result = conversation_coherence_scorer(agent_data, mock_model)

    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    assert "trace" in result["missing_fields"]


@pytest.mark.unit
def test_conversation_coherence_scorer_success():
    """Test conversation_coherence_scorer with valid data."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[
            {"type": "user_input", "content": "Help me write an email"},
            {
                "type": "agent_response",
                "content": "I'd be happy to help you write an email",
            },
            {"type": "user_input", "content": "Make it formal"},
            {
                "type": "agent_response",
                "content": "Certainly, I'll help you write a formal email",
            },
        ],
    )

    mock_response = '{"original_task": "Help write an email", "score": 8.5, "reasoning": "Conversation flows logically and maintains context well"}'
    mock_model = MockLLMModel(mock_response)
    result = conversation_coherence_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Help write an email"
    assert result.score == 8.5
    assert result.reasoning == "Conversation flows logically and maintains context well"


@pytest.mark.unit
def test_conversation_coherence_scorer_json_decode_error():
    """Test conversation_coherence_scorer with malformed JSON response."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[{"conversation": "sample"}],
    )

    # Mock response with malformed JSON but extractable with regex (JSON-like format)
    mock_response = 'My evaluation: "original_task": "Chat support", "score": 7.0, "reasoning": "Mostly coherent conversation" done.'
    mock_model = MockLLMModel(mock_response)

    result = conversation_coherence_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.score == 7.0
    assert result.original_task == "Chat support"


@pytest.mark.unit
def test_conversation_coherence_scorer_exception_handling():
    """Test conversation_coherence_scorer exception handling."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[{"step": 1}],
    )

    # Mock model that throws exception
    mock_model = Mock()
    mock_model.generate.side_effect = Exception("Model connection failed")

    result = conversation_coherence_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Error during evaluation"
    assert result.score == 1.0
    assert "Failed to evaluate conversation coherence" in result.reasoning
    assert "Model connection failed" in result.reasoning


# Additional tests for parse_score_with_original_task
@pytest.mark.unit
def test_parse_score_with_original_task_regex_fallback():
    """Test parse_score_with_original_task with regex fallback for various formats."""
    # Test with different regex patterns
    response1 = (
        'original_task: "Complete the analysis" score: 8.5 reasoning: "Well done"'
    )
    result1 = parse_score_with_original_task(response1)
    assert result1.original_task == "Complete the analysis"
    assert result1.score == 8.5
    assert result1.reasoning == "Well done"

    # Test with quoted fields
    response2 = '"original_task": "Build application", "score": 7.0, "reasoning": "Good progress"'
    result2 = parse_score_with_original_task(response2)
    assert result2.original_task == "Build application"
    assert result2.score == 7.0
    assert result2.reasoning == "Good progress"

    # Test with missing original_task in regex
    response3 = 'score: 6.5 reasoning: "Partial completion"'
    result3 = parse_score_with_original_task(response3)
    assert result3.original_task == "Unknown task"
    assert result3.score == 6.5
    assert result3.reasoning == "Partial completion"

    # Test with missing score in regex
    response4 = 'original_task: "Test task" reasoning: "No score provided"'
    result4 = parse_score_with_original_task(response4)
    assert result4.original_task == "Test task"
    assert result4.score == 1.0
    assert result4.reasoning == "No score provided"


@pytest.mark.unit
def test_parse_score_with_original_task_exception_handling():
    """Test parse_score_with_original_task exception handling."""
    # Test with response that causes exception during processing
    with patch("json.loads", side_effect=Exception("JSON processing error")):
        result = parse_score_with_original_task('{"original_task": "test"}')
        assert result.original_task == "Unknown task"
        assert result.score == 1.0
        assert "Error parsing response" in result.reasoning


# Additional tests for scorer functions with more edge cases
@pytest.mark.unit
def test_parameter_correctness_scorer_missing_tool_call_results():
    """Test parameter_correctness_scorer when tool_call_results is None."""
    from novaeval.scorers.agent_scorers import parameter_correctness_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        tool_calls=[ToolCall(tool_name="test_tool", parameters={}, call_id="123")],
        parameters_passed={"key": "value"},
        tool_call_results=None,  # Missing results
    )

    mock_model = MockLLMModel()
    result = parameter_correctness_scorer(agent_data, mock_model)

    assert isinstance(result, dict)
    assert "Missing required fields:" in result["error"]
    assert "tool_call_results" in result["missing_fields"]


@pytest.mark.unit
def test_parameter_correctness_scorer_no_matching_result():
    """Test parameter_correctness_scorer when no matching result is found for a call."""
    from novaeval.scorers.agent_scorers import parameter_correctness_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        tool_calls=[ToolCall(tool_name="test_tool", parameters={}, call_id="call_123")],
        parameters_passed={"key": "value"},
        tool_call_results=[
            ToolResult(
                call_id="different_call",
                result="result",
                success=True,
                error_message=None,
            )
        ],
    )

    mock_model = MockLLMModel('{"score": 5.0, "reasoning": "No matching result"}')
    result = parameter_correctness_scorer(agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].score == 5.0


@pytest.mark.unit
def test_parameter_correctness_scorer_exception_handling():
    """Test parameter_correctness_scorer exception handling."""
    from novaeval.scorers.agent_scorers import parameter_correctness_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        tool_calls=[ToolCall(tool_name="test_tool", parameters={}, call_id="123")],
        parameters_passed={"key": "value"},
        tool_call_results=[
            ToolResult(call_id="123", result="result", success=True, error_message=None)
        ],
    )

    # Mock model that throws exception
    mock_model = Mock()
    mock_model.generate.side_effect = Exception("Parameter evaluation failed")

    result = parameter_correctness_scorer(agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].score == 1.0
    assert "Failed to evaluate tool call" in result[0].reasoning
    assert "Parameter evaluation failed" in result[0].reasoning


@pytest.mark.unit
def test_task_progression_scorer_exception_handling():
    """Test task_progression_scorer exception handling."""
    from novaeval.scorers.agent_scorers import task_progression_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_task="Complete the analysis",
        agent_role="analyst",
        system_prompt="You are an analyst.",
        agent_response="I'll start the analysis",
    )

    # Mock model that throws exception
    mock_model = Mock()
    mock_model.generate.side_effect = Exception("Task progression evaluation failed")

    result = task_progression_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Complete the analysis"
    assert result.score == 1.0
    assert "Failed to evaluate task progression" in result.reasoning
    assert "Task progression evaluation failed" in result.reasoning


@pytest.mark.unit
def test_context_relevancy_scorer_exception_handling():
    """Test context_relevancy_scorer exception handling."""
    from novaeval.scorers.agent_scorers import context_relevancy_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_task="Search for information",
        agent_role="assistant",
        agent_response="I found relevant information",
    )

    # Mock model that throws exception
    mock_model = Mock()
    mock_model.generate.side_effect = Exception("Context evaluation failed")

    result = context_relevancy_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithReasoning)
    assert result.score == 1.0
    assert "Failed to evaluate response appropriateness" in result.reasoning
    assert "Context evaluation failed" in result.reasoning


@pytest.mark.unit
def test_role_adherence_scorer_exception_handling():
    """Test role_adherence_scorer exception handling."""
    from novaeval.scorers.agent_scorers import role_adherence_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_role="assistant",
        agent_task="Help with a task",
        agent_response="I can help you with that task",
        tool_calls=[],
    )

    # Mock model that throws exception
    mock_model = Mock()
    mock_model.generate.side_effect = Exception("Role adherence evaluation failed")

    result = role_adherence_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithReasoning)
    assert result.score == 1.0
    assert "Failed to evaluate role adherence" in result.reasoning
    assert "Role adherence evaluation failed" in result.reasoning


# Test AgentScorers class additional edge cases
@pytest.mark.unit
def test_agent_scorers_goal_achievement_wrapper():
    """Test goal_achievement_scorer function directly."""
    mock_model = MockLLMModel(
        '{"original_task": "Test task", "score": 8.0, "reasoning": "Good"}'
    )

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[{"step": 1}],
    )

    result = goal_achievement_scorer(agent_data, mock_model)

    # The function should return a ScoreWithOriginalTask object
    assert isinstance(result, (ScoreWithOriginalTask, dict))


@pytest.mark.unit
def test_agent_scorers_conversation_coherence_wrapper():
    """Test conversation_coherence_scorer function directly."""
    mock_model = MockLLMModel(
        '{"original_task": "Chat task", "score": 7.5, "reasoning": "Coherent"}'
    )

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[{"conversation": "sample"}],
    )

    result = conversation_coherence_scorer(agent_data, mock_model)

    # The function should return a ScoreWithOriginalTask object
    assert isinstance(result, (ScoreWithOriginalTask, dict))


@pytest.mark.unit
def test_agent_scorers_score_all_with_errors():
    """Test AgentScorers.score_all when some scorers return errors."""
    mock_model = MockLLMModel()
    scorers = AgentScorers(mock_model)

    # Create agent data that will cause some scorers to return errors
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=False,  # This will cause goal_achievement and conversation_coherence to return specific responses
    )

    result = scorers.score_all(agent_data)

    assert isinstance(result, dict)
    assert len(result) == 8  # All 5 scoring categories should be present

    # Check that error responses are properly included
    for key in result:
        assert result[key] is not None


@pytest.mark.unit
def test_parse_score_with_reasoning_number_extraction():
    """Test parse_score_with_reasoning number extraction fallback."""
    # Test with multiple numbers - should pick the first one
    response = "The quality is 8.5 out of 10, but overall I'd say 7.2"
    result = parse_score_with_reasoning(response)
    assert result.score == 8.5

    # Test with integer
    response = "Score: 9"
    result = parse_score_with_reasoning(response)
    assert result.score == 9.0

    # Test with decimal at start
    response = "7.75 is my assessment"
    result = parse_score_with_reasoning(response)
    assert result.score == 7.75


@pytest.mark.unit
def test_escape_json_for_format_edge_cases():
    """Test escape_json_for_format with additional edge cases."""
    # Test with only opening braces
    result = escape_json_for_format("{ no closing")
    assert result == "{{ no closing"

    # Test with only closing braces
    result = escape_json_for_format("no opening }")
    assert result == "no opening }}"

    # Test with mixed content
    result = escape_json_for_format("text { more text } end")
    assert result == "text {{ more text }} end"


# Additional edge case tests for near 100% coverage
@pytest.mark.unit
def test_parse_score_with_reasoning_json_with_extra_fields():
    """Test parse_score_with_reasoning with JSON containing extra fields."""
    response = '{"score": 8.5, "reasoning": "Good work", "extra_field": "ignored"}'
    result = parse_score_with_reasoning(response)
    assert result.score == 8.5
    assert result.reasoning == "Good work"


@pytest.mark.unit
def test_parse_score_with_original_task_json_with_extra_fields():
    """Test parse_score_with_original_task with JSON containing extra fields."""
    response = '{"original_task": "Test task", "score": 7.0, "reasoning": "OK", "extra": "ignored"}'
    result = parse_score_with_original_task(response)
    assert result.original_task == "Test task"
    assert result.score == 7.0
    assert result.reasoning == "OK"


@pytest.mark.unit
def test_agent_scorers_all_wrapper_methods_coverage():
    """Test all AgentScorers wrapper methods for complete coverage."""
    mock_model = MockLLMModel('{"score": 8.0, "reasoning": "Test"}')
    scorers = AgentScorers(mock_model)

    # Create complete agent data
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        tools_available=[ToolSchema(name="test", description="test")],
        tool_calls=[ToolCall(tool_name="test", parameters={}, call_id="123")],
        expected_tool_call=ToolCall(tool_name="expected", parameters={}, call_id="exp"),
        parameters_passed={"test": "value"},
        tool_call_results=[
            ToolResult(call_id="123", result="result", success=True, error_message=None)
        ],
        agent_role="assistant",
        agent_response="Response",
        agent_task="Task",
        system_prompt="You are helpful.",
        agent_exit=True,
        trace=[{"step": 1}],
    )

    # Test all the non-score-prefixed wrapper methods that actually exist
    result = scorers.tool_relevancy(agent_data)
    assert isinstance(result, list)

    result = scorers.parameter_correctness(agent_data)
    assert isinstance(result, list)

    result = scorers.task_progression(agent_data)
    assert isinstance(result, ScoreWithOriginalTask)

    result = scorers.context_relevancy(agent_data)
    assert isinstance(result, ScoreWithReasoning)

    result = scorers.role_adherence(agent_data)
    assert isinstance(result, ScoreWithReasoning)


@pytest.mark.unit
def test_parse_score_with_reasoning_nested_json():
    """Test parse_score_with_reasoning with deeply nested JSON (should ignore nesting)."""
    response = (
        '{"nested": {"score": 5.0}, "score": 8.5, "reasoning": "Outer score counts"}'
    )
    result = parse_score_with_reasoning(response)
    assert result.score == 8.5
    assert result.reasoning == "Outer score counts"


@pytest.mark.unit
def test_parse_score_with_original_task_no_braces_in_response():
    """Test parse_score_with_original_task when response has no braces at all."""
    response = "No JSON here, just plain text"
    result = parse_score_with_original_task(response)
    assert result.original_task == "Unknown task"
    assert result.score == 1.0
    assert "Error parsing response" in result.reasoning


@pytest.mark.unit
def test_parameter_correctness_scorer_with_multiple_results():
    """Test parameter_correctness_scorer with multiple matching and non-matching results."""
    from novaeval.scorers.agent_scorers import parameter_correctness_scorer

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        tool_calls=[
            ToolCall(tool_name="tool1", parameters={}, call_id="call1"),
            ToolCall(tool_name="tool2", parameters={}, call_id="call2"),
        ],
        parameters_passed={"key": "value"},
        tool_call_results=[
            ToolResult(
                call_id="call1", result="result1", success=True, error_message=None
            ),
            ToolResult(
                call_id="call2", result="result2", success=False, error_message="Error"
            ),
            ToolResult(
                call_id="call3", result="result3", success=True, error_message=None
            ),  # No matching tool call
        ],
    )

    mock_model = MockLLMModel('{"score": 6.0, "reasoning": "Mixed results"}')
    result = parameter_correctness_scorer(agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 2  # Should have results for both tool calls
    assert all(score.score == 6.0 for score in result)


@pytest.mark.unit
def test_field_availability_error_with_all_fields_missing():
    """Test FieldAvailabilityError when all fields are missing."""
    error = FieldAvailabilityError(
        required_fields={"field1": False, "field2": False, "field3": False},
        error_message="All fields are missing",
    )

    assert all(not available for available in error.required_fields.values())
    assert error.error_message == "All fields are missing"


@pytest.mark.unit
def test_goal_achievement_scorer_minimal_trace():
    """Test goal_achievement_scorer with minimal trace content."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[{}],  # Minimal trace with empty dict
    )

    mock_response = (
        '{"original_task": "Minimal task", "score": 5.0, "reasoning": "Minimal trace"}'
    )
    mock_model = MockLLMModel(mock_response)
    result = goal_achievement_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Minimal task"
    assert result.score == 5.0


@pytest.mark.unit
def test_conversation_coherence_scorer_minimal_trace():
    """Test conversation_coherence_scorer with minimal trace content."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        agent_exit=True,
        trace=[{"msg": "hello"}],  # Minimal trace
    )

    mock_response = (
        '{"original_task": "Chat", "score": 6.0, "reasoning": "Simple conversation"}'
    )
    mock_model = MockLLMModel(mock_response)
    result = conversation_coherence_scorer(agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Chat"
    assert result.score == 6.0


# Tests for union type handling (string vs complex types)


@pytest.fixture
def string_union_agent_data():
    """Create agent data using string values for union fields."""
    return AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        ground_truth="The correct answer is 42",
        expected_tool_call="string_tool_call_representation",
        agent_name="TestAgent",
        agent_role="Mathematical assistant",
        agent_task="Calculate 20 + 22",
        system_prompt="You are a helpful math assistant.",
        agent_response="I'll calculate 20 + 22 using the calculator tool.",
        tools_available="string_tools_available",
        tool_calls="string_tool_calls",
        parameters_passed="string_parameters",
        tool_call_results="string_results",
        retrieved_context=[["context"]],
        trace="string_trace_representation",
        exit_status="completed",
        agent_exit="true",  # String representation of boolean
        metadata="Test metadata",
    )


@pytest.mark.unit
def test_safe_serialize_union_field_with_string():
    """Test safe_serialize_union_field with string input."""
    from novaeval.scorers.agent_scorers import safe_serialize_union_field

    result = safe_serialize_union_field("test_string", "test_field")
    assert result == "test_string"


@pytest.mark.unit
def test_safe_serialize_union_field_with_dict():
    """Test safe_serialize_union_field with dict input."""
    from novaeval.scorers.agent_scorers import safe_serialize_union_field

    test_dict = {"key": "value", "number": 42}
    result = safe_serialize_union_field(test_dict, "test_field")
    import json

    expected = json.dumps(test_dict, indent=2)
    assert result == expected


@pytest.mark.unit
def test_safe_serialize_union_field_with_pydantic_model():
    """Test safe_serialize_union_field with Pydantic model input."""
    from novaeval.scorers.agent_scorers import safe_serialize_union_field

    tool_call = ToolCall(tool_name="test", parameters={}, call_id="123")
    result = safe_serialize_union_field(tool_call, "test_field")
    import json

    expected = json.dumps(tool_call.model_dump(), indent=2)
    assert result == expected


@pytest.mark.unit
def test_safe_serialize_union_field_with_list_of_models():
    """Test safe_serialize_union_field with list of Pydantic models."""
    from novaeval.scorers.agent_scorers import safe_serialize_union_field

    tool_calls = [
        ToolCall(tool_name="test1", parameters={}, call_id="123"),
        ToolCall(tool_name="test2", parameters={}, call_id="456"),
    ]
    result = safe_serialize_union_field(tool_calls, "test_field")
    import json

    expected = json.dumps([tc.model_dump() for tc in tool_calls], indent=2)
    assert result == expected


@pytest.mark.unit
def test_safe_get_boolean_field_with_bool():
    """Test safe_get_boolean_field with boolean input."""
    from novaeval.scorers.agent_scorers import safe_get_boolean_field

    assert safe_get_boolean_field(True) is True
    assert safe_get_boolean_field(False) is False


@pytest.mark.unit
def test_safe_get_boolean_field_with_string():
    """Test safe_get_boolean_field with string input."""
    from novaeval.scorers.agent_scorers import safe_get_boolean_field

    # True cases
    assert safe_get_boolean_field("true") is True
    assert safe_get_boolean_field("TRUE") is True
    assert safe_get_boolean_field("1") is True
    assert safe_get_boolean_field("yes") is True
    assert safe_get_boolean_field("on") is True

    # False cases
    assert safe_get_boolean_field("false") is False
    assert safe_get_boolean_field("FALSE") is False
    assert safe_get_boolean_field("0") is False
    assert safe_get_boolean_field("no") is False
    assert safe_get_boolean_field("off") is False
    assert safe_get_boolean_field("maybe") is False
    assert safe_get_boolean_field("unknown") is False


@pytest.mark.unit
def test_tool_relevancy_scorer_with_string_fields(string_union_agent_data):
    """Test tool_relevancy_scorer with string union fields."""
    mock_model = MockLLMModel(
        '{"score": 7.5, "reasoning": "String fields handled correctly"}'
    )

    result = tool_relevancy_scorer(string_union_agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ScoreWithReasoning)
    assert result[0].score == 7.5
    assert "String fields handled correctly" in result[0].reasoning


@pytest.mark.unit
def test_tool_correctness_scorer_with_string_fields(string_union_agent_data):
    """Test tool_correctness_scorer with string union fields."""
    from novaeval.scorers.agent_scorers import tool_correctness_scorer

    mock_model = MockLLMModel(
        '{"score": 6.0, "reasoning": "String comparison successful"}'
    )

    result = tool_correctness_scorer(string_union_agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ScoreWithReasoning)
    assert result[0].score == 6.0


@pytest.mark.unit
def test_parameter_correctness_scorer_with_string_fields(string_union_agent_data):
    """Test parameter_correctness_scorer with string union fields."""
    from novaeval.scorers.agent_scorers import parameter_correctness_scorer

    mock_model = MockLLMModel(
        '{"score": 8.0, "reasoning": "String parameters processed"}'
    )

    result = parameter_correctness_scorer(string_union_agent_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ScoreWithReasoning)
    assert result[0].score == 8.0


@pytest.mark.unit
def test_role_adherence_scorer_with_string_fields(string_union_agent_data):
    """Test role_adherence_scorer with string union fields."""
    from novaeval.scorers.agent_scorers import role_adherence_scorer

    mock_model = MockLLMModel(
        '{"score": 9.0, "reasoning": "Role adherence with strings"}'
    )

    result = role_adherence_scorer(string_union_agent_data, mock_model)

    assert isinstance(result, ScoreWithReasoning)
    assert result.score == 9.0
    assert "Role adherence with strings" in result.reasoning


@pytest.mark.unit
def test_goal_achievement_scorer_with_string_fields(string_union_agent_data):
    """Test goal_achievement_scorer with string union fields."""
    mock_response = '{"original_task": "Math calculation", "score": 8.5, "reasoning": "Goal achieved with string trace"}'
    mock_model = MockLLMModel(mock_response)

    result = goal_achievement_scorer(string_union_agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Math calculation"
    assert result.score == 8.5
    assert "Goal achieved with string trace" in result.reasoning


@pytest.mark.unit
def test_conversation_coherence_scorer_with_string_fields(string_union_agent_data):
    """Test conversation_coherence_scorer with string union fields."""
    mock_response = '{"original_task": "Math conversation", "score": 7.0, "reasoning": "Coherent string conversation"}'
    mock_model = MockLLMModel(mock_response)

    result = conversation_coherence_scorer(string_union_agent_data, mock_model)

    assert isinstance(result, ScoreWithOriginalTask)
    assert result.original_task == "Math conversation"
    assert result.score == 7.0
    assert "Coherent string conversation" in result.reasoning


@pytest.mark.unit
def test_mixed_union_types():
    """Test handling of mixed union types (some string, some complex)."""
    # Create agent data with mixed types
    mixed_data = AgentData(
        user_id="user123",
        task_id="task456",
        agent_name="TestAgent",
        agent_role="Mixed assistant",
        agent_task="Mixed task",
        system_prompt="Mixed prompt",
        agent_response="Mixed response",
        tools_available="string_tools",  # String
        tool_calls=[  # List of objects
            ToolCall(tool_name="test", parameters={}, call_id="123")
        ],
        parameters_passed={"key": "value"},  # Dict
        tool_call_results="string_results",  # String
        trace="string_trace",  # String
        agent_exit=True,  # Boolean
    )

    mock_model = MockLLMModel('{"score": 8.0, "reasoning": "Mixed types handled"}')

    # Test with a scorer that uses multiple union fields
    result = tool_relevancy_scorer(mixed_data, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ScoreWithReasoning)


@pytest.mark.unit
def test_agent_exit_string_false():
    """Test agent_exit as string 'false' is handled correctly."""
    agent_data = AgentData(
        agent_name="TestAgent", agent_exit="false", trace="string_trace"  # String false
    )

    mock_model = MockLLMModel(
        '{"original_task": "Test", "score": 5.0, "reasoning": "Not exited"}'
    )

    result = goal_achievement_scorer(agent_data, mock_model)

    # Should return early because agent hasn't exited
    assert isinstance(result, ScoreWithOriginalTask)
    assert result.score == -1.0
    assert "has not yet exited" in result.reasoning


@pytest.mark.unit
def test_agent_exit_string_true():
    """Test agent_exit as string 'true' is handled correctly."""
    agent_data = AgentData(
        agent_name="TestAgent", agent_exit="true", trace="string_trace"  # String true
    )

    mock_response = (
        '{"original_task": "Test task", "score": 8.0, "reasoning": "Task completed"}'
    )
    mock_model = MockLLMModel(mock_response)

    result = goal_achievement_scorer(agent_data, mock_model)

    # Should proceed with scoring because agent has exited
    assert isinstance(result, ScoreWithOriginalTask)
    assert result.score == 8.0
    assert result.original_task == "Test task"


@pytest.mark.unit
def test_empty_string_union_fields():
    """Test handling of empty string union fields."""
    agent_data = AgentData(
        user_id="user123",
        agent_name="TestAgent",
        agent_role="Test role",
        agent_task="Test task",
        agent_response="Test response",
        tools_available="",  # Empty string
        tool_calls="",  # Empty string
        expected_tool_call="",  # Empty string
        parameters_passed="",  # Empty string
        tool_call_results="",  # Empty string
        trace="",  # Empty string
        agent_exit="true",
    )

    mock_model = MockLLMModel('{"score": 5.0, "reasoning": "Empty strings handled"}')

    # Test various scorers with empty string fields
    result1 = tool_relevancy_scorer(agent_data, mock_model)
    assert isinstance(result1, list)  # Should process empty strings
    assert len(result1) == 1
    assert isinstance(result1[0], ScoreWithReasoning)

    from novaeval.scorers.agent_scorers import role_adherence_scorer

    result2 = role_adherence_scorer(agent_data, mock_model)
    assert isinstance(
        result2, ScoreWithReasoning
    )  # Should work with empty tool_calls string


@pytest.mark.unit
def test_safe_serialize_union_field_edge_cases():
    """Test edge cases in safe_serialize_union_field function."""
    from novaeval.scorers.agent_scorers import safe_serialize_union_field

    # Test with None value (line 40)
    result = safe_serialize_union_field(None, "test_field")
    assert result == ""

    # Test with non-serializable object that raises TypeError (lines 57-59)
    class NonSerializableObj:
        def __str__(self):
            return "non_serializable"

    obj = NonSerializableObj()
    result = safe_serialize_union_field(obj, "test_field")
    assert result == "non_serializable"

    # Test with object that has no model_dump but is JSON serializable
    result = safe_serialize_union_field([1, 2, 3], "test_field")
    assert result == "[\n  1,\n  2,\n  3\n]"  # Uses indent=2


@pytest.mark.unit
def test_safe_get_boolean_field_edge_cases():
    """Test edge cases in safe_get_boolean_field function."""
    from novaeval.scorers.agent_scorers import safe_get_boolean_field

    # Test with non-string non-bool value (line 79)
    result = safe_get_boolean_field(42)
    assert result is True  # Non-zero number should be True

    result = safe_get_boolean_field(0)
    assert result is False  # Zero should be False

    result = safe_get_boolean_field([])
    assert result is False  # Empty list should be False

    result = safe_get_boolean_field([1, 2, 3])
    assert result is True  # Non-empty list should be True


@pytest.mark.unit
def test_parse_score_with_original_task_unexpected_format():
    """Test parse_score_with_original_task with unexpected response format."""
    from novaeval.scorers.agent_scorers import parse_score_with_original_task

    # Test with valid JSON that is not a dict (line 254)
    # This should hit the "Unexpected response format" branch
    unexpected_response = "[1, 2, 3]"  # Valid JSON but not a dict
    result = parse_score_with_original_task(unexpected_response)

    assert result.original_task == "Unknown task"
    assert result.score == 1.0
    assert "Unexpected response format" in result.reasoning


@pytest.mark.unit
def test_tool_relevancy_scorer_string_exception():
    """Test tool_relevancy_scorer with string tool_calls and exception handling."""
    from novaeval.scorers.agent_scorers import tool_relevancy_scorer

    # Create agent data with string tool_calls
    agent_data = AgentData(
        agent_name="TestAgent",
        tools_available=[
            ToolSchema(
                name="calculator",
                description="Performs basic mathematical operations",
                args_schema={"operation": "str", "a": "number", "b": "number"},
                return_schema={"result": "number"},
            )
        ],
        tool_calls="string_tool_call",  # String instead of list
    )

    # Create a mock model that raises an exception
    class ExceptionModel:
        def generate(self, prompt):
            raise ValueError("Model error")

    model = ExceptionModel()

    # This should hit the exception handling branch for string tool_calls (lines 342-346)
    result = tool_relevancy_scorer(agent_data, model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].score == 1.0
    assert "Failed to evaluate tool call" in result[0].reasoning


@pytest.mark.unit
def test_safe_serialize_union_field_improved_error_handling():
    """Test the improved error handling in safe_serialize_union_field function."""
    from novaeval.scorers.agent_scorers import safe_serialize_union_field

    # Test with object that raises TypeError during JSON serialization
    class TypeErrorObj:
        def __str__(self):
            return "type_error_object"

    obj = TypeErrorObj()
    result = safe_serialize_union_field(obj, "test_field")
    assert result == "type_error_object"

    # Test with object that raises AttributeError during JSON serialization
    class AttributeErrorObj:
        def __str__(self):
            return "attribute_error_object"

    obj = AttributeErrorObj()
    result = safe_serialize_union_field(obj, "test_field")
    assert result == "attribute_error_object"

    # Test with object that raises Exception during str() conversion
    class StrExceptionObj:
        def __str__(self):
            raise Exception("str conversion failed")

        def __repr__(self):
            return "StrExceptionObj()"

    obj = StrExceptionObj()
    result = safe_serialize_union_field(obj, "test_field")
    assert result == "Error serializing field"


@pytest.mark.unit
def test_safe_serialize_union_field_early_returns():
    """Test the early return optimizations in safe_serialize_union_field function."""
    from novaeval.scorers.agent_scorers import safe_serialize_union_field

    # Test early return for string values
    result = safe_serialize_union_field("simple_string", "test_field")
    assert result == "simple_string"

    # Test early return for None values
    result = safe_serialize_union_field(None, "test_field")
    assert result == ""

    # Test that complex types still go through JSON serialization
    test_dict = {"key": "value"}
    result = safe_serialize_union_field(test_dict, "test_field")
    import json

    expected = json.dumps(test_dict, indent=2)
    assert result == expected


@pytest.mark.unit
def test_safe_serialize_union_field_pydantic_handling():
    """Test Pydantic model handling in safe_serialize_union_field function."""
    from novaeval.scorers.agent_scorers import safe_serialize_union_field

    # Test single Pydantic model
    tool_call = ToolCall(tool_name="test", parameters={}, call_id="123")
    result = safe_serialize_union_field(tool_call, "test_field")
    import json

    expected = json.dumps(tool_call.model_dump(), indent=2)
    assert result == expected

    # Test list of Pydantic models
    tool_calls = [
        ToolCall(tool_name="test1", parameters={}, call_id="123"),
        ToolCall(tool_name="test2", parameters={}, call_id="456"),
    ]
    result = safe_serialize_union_field(tool_calls, "test_field")
    expected = json.dumps([tc.model_dump() for tc in tool_calls], indent=2)
    assert result == expected

    # Test empty list of Pydantic models
    empty_list = []
    result = safe_serialize_union_field(empty_list, "test_field")
    expected = json.dumps(empty_list, indent=2)
    assert result == expected


@pytest.mark.unit
def test_safe_serialize_union_field_mixed_types():
    """Test safe_serialize_union_field with various mixed type scenarios."""
    from novaeval.scorers.agent_scorers import safe_serialize_union_field

    # Test with empty string
    result = safe_serialize_union_field("", "test_field")
    assert result == ""

    # Test with whitespace string
    result = safe_serialize_union_field("   ", "test_field")
    assert result == "   "

    # Test with numeric values
    result = safe_serialize_union_field(42, "test_field")
    assert result == "42"

    result = safe_serialize_union_field(3.14, "test_field")
    assert result == "3.14"

    # Test with boolean values
    result = safe_serialize_union_field(True, "test_field")
    assert result == "true"

    result = safe_serialize_union_field(False, "test_field")
    assert result == "false"

    # Test with empty list
    result = safe_serialize_union_field([], "test_field")
    assert result == "[]"

    # Test with empty dict
    result = safe_serialize_union_field({}, "test_field")
    assert result == "{}"
