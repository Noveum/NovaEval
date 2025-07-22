import pytest
from pydantic import ValidationError

from novaeval.agents import AgentData, ToolCall, ToolResult, ToolSchema


@pytest.mark.unit
def test_tool_schema_minimal():
    schema = ToolSchema(name="calculator", description="Adds two numbers")
    assert schema.name == "calculator"
    assert schema.args_schema is None
    assert schema.return_schema is None


@pytest.mark.unit
def test_tool_call_valid():
    call = ToolCall(
        tool_name="calculator", parameters={"x": 1, "y": 2}, call_id="abc123"
    )
    assert call.tool_name == "calculator"
    assert call.parameters["x"] == 1


@pytest.mark.unit
def test_tool_result_successful():
    result = ToolResult(call_id="abc123", result=3, success=True)
    assert result.success is True
    assert result.error_message is None


@pytest.mark.unit
def test_agent_data_complete():
    agent = AgentData(
        agent_name="EvalBot",
        agent_role="evaluator",
        agent_task="Summarize",
        agent_response="Summary",
        trace=[{"step": 1}],
        tools_available=[ToolSchema(name="calculator", description="Adds")],
        tool_calls=[
            ToolCall(
                tool_name="calculator", parameters={"x": 1, "y": 2}, call_id="call1"
            )
        ],
        parameters_passed={"x": 1, "y": 2},
        tool_call_results=[ToolResult(call_id="call1", result=3, success=True)],
        retrieval_context="Math context",
        metadata="metadata string",
    )
    assert agent.agent_name == "EvalBot"
    assert len(agent.tools_available) == 1
    assert agent.retrieval_context == "Math context"


@pytest.mark.unit
def test_agent_data_missing_required_fields():
    with pytest.raises(ValidationError):
        AgentData(
            agent_name="Bot",
            agent_role="helper",
            tools_available=[],
            tool_calls=[],
            parameters_passed={},
        )
