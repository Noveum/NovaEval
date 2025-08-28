import pytest

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
        user_id="user42",
        task_id="task99",
        turn_id="turn7",
        ground_truth="expected answer",
        expected_tool_call=ToolCall(
            tool_name="calculator", parameters={"x": 1, "y": 2}, call_id="call1"
        ),
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
        retrieval_query=["What is 1+2?"],
        retrieved_context=[["Math context"]],
        exit_status="completed",
        agent_exit=True,
        metadata="metadata string",
    )
    assert agent.user_id == "user42"
    assert agent.task_id == "task99"
    assert agent.turn_id == "turn7"
    assert agent.ground_truth == "expected answer"
    assert agent.expected_tool_call is not None
    assert agent.expected_tool_call.tool_name == "calculator"
    assert agent.agent_name == "EvalBot"
    assert len(agent.tools_available) == 1
    assert agent.retrieved_context == [["Math context"]]
    assert agent.exit_status == "completed"
    assert agent.agent_exit is True


@pytest.mark.unit
def test_agent_data_missing_required_fields():
    agent = AgentData(
        agent_name="Bot",
        agent_role="helper",
        tools_available=[],
        tool_calls=[],
        parameters_passed={},
    )
    assert agent.user_id is None
    assert agent.task_id is None
    assert agent.turn_id is None
    assert agent.ground_truth is None
    assert agent.expected_tool_call is None
    assert agent.agent_name == "Bot"
    assert agent.agent_role == "helper"
    assert agent.tools_available == []
    assert agent.tool_calls == []
    assert agent.parameters_passed == {}
    assert agent.tool_call_results is None
    assert agent.retrieval_query is None
    assert agent.retrieved_context is None
    assert agent.exit_status is None
    assert agent.agent_exit is False
    assert agent.metadata is None


@pytest.mark.unit
def test_tool_schema_missing_required_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolSchema(description="desc only")
    with pytest.raises(ValidationError):
        ToolSchema(name="n")


@pytest.mark.unit
def test_tool_schema_extra_fields():
    schema = ToolSchema(name="n", description="d", extra_field=123)
    assert not hasattr(schema, "extra_field")


@pytest.mark.unit
def test_tool_schema_args_and_return_schema():
    schema = ToolSchema(
        name="n", description="d", args_schema={"a": "b"}, return_schema={"r": "s"}
    )
    assert schema.args_schema == {"a": "b"}
    assert schema.return_schema == {"r": "s"}


@pytest.mark.unit
def test_tool_call_missing_required_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolCall(parameters={}, call_id="id")
    with pytest.raises(ValidationError):
        ToolCall(tool_name="t", call_id="id")
    with pytest.raises(ValidationError):
        ToolCall(tool_name="t", parameters={})


@pytest.mark.unit
def test_tool_call_extra_fields():
    call = ToolCall(tool_name="t", parameters={}, call_id="id", extra_field=1)
    assert not hasattr(call, "extra_field")


@pytest.mark.unit
def test_tool_result_missing_required_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolResult(call_id="id", result=1)
    with pytest.raises(ValidationError):
        ToolResult(result=1, success=True)
    with pytest.raises(ValidationError):
        ToolResult(call_id="id", success=True)


@pytest.mark.unit
def test_tool_result_extra_fields():
    result = ToolResult(call_id="id", result=1, success=True, extra_field=1)
    assert not hasattr(result, "extra_field")


@pytest.mark.unit
def test_tool_result_error_message():
    result = ToolResult(call_id="id", result=1, success=False, error_message="fail")
    assert result.error_message == "fail"


@pytest.mark.unit
def test_agent_data_empty_lists_and_dicts():
    agent = AgentData(tools_available=[], tool_calls=[], parameters_passed={})
    assert agent.tools_available == []
    assert agent.tool_calls == []
    assert agent.parameters_passed == {}


@pytest.mark.unit
def test_agent_data_extra_fields():
    agent = AgentData(
        agent_name="Bot",
        tools_available=[],
        tool_calls=[],
        parameters_passed={},
        extra_field=123,
    )
    assert not hasattr(agent, "extra_field")


@pytest.mark.unit
def test_agent_data_serialization():
    agent = AgentData(
        agent_name="Bot", tools_available=[], tool_calls=[], parameters_passed={}
    )
    data = agent.model_dump()
    agent2 = AgentData.model_validate(data)
    assert agent2 == agent


@pytest.mark.unit
def test_agent_data_defaults():
    agent = AgentData()
    assert agent.tools_available == []
    assert agent.tool_calls == []
    assert agent.parameters_passed == {}
    assert agent.tool_call_results is None
    assert agent.exit_status is None
    assert agent.agent_exit is False
    assert agent.metadata is None


@pytest.mark.unit
def test_tool_schema_model_dump_and_validate():
    schema = ToolSchema(name="n", description="d")
    dumped = schema.model_dump()
    loaded = ToolSchema.model_validate(dumped)
    assert loaded == schema


@pytest.mark.unit
def test_tool_call_model_dump_and_validate():
    call = ToolCall(tool_name="t", parameters={}, call_id="id")
    dumped = call.model_dump()
    loaded = ToolCall.model_validate(dumped)
    assert loaded == call


@pytest.mark.unit
def test_tool_result_model_dump_and_validate():
    result = ToolResult(call_id="id", result=1, success=True)
    dumped = result.model_dump()
    loaded = ToolResult.model_validate(dumped)
    assert loaded == result


@pytest.mark.unit
def test_tool_schema_invalid_types():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolSchema(name=123, description=456)
    with pytest.raises(ValidationError):
        ToolSchema(name="n", description=1.2)
    with pytest.raises(ValidationError):
        ToolSchema(name="n", description="d", args_schema="not a dict")
    with pytest.raises(ValidationError):
        ToolSchema(name="n", description="d", return_schema=[1, 2, 3])


@pytest.mark.unit
def test_tool_call_invalid_types():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolCall(tool_name=123, parameters={}, call_id="id")
    with pytest.raises(ValidationError):
        ToolCall(tool_name="t", parameters="not a dict", call_id="id")
    with pytest.raises(ValidationError):
        ToolCall(tool_name="t", parameters={}, call_id=123)


@pytest.mark.unit
def test_tool_result_invalid_types():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolResult(call_id=123, result=1, success=True)
    with pytest.raises(ValidationError):
        ToolResult(call_id="id", result=1, success="not bool")
    with pytest.raises(ValidationError):
        ToolResult(call_id="id", result=1, success=True, error_message=123)


@pytest.mark.unit
def test_agent_data_invalid_types():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentData(user_id=123)
    with pytest.raises(ValidationError):
        AgentData(tools_available={})
    with pytest.raises(ValidationError):
        AgentData(tool_calls={})
    with pytest.raises(ValidationError):
        AgentData(parameters_passed=[])


@pytest.mark.unit
def test_agent_data_trace_edge_cases():
    agent = AgentData(trace=None)
    assert agent.trace is None
    agent = AgentData(trace=[])
    assert agent.trace == []
    agent = AgentData(trace=[{"step": 1}, {"step": 2}])
    assert agent.trace == [{"step": 1}, {"step": 2}]


@pytest.mark.unit
def test_agent_data_tool_call_results_edge_cases():
    agent = AgentData(tool_call_results=None)
    assert agent.tool_call_results is None
    agent = AgentData(tool_call_results=[])
    assert agent.tool_call_results == []
    agent = AgentData(
        tool_call_results=[ToolResult(call_id="id", result=1, success=True)]
    )
    assert len(agent.tool_call_results) == 1


@pytest.mark.unit
def test_agent_data_retrieval_fields_edge_cases():
    agent = AgentData(retrieval_query=None, retrieved_context=None)
    assert agent.retrieval_query is None
    assert agent.retrieved_context is None
    agent = AgentData(retrieval_query=["q"], retrieved_context=[["ctx"]])
    assert agent.retrieval_query == ["q"]
    assert agent.retrieved_context == [["ctx"]]


@pytest.mark.unit
def test_agent_data_exit_fields_defaults():
    agent = AgentData()
    assert agent.exit_status is None
    assert agent.agent_exit is False


@pytest.mark.unit
def test_agent_data_exit_fields_valid_values():
    agent = AgentData(exit_status="success", agent_exit=True)
    assert agent.exit_status == "success"
    assert agent.agent_exit is True

    agent = AgentData(exit_status="error", agent_exit=False)
    assert agent.exit_status == "error"
    assert agent.agent_exit is False


@pytest.mark.unit
def test_agent_data_exit_fields_invalid_types():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentData(exit_status=123)
    # agent_exit can be bool or str, so "not_boolean" should be valid
    # with pytest.raises(ValidationError):
    #     AgentData(agent_exit="not_boolean")
    with pytest.raises(ValidationError):
        AgentData(agent_exit=[1, 2, 3])  # List cannot be converted to boolean


@pytest.mark.unit
def test_agent_data_exit_fields_serialization():
    agent = AgentData(exit_status="timeout", agent_exit=True)
    data = agent.model_dump()
    agent2 = AgentData.model_validate(data)
    assert agent2.exit_status == "timeout"
    assert agent2.agent_exit is True
    assert agent2 == agent
