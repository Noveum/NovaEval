from typing import Any, Optional, Union

from pydantic import BaseModel


class ToolSchema(BaseModel):
    name: str
    description: str
    args_schema: Optional[dict[str, Any]] = None  # Schema description as dict
    return_schema: Optional[dict[str, Any]] = None  # Return schema description as dict


class ToolCall(BaseModel):
    tool_name: str
    parameters: dict[str, Any]
    call_id: str


class ToolResult(BaseModel):
    call_id: str
    result: Any
    success: bool
    error_message: Optional[str] = None


class AgentData(BaseModel):
    user_id: Optional[str] = (
        None  # unique to each user (each project from a user gets a diff id)
    )
    task_id: Optional[str] = None  # unique to each trace
    turn_id: Optional[str] = None  # unique to each span/turn

    ground_truth: Optional[str] = None
    expected_tool_call: Optional[Union[ToolCall, str]] = None

    agent_name: Optional[str] = None  # unique to each agent (for the same user)

    agent_role: Optional[str] = None
    agent_task: Optional[str] = None  # has the current input.

    system_prompt: Optional[str] = None
    agent_response: Optional[str] = None
    trace: Optional[Union[list[dict[str, Any]], str]] = (
        None  # we might need a method to parse this, will do once the trace is formalized  # will have all the past context. useful for evaluating the agent
    )

    tools_available: Union[list[ToolSchema], str] = []
    tool_calls: Union[list[ToolCall], str] = []
    parameters_passed: Union[dict[str, Any], str] = {}  # JSON-like dict
    tool_call_results: Optional[Union[list[ToolResult], str]] = None

    retrieval_query: Optional[list[str]] = None  # list of queries, made to Vector DB
    retrieved_context: Optional[list[list[str]]] = (
        None  # List of responses received from Vector DB for each query, (generally KNN is used, so len will be K)
    )

    exit_status: Optional[str] = None
    agent_exit: Union[bool, str] = False

    metadata: Optional[str] = None
