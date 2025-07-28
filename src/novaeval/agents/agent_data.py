from typing import Any, Optional

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
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    turn_id: Optional[str] = None

    ground_truth: Optional[str] = None
    expected_tool_call: Optional[ToolCall] = None

    agent_name: Optional[str] = None
    agent_role: Optional[str] = None
    agent_task: Optional[str] = None

    system_prompt: Optional[str] = None
    agent_response: Optional[str] = None
    trace: Optional[list[dict[str, Any]]] = (
        None  # we might need a method to parse this, will do once the trace is formalized
    )

    tools_available: list[ToolSchema] = []
    tool_calls: list[ToolCall] = []
    parameters_passed: dict[str, Any] = {}  # JSON-like dict
    tool_call_results: Optional[list[ToolResult]] = None

    retrieval_query: Optional[str] = None
    retrieved_context: Optional[str] = None

    metadata: Optional[str] = None

#TODO:
#add Exit_status .