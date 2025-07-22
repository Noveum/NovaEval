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
    agent_name: str
    agent_role: str
    agent_task: Optional[str] = None
    agent_response: str
    trace: Optional[list[dict[str, Any]]] = None  # List of JSON-like dicts
    tools_available: list[ToolSchema]
    tool_calls: list[ToolCall]
    parameters_passed: dict[str, Any]  # JSON-like dict
    tool_call_results: Optional[list[ToolResult]] = None
    retrieval_context: str
    metadata: Optional[str]
