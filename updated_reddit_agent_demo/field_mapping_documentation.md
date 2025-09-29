# Field Mapping Documentation: dataset.json → AgentData

This document outlines how fields from the `dataset.json` file will be mapped to the `AgentData` Pydantic model for evaluation purposes.

## Overview

The `dataset.json` file contains spans from a Langchain agent execution trace. **Each span becomes one AgentData record.**

Span types identified by attribute prefixes:
- **Agent spans**: Have `agent.*` attributes (chain execution, agent actions)
- **LLM spans**: Have `llm.*` attributes (language model calls)  
- **Tool spans**: Have `tool.*` attributes (individual tool executions)

## Field Mapping Strategy

### Direct Mappings
| AgentData Field | Dataset.json Path | Notes |
|----------------|-------------------|--------|
| `task_id` | `trace_id` | Trace identifier |
| `turn_id` | `span_id` | Individual span identifier |
| `exit_status` | `status` | Span execution status |
| `user_id` | `metadata.user_id` | Always null in dataset |

### Set to None (Always)
- `ground_truth` → None
- `system_prompt` → None  
- `metadata` → None

### Span Type Specific Mappings

#### Agent Name (Based on Span Type)
- Agent spans → `"agent"`
- LLM spans → `"llm"`
- Tool spans → `"tool"`

#### Agent Task
- Agent spans: `attributes.chain.inputs.input`
- Other spans: None

#### Agent Response (Span Type Dependent)
- Agent spans: `attributes.agent.output.finish.return_values.output`
- LLM spans: `attributes.llm.output.response[0]`
- Tool spans: `attributes.tool.output.output`

#### Trace Field (Events Dump)
- Agent spans with events: `json.dumps(events)`
- Other spans: None

### Tool Information

#### Tools Available
- Extract from LLM prompts using regex to parse tool signatures
- Create ToolSchema objects with name, description, args_schema
- Other spans: empty list `[]`

#### Tool Calls  
- Agent spans: Extract from `attributes.agent.output.action.*`:
  - `tool_name`: `agent.output.action.tool`
  - `parameters`: `agent.output.action.tool_input`
  - `call_id`: `span_id`
- Other spans: empty list `[]`

#### Tool Call Results
- Tool spans: Create ToolResult:
  - `call_id`: `span_id`
  - `result`: `tool.output.output`
  - `success`: Based on `status == "ok"`
- Other spans: empty list `[]`

### Retrieval Fields (Conditional)

#### Retrieval Query & Retrieved Context
- **If** tool call is `langchain_retriever`:
  - `retrieval_query`: `[tool_input_parameter]`
  - `retrieved_context`: `[[tool_result]]`
- **Else**: Both None

### Parameters Passed
- Agent spans: `attributes.agent.output.action.tool_input` as dict
- Tool spans: `attributes.tool.input.*` as dict  
- LLM spans: Empty dict `{}`

### Agent Exit
- Agent spans: `True` if events contain `agent_finish`
- Other spans: `False`

### Fields to Ignore/Drop

#### Ignored Fields (Not Relevant)
- `metadata.*` - All null/empty as mentioned
- `sdk.*` - SDK version information
- `project`, `environment` - Deployment metadata  
- `links` - Empty span linking information
- `trace_*` - Duplicate trace-level information at span level
- `parent_span_id` - Relationship info not needed for evaluation
- `start_time`, `end_time`, `duration_ms` - Timing info not needed
- `langchain.run_id` - Internal Langchain IDs

#### Fields Set to None/Default
- `ground_truth` - Not available in trace data
- `expected_tool_call` - Not available in trace data  
- `retrieval_query` - No retrieval information in current spans
- `retrieved_context` - No retrieval information in current spans
- `metadata` - Keep as None since source metadata is empty

## Processing Strategy

### Span Aggregation
Since the dataset contains multiple spans per trace, we'll need to:

1. **Group spans by `trace_id`** - Each trace becomes one AgentData record
2. **Identify span roles**:
   - Chain spans → Overall agent behavior and final outputs
   - LLM spans → System prompts and intermediate reasoning
   - Tool spans → Tool executions and results
3. **Aggregate information**:
   - Combine all tool calls within a trace
   - Use the final chain span for agent_response
   - Extract system prompt from first LLM span
   - Collect all tool results

### Field Trimming Examples
- `attributes.agent.output.action.tool` → `tool_name` in ToolCall
- `attributes.agent.output.action.tool_input` → `parameters` in ToolCall  
- `attributes.llm.input.prompts[0]` → Extract system_prompt portion
- `attributes.tool.output.output` → `result` in ToolResult

## Implementation Notes

1. **JSON Parsing**: Many fields will need JSON parsing for complex structures
2. **String vs Object**: Use Union types (str or actual objects) for flexibility
3. **Error Handling**: Handle missing fields gracefully with None values
4. **Validation**: Ensure tool calls and results have matching IDs where possible

## Expected Output Structure

Each trace will produce one AgentData record with:
- Basic identification (task_id=trace_id, turn_id=main_span_id)
- Agent behavior (task, response, exit status)
- Tool usage (available tools, calls made, results received)
- System context (prompt, agent name/role)
- Execution trace for detailed analysis
