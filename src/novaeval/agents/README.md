# Agent Scorers

This module provides G-Eval based scoring functions for evaluating agent performance across multiple dimensions.

## Overview

The agent scoring system evaluates agent behavior using LLM-based scoring with the following scorers:

1. **Tool Relevancy Scorer** - Evaluates how relevant tool calls are given available tools
2. **Tool Correctness Scorer** - Compares tool calls against expected tool calls
3. **Parameter Correctness Scorer** - Evaluates whether correct parameters were passed to tools
4. **Task Progression Scorer** - Measures how well the agent progressed on assigned tasks
5. **Context Relevancy Scorer** - Assesses if retrieved context is helpful for the task
6. **Role Adherence Scorer** - Evaluates whether the agent's tool calls and response adhere to its assigned role
7. **Goal Achievement Scorer** - Measures how well the agent achieved its original goal (requires agent to have exited)
8. **Conversation Coherence Scorer** - Evaluates the coherence and logical flow of the agent's conversation

## Files

- `agent_data.py` - Contains the `AgentData` model with all agent information and supporting models (`ToolSchema`, `ToolCall`, `ToolResult`)
- `agent_scorers_system_prompts.py` - System prompts for G-Eval scoring
- `scorers/agent_scorers.py` - Main scoring functions and `AgentScorers` class
- `datasets/agent_dataset.py` - `AgentDataset` class for loading and managing agent data

## Usage

### Basic Usage

```python
import asyncio
from novaeval.agents.agent_data import AgentData
from novaeval.scorers.agent_scorers import AgentScorers
from novaeval.models.openai import OpenAIModel

async def score_agent():
    # Initialize model
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key="your-key")
    
    # Create scorers
    scorers = AgentScorers(model)
    
    # Create agent data
    agent_data = AgentData(
        agent_task="Solve math problem",
        agent_role="Math assistant",
        system_prompt="You are a helpful math assistant",
        agent_response="The answer is 42",
        tools_available=[{"name": "calculator", "description": "Perform calculations"}],
        tool_calls=[{"tool_name": "calculator", "parameters": {"expression": "2+2"}, "call_id": "1"}],
        # ... other fields
    )
    
    # Score all aspects
    results = await scorers.score_all(agent_data)
    print(results)

# Run the scoring
asyncio.run(score_agent())
```

### Individual Scorers

```python
# Score specific aspects
tool_relevancy = await scorers.score_tool_relevancy(agent_data)
tool_correctness = await scorers.score_tool_correctness(agent_data)
parameter_correctness = await scorers.score_parameter_correctness(agent_data)
task_progression = await scorers.score_task_progression(agent_data)
context_relevancy = await scorers.score_context_relevancy(agent_data)
role_adherence = await scorers.score_role_adherence(agent_data)

# Advanced scorers (require agent to have exited)
goal_achievement = await scorers.goal_achievement_scorer(agent_data, model)
conversation_coherence = await scorers.conversation_coherence_scorer(agent_data, model)
```

### Working with Datasets

```python
from novaeval.datasets.agent_dataset import AgentDataset

# Load data from CSV
dataset = AgentDataset()
dataset.ingest_from_csv("agent_data.csv")

# Load data from JSON
dataset.ingest_from_json("agent_data.json")

# Stream large datasets
for chunk in dataset.stream_from_csv("large_agent_data.csv", chunk_size=1000):
    for agent_data in chunk:
        results = await scorers.score_all(agent_data)
        # Process results...

# Export results
dataset.export_to_csv("scored_results.csv")
dataset.export_to_json("scored_results.json")
```

## Scoring Details

### Tool Relevancy Scorer
- **Input**: `tools_available`, `tool_calls`
- **Output**: List of scores (1-10) for each tool call
- **Evaluates**: Appropriateness and timing of tool usage

### Tool Correctness Scorer
- **Input**: `expected_tool_call`, `tool_calls`
- **Output**: List of scores (1-10) for each tool call
- **Evaluates**: Correctness compared to expected behavior

### Parameter Correctness Scorer
- **Input**: `tool_calls`, `parameters_passed`, `tool_call_results`
- **Output**: List of scores (1-10) for each tool call
- **Evaluates**: Whether correct parameters were used

### Task Progression Scorer
- **Input**: `agent_task`, `agent_role`, `system_prompt`, `agent_response`
- **Output**: Single score (1-5) with original task extraction
- **Evaluates**: Progress toward task completion

### Context Relevancy Scorer
- **Input**: `agent_task`, `agent_role`, `agent_response`
- **Output**: Single score (1-10)
- **Evaluates**: Appropriateness of agent response given task and role

### Role Adherence Scorer
- **Input**: `agent_role`, `agent_task`, `agent_response`, `tool_calls`
- **Output**: Single score (1-10)
- **Evaluates**: Whether agent's actions align with its assigned role

### Goal Achievement Scorer
- **Input**: `trace`, `agent_exit` (must be True)
- **Output**: Single score (1-10) with original task extraction
- **Evaluates**: How well the agent achieved its original goal

### Conversation Coherence Scorer
- **Input**: `trace`, `agent_exit` (must be True)
- **Output**: Single score (1-10) with original task extraction
- **Evaluates**: Coherence and logical flow of the conversation

## Data Models

### AgentData
The main data model containing all agent information:

```python
class AgentData(BaseModel):
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    turn_id: Optional[str] = None
    ground_truth: Optional[str] = None
    expected_tool_call: Optional[Union[ToolCall, str]] = None
    agent_name: Optional[str] = None
    agent_role: Optional[str] = None
    agent_task: Optional[str] = None
    system_prompt: Optional[str] = None
    agent_response: Optional[str] = None
    trace: Optional[Union[list[dict[str, Any]], str]] = None
    tools_available: Union[list[ToolSchema], str] = []
    tool_calls: Union[list[ToolCall], str] = []
    parameters_passed: Union[dict[str, Any], str] = {}
    tool_call_results: Optional[Union[list[ToolResult], str]] = None
    retrieval_query: Optional[list[str]] = None  # List of queries made to Vector DB
    retrieved_context: Optional[list[list[str]]] = None  # List of responses received from Vector DB for each query, (generally KNN is used, so len will be K)
    exit_status: Optional[str] = None
    agent_exit: Union[bool, str] = False
    metadata: Optional[str] = None
```

### Supporting Models
- `ToolSchema`: Defines tool structure with name, description, and schemas
- `ToolCall`: Represents a tool call with tool name, parameters, and call ID
- `ToolResult`: Contains tool call results with success status and error messages

## Error Handling

When required fields are missing, scorers return error dictionaries:

```python
{
    "error": "Missing required fields",
    "required_fields": {"field1": True, "field2": False},
    "missing_fields": ["field2"]
}
```

## LLM Model Support

The scorers work with any LLM model that implements the `BaseModel` interface:

- OpenAI models (`OpenAIModel`)
- Anthropic models (`AnthropicModel`)
- Azure OpenAI (`AzureOpenAIModel`)
- Google Gemini (`GeminiModel`)

## Examples

See `examples/agent_scoring_example.py` for a complete usage example.

## Architecture

The scoring system follows the G-Eval architecture:

1. **System Prompts** - Structured evaluation criteria with step-by-step instructions
2. **LLM Evaluation** - Uses chain-of-thought reasoning for scoring
3. **JSON Output** - Constrains output to structured numerical scores
4. **Error Handling** - Graceful handling of missing fields and parsing errors
5. **Flexible Data Types** - Supports both structured objects and string representations for backward compatibility 