# Agent Scorers

This module provides G-Eval based scoring functions for evaluating agent performance across multiple dimensions.

## Overview

The agent scoring system evaluates agent behavior using LLM-based scoring with the following scorers:

1. **Tool Relevancy Scorer** - Evaluates how relevant tool calls are given available tools
2. **Tool Correctness Scorer** - Compares tool calls against expected tool calls
3. **Parameter Correctness Scorer** - Evaluates whether correct parameters were passed to tools
4. **Task Progression Scorer** - Measures how well the agent progressed on assigned tasks
5. **Context Relevancy Scorer** - Assesses if retrieved context is helpful for the task

## Files

- `agent_data.py` - Contains the `AgentData` model with all agent information
- `agent_scorers.py` - Main scoring functions and `AgentScorers` class
- `agent_scorers_system_prompts.py` - System prompts for G-Eval scoring

## Usage

### Basic Usage

```python
import asyncio
from novaeval.agents.agent_data import AgentData
from novaeval.agents.agent_scorers import AgentScorers
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
- **Output**: Single score (1-5)
- **Evaluates**: Progress toward task completion

### Context Relevancy Scorer
- **Input**: `agent_task`, `retrieved_context`
- **Output**: Single score (1-10)
- **Evaluates**: Usefulness of retrieved context

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