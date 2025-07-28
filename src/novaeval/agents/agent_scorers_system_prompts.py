"""
System prompts for agent scorers using G-Eval architecture.

This module contains all the system prompts used by the agent scoring functions.
"""

# Tool Relevancy Scorer Prompt
TOOL_RELEVANCY_PROMPT = """# Tool Relevancy Evaluation

## Task
Evaluate how relevant each tool call is given the available tools and the agent's context.

## Criteria
Rate the relevancy of each tool call on a scale of 1-10, where:
- 1-3: Tool call is completely irrelevant or inappropriate for the context
- 4-6: Tool call is somewhat relevant but not optimal or has questionable timing
- 7-8: Tool call is relevant and appropriate for the context
- 9-10: Tool call is highly relevant, well-timed, and optimal for the situation

## Available Tools
{tools_available}

## Tool Calls Made
{tool_calls}

## Instructions
For each tool call, evaluate:
1. Is this tool appropriate for the current context?
2. Is the timing of the tool call reasonable?
3. Does the tool help progress toward the goal?
4. Are there better alternatives available?

Provide a relevancy score from 1-10 for each tool call as a JSON array of numbers.
Example: [8.5, 6.0, 9.2]

Your response should be ONLY the JSON array of scores, nothing else.
"""

# Tool Correctness Scorer Prompt  
TOOL_CORRECTNESS_PROMPT = """# Tool Correctness Evaluation

## Task
Evaluate how correct each tool call is compared to the expected tool call.

## Criteria
Rate the correctness of each tool call on a scale of 1-10, where:
- 1-3: Tool call is completely incorrect or wrong tool used
- 4-6: Tool call is partially correct but has significant issues
- 7-8: Tool call is mostly correct with minor issues
- 9-10: Tool call is completely correct and matches expectations

## Expected Tool Call
{expected_tool_call}

## Actual Tool Calls Made
{tool_calls}

## Instructions
For each tool call, evaluate:
1. Is the correct tool being used?
2. Are the parameters appropriate?
3. Is the call_id properly structured?
4. Does it match the expected behavior?

Provide a correctness score from 1-10 for each tool call as a JSON array of numbers.
Example: [7.5, 9.0, 4.2]

Your response should be ONLY the JSON array of scores, nothing else.
"""

# Parameter Correctness Scorer Prompt
PARAMETER_CORRECTNESS_PROMPT = """# Parameter Correctness Evaluation

## Task
Evaluate whether the correct parameters were passed to each tool call based on the tool results.

## Criteria
Rate the parameter correctness on a scale of 1-10, where:
- 1-3: Parameters are completely wrong or missing critical information
- 4-6: Parameters are partially correct but have significant issues
- 7-8: Parameters are mostly correct with minor issues
- 9-10: Parameters are completely correct and optimal

## Tool Calls with Parameters
{tool_calls_with_parameters}

## Tool Call Results
{tool_call_results}

## Instructions
For each tool call, evaluate:
1. Are all required parameters provided?
2. Are the parameter values appropriate and correctly formatted?
3. Do the parameters match what the tool expects?
4. Did the parameters lead to successful tool execution?

Provide a parameter correctness score from 1-10 for each tool call as a JSON array of numbers.
Example: [8.0, 6.5, 9.5]

Your response should be ONLY the JSON array of scores, nothing else.
"""

# Task Progression Scorer Prompt
TASK_PROGRESSION_PROMPT = """# Task Progression Evaluation

## Task
Evaluate whether the agent has made meaningful progress on the assigned task.

## Criteria
Rate the task progression on a scale of 1-5, where:
- 1: No progress made, response is off-topic or unhelpful
- 2: Minimal progress, some understanding shown but no concrete advancement
- 3: Moderate progress, clear advancement but incomplete or inefficient
- 4: Good progress, significant advancement with minor gaps
- 5: Excellent progress, substantial advancement toward task completion

## Agent Information
**Agent Role:** {agent_role}
**Agent Task:** {agent_task}
**System Prompt:** {system_prompt}
**Agent Response:** {agent_response}

## Instructions
Evaluate:
1. Does the agent understand its role and task?
2. Is the response aligned with the assigned task?
3. Has the agent made concrete progress toward the goal?
4. Is the approach efficient and logical?
5. How much closer is the agent to completing the task?

Provide a single progression score from 1-5 as a JSON number.
Example: 4.2

Your response should be ONLY the JSON number, nothing else.
"""

# Context Relevancy Scorer Prompt
CONTEXT_RELEVANCY_PROMPT = """# Context Relevancy Evaluation

## Task
Evaluate whether the retrieved context is helpful and relevant for the agent's task.

## Criteria
Rate the context relevancy on a scale of 1-10, where:
- 1-3: Context is completely irrelevant or misleading for the task
- 4-6: Context is somewhat relevant but not very helpful
- 7-8: Context is relevant and helpful for the task
- 9-10: Context is highly relevant and provides excellent information for the task

## Agent Task
{agent_task}

## Retrieved Context
{retrieved_context}

## Instructions
Evaluate:
1. Is the context directly related to the agent's task?
2. Does the context provide useful information for task completion?
3. Is the context accurate and up-to-date?
4. Would this context help the agent make better decisions?
5. Is there any misleading or irrelevant information?

Provide a single relevancy score from 1-10 as a JSON number.
Example: 7.8

Your response should be ONLY the JSON number, nothing else.
""" 