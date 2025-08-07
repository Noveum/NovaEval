"""
System prompts for agent scorers using G-Eval architecture.

This module contains all the system prompts used by the agent scoring functions.
"""

# Tool Relevancy Scorer Prompt
TOOL_RELEVANCY_PROMPT = """# Tool Relevancy Evaluation

## Task
Evaluate how relevant the tool call is given the available tools and the agent's context.

## Criteria
Rate the relevancy of the tool call on a scale of 1-10, where:
- 1-3: Tool call is completely irrelevant or inappropriate for the context
- 4-6: Tool call is somewhat relevant but not optimal or has questionable timing
- 7-8: Tool call is relevant and appropriate for the context
- 9-10: Tool call is highly relevant, well-timed, and optimal for the situation

## Available Tools
{tools_available}

## Tool Call Made
{tool_calls}

## Instructions
For the tool call, evaluate:
1. Is this tool appropriate for the current context?
2. Is the timing of the tool call reasonable?
3. Does the tool help progress toward the goal?
4. Are there better alternatives available?

The reasoning should be of 50-60 words. It should not be longer than 75 words.

Provide a single relevancy score from 1-10 for the tool call along with your reasoning. IMPORTANT: Use decimal scores (e.g., 7.5, 8.2, 9.1) rather than round numbers (e.g., 7.0, 8.0, 9.0) to provide more nuanced evaluation.

Format your response as a JSON object:
{{
  "score": 8.5,
  "reasoning": "The calculator tool is highly relevant for this mathematical task as it directly addresses the user's need to perform addition. The timing is appropriate and this is the optimal tool choice among the available options."
}}

Your response should be ONLY the JSON object, nothing else.
"""

# Tool Correctness Scorer Prompt
TOOL_CORRECTNESS_PROMPT = """# Tool Correctness Evaluation

## Task
Evaluate how correct the tool call is compared to the expected tool call.

## Criteria
Rate the correctness of the tool call on a scale of 1-10, where:
- 1-3: Tool call is completely incorrect or wrong tool used
- 4-6: Tool call is partially correct but has significant issues
- 7-8: Tool call is mostly correct with minor issues
- 9-10: Tool call is completely correct and matches expectations

## Expected Tool Call
{expected_tool_call}

## Actual Tool Call Made
{tool_calls}

## Instructions
For the tool call, evaluate:
1. Is the correct tool being used?
2. Are the parameters appropriate?
3. Is the call_id properly structured?
4. Does it match the expected behavior?

The reasoning should be of 50-60 words. It should not be longer than 75 words.

Provide a single correctness score from 1-10 for the tool call along with your reasoning. IMPORTANT: Use decimal scores (e.g., 7.5, 8.2, 9.1) rather than round numbers (e.g., 7.0, 8.0, 9.0) to provide more nuanced evaluation.

Format your response as a JSON object:
{{
  "score": 7.5,
  "reasoning": "The tool call uses the correct tool name and most parameters match the expected call, but there is a minor discrepancy in the call_id format that reduces the overall correctness score."
}}

Your response should be ONLY the JSON object, nothing else.
"""

# Parameter Correctness Scorer Prompt
PARAMETER_CORRECTNESS_PROMPT = """# Parameter Correctness Evaluation

## Task
Evaluate whether the correct parameters were passed to the tool call based on the tool results.

## Criteria
Rate the parameter correctness on a scale of 1-10, where:
- 1-3: Parameters are completely wrong or missing critical information
- 4-6: Parameters are partially correct but have significant issues
- 7-8: Parameters are mostly correct with minor issues
- 9-10: Parameters are completely correct and optimal

## Tool Call with Parameters
{tool_calls_with_parameters}

## Tool Call Result
{tool_call_results}

## Instructions
For the tool call, evaluate:
1. Are all required parameters provided?
2. Are the parameter values appropriate and correctly formatted?
3. Do the parameters match what the tool expects?
4. Did the parameters lead to successful tool execution?

The reasoning should be of 50-60 words. It should not be longer than 75 words.

Provide a single parameter correctness score from 1-10 for the tool call along with your reasoning. IMPORTANT: Use decimal scores (e.g., 7.5, 8.2, 9.1) rather than round numbers (e.g., 7.0, 8.0, 9.0) to provide more nuanced evaluation.

Format your response as a JSON object:
{{
  "score": 8.0,
  "reasoning": "All required parameters are provided with correct types and values. The parameters successfully led to tool execution, though there could be minor improvements in parameter naming conventions."
}}

Your response should be ONLY the JSON object, nothing else.
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

The reasoning should be of 50-60 words. It should not be longer than 75 words.

Provide a single progression score from 1-5 along with your reasoning. IMPORTANT: Use decimal scores (e.g., 3.5, 4.2, 4.8) rather than round numbers (e.g., 3.0, 4.0, 5.0) to provide more nuanced evaluation.

Format your response as a JSON object:
{{
  "score": 4.2,
  "reasoning": "The agent demonstrates good understanding of its role and has made significant progress toward completing the assigned task. The approach is logical and efficient, with only minor gaps in execution that prevent a perfect score."
}}

Your response should be ONLY the JSON object, nothing else.
"""

# Context Relevancy Scorer Prompt
CONTEXT_RELEVANCY_PROMPT = """# Context Relevancy Evaluation

## Task
Evaluate whether the agent response is appropriate and relevant given the agent's task and role.

## Criteria
Rate the response appropriateness on a scale of 1-10, where:
- 1-3: Response is completely inappropriate or irrelevant for the task and role
- 4-6: Response is somewhat appropriate but not well-aligned with task and role
- 7-8: Response is appropriate and well-aligned with the task and role
- 9-10: Response is highly appropriate, perfectly aligned with both task and role

## Agent Information
**Agent Role:** {agent_role}
**Agent Task:** {agent_task}

## Agent Response
{agent_response}

## Instructions
Evaluate:
1. Is the response directly relevant to the assigned task?
2. Does the response align with the agent's designated role and expertise?
3. Is the response appropriate in tone and content for the role?
4. Does the response demonstrate understanding of the task requirements?
5. Is the response helpful in progressing toward task completion?

The reasoning should be of 50-60 words. It should not be longer than 75 words.

Provide a single appropriateness score from 1-10 along with your reasoning. IMPORTANT: Use decimal scores (e.g., 7.5, 8.2, 9.1) rather than round numbers (e.g., 7.0, 8.0, 9.0) to provide more nuanced evaluation.

Format your response as a JSON object:
{{
  "score": 7.8,
  "reasoning": "The agent response is directly relevant to the assigned task and demonstrates good alignment with the designated role. The tone and content are appropriate for the role, though there could be minor improvements in demonstrating deeper task understanding to achieve a higher score."
}}

Your response should be ONLY the JSON object, nothing else.
"""

# Role Adherence Scorer Prompt
ROLE_ADHERENCE_PROMPT = """# Role Adherence Evaluation

## Task
Evaluate whether the agent's tool calls and response adhere to its assigned role and task.

## Criteria
Rate the role adherence on a scale of 1-10, where:
- 1-3: Agent completely ignores its role, acts inappropriately or contradicts its assigned task
- 4-6: Agent partially follows its role but has significant deviations or inconsistencies
- 7-8: Agent mostly adheres to its role with minor deviations
- 9-10: Agent perfectly adheres to its role and task, maintaining consistency throughout

## Agent Information
**Agent Role:** {agent_role}
**Agent Task:** {agent_task}

## Agent Response
{agent_response}

## Tool Calls Made
{tool_calls}

## Instructions
Evaluate:
1. Does the agent maintain its assigned role throughout the interaction?
2. Are the tool calls appropriate for the agent's role and expertise?
3. Is the response content consistent with what's expected from this role?
4. Does the agent stay focused on its assigned task?
5. Are there any role contradictions or inappropriate behaviors?

The reasoning should be of 50-60 words. It should not be longer than 75 words.

Provide a single adherence score from 1-10 along with your reasoning. IMPORTANT: Use decimal scores (e.g., 7.5, 8.2, 9.1) rather than round numbers (e.g., 7.0, 8.0, 9.0) to provide more nuanced evaluation.

Format your response as a JSON object:
{{
  "score": 8.5,
  "reasoning": "The agent consistently maintains its assigned role as a data analyst and stays focused on the data analysis task. The tool calls are appropriate for this role, using statistical tools effectively. Minor deduction for one instance where the response could have been more aligned with the analytical tone expected from this role."
}}

Your response should be ONLY the JSON object, nothing else.
"""
