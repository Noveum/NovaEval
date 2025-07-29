"""
Agent scorers for evaluating agent performance using G-Eval architecture.

This module contains scoring functions for various aspects of agent behavior
including tool usage, task progression, and context relevancy.
"""

import json
from typing import Any, Union

from pydantic import BaseModel, Field

from novaeval.agents.agent_data import AgentData
from novaeval.agents.agent_scorers_system_prompts import (
    CONTEXT_RELEVANCY_PROMPT,
    PARAMETER_CORRECTNESS_PROMPT,
    ROLE_ADHERENCE_PROMPT,
    TASK_PROGRESSION_PROMPT,
    TOOL_CORRECTNESS_PROMPT,
    TOOL_RELEVANCY_PROMPT,
)
from novaeval.models.base import BaseModel as LLMModel


class ScoreWithReasoning(BaseModel):
    """Pydantic model for a single score with reasoning."""
    
    score: float = Field(description="Numerical score")
    reasoning: str = Field(description="Explanation for the score")


class ScoreWithOriginalTask(BaseModel):
    """Pydantic model for agent evaluation scores that include original task extraction."""
    
    original_task: str = Field(description="The original task identified from the trace")
    score: float = Field(description="Numerical score (1-10)")
    reasoning: str = Field(description="Explanation for the score")


def escape_json_for_format(json_str: str) -> str:
    """Escape JSON string for use in .format() method."""
    return json_str.replace("{", "{{").replace("}", "}}")


def parse_score_with_reasoning(response: str) -> ScoreWithReasoning:
    """
    Parse LLM response to extract score and reasoning with fallback handling.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        ScoreWithReasoning object
    """
    import re
    
    try:
        # Clean and parse JSON response
        cleaned_response = response.strip()
        
        # Try to extract JSON from response if it's embedded in text
        if "{" in cleaned_response and "}" in cleaned_response:
            start_idx = cleaned_response.find("{")
            end_idx = cleaned_response.rfind("}") + 1
            json_str = cleaned_response[start_idx:end_idx]
        else:
            json_str = cleaned_response
        
        try:
            parsed_response = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: try to extract score from response text using regex
            score_match = re.search(r'["\']?score["\']?\s*:\s*([0-9.]+)', cleaned_response)
            reasoning_match = re.search(r'["\']?reasoning["\']?\s*:\s*["\']([^"\']*)["\']', cleaned_response)
            
            if score_match:
                return ScoreWithReasoning(
                    score=float(score_match.group(1)),
                    reasoning=reasoning_match.group(1) if reasoning_match else "No reasoning provided"
                )
            else:
                # Try to find any number in the response as a score
                number_match = re.search(r'\b([0-9.]+)\b', cleaned_response)
                if number_match:
                    return ScoreWithReasoning(
                        score=float(number_match.group(1)),
                        reasoning=f"Extracted score from response: {cleaned_response[:100]}..."
                    )
                else:
                    return ScoreWithReasoning(
                        score=1.0,
                        reasoning=f"Could not parse response: {cleaned_response[:100]}..."
                    )
        
        # Extract score and reasoning from parsed JSON
        if isinstance(parsed_response, dict) and "score" in parsed_response and "reasoning" in parsed_response:
            return ScoreWithReasoning(
                score=float(parsed_response["score"]),
                reasoning=str(parsed_response["reasoning"])
            )
        elif isinstance(parsed_response, dict) and "score" in parsed_response:
            return ScoreWithReasoning(
                score=float(parsed_response["score"]),
                reasoning="No reasoning provided in response"
            )
        else:
            # Check if it's just a number
            if isinstance(parsed_response, (int, float)):
                return ScoreWithReasoning(
                    score=float(parsed_response),
                    reasoning="Score provided without reasoning"
                )
            else:
                return ScoreWithReasoning(
                    score=1.0,
                    reasoning=f"Unexpected response format: {str(parsed_response)}"
                )
        
    except Exception as e:
        return ScoreWithReasoning(
            score=1.0,
            reasoning=f"Failed to parse response: {str(e)}"
        )


class ScoreListResponse(BaseModel):
    """Pydantic model to constrain LLM output to a list of scores with reasoning."""
    
    scores: list[ScoreWithReasoning] = Field(description="List of scores with reasoning")


class SingleScoreResponse(BaseModel):
    """Pydantic model to constrain LLM output to a single score with reasoning."""
    
    score: float = Field(description="Numerical score")
    reasoning: str = Field(description="Explanation for the score")


class FieldAvailabilityError(BaseModel):
    """Model for representing missing field information."""
    
    required_fields: dict[str, bool] = Field(
        description="Dictionary mapping field names to their availability status"
    )
    error_message: str = Field(
        description="Description of which fields are missing"
    )


def tool_relevancy_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[list[ScoreWithReasoning], dict[str, Any]]:
    """
    Score the relevancy of tool calls given available tools.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        List of ScoreWithReasoning objects for each tool call, or error dict if fields missing
    """
    required_fields = {
        "tools_available": agent_data.tools_available is not None,
        "tool_calls": agent_data.tool_calls is not None and len(agent_data.tool_calls) > 0
    }
    
    # Check if all required fields are available
    if not all(required_fields.values()):
        missing_fields = [field for field, available in required_fields.items() if not available]
        return {
            "error": "Missing required fields",
            "required_fields": required_fields,
            "missing_fields": missing_fields
        }
    
    # Format the available tools once (they're the same for all calls)
    tools_available_str = escape_json_for_format(
        json.dumps([tool.dict() for tool in agent_data.tools_available], indent=2)
    )
    
    scores = []
    
    # Iterate over each tool call individually
    for tool_call in agent_data.tool_calls:
        # Format just this single tool call
        single_tool_call_str = escape_json_for_format(json.dumps(tool_call.dict(), indent=2))
        
        # Create prompt for this specific tool call
        prompt = TOOL_RELEVANCY_PROMPT.format(
            tools_available=tools_available_str,
            tool_calls=f"[{single_tool_call_str}]"  # Wrap in array brackets for consistency
        )
        
        try:
            response = model.generate(prompt)
            score_obj = parse_score_with_reasoning(response)
            scores.append(score_obj)
            
        except Exception as e:
            # Return default low score if parsing fails for this tool call
            default_score = ScoreWithReasoning(
                score=1.0,
                reasoning=f"Failed to evaluate tool call: {str(e)}"
            )
            scores.append(default_score)
    
    return scores


def tool_correctness_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[list[ScoreWithReasoning], dict[str, Any]]:
    """
    Score the correctness of tool calls compared to expected tool call.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        List of ScoreWithReasoning objects for each tool call, or error dict if fields missing
    """
    required_fields = {
        "expected_tool_call": agent_data.expected_tool_call is not None,
        "tool_calls": agent_data.tool_calls is not None and len(agent_data.tool_calls) > 0
    }
    
    # Check if all required fields are available
    if not all(required_fields.values()):
        missing_fields = [field for field, available in required_fields.items() if not available]
        return {
            "error": "Missing required fields",
            "required_fields": required_fields,
            "missing_fields": missing_fields
        }
    
    # Format the expected call once (same for all comparisons)
    expected_call_str = escape_json_for_format(
        json.dumps(agent_data.expected_tool_call.dict(), indent=2)
    )
    
    scores = []
    
    # Iterate over each tool call individually
    for tool_call in agent_data.tool_calls:
        # Format just this single tool call
        single_tool_call_str = escape_json_for_format(json.dumps(tool_call.dict(), indent=2))
        
        # Create prompt for this specific tool call comparison
        prompt = TOOL_CORRECTNESS_PROMPT.format(
            expected_tool_call=expected_call_str,
            tool_calls=f"[{single_tool_call_str}]"  # Wrap in array brackets for consistency
        )
        
        try:
            response = model.generate(prompt)
            score_obj = parse_score_with_reasoning(response)
            scores.append(score_obj)
            
        except Exception as e:
            # Return default low score if parsing fails for this tool call
            default_score = ScoreWithReasoning(
                score=1.0,
                reasoning=f"Failed to evaluate tool call: {str(e)}"
            )
            scores.append(default_score)
    
    return scores


def parameter_correctness_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[list[ScoreWithReasoning], dict[str, Any]]:
    """
    Score the correctness of parameters passed to tool calls.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        List of ScoreWithReasoning objects for each tool call, or error dict if fields missing
    """
    required_fields = {
        "tool_calls": agent_data.tool_calls is not None and len(agent_data.tool_calls) > 0,
        "parameters_passed": agent_data.parameters_passed is not None,
        "tool_call_results": agent_data.tool_call_results is not None
    }
    
    # Check if all required fields are available
    if not all(required_fields.values()):
        missing_fields = [field for field, available in required_fields.items() if not available]
        return {
            "error": "Missing required fields",
            "required_fields": required_fields,
            "missing_fields": missing_fields
        }
    
    # Create a mapping of call_id to results for easier lookup
    results_by_call_id = {result.call_id: result for result in agent_data.tool_call_results}
    
    scores = []
    
    # Iterate over each tool call individually
    for tool_call in agent_data.tool_calls:
        # Find the corresponding result for this tool call
        corresponding_result = results_by_call_id.get(tool_call.call_id)
        
        # Create individual tool call with parameters
        call_with_params = tool_call.dict()
        call_with_params["mapped_parameters"] = agent_data.parameters_passed
        
        # Format just this single tool call and its result
        single_tool_call_str = escape_json_for_format(json.dumps(call_with_params, indent=2))
        single_result_str = escape_json_for_format(
            json.dumps(corresponding_result.dict() if corresponding_result else {}, indent=2)
        )
        
        # Create prompt for this specific tool call
        prompt = PARAMETER_CORRECTNESS_PROMPT.format(
            tool_calls_with_parameters=f"[{single_tool_call_str}]",  # Wrap in array brackets
            tool_call_results=f"[{single_result_str}]"  # Wrap in array brackets
        )
        
        try:
            response = model.generate(prompt)
            score_obj = parse_score_with_reasoning(response)
            scores.append(score_obj)
            
        except Exception as e:
            # Return default low score if parsing fails for this tool call
            default_score = ScoreWithReasoning(
                score=1.0,
                reasoning=f"Failed to evaluate tool call: {str(e)}"
            )
            scores.append(default_score)
    
    return scores


def task_progression_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[ScoreWithReasoning, dict[str, Any]]:
    """
    Score how well the agent has progressed on the assigned task.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        ScoreWithReasoning object with progression score (1-5), or error dict if fields missing
    """
    required_fields = {
        "agent_task": agent_data.agent_task is not None,
        "agent_role": agent_data.agent_role is not None,
        "system_prompt": agent_data.system_prompt is not None,
        "agent_response": agent_data.agent_response is not None
    }
    
    # Check if all required fields are available
    if not all(required_fields.values()):
        missing_fields = [field for field, available in required_fields.items() if not available]
        return {
            "error": "Missing required fields",
            "required_fields": required_fields,
            "missing_fields": missing_fields
        }
    
    prompt = TASK_PROGRESSION_PROMPT.format(
        agent_role=agent_data.agent_role,
        agent_task=agent_data.agent_task,
        system_prompt=agent_data.system_prompt,
        agent_response=agent_data.agent_response
    )
    
    try:
        response = model.generate(prompt)
        return parse_score_with_reasoning(response)
        
    except Exception as e:
        # Return default low score if parsing fails
        return ScoreWithReasoning(
            score=1.0,
            reasoning=f"Failed to evaluate task progression: {str(e)}"
        )


def context_relevancy_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[ScoreWithReasoning, dict[str, Any]]:
    """
    Score the appropriateness of the agent response given the agent's task and role.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        ScoreWithReasoning object with appropriateness score (1-10), or error dict if fields missing
    """
    required_fields = {
        "agent_task": agent_data.agent_task is not None,
        "agent_role": agent_data.agent_role is not None,
        "agent_response": agent_data.agent_response is not None
    }
    
    # Check if all required fields are available
    if not all(required_fields.values()):
        missing_fields = [field for field, available in required_fields.items() if not available]
        return {
            "error": "Missing required fields", 
            "required_fields": required_fields,
            "missing_fields": missing_fields
        }
    
    prompt = CONTEXT_RELEVANCY_PROMPT.format(
        agent_task=agent_data.agent_task,
        agent_role=agent_data.agent_role,
        agent_response=agent_data.agent_response
    )
    
    try:
        response = model.generate(prompt)
        return parse_score_with_reasoning(response)
        
    except Exception as e:
        # Return default low score if parsing fails
        return ScoreWithReasoning(
            score=1.0,
            reasoning=f"Failed to evaluate response appropriateness: {str(e)}"
        )


def role_adherence_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[ScoreWithReasoning, dict[str, Any]]:
    """
    Score whether the agent's tool calls and response adhere to its assigned role and task.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        ScoreWithReasoning object with adherence score (1-10), or error dict if fields missing
    """
    required_fields = {
        "agent_role": agent_data.agent_role is not None,
        "agent_task": agent_data.agent_task is not None,
        "agent_response": agent_data.agent_response is not None,
        "tool_calls": agent_data.tool_calls is not None
    }
    
    # Check if all required fields are available
    if not all(required_fields.values()):
        missing_fields = [field for field, available in required_fields.items() if not available]
        return {
            "error": "Missing required fields",
            "required_fields": required_fields,
            "missing_fields": missing_fields
        }
    
    # Format tool calls for the prompt
    tool_calls_str = escape_json_for_format(
        json.dumps([tool_call.dict() for tool_call in agent_data.tool_calls], indent=2)
    )
    
    prompt = ROLE_ADHERENCE_PROMPT.format(
        agent_role=agent_data.agent_role,
        agent_task=agent_data.agent_task,
        agent_response=agent_data.agent_response,
        tool_calls=tool_calls_str
    )
    
    try:
        response = model.generate(prompt)
        return parse_score_with_reasoning(response)
        
    except Exception as e:
        # Return default low score if parsing fails
        return ScoreWithReasoning(
            score=1.0,
            reasoning=f"Failed to evaluate role adherence: {str(e)}"
        )


def goal_achievement_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[ScoreWithOriginalTask, dict[str, Any]]:
    """
    Score how well the agent achieved its original goal using G-Eval structure.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        ScoreWithOriginalTask object with goal achievement score (1-10), or error dict if fields missing
    """
    # Check if agent has exited - will be implemented when boolean is added
    # TODO: Uncomment when agent_exit boolean is added to AgentData
    # if hasattr(agent_data, 'agent_exit') and not agent_data.agent_exit:
    #     return ScoreWithOriginalTask(
    #         original_task="N/A - Agent has not exited",
    #         score=-1.0,
    #         reasoning="The agent has not yet exited"
    #     )
    
    required_fields = {
        "trace": agent_data.trace is not None and len(agent_data.trace) > 0
    }
    
    # Check if all required fields are available
    if not all(required_fields.values()):
        missing_fields = [field for field, available in required_fields.items() if not available]
        return {
            "error": "Missing required fields",
            "required_fields": required_fields,
            "missing_fields": missing_fields
        }
    
    # Format the trace for the prompt
    trace_str = json.dumps(agent_data.trace, indent=2)
    
    # G-Eval structured prompt for goal achievement
    prompt = f"""# Goal Achievement Evaluation Task

## Criteria:
Evaluate how well the agent achieved its original goal based on the complete interaction trace. The agent's performance should be measured against the initial task requirements and expected outcomes.

## Score Range: 1 to 10
- 1-2: Completely failed to achieve the goal
- 3-4: Made minimal progress toward the goal
- 5-6: Made significant progress but didn't fully achieve the goal
- 7-8: Largely achieved the goal with minor issues
- 9-10: Completely achieved the goal successfully

## Evaluation Steps:
1. Identify the original task/goal from the trace
2. Analyze the agent's actions and responses throughout the interaction
3. Assess how well the agent's final state aligns with the original goal
4. Consider the effectiveness and efficiency of the agent's approach
5. Provide a final score and detailed reasoning

## Agent Trace:
{trace_str}

## Instructions:
Please evaluate the agent's goal achievement step by step following the evaluation steps above.
First, identify and extract the original task from the trace.
Then provide your reasoning for the score, and finally give a score from 1 to 10.

Format your response as JSON:
{{
    "original_task": "[extracted original task from trace]",
    "score": [numerical score 1-10],
    "reasoning": "[detailed explanation of the score based on goal achievement]"
}}"""
    
    try:
        response = model.generate(prompt)
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response.strip())
            return ScoreWithOriginalTask(
                original_task=str(parsed_response.get("original_task", "Unknown task")),
                score=float(parsed_response.get("score", 1.0)),
                reasoning=str(parsed_response.get("reasoning", "No reasoning provided"))
            )
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            import re
            
            # Try to extract fields using regex
            original_task_match = re.search(r'"original_task":\s*"([^"]*)"', response)
            score_match = re.search(r'"score":\s*([0-9.]+)', response)
            reasoning_match = re.search(r'"reasoning":\s*"([^"]*)"', response)
            
            return ScoreWithOriginalTask(
                original_task=original_task_match.group(1) if original_task_match else "Could not extract task",
                score=float(score_match.group(1)) if score_match else 1.0,
                reasoning=reasoning_match.group(1) if reasoning_match else f"Could not parse response: {response[:200]}..."
            )
            
    except Exception as e:
        return ScoreWithOriginalTask(
            original_task="Error during evaluation",
            score=1.0,
            reasoning=f"Failed to evaluate goal achievement: {str(e)}"
        )


def conversation_coherence_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[ScoreWithOriginalTask, dict[str, Any]]:
    """
    Score the coherence and logical flow of the agent's conversation using the trace.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        ScoreWithOriginalTask object with coherence score (1-10), or error dict if fields missing
    """
    # Check if agent has exited - will be implemented when boolean is added
    # TODO: Uncomment when agent_exit boolean is added to AgentData
    # if hasattr(agent_data, 'agent_exit') and not agent_data.agent_exit:
    #     return ScoreWithOriginalTask(
    #         original_task="N/A - Agent has not exited",
    #         score=-1.0,
    #         reasoning="The agent has not yet exited"
    #     )
    
    required_fields = {
        "trace": agent_data.trace is not None and len(agent_data.trace) > 0
    }
    
    # Check if all required fields are available
    if not all(required_fields.values()):
        missing_fields = [field for field, available in required_fields.items() if not available]
        return {
            "error": "Missing required fields",
            "required_fields": required_fields,
            "missing_fields": missing_fields
        }
    
    # Format the trace for the prompt
    trace_str = json.dumps(agent_data.trace, indent=2)
    
    # Prompt for conversation coherence evaluation
    prompt = f"""# Conversation Coherence Evaluation Task

## Criteria:
Evaluate the coherence and logical flow of the agent's conversation based on the complete interaction trace. Focus on how well the agent maintains context, responds appropriately to inputs, and creates a logical conversational flow.

## Score Range: 1 to 10
- 1-2: Completely incoherent conversation with no logical flow
- 3-4: Poor coherence with many inconsistencies and context loss
- 5-6: Moderate coherence with some logical flow but noticeable issues
- 7-8: Good coherence with clear logical flow and minimal issues
- 9-10: Excellent coherence with perfect logical flow and context maintenance

## Evaluation Steps:
1. Identify the original task/goal from the trace
2. Analyze the conversational flow and context maintenance
3. Check for logical consistency in responses and actions
4. Assess how well the agent maintains context throughout the interaction
5. Evaluate the overall coherence of the conversation
6. Provide a final score and detailed reasoning

## Agent Trace:
{trace_str}

## Instructions:
Please evaluate the agent's conversation coherence step by step following the evaluation steps above.
First, identify and extract the original task from the trace.
Then analyze the conversational flow and provide your reasoning for the score.
Finally, give a score from 1 to 10.

Format your response as JSON:
{{
    "original_task": "[extracted original task from trace]",
    "score": [numerical score 1-10],
    "reasoning": "[detailed explanation of the coherence score based on conversational flow and context maintenance]"
}}"""
    
    try:
        response = model.generate(prompt)
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response.strip())
            return ScoreWithOriginalTask(
                original_task=str(parsed_response.get("original_task", "Unknown task")),
                score=float(parsed_response.get("score", 1.0)),
                reasoning=str(parsed_response.get("reasoning", "No reasoning provided"))
            )
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            import re
            
            # Try to extract fields using regex
            original_task_match = re.search(r'"original_task":\s*"([^"]*)"', response)
            score_match = re.search(r'"score":\s*([0-9.]+)', response)
            reasoning_match = re.search(r'"reasoning":\s*"([^"]*)"', response)
            
            return ScoreWithOriginalTask(
                original_task=original_task_match.group(1) if original_task_match else "Could not extract task",
                score=float(score_match.group(1)) if score_match else 1.0,
                reasoning=reasoning_match.group(1) if reasoning_match else f"Could not parse response: {response[:200]}..."
            )
            
    except Exception as e:
        return ScoreWithOriginalTask(
            original_task="Error during evaluation",
            score=1.0,
            reasoning=f"Failed to evaluate conversation coherence: {str(e)}"
        )


# Convenience class to group all scorers
class AgentScorers:
    """Collection of all agent scoring functions."""
    
    def __init__(self, model: LLMModel):
        """
        Initialize the agent scorers with an LLM model.
        
        Args:
            model: LLM model to use for all scoring operations
        """
        self.model = model
    
    def score_tool_relevancy(self, agent_data: AgentData) -> Union[list[ScoreWithReasoning], dict[str, Any]]:
        """Score tool call relevancy."""
        return tool_relevancy_scorer(agent_data, self.model)
    
    def score_tool_correctness(self, agent_data: AgentData) -> Union[list[ScoreWithReasoning], dict[str, Any]]:
        """Score tool call correctness."""
        return tool_correctness_scorer(agent_data, self.model)
    
    def score_parameter_correctness(self, agent_data: AgentData) -> Union[list[ScoreWithReasoning], dict[str, Any]]:
        """Score parameter correctness."""
        return parameter_correctness_scorer(agent_data, self.model)
    
    def score_task_progression(self, agent_data: AgentData) -> Union[ScoreWithReasoning, dict[str, Any]]:
        """Score task progression."""
        return task_progression_scorer(agent_data, self.model)
    
    def score_context_relevancy(self, agent_data: AgentData) -> Union[ScoreWithReasoning, dict[str, Any]]:
        """Score response appropriateness given task and role."""
        return context_relevancy_scorer(agent_data, self.model)
    
    def score_role_adherence(self, agent_data: AgentData) -> Union[ScoreWithReasoning, dict[str, Any]]:
        """Score role adherence."""
        return role_adherence_scorer(agent_data, self.model)
    
    def score_goal_achievement(self, agent_data: AgentData) -> Union[ScoreWithOriginalTask, dict[str, Any]]:
        """Score goal achievement."""
        return goal_achievement_scorer(agent_data, self.model)

    def score_conversation_coherence(self, agent_data: AgentData) -> Union[ScoreWithOriginalTask, dict[str, Any]]:
        """Score conversation coherence."""
        return conversation_coherence_scorer(agent_data, self.model)
    
    def score_all(self, agent_data: AgentData) -> dict[str, Any]:
        """
        Run all applicable scorers on the agent data.
        
        Args:
            agent_data: AgentData object to score
            
        Returns:
            Dictionary with all scoring results
        """
        results = {}
        
        # Score tool relevancy if applicable
        tool_relevancy_result = self.score_tool_relevancy(agent_data)
        results["tool_relevancy"] = tool_relevancy_result
        
        # Score tool correctness if applicable
        tool_correctness_result = self.score_tool_correctness(agent_data)
        results["tool_correctness"] = tool_correctness_result
        
        # Score parameter correctness if applicable  
        parameter_correctness_result = self.score_parameter_correctness(agent_data)
        results["parameter_correctness"] = parameter_correctness_result
        
        # Score task progression if applicable
        task_progression_result = self.score_task_progression(agent_data)
        results["task_progression"] = task_progression_result
        
        # Score context relevancy if applicable
        context_relevancy_result = self.score_context_relevancy(agent_data)
        results["context_relevancy"] = context_relevancy_result
        
        # Score role adherence if applicable
        role_adherence_result = self.score_role_adherence(agent_data)
        results["role_adherence"] = role_adherence_result

        # Score goal achievement if applicable
        goal_achievement_result = self.score_goal_achievement(agent_data)
        results["goal_achievement"] = goal_achievement_result

        # Score conversation coherence if applicable
        conversation_coherence_result = self.score_conversation_coherence(agent_data)
        results["conversation_coherence"] = conversation_coherence_result
        
        return results 