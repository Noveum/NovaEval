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
    TASK_PROGRESSION_PROMPT,
    TOOL_CORRECTNESS_PROMPT,
    TOOL_RELEVANCY_PROMPT,
)
from novaeval.models.base import BaseModel as LLMModel


class ScoreListResponse(BaseModel):
    """Pydantic model to constrain LLM output to a list of floats."""
    
    scores: list[float] = Field(description="List of scores as floats")


class SingleScoreResponse(BaseModel):
    """Pydantic model to constrain LLM output to a single float."""
    
    score: float = Field(description="Single score as float")


class FieldAvailabilityError(BaseModel):
    """Model for representing missing field information."""
    
    required_fields: dict[str, bool] = Field(
        description="Dictionary mapping field names to their availability status"
    )
    error_message: str = Field(
        description="Description of which fields are missing"
    )


async def tool_relevancy_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[list[float], dict[str, Any]]:
    """
    Score the relevancy of tool calls given available tools.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        List of relevancy scores (1-10) for each tool call, or error dict if fields missing
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
    
    # Format the tools and calls for the prompt
    tools_available_str = json.dumps([tool.dict() for tool in agent_data.tools_available], indent=2)
    tool_calls_str = json.dumps([call.dict() for call in agent_data.tool_calls], indent=2)
    
    prompt = TOOL_RELEVANCY_PROMPT.format(
        tools_available=tools_available_str,
        tool_calls=tool_calls_str
    )
    
    try:
        response = await model.generate(prompt)
        # Parse JSON response
        scores = json.loads(response.strip())
        
        # Ensure we have the right number of scores
        if len(scores) != len(agent_data.tool_calls):
            # If mismatch, pad or truncate to match tool_calls length
            while len(scores) < len(agent_data.tool_calls):
                scores.append(1.0)  # Default low score for missing
            scores = scores[:len(agent_data.tool_calls)]
        
        return [float(score) for score in scores]
        
    except (json.JSONDecodeError, ValueError, Exception) as e:
        # Return default low scores if parsing fails
        return [1.0] * len(agent_data.tool_calls)


async def tool_correctness_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[list[float], dict[str, Any]]:
    """
    Score the correctness of tool calls compared to expected tool call.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        List of correctness scores (1-10) for each tool call, or error dict if fields missing
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
    
    # Format the expected call and actual calls for the prompt
    expected_call_str = json.dumps(agent_data.expected_tool_call.dict(), indent=2)
    tool_calls_str = json.dumps([call.dict() for call in agent_data.tool_calls], indent=2)
    
    prompt = TOOL_CORRECTNESS_PROMPT.format(
        expected_tool_call=expected_call_str,
        tool_calls=tool_calls_str
    )
    
    try:
        response = await model.generate(prompt)
        # Parse JSON response
        scores = json.loads(response.strip())
        
        # Ensure we have the right number of scores
        if len(scores) != len(agent_data.tool_calls):
            while len(scores) < len(agent_data.tool_calls):
                scores.append(1.0)  # Default low score for missing
            scores = scores[:len(agent_data.tool_calls)]
        
        return [float(score) for score in scores]
        
    except (json.JSONDecodeError, ValueError, Exception) as e:
        # Return default low scores if parsing fails
        return [1.0] * len(agent_data.tool_calls)


async def parameter_correctness_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[list[float], dict[str, Any]]:
    """
    Score the correctness of parameters passed to tool calls.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        List of parameter correctness scores (1-10) for each tool call, or error dict if fields missing
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
    
    # Combine tool calls with their parameters
    tool_calls_with_params = []
    for call in agent_data.tool_calls:
        call_with_params = call.dict()
        # Map parameters_passed to specific calls if possible
        call_with_params["mapped_parameters"] = agent_data.parameters_passed
        tool_calls_with_params.append(call_with_params)
    
    tool_calls_str = json.dumps(tool_calls_with_params, indent=2)
    tool_results_str = json.dumps([result.dict() for result in agent_data.tool_call_results], indent=2)
    
    prompt = PARAMETER_CORRECTNESS_PROMPT.format(
        tool_calls_with_parameters=tool_calls_str,
        tool_call_results=tool_results_str
    )
    
    try:
        response = await model.generate(prompt)
        # Parse JSON response
        scores = json.loads(response.strip())
        
        # Ensure we have the right number of scores
        if len(scores) != len(agent_data.tool_calls):
            while len(scores) < len(agent_data.tool_calls):
                scores.append(1.0)  # Default low score for missing
            scores = scores[:len(agent_data.tool_calls)]
        
        return [float(score) for score in scores]
        
    except (json.JSONDecodeError, ValueError, Exception) as e:
        # Return default low scores if parsing fails
        return [1.0] * len(agent_data.tool_calls)


async def task_progression_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[float, dict[str, Any]]:
    """
    Score how well the agent has progressed on the assigned task.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        Single progression score (1-5), or error dict if fields missing
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
        response = await model.generate(prompt)
        # Parse JSON response
        score = json.loads(response.strip())
        return float(score)
        
    except (json.JSONDecodeError, ValueError, Exception) as e:
        # Return default low score if parsing fails
        return 1.0


async def context_relevancy_scorer(
    agent_data: AgentData, model: LLMModel
) -> Union[float, dict[str, Any]]:
    """
    Score the relevancy of retrieved context for the agent's task.
    
    Args:
        agent_data: AgentData object containing agent information
        model: LLM model to use for scoring
        
    Returns:
        Single relevancy score (1-10), or error dict if fields missing
    """
    required_fields = {
        "agent_task": agent_data.agent_task is not None,
        "retrieved_context": agent_data.retrieved_context is not None
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
        retrieved_context=agent_data.retrieved_context
    )
    
    try:
        response = await model.generate(prompt)
        # Parse JSON response
        score = json.loads(response.strip())
        return float(score)
        
    except (json.JSONDecodeError, ValueError, Exception) as e:
        # Return default low score if parsing fails
        return 1.0


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
    
    async def score_tool_relevancy(self, agent_data: AgentData) -> Union[list[float], dict[str, Any]]:
        """Score tool call relevancy."""
        return await tool_relevancy_scorer(agent_data, self.model)
    
    async def score_tool_correctness(self, agent_data: AgentData) -> Union[list[float], dict[str, Any]]:
        """Score tool call correctness."""
        return await tool_correctness_scorer(agent_data, self.model)
    
    async def score_parameter_correctness(self, agent_data: AgentData) -> Union[list[float], dict[str, Any]]:
        """Score parameter correctness."""
        return await parameter_correctness_scorer(agent_data, self.model)
    
    async def score_task_progression(self, agent_data: AgentData) -> Union[float, dict[str, Any]]:
        """Score task progression."""
        return await task_progression_scorer(agent_data, self.model)
    
    async def score_context_relevancy(self, agent_data: AgentData) -> Union[float, dict[str, Any]]:
        """Score context relevancy."""
        return await context_relevancy_scorer(agent_data, self.model)
    
    async def score_all(self, agent_data: AgentData) -> dict[str, Any]:
        """
        Run all applicable scorers on the agent data.
        
        Args:
            agent_data: AgentData object to score
            
        Returns:
            Dictionary with all scoring results
        """
        results = {}
        
        # Score tool relevancy if applicable
        tool_relevancy_result = await self.score_tool_relevancy(agent_data)
        results["tool_relevancy"] = tool_relevancy_result
        
        # Score tool correctness if applicable
        tool_correctness_result = await self.score_tool_correctness(agent_data)
        results["tool_correctness"] = tool_correctness_result
        
        # Score parameter correctness if applicable  
        parameter_correctness_result = await self.score_parameter_correctness(agent_data)
        results["parameter_correctness"] = parameter_correctness_result
        
        # Score task progression if applicable
        task_progression_result = await self.score_task_progression(agent_data)
        results["task_progression"] = task_progression_result
        
        # Score context relevancy if applicable
        context_relevancy_result = await self.score_context_relevancy(agent_data)
        results["context_relevancy"] = context_relevancy_result
        
        return results 