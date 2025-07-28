"""
Example usage of agent scorers for evaluating agent performance.

This script demonstrates how to use the agent scoring functions to evaluate
various aspects of agent behavior.
"""

import asyncio
from typing import Any

from novaeval.agents.agent_data import AgentData, ToolCall, ToolResult, ToolSchema
from novaeval.agents.agent_scorers import AgentScorers
from novaeval.models.openai import OpenAIModel


async def main():
    """Demonstrate agent scoring functionality."""
    
    # Initialize OpenAI model for scoring (you can use any LLM model)
    model = OpenAIModel(
        model_name="gpt-3.5-turbo",
        api_key="your-api-key-here"  # Replace with actual API key
    )
    
    # Create agent scorers instance
    scorers = AgentScorers(model)
    
    # Create sample agent data
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        ground_truth="The correct answer is 42",
        expected_tool_call=ToolCall(
            tool_name="calculator",
            parameters={"operation": "add", "a": 20, "b": 22},
            call_id="call_001"
        ),
        agent_name="MathAgent",
        agent_role="Mathematical assistant that helps solve problems",
        agent_task="Calculate the sum of 20 and 22",
        system_prompt="You are a helpful math assistant. Use tools when needed.",
        agent_response="I'll help you calculate 20 + 22. Let me use the calculator tool.",
        tools_available=[
            ToolSchema(
                name="calculator",
                description="Performs basic mathematical operations",
                args_schema={"operation": "str", "a": "number", "b": "number"},
                return_schema={"result": "number"}
            ),
            ToolSchema(
                name="memory",
                description="Stores information for later use",
                args_schema={"key": "str", "value": "str"},
                return_schema={"success": "bool"}
            )
        ],
        tool_calls=[
            ToolCall(
                tool_name="calculator",
                parameters={"operation": "add", "a": 20, "b": 22},
                call_id="call_001"
            )
        ],
        parameters_passed={"operation": "add", "a": 20, "b": 22},
        tool_call_results=[
            ToolResult(
                call_id="call_001",
                result=42,
                success=True,
                error_message=None
            )
        ],
        retrieved_context="Mathematical operations: Addition is the process of combining two or more numbers to get their sum.",
        metadata="Sample evaluation data",
        exit_status="success"
    )
    
    print("=== Agent Scoring Example ===\n")
    
    # Score individual aspects
    print("1. Tool Relevancy Scoring:")
    tool_relevancy = await scorers.score_tool_relevancy(agent_data)
    print(f"   Result: {tool_relevancy}\n")
    
    print("2. Tool Correctness Scoring:")
    tool_correctness = await scorers.score_tool_correctness(agent_data)
    print(f"   Result: {tool_correctness}\n")
    
    print("3. Parameter Correctness Scoring:")
    param_correctness = await scorers.score_parameter_correctness(agent_data)
    print(f"   Result: {param_correctness}\n")
    
    print("4. Task Progression Scoring:")
    task_progression = await scorers.score_task_progression(agent_data)
    print(f"   Result: {task_progression}\n")
    
    print("5. Context Relevancy Scoring:")
    context_relevancy = await scorers.score_context_relevancy(agent_data)
    print(f"   Result: {context_relevancy}\n")
    
    # Score all aspects at once
    print("6. All Scores:")
    all_scores = await scorers.score_all(agent_data)
    print(f"   Results: {all_scores}\n")
    
    # Example with missing fields
    print("=== Example with Missing Fields ===\n")
    incomplete_data = AgentData(
        agent_name="IncompleteAgent",
        # Missing required fields for most scorers
    )
    
    print("Tool Relevancy with missing fields:")
    missing_result = await scorers.score_tool_relevancy(incomplete_data)
    print(f"   Result: {missing_result}\n")


async def example_with_different_model():
    """Example using a different model (Anthropic Claude)."""
    from novaeval.models.anthropic import AnthropicModel
    
    # Initialize Claude model
    model = AnthropicModel(
        model_name="claude-3-sonnet-20240229",
        api_key="your-anthropic-api-key-here"  # Replace with actual API key
    )
    
    scorers = AgentScorers(model)
    
    # Create minimal agent data for context scoring
    agent_data = AgentData(
        agent_task="Write a Python function to sort a list",
        retrieved_context="Python's sorted() function returns a new sorted list from the items in an iterable."
    )
    
    print("=== Using Claude Model ===")
    context_score = await scorers.score_context_relevancy(agent_data)
    print(f"Context Relevancy Score: {context_score}")


if __name__ == "__main__":
    print("Agent Scoring Example")
    print("Note: Make sure to set your API keys before running this example.\n")
    
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to run Claude example
    # asyncio.run(example_with_different_model()) 