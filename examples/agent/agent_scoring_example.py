"""
Example usage of agent scorers for evaluating agent performance.

This script demonstrates how to use the agent scoring functions to evaluate
various aspects of agent behavior.
"""

import os

from novaeval.agents.agent_data import AgentData, ToolCall, ToolResult, ToolSchema
from novaeval.models.openai import OpenAIModel
from novaeval.scorers.agent_scorers import AgentScorers


def main():
    """Demonstrate agent scoring functionality."""

    # Initialize OpenAI model for scoring (you can use any LLM model)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

    # Test basic LLM functionality
    print("Testing basic LLM functionality...")
    try:
        test_response = model.generate("Please respond with just the number 42.")
        print(f"   LLM Test Response: {test_response.strip()}\n")
    except Exception as e:
        print(f"   LLM Test Failed: {e}\n")
        return

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
            call_id="call_001",
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
                return_schema={"result": "number"},
            ),
            ToolSchema(
                name="memory",
                description="Stores information for later use",
                args_schema={"key": "str", "value": "str"},
                return_schema={"success": "bool"},
            ),
        ],
        tool_calls=[
            ToolCall(
                tool_name="calculator",
                parameters={"operation": "add", "a": 20, "b": 22},
                call_id="call_001",
            )
        ],
        parameters_passed={"operation": "add", "a": 20, "b": 22},
        tool_call_results=[
            ToolResult(call_id="call_001", result=42, success=True, error_message=None)
        ],
        retrieved_context=[
            [
                "Mathematical operations: Addition is the process of combining two or more numbers to get their sum."
            ]
        ],
        agent_exit=True,  # Agent has completed the task
        metadata="Sample evaluation data",
    )

    print("=== Agent Scoring Example ===\n")

    # Score individual aspects
    print("1. Tool Relevancy Scoring:")
    try:
        tool_relevancy = scorers.score_tool_relevancy(agent_data)
        if isinstance(tool_relevancy, list):
            for i, score_obj in enumerate(tool_relevancy, 1):
                print(
                    f"   Tool Call {i}: Score={score_obj.score}, Reasoning='{score_obj.reasoning}'"
                )
        else:
            print(f"   Error: {tool_relevancy}")
        print()
    except Exception as e:
        print(f"   Error: {e}\n")

    print("2. Tool Correctness Scoring:")
    # Use the standalone function since AgentScorers doesn't have this method
    from novaeval.scorers.agent_scorers import tool_correctness_scorer

    tool_correctness = tool_correctness_scorer(agent_data, model)
    if isinstance(tool_correctness, list):
        for i, score_obj in enumerate(tool_correctness, 1):
            print(
                f"   Tool Call {i}: Score={score_obj.score}, Reasoning='{score_obj.reasoning}'"
            )
    else:
        print(f"   Error: {tool_correctness}")
    print()

    print("3. Parameter Correctness Scoring:")
    param_correctness = scorers.score_parameter_correctness(agent_data)
    if isinstance(param_correctness, list):
        for i, score_obj in enumerate(param_correctness, 1):
            print(
                f"   Tool Call {i}: Score={score_obj.score}, Reasoning='{score_obj.reasoning}'"
            )
    else:
        print(f"   Error: {param_correctness}")
    print()

    print("4. Task Progression Scoring:")
    task_progression = scorers.score_task_progression(agent_data)
    if hasattr(task_progression, "score"):
        print(f"   Score: {task_progression.score}")
        print(f"   Reasoning: {task_progression.reasoning}")
    else:
        print(f"   Error: {task_progression}")
    print()

    print("5. Context Relevancy Scoring:")
    context_relevancy = scorers.score_context_relevancy(agent_data)
    if hasattr(context_relevancy, "score"):
        print(f"   Score: {context_relevancy.score}")
        print(f"   Reasoning: {context_relevancy.reasoning}")
    else:
        print(f"   Error: {context_relevancy}")
    print()

    print("6. Role Adherence Scoring:")
    role_adherence = scorers.score_role_adherence(agent_data)
    if hasattr(role_adherence, "score"):
        print(f"   Score: {role_adherence.score}")
        print(f"   Reasoning: {role_adherence.reasoning}")
    else:
        print(f"   Error: {role_adherence}")
    print()

    # Add simple trace for goal achievement and conversation coherence scoring
    agent_data.trace = [
        {"type": "user_input", "content": "Calculate the sum of 20 and 22"},
        {
            "type": "agent_response",
            "content": "I'll help you calculate 20 + 22. Let me use the calculator tool.",
        },
        {
            "type": "tool_call",
            "tool": "calculator",
            "parameters": {"operation": "add", "a": 20, "b": 22},
        },
        {"type": "tool_result", "result": 42},
        {"type": "agent_response", "content": "The sum of 20 and 22 is 42."},
    ]

    print("7. Goal Achievement Scoring:")
    # Use the standalone function since AgentScorers doesn't have this method
    from novaeval.scorers.agent_scorers import goal_achievement_scorer

    goal_achievement = goal_achievement_scorer(agent_data, model)
    if hasattr(goal_achievement, "score"):
        print(f"   Original Task: {goal_achievement.original_task}")
        print(f"   Score: {goal_achievement.score}/10.0")
        print(f"   Reasoning: {goal_achievement.reasoning}")
    else:
        print(f"   Error: {goal_achievement}")
    print()

    print("8. Conversation Coherence Scoring:")
    # Use the standalone function since AgentScorers doesn't have this method
    from novaeval.scorers.agent_scorers import conversation_coherence_scorer

    conversation_coherence = conversation_coherence_scorer(agent_data, model)
    if hasattr(conversation_coherence, "score"):
        print(f"   Original Task: {conversation_coherence.original_task}")
        print(f"   Score: {conversation_coherence.score}/10.0")
        print(f"   Reasoning: {conversation_coherence.reasoning}")
    else:
        print(f"   Error: {conversation_coherence}")
    print()

    # Score all aspects at once
    print("9. All Scores:")
    all_scores = scorers.score_all(agent_data)
    print(f"   Results: {all_scores}\n")

    # Example with missing fields
    print("=== Example with Missing Fields ===\n")
    incomplete_data = AgentData(
        agent_name="IncompleteAgent",
        agent_exit=False,  # Agent has not finished the task yet
        # Missing required fields for most scorers
    )

    print("Tool Relevancy with missing fields:")
    missing_result = scorers.score_tool_relevancy(incomplete_data)
    print(f"   Result: {missing_result}\n")

    # Example showing behavior when agent hasn't exited
    print("=== Example with Agent Not Exited ===\n")
    not_exited_data = AgentData(
        agent_task="Calculate complex equation",
        agent_role="Math assistant",
        agent_response="I'm working on this problem...",
        agent_exit=False,  # Agent is still working
        trace=[
            {"type": "user_input", "content": "Calculate a complex equation"},
            {"type": "agent_response", "content": "I'm working on this problem..."},
        ],
    )

    print("Goal Achievement Scoring for non-exited agent:")
    goal_result = goal_achievement_scorer(not_exited_data, model)
    if hasattr(goal_result, "score"):
        print(f"   Original Task: {goal_result.original_task}")
        print(f"   Score: {goal_result.score}")
        print(f"   Reasoning: {goal_result.reasoning}")
    else:
        print(f"   Result: {goal_result}")
    print()

    print("Conversation Coherence Scoring for non-exited agent:")
    coherence_result = conversation_coherence_scorer(not_exited_data, model)
    if hasattr(coherence_result, "score"):
        print(f"   Original Task: {coherence_result.original_task}")
        print(f"   Score: {coherence_result.score}")
        print(f"   Reasoning: {coherence_result.reasoning}")
    else:
        print(f"   Result: {coherence_result}")
    print()


def example_with_different_model():
    """Example using a different model (Anthropic Claude)."""
    from novaeval.models.anthropic import AnthropicModel

    # Initialize Claude model
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

    model = AnthropicModel(model_name="claude-3-sonnet-20240229", api_key=api_key)

    scorers = AgentScorers(model)

    # Create minimal agent data for context scoring
    agent_data = AgentData(
        agent_task="Write a Python function to sort a list",
        retrieved_context=[
            [
                "Python's sorted() function returns a new sorted list from the items in an iterable."
            ]
        ],
        agent_exit=True,  # Agent has completed this simple task
    )

    print("=== Using Claude Model ===")
    context_score = scorers.score_context_relevancy(agent_data)
    print(f"Context Relevancy Score: {context_score}")


if __name__ == "__main__":
    print("Agent Scoring Example")
    print("Note: Using API keys from environment variables (OPENAI_API_KEY).\n")

    # Run the main example
    main()

    # Uncomment to run Claude example (requires ANTHROPIC_API_KEY)
    # example_with_different_model()
