"""
Example demonstrating agent scoring with multiple tool calls.

This shows how each tool call is evaluated individually.
"""

import os

from novaeval.agents.agent_data import AgentData, ToolCall, ToolResult, ToolSchema
from novaeval.models.openai import OpenAIModel
from novaeval.scorers.agent_scorers import AgentScorers


def main():
    """Demonstrate agent scoring with multiple tool calls."""

    # Initialize OpenAI model for scoring
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

    scorers = AgentScorers(model)

    # Create sample agent data with multiple tool calls
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        ground_truth="The correct answer involves using calculator and memory tools",
        expected_tool_call=ToolCall(
            tool_name="calculator",
            parameters={"operation": "add", "a": 20, "b": 22},
            call_id="call_001",
        ),
        agent_name="MathAgent",
        agent_role="Mathematical assistant that helps solve problems and stores results",
        agent_task="Calculate 20 + 22 and store the result in memory",
        system_prompt="You are a helpful math assistant. Use tools when needed.",
        agent_response="I'll calculate 20 + 22 and store the result for you.",
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
            ToolSchema(
                name="weather",
                description="Gets weather information",
                args_schema={"location": "str"},
                return_schema={"weather": "str", "temperature": "number"},
            ),
        ],
        tool_calls=[
            # Good tool call - matches expected
            ToolCall(
                tool_name="calculator",
                parameters={"operation": "add", "a": 20, "b": 22},
                call_id="call_001",
            ),
            # Appropriate follow-up tool call
            ToolCall(
                tool_name="memory",
                parameters={"key": "calculation_result", "value": "42"},
                call_id="call_002",
            ),
            # Irrelevant tool call for the task
            ToolCall(
                tool_name="weather",
                parameters={"location": "New York"},
                call_id="call_003",
            ),
        ],
        parameters_passed={
            "operation": "add",
            "a": 20,
            "b": 22,
            "key": "calculation_result",
            "value": "42",
            "location": "New York",
        },
        tool_call_results=[
            ToolResult(call_id="call_001", result=42, success=True, error_message=None),
            ToolResult(
                call_id="call_002",
                result={"success": True},
                success=True,
                error_message=None,
            ),
            ToolResult(
                call_id="call_003",
                result={"weather": "sunny", "temperature": 75},
                success=True,
                error_message=None,
            ),
        ],
        retrieval_query=["Calculate 20 + 22 and store the result"],
        retrieved_context=[
            [
                "Mathematical operations: Addition combines numbers. Memory storage helps retain important results."
            ]
        ],
        agent_exit=True,  # Agent has completed the task
        metadata="Multi-tool evaluation example",
    )

    print("=== Multi-Tool Agent Scoring Example ===\n")
    print(f"Agent Task: {agent_data.agent_task}")
    print(f"Number of Tool Calls: {len(agent_data.tool_calls)}")
    print("Tool Calls:")
    for i, call in enumerate(agent_data.tool_calls, 1):
        print(f"  {i}. {call.tool_name}({call.parameters}) -> {call.call_id}")
    print()

    # Score individual aspects
    print("1. Tool Relevancy Scoring (individual evaluation):")
    tool_relevancy = scorers.score_tool_relevancy(agent_data)
    if isinstance(tool_relevancy, list):
        relevancy_scores = []
        for i, score_obj in enumerate(tool_relevancy, 1):
            call = agent_data.tool_calls[i - 1]
            print(f"   Tool Call {i} ({call.tool_name}): {score_obj.score}")
            print(f"      Reasoning: {score_obj.reasoning}")
            relevancy_scores.append(score_obj.score)
        print(
            f"   Average Relevancy: {sum(relevancy_scores) / len(relevancy_scores):.2f}\n"
        )
    else:
        print(f"   Error: {tool_relevancy}\n")
        relevancy_scores = []

    print("2. Tool Correctness Scoring (vs expected):")
    # Use the standalone function since AgentScorers doesn't have this method
    from novaeval.scorers.agent_scorers import tool_correctness_scorer

    tool_correctness = tool_correctness_scorer(agent_data, model)
    if isinstance(tool_correctness, list):
        correctness_scores = []
        for i, score_obj in enumerate(tool_correctness, 1):
            call = agent_data.tool_calls[i - 1]
            print(f"   Tool Call {i} ({call.tool_name}): {score_obj.score}")
            print(f"      Reasoning: {score_obj.reasoning}")
            correctness_scores.append(score_obj.score)
        print(
            f"   Average Correctness: {sum(correctness_scores) / len(correctness_scores):.2f}\n"
        )
    else:
        print(f"   Error: {tool_correctness}\n")
        correctness_scores = []

    print("3. Parameter Correctness Scoring:")
    param_correctness = scorers.score_parameter_correctness(agent_data)
    if isinstance(param_correctness, list):
        param_scores = []
        for i, score_obj in enumerate(param_correctness, 1):
            call = agent_data.tool_calls[i - 1]
            result = next(
                (
                    r
                    for r in (agent_data.tool_call_results or [])
                    if r.call_id == call.call_id
                ),
                None,
            )
            success_status = "✓" if result and result.success else "✗"
            print(
                f"   Tool Call {i} ({call.tool_name}) {success_status}: {score_obj.score}"
            )
            print(f"      Reasoning: {score_obj.reasoning}")
            param_scores.append(score_obj.score)
        print(
            f"   Average Parameter Correctness: {sum(param_scores) / len(param_scores):.2f}\n"
        )
    else:
        print(f"   Error: {param_correctness}\n")
        param_scores = []

    print("4. Task Progression Scoring:")
    task_progression = scorers.score_task_progression(agent_data)
    if hasattr(task_progression, "score"):
        print(f"   Overall Task Progress: {task_progression.score}/5.0")
        print(f"   Reasoning: {task_progression.reasoning}\n")
    else:
        print(f"   Error: {task_progression}\n")

    print("5. Context Relevancy Scoring:")
    context_relevancy = scorers.score_context_relevancy(agent_data)
    if hasattr(context_relevancy, "score"):
        print(f"   Context Relevancy: {context_relevancy.score}/10.0")
        print(f"   Reasoning: {context_relevancy.reasoning}\n")
    else:
        print(f"   Error: {context_relevancy}\n")

    print("6. Role Adherence Scoring:")
    role_adherence = scorers.score_role_adherence(agent_data)
    if hasattr(role_adherence, "score"):
        print(f"   Role Adherence: {role_adherence.score}/10.0")
        print(f"   Reasoning: {role_adherence.reasoning}\n")
    else:
        print(f"   Error: {role_adherence}\n")

    # Add trace for goal achievement and conversation coherence scoring
    agent_data.trace = [
        {
            "type": "user_input",
            "content": "Calculate 20 + 22 and store the result in memory",
        },
        {
            "type": "agent_response",
            "content": "I'll calculate 20 + 22 and store the result for you.",
        },
        {
            "type": "tool_call",
            "tool": "calculator",
            "parameters": {"operation": "add", "a": 20, "b": 22},
        },
        {"type": "tool_result", "result": 42},
        {
            "type": "tool_call",
            "tool": "memory",
            "parameters": {"key": "calculation_result", "value": "42"},
        },
        {"type": "tool_result", "result": {"success": True}},
        {
            "type": "tool_call",
            "tool": "weather",
            "parameters": {"location": "New York"},
        },
        {"type": "tool_result", "result": {"weather": "sunny", "temperature": 75}},
        {
            "type": "agent_response",
            "content": "I've calculated 20 + 22 = 42 and stored it in memory. I also checked the weather for some reason.",
        },
    ]

    print("7. Goal Achievement Scoring:")
    # Use the standalone function since AgentScorers doesn't have this method
    from novaeval.scorers.agent_scorers import goal_achievement_scorer

    goal_achievement = goal_achievement_scorer(agent_data, model)
    if hasattr(goal_achievement, "score"):
        print(f"   Original Task: {goal_achievement.original_task}")
        print(f"   Goal Achievement: {goal_achievement.score}/10.0")
        print(f"   Reasoning: {goal_achievement.reasoning}\n")
    else:
        print(f"   Error: {goal_achievement}\n")

    print("8. Conversation Coherence Scoring:")
    # Use the standalone function since AgentScorers doesn't have this method
    from novaeval.scorers.agent_scorers import conversation_coherence_scorer

    conversation_coherence = conversation_coherence_scorer(agent_data, model)
    if hasattr(conversation_coherence, "score"):
        print(f"   Original Task: {conversation_coherence.original_task}")
        print(f"   Conversation Coherence: {conversation_coherence.score}/10.0")
        print(f"   Reasoning: {conversation_coherence.reasoning}\n")
    else:
        print(f"   Error: {conversation_coherence}\n")

    # Summary
    print("=== Summary ===")
    if (
        isinstance(tool_relevancy, list)
        and isinstance(tool_correctness, list)
        and isinstance(param_correctness, list)
        and relevancy_scores
        and correctness_scores
        and param_scores
    ):

        max_relevancy_idx = relevancy_scores.index(max(relevancy_scores))
        max_correctness_idx = correctness_scores.index(max(correctness_scores))
        max_param_idx = param_scores.index(max(param_scores))
        min_relevancy_idx = relevancy_scores.index(min(relevancy_scores))

        print(
            f"Most Relevant Tool: {agent_data.tool_calls[max_relevancy_idx].tool_name} ({max(relevancy_scores)})"
        )
        print(
            f"Most Correct Tool: {agent_data.tool_calls[max_correctness_idx].tool_name} ({max(correctness_scores)})"
        )
        print(
            f"Best Parameters: {agent_data.tool_calls[max_param_idx].tool_name} ({max(param_scores)})"
        )
        print(
            f"Least Relevant Tool: {agent_data.tool_calls[min_relevancy_idx].tool_name} ({min(relevancy_scores)})"
        )


if __name__ == "__main__":
    print("Multi-Tool Agent Scoring Example")
    print("Note: Using API keys from environment variables (OPENAI_API_KEY).\n")

    main()
