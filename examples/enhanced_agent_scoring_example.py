"""
Enhanced Agent Scoring Example

This example demonstrates the new AgentScorers class that inherits from BaseScorer
and supports:
1. Enum-based scorer selection
2. Custom scorer functions
3. BaseScorer interface compatibility
"""

import os

from novaeval.agents import AgentData, ToolCall, ToolSchema
from novaeval.models.openai import OpenAIModel
from novaeval.scorers.agent_scorers import AgentScorers


def custom_tool_efficiency_scorer(agent_data: AgentData, _model) -> dict:
    """
    Custom scorer example: Evaluate tool usage efficiency.

    Args:
        agent_data: AgentData object
        model: LLM model for scoring

    Returns:
        Dict with score and reasoning
    """
    if not agent_data.tool_calls:
        return {"score": 0.0, "reasoning": "No tool calls made"}

    # Simple efficiency metric: fewer calls = higher efficiency
    # In practice, this would be more sophisticated
    num_calls = len(agent_data.tool_calls)
    if num_calls <= 2:
        score = 10.0
    elif num_calls <= 4:
        score = 7.0
    elif num_calls <= 6:
        score = 5.0
    else:
        score = 2.0

    return {
        "score": score,
        "reasoning": f"Made {num_calls} tool calls. Efficiency score based on call count.",
    }


def custom_response_length_scorer(agent_data: AgentData, _model) -> float:
    """
    Custom scorer example: Evaluate response length appropriateness.

    Args:
        agent_data: AgentData object
        model: LLM model for scoring

    Returns:
        Float score
    """
    if not agent_data.agent_response:
        return 0.0

    response_length = len(agent_data.agent_response.split())

    # Ideal response length is 20-100 words
    if 20 <= response_length <= 100:
        return 10.0
    elif 10 <= response_length < 20 or 100 < response_length <= 150:
        return 7.0
    elif 5 <= response_length < 10 or 150 < response_length <= 200:
        return 5.0
    else:
        return 3.0


def main():
    """Demonstrate enhanced agent scoring functionality."""

    # Initialize OpenAI model for scoring
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)

    # Create sample agent data
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        turn_id="turn789",
        ground_truth="The weather in Paris is sunny with 22°C",
        expected_tool_call=ToolCall(
            tool_name="get_weather",
            parameters={"city": "Paris", "unit": "celsius"},
            call_id="call_001",
        ),
        agent_name="WeatherAgent",
        agent_role="Weather information assistant",
        agent_task="Get the current weather in Paris",
        system_prompt="You are a helpful weather assistant. Provide accurate weather information.",
        agent_response="I'll check the current weather in Paris for you. The weather in Paris is currently sunny with a temperature of 22°C. It's a beautiful day!",
        tools_available=[
            ToolSchema(
                name="get_weather",
                description="Get current weather for a city",
                args_schema={
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "description": "celsius or fahrenheit"},
                },
            )
        ],
        tool_calls=[
            ToolCall(
                tool_name="get_weather",
                parameters={"city": "Paris", "unit": "celsius"},
                call_id="call_001",
            )
        ],
        parameters_passed={"city": "Paris", "unit": "celsius"},
        trace=[
            {"role": "user", "content": "What's the weather like in Paris?"},
            {"role": "assistant", "content": "I'll check the weather for you."},
            {"role": "tool", "content": "Weather: sunny, 22°C"},
            {
                "role": "assistant",
                "content": "The weather in Paris is sunny with 22°C.",
            },
        ],
        agent_exit=True,
    )

    print("=== Enhanced Agent Scoring Examples ===\n")

    # Example 1: Use all available scorers (default behavior)
    print("1. Using all available scorers:")
    all_scorers = AgentScorers(model)

    # Using the score_all method
    scores = all_scorers.score_all(agent_data)
    print("   All scores:", scores)

    # Using the convenience method
    detailed_scores = all_scorers.score_all(agent_data)
    print("   First scorer details:", list(detailed_scores.keys())[:3])
    print()

    # Example 2: Use only specific scorers (using individual method calls)
    print("2. Using only selected scorers:")
    selected_scorers = AgentScorers(model)

    # Call specific scorers individually
    tool_relevancy = selected_scorers.score_tool_relevancy(agent_data)
    context_relevancy = selected_scorers.score_context_relevancy(agent_data)
    role_adherence = selected_scorers.score_role_adherence(agent_data)

    print("   Tool relevancy:", tool_relevancy)
    print("   Context relevancy:", context_relevancy)
    print("   Role adherence:", role_adherence)
    print()

    # Example 3: Use custom scorers only
    print("3. Using custom scorers:")
    # Call custom scorers directly
    custom_tool_score = custom_tool_efficiency_scorer(agent_data, model)
    custom_length_score = custom_response_length_scorer(agent_data, model)

    print("   Custom tool efficiency score:", custom_tool_score)
    print("   Custom response length score:", custom_length_score)
    print()

    # Example 4: Mix of built-in and custom scorers
    print("4. Using mixed scorers (built-in + custom):")
    mixed_scorers = AgentScorers(model)

    # Built-in scorers
    tool_relevancy = mixed_scorers.score_tool_relevancy(agent_data)
    # Custom scorers
    custom_tool_score = custom_tool_efficiency_scorer(agent_data, model)
    custom_length_score = custom_response_length_scorer(agent_data, model)

    print("   Built-in tool relevancy:", tool_relevancy)
    print("   Custom tool efficiency:", custom_tool_score)
    print("   Custom response length:", custom_length_score)
    print()

    # Example 5: Batch processing example
    print("5. Batch processing example:")
    batch_data = [agent_data, agent_data]  # Same data for demo

    # Simulate batch scoring
    batch_scores = []
    for data in batch_data:
        score = mixed_scorers.score_all(data)
        batch_scores.append(score)

    print("   Batch scores:", len(batch_scores), "items processed")
    print("   First item tool_relevancy:", batch_scores[0].get("tool_relevancy", "N/A"))
    print()

    # Example 6: Error handling
    print("6. Error handling example:")
    try:
        # This should fail - incomplete agent data
        incomplete_data = AgentData(agent_name="Incomplete")
        mixed_scorers.score_tool_relevancy(incomplete_data)
    except Exception as e:
        print("   Expected error:", str(e))

    # Example 7: Backward compatibility
    print("\n7. Backward compatibility:")
    legacy_scorer = AgentScorers(model)

    # Method calls work
    tool_relevancy_result = legacy_scorer.score_tool_relevancy(agent_data)
    print("   Tool relevancy method works:", type(tool_relevancy_result).__name__)

    # Shorthand methods still work
    context_result = legacy_scorer.context_relevancy(agent_data)
    print("   Shorthand method works:", type(context_result).__name__)


if __name__ == "__main__":
    main()
