"""
Enhanced Agent Scoring Example

This example demonstrates the new AgentScorers class that inherits from BaseScorer
and supports:
1. Enum-based scorer selection
2. Custom scorer functions
3. BaseScorer interface compatibility
"""

import os

from novaeval.agents import AgentData, AgentScorers, AgentScorerType, ToolCall, ToolSchema
from novaeval.models.openai import OpenAIModel


def custom_tool_efficiency_scorer(agent_data: AgentData, model) -> dict:
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
        "reasoning": f"Made {num_calls} tool calls. Efficiency score based on call count."
    }


def custom_response_length_scorer(agent_data: AgentData, model) -> float:
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
            {"role": "assistant", "content": "The weather in Paris is sunny with 22°C."},
        ],
        agent_exit=True
    )
    
    print("=== Enhanced Agent Scoring Examples ===\n")
    
    # Example 1: Use all available scorers (default behavior)
    print("1. Using all available scorers:")
    all_scorers = AgentScorers(model)
    
    # Using the BaseScorer interface
    scores = all_scorers.score("", "", {"agent_data": agent_data})
    print("   Numeric scores:", scores)
    
    # Using the convenience method
    detailed_scores = all_scorers.score_all(agent_data)
    print("   First scorer details:", list(detailed_scores.keys())[:3])
    print()
    
    # Example 2: Use only specific scorers
    print("2. Using only selected scorers:")
    selected_scorers = AgentScorers(
        model,
        scorers=[
            AgentScorerType.TOOL_RELEVANCY,
            AgentScorerType.CONTEXT_RELEVANCY,
            AgentScorerType.ROLE_ADHERENCE
        ]
    )
    
    scores = selected_scorers.score("", "", {"agent_data": agent_data})
    print("   Selected scores:", scores)
    print()
    
    # Example 3: Use custom scorers only
    print("3. Using custom scorers:")
    custom_scorers = AgentScorers(
        model,
        scorers=[],  # No built-in scorers
        custom_scorers=[
            custom_tool_efficiency_scorer,
            custom_response_length_scorer
        ]
    )
    
    scores = custom_scorers.score("", "", {"agent_data": agent_data})
    print("   Custom scores:", scores)
    print()
    
    # Example 4: Mix of built-in and custom scorers
    print("4. Using mixed scorers (built-in + custom):")
    mixed_scorers = AgentScorers(
        model,
        scorers=[
            AgentScorerType.TOOL_RELEVANCY,
            AgentScorerType.GOAL_ACHIEVEMENT
        ],
        custom_scorers=[
            custom_tool_efficiency_scorer,
            custom_response_length_scorer
        ]
    )
    
    scores = mixed_scorers.score("", "", {"agent_data": agent_data})
    print("   Mixed scores:", scores)
    print()
    
    # Example 5: Using as a traditional BaseScorer in batch processing
    print("5. Batch processing example:")
    batch_contexts = [
        {"agent_data": agent_data},
        {"agent_data": agent_data},  # Same data for demo
    ]
    
    # Simulate batch scoring
    batch_scores = []
    for context in batch_contexts:
        score = mixed_scorers.score("", "", context)
        batch_scores.append(score)
    
    print("   Batch scores:", len(batch_scores), "items processed")
    print("   Average tool_relevancy:", 
          sum(s.get("tool_relevancy", 0) for s in batch_scores) / len(batch_scores))
    print()
    
    # Example 6: Error handling
    print("6. Error handling example:")
    try:
        # This should fail - no agent_data in context
        mixed_scorers.score("", "", {})
    except ValueError as e:
        print("   Expected error:", str(e))
    
    # Example 7: Backward compatibility
    print("\n7. Backward compatibility:")
    legacy_scorer = AgentScorers(model)
    
    # Old-style method calls still work
    tool_relevancy_result = legacy_scorer.score_tool_relevancy(agent_data)
    print("   Legacy method works:", type(tool_relevancy_result).__name__)
    
    # Shorthand methods still work
    context_result = legacy_scorer.context_relevancy(agent_data)
    print("   Shorthand method works:", type(context_result).__name__)


if __name__ == "__main__":
    main() 