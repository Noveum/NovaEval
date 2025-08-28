"""
Test script for the aggregator functions.

This script demonstrates how to use the aggregator functions to process
evaluation results from the AgentEvaluator.
"""

from pathlib import Path

from novaeval.evaluators.aggregators import (
    aggregate_by_agent_name,
    aggregate_by_task,
    aggregate_by_user,
    mean_callable,
)


def range_aggregator(scores: list[float]) -> float:
    """Custom aggregation function - calculates range (max - min)."""
    if not scores:
        return 0.0
    return max(scores) - min(scores)


def main():
    """Test the aggregator functions."""

    # Check if we have evaluation results
    results_file = Path("results/agent_evaluation/agent_evaluation_results.csv")

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run the agent evaluator example first to generate results.")
        return

    print("Testing aggregator functions...")
    print(f"Input file: {results_file}")

    # Test 1: Aggregate by task (memory-based)
    print("\n1. Testing aggregate_by_task (memory-based)...")
    task_output = Path("results/agent_evaluation/task_aggregation.csv")
    aggregate_by_task(
        input_file=results_file,
        output_filename=task_output,
        callable_func=mean_callable,
        streaming=False,
    )
    print(f"Task aggregation saved to: {task_output}")

    # Test 2: Aggregate by user (memory-based)
    print("\n2. Testing aggregate_by_user (memory-based)...")
    user_output = Path("results/agent_evaluation/user_aggregation.csv")
    aggregate_by_user(
        input_file=results_file,
        output_filename=user_output,
        callable_func=mean_callable,
        streaming=False,
    )
    print(f"User aggregation saved to: {user_output}")

    # Test 3: Aggregate by agent name (memory-based)
    print("\n3. Testing aggregate_by_agent_name (memory-based)...")
    agent_output = Path("results/agent_evaluation/agent_aggregation.csv")
    aggregate_by_agent_name(
        input_file=results_file,
        output_filename=agent_output,
        callable_func=mean_callable,
        streaming=False,
    )
    print(f"Agent aggregation saved to: {agent_output}")

    # Test 4: Aggregate by task with custom function
    print("\n4. Testing aggregate_by_task with custom function (range)...")
    task_range_output = Path("results/agent_evaluation/task_range_aggregation.csv")
    aggregate_by_task(
        input_file=results_file,
        output_filename=task_range_output,
        callable_func=range_aggregator,
        streaming=False,
    )
    print(f"Task range aggregation saved to: {task_range_output}")

    # Test 4b: Aggregate by task with built-in functions
    print("\n4b. Testing aggregate_by_task with built-in functions...")
    task_max_output = Path("results/agent_evaluation/task_max_aggregation.csv")
    aggregate_by_task(
        input_file=results_file,
        output_filename=task_max_output,
        callable_func=max,
        streaming=False,
    )
    print(f"Task max aggregation saved to: {task_max_output}")

    task_min_output = Path("results/agent_evaluation/task_min_aggregation.csv")
    aggregate_by_task(
        input_file=results_file,
        output_filename=task_min_output,
        callable_func=min,
        streaming=False,
    )
    print(f"Task min aggregation saved to: {task_min_output}")

    # Test 5: Streaming aggregation by task
    print("\n5. Testing aggregate_by_task (streaming)...")
    task_streaming_output = Path(
        "results/agent_evaluation/task_streaming_aggregation.csv"
    )
    aggregate_by_task(
        input_file=results_file,
        output_filename=task_streaming_output,
        callable_func=mean_callable,
        streaming=True,
        chunk_size=5,
    )
    print(f"Task streaming aggregation saved to: {task_streaming_output}")

    # Test 6: JSON output
    print("\n6. Testing JSON output...")
    task_json_output = Path("results/agent_evaluation/task_aggregation.json")
    aggregate_by_task(
        input_file=results_file,
        output_filename=task_json_output,
        callable_func=mean_callable,
        streaming=False,
    )
    print(f"Task JSON aggregation saved to: {task_json_output}")

    print("\nAll aggregator tests completed!")

    # Show the results
    print("\nGenerated files:")
    output_dir = Path("results/agent_evaluation")
    for file in output_dir.glob("*aggregation*"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
