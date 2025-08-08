#!/usr/bin/env python3
"""
Example demonstrating the new ijson-based streaming functionality for JSON array files.

This example shows how the aggregators now use true streaming with ijson instead of
loading the entire JSON file into memory first.
"""

import json
import tempfile
from pathlib import Path

from novaeval.evaluators.aggregators import (
    aggregate_by_agent_name,
    aggregate_by_task,
    aggregate_by_user,
    mean_callable,
)


def create_sample_data():
    """Create sample evaluation data in JSON array format."""
    return [
        {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "accuracy_score": 0.8,
            "relevance_score": 0.9,
            "helpfulness_score": 0.7,
        },
        {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn2",
            "agent_name": "agent1",
            "accuracy_score": 0.9,
            "relevance_score": 0.8,
            "helpfulness_score": 0.8,
        },
        {
            "user_id": "user2",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent2",
            "accuracy_score": 0.7,
            "relevance_score": 0.6,
            "helpfulness_score": 0.9,
        },
        {
            "user_id": "user2",
            "task_id": "task2",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "accuracy_score": 0.6,
            "relevance_score": 0.7,
            "helpfulness_score": 0.8,
        },
        {
            "user_id": "user3",
            "task_id": "task2",
            "turn_id": "turn1",
            "agent_name": "agent2",
            "accuracy_score": 0.9,
            "relevance_score": 0.9,
            "helpfulness_score": 0.9,
        },
    ]


def main():
    """Demonstrate the new ijson streaming functionality."""
    print("=== NovaEval ijson Streaming Example ===\n")

    # Create sample data
    data = create_sample_data()

    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save data as JSON array
        input_file = temp_path / "evaluation_results.json"
        with open(input_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Created sample data with {len(data)} evaluation records")
        print(f"Input file: {input_file}")
        print()

        # Demonstrate task aggregation with streaming
        print("1. Aggregating by task (streaming mode):")
        task_output = temp_path / "task_aggregation.csv"
        aggregate_by_task(
            input_file=input_file,
            output_filename=task_output,
            callable_func=mean_callable,
            streaming=True,
        )

        # Read and display results
        import pandas as pd

        task_results = pd.read_csv(task_output)
        print(f"   Results saved to: {task_output}")
        print("   Task aggregation results:")
        print(task_results.to_string(index=False))
        print()

        # Demonstrate user aggregation with streaming
        print("2. Aggregating by user (streaming mode):")
        user_output = temp_path / "user_aggregation.csv"
        aggregate_by_user(
            input_file=input_file,
            output_filename=user_output,
            callable_func=mean_callable,
            streaming=True,
        )

        user_results = pd.read_csv(user_output)
        print(f"   Results saved to: {user_output}")
        print("   User aggregation results:")
        print(user_results.to_string(index=False))
        print()

        # Demonstrate agent aggregation with streaming
        print("3. Aggregating by agent (streaming mode):")
        agent_output = temp_path / "agent_aggregation.csv"
        aggregate_by_agent_name(
            input_file=input_file,
            output_filename=agent_output,
            callable_func=mean_callable,
            streaming=True,
        )

        agent_results = pd.read_csv(agent_output)
        print(f"   Results saved to: {agent_output}")
        print("   Agent aggregation results:")
        print(agent_results.to_string(index=False))
        print()

        # Demonstrate multiple aggregation functions
        print("4. Aggregating with multiple functions (streaming mode):")

        def max_callable(scores):
            return max(scores) if scores else 0.0

        def min_callable(scores):
            return min(scores) if scores else 0.0

        multi_output = temp_path / "multi_function_aggregation.csv"
        aggregate_by_task(
            input_file=input_file,
            output_filename=multi_output,
            callable_func=[mean_callable, max_callable, min_callable],
            streaming=True,
        )

        multi_results = pd.read_csv(multi_output)
        print(f"   Results saved to: {multi_output}")
        print("   Multi-function aggregation results:")
        print(multi_results.to_string(index=False))
        print()

        print("=== Key Benefits of ijson Streaming ===")
        print("✅ True streaming: Only processes one JSON object at a time")
        print("✅ Memory efficient: Doesn't load entire file into memory")
        print("✅ Fast: Processes large JSON files without memory issues")
        print("✅ Supports JSON array format: [object1, object2, ...]")
        print("✅ Error handling: Gracefully handles malformed JSON")
        print("✅ Column detection: Automatically detects available columns")


if __name__ == "__main__":
    main()
