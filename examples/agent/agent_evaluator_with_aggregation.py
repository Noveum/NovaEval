"""
Test script for AgentEvaluator with aggregation functionality.
"""

import os
from pathlib import Path

from novaeval.datasets.agent_dataset import AgentDataset
from novaeval.evaluators.agent_evaluator import AgentEvaluator
from novaeval.evaluators.aggregators import mean_callable
from novaeval.models.gemini import GeminiModel
from novaeval.scorers.agent_scorers import (
    context_relevancy_scorer,
    role_adherence_scorer,
    tool_relevancy_scorer,
)


def main():
    """Test the AgentEvaluator with aggregation."""

    # Set up output directory
    output_dir = Path("./results/agent_evaluation_with_aggregation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize agent dataset
    agent_dataset = AgentDataset()

    # Load data from the simple CSV file
    csv_file = "examples/simple_agent_dataset.csv"

    if os.path.exists(csv_file):
        print(f"Loading data from {csv_file}")
        agent_dataset.ingest_from_csv(
            csv_file,
            user_id="user_id",
            task_id="task_id",
            turn_id="turn_id",
            agent_name="agent_name",
            agent_role="agent_role",
            agent_task="agent_task",
            system_prompt="system_prompt",
            agent_response="agent_response",
            trace="trace",
            tools_available="tools_available",
            tool_calls="tool_calls",
            parameters_passed="parameters_passed",
            tool_call_results="tool_call_results",
            retrieval_query=["retrieval_query"],
            retrieved_context=[["retrieved_context"]],
            exit_status="exit_status",
            agent_exit="agent_exit",
            ground_truth="ground_truth",
            expected_tool_call="expected_tool_call",
            metadata="metadata",
        )
        print(f"Loaded {len(agent_dataset.data)} samples")
    else:
        print(f"CSV file not found: {csv_file}")
        return

    # Initialize models
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return

    model = GeminiModel(
        model_name="gemini-2.5-flash", api_key=gemini_api_key, temperature=0.0
    )
    print("Using Gemini model for evaluation")

    # Initialize scoring functions (using fewer for faster testing)
    scoring_functions = [
        tool_relevancy_scorer,
        context_relevancy_scorer,
        role_adherence_scorer,
    ]
    print(f"Using {len(scoring_functions)} scoring functions:")
    for func in scoring_functions:
        print(f"  - {func.__name__}")

    # Define aggregator functions
    aggregator_functions = [
        max,
        min,
        mean_callable,
    ]
    print(f"Using {len(aggregator_functions)} aggregator functions:")
    for func in aggregator_functions:
        print(f"  - {func.__name__}")

    # Create the agent evaluator
    evaluator = AgentEvaluator(
        agent_dataset=agent_dataset,
        models=[model],
        scoring_functions=scoring_functions,
        output_dir=output_dir,
        stream=False,
        include_reasoning=True,
        config={
            "log_level": "INFO",
        },
    )

    # Run evaluation with aggregation
    print("Starting agent evaluation with aggregation...")
    evaluator.run_all(
        save_every=5,
        file_type="csv",
        aggregate_by_task=True,
        aggregate_by_user=False,  # Skip user aggregation for now
        aggregate_by_agent_name=True,
        aggregator_functions=aggregator_functions,
        aggregation_chunk_size=100,
    )

    print(f"Evaluation completed! Results saved to {output_dir}")

    # List generated files
    print("\nGenerated files:")
    for file in output_dir.glob("*.csv"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
