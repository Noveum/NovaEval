"""
Example usage of the AgentEvaluator.

This script demonstrates how to use the AgentEvaluator to evaluate
agent datasets using agent scorers.
"""

import os
from pathlib import Path

from novaeval.datasets.agent_dataset import AgentDataset
from novaeval.evaluators.agent_evaluator import AgentEvaluator
from novaeval.models.gemini import GeminiModel
from novaeval.scorers.agent_scorers import (
    context_relevancy_scorer,
    parameter_correctness_scorer,
    role_adherence_scorer,
    task_progression_scorer,
    tool_correctness_scorer,
    tool_relevancy_scorer,
)


def main():
    """Run the agent evaluator example."""

    # Set up output directory
    output_dir = Path("./results/agent_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize agent dataset
    agent_dataset = AgentDataset()

    # Load data from the simple CSV file
    csv_file = "examples/simple_agent_dataset.csv"

    if os.path.exists(csv_file):
        print(f"Loading data from {csv_file}")
        # Map CSV columns to AgentData fields
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
        print("Please ensure the dummy dataset file exists.")
        return

    # Initialize models
    # You'll need to set your Gemini API key as an environment variable
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return

    model = GeminiModel(
        model_name="gemini-2.5-flash",  # Using the faster, cheaper model
        api_key=gemini_api_key,
        temperature=0.0,  # For consistent evaluation results
    )
    print("Using Gemini model for evaluation")

    # Initialize scoring functions
    scoring_functions = [
        tool_relevancy_scorer,
        tool_correctness_scorer,
        parameter_correctness_scorer,
        task_progression_scorer,
        context_relevancy_scorer,
        role_adherence_scorer,
    ]
    print(f"Using {len(scoring_functions)} scoring functions:")
    for func in scoring_functions:
        print(f"  - {func.__name__}")
    # Create the agent evaluator
    evaluator = AgentEvaluator(
        agent_dataset=agent_dataset,
        models=[model],
        scoring_functions=scoring_functions,
        output_dir=output_dir,
        stream=False,  # Set to True for large datasets
        include_reasoning=True,  # Set to False to exclude reasoning
        config={
            "log_level": "INFO",
            "save_every": 5,  # Save results every 5 samples
        },
    )

    # Run the evaluation
    print("Starting agent evaluation...")
    evaluator.run_all(
        save_every=5, file_type="csv"  # Save every 5 samples  # or "json"
    )

    print(f"Evaluation completed! Results saved to {output_dir}")

    # You can also run individual samples
    print("\nExample of evaluating a single sample:")
    sample = next(agent_dataset.get_datapoint())
    sample_result = evaluator.evaluate_sample(sample, model)
    print(f"Sample result: {sample_result}")


if __name__ == "__main__":
    main()
