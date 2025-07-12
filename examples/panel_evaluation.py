#!/usr/bin/env python3
"""
Panel of LLMs Evaluation with Latest Models - Updated for NovaEval

This example demonstrates the Panel of Judges functionality using the latest
OpenAI and Anthropic models within the NovaEval framework.
"""

from novaeval import Evaluator
from novaeval.datasets import CustomDataset
from novaeval.datasets.huggingface import HuggingFaceDataset
from novaeval.models import AnthropicModel, OpenAIModel
from novaeval.scorers.panel_judge import (
    AggregationMethod,
    JudgeConfig,
    PanelOfJudgesScorer,
)


def create_latest_models_evaluation():
    """Create evaluation setup with latest models."""

    # Set up the models to be evaluated (latest versions)
    models_to_evaluate = [
        OpenAIModel(model_name="gpt-4o", temperature=0.0),
        AnthropicModel(model_name="claude-3-5-sonnet-20241022", temperature=0.0),
        OpenAIModel(model_name="gpt-4o-mini", temperature=0.0),
    ]

    # Configure the judge panel with latest models
    judge_panel = [
        JudgeConfig(
            model=OpenAIModel(model_name="gpt-4o", temperature=0.0),
            weight=1.5,
            name="GPT-4o Expert",
            specialty="accuracy_and_reasoning",
        ),
        JudgeConfig(
            model=AnthropicModel(
                model_name="claude-3-5-sonnet-20241022", temperature=0.0
            ),
            weight=1.5,
            name="Claude-3.5-Sonnet Expert",
            specialty="clarity_and_helpfulness",
        ),
        JudgeConfig(
            model=OpenAIModel(model_name="gpt-4o-mini", temperature=0.1),
            weight=1.0,
            name="GPT-4o-Mini Judge",
            specialty="completeness_and_efficiency",
        ),
        JudgeConfig(
            model=AnthropicModel(
                model_name="claude-3-5-haiku-20241022", temperature=0.1
            ),
            weight=0.8,
            name="Claude-3.5-Haiku Judge",
            specialty="conciseness_and_relevance",
        ),
    ]

    # Create the panel scorer
    panel_scorer = PanelOfJudgesScorer(
        judges=judge_panel,
        aggregation_method=AggregationMethod.WEIGHTED_MEAN,
        threshold=0.8,
        require_consensus=True,
        consensus_threshold=0.7,
        evaluation_criteria="overall quality, factual accuracy, helpfulness, and clarity",
    )

    return models_to_evaluate, panel_scorer


def create_reasoning_panel():
    """Create a specialized reasoning panel with latest models."""

    reasoning_judges = [
        JudgeConfig(
            model=OpenAIModel(model_name="o1-preview", temperature=0.0),
            weight=2.0,
            name="o1-Preview Reasoning Expert",
            specialty="complex_reasoning_and_analysis",
        ),
        JudgeConfig(
            model=OpenAIModel(model_name="gpt-4o", temperature=0.0),
            weight=1.5,
            name="GPT-4o General Expert",
            specialty="accuracy_and_completeness",
        ),
        JudgeConfig(
            model=AnthropicModel(
                model_name="claude-3-5-sonnet-20241022", temperature=0.0
            ),
            weight=1.5,
            name="Claude-3.5-Sonnet Clarity Expert",
            specialty="clarity_and_explanation",
        ),
        JudgeConfig(
            model=OpenAIModel(model_name="o1-mini", temperature=0.0),
            weight=1.0,
            name="o1-Mini Reasoning Judge",
            specialty="logical_consistency",
        ),
    ]

    reasoning_panel = PanelOfJudgesScorer(
        judges=reasoning_judges,
        aggregation_method=AggregationMethod.WEIGHTED_MEAN,
        threshold=0.85,
        require_consensus=True,
        consensus_threshold=0.75,
        evaluation_criteria="reasoning quality, logical consistency, accuracy, and explanation clarity",
    )

    return reasoning_panel


def run_basic_evaluation():
    """Run basic evaluation with custom dataset."""

    print("🏛️ Running Panel of Judges Evaluation with Latest Models")
    print("=" * 60)

    # Get models and panel scorer
    models_to_evaluate, panel_scorer = create_latest_models_evaluation()

    # Set up dataset
    dataset = CustomDataset(data_source="./test_data/qa_dataset.jsonl")

    # Run evaluation
    evaluator = Evaluator(
        dataset=dataset,
        models=models_to_evaluate,
        scorers=[panel_scorer],
        output_dir="./results/panel_evaluation_latest",
    )

    results = evaluator.run()

    # Display results
    print("\n📊 Evaluation Results:")
    for model_name, model_results in results["model_results"].items():
        panel_score = model_results["scores"]["panel_judge"]["mean"]
        # Safely get consensus level from metadata
        panel_judge_scores = model_results["scores"]["panel_judge"]
        consensus = panel_judge_scores.get("metadata", {}).get("consensus_level", "N/A")

        print(f"  {model_name}:")
        print(f"    Panel Score: {panel_score:.3f}")
        if consensus != "N/A":
            print(f"    Consensus Level: {consensus:.3f}")
        else:
            print(f"    Consensus Level: {consensus}")

    return results


def run_huggingface_evaluation():
    """Run evaluation with interesting HuggingFace dataset."""

    print("\n🤗 Running Evaluation on HuggingFace Dataset")
    print("=" * 50)

    # Get models and panel scorer
    models_to_evaluate, panel_scorer = create_latest_models_evaluation()

    # Set up HuggingFace dataset (Berkeley Nectar)
    dataset = HuggingFaceDataset(
        dataset_name="berkeley-nest/Nectar",
        split="train",
        input_column="prompt",
        num_samples=25,
        preprocessing_fn=lambda x: {
            "input": x["prompt"]
            .split("\n\nHuman:")[-1]
            .split("\n\nAssistant:")[0]
            .strip(),
            "expected": x["answers"][0]["answer"] if x["answers"] else "",
            "metadata": {
                "source": "berkeley-nest/Nectar",
                "good_natured": x.get("good_natured", True),
                "num_responses": len(x["answers"]),
            },
        },
    )

    # Run evaluation
    evaluator = Evaluator(
        dataset=dataset,
        models=models_to_evaluate,
        scorers=[panel_scorer],
        output_dir="./results/nectar_evaluation_latest",
    )

    results = evaluator.run()

    # Display results
    print("\n📊 HuggingFace Dataset Results:")
    for model_name, model_results in results["model_results"].items():
        panel_score = model_results["scores"]["panel_judge"]["mean"]
        panel_judge_scores = model_results["scores"]["panel_judge"]
        consensus = panel_judge_scores.get("metadata", {}).get("consensus_level", "N/A")

        print(f"  {model_name}:")
        print(f"    Panel Score: {panel_score:.3f}")
        if consensus != "N/A":
            print(f"    Consensus Level: {consensus:.3f}")
        else:
            print(f"    Consensus Level: {consensus}")

    return results


def run_reasoning_evaluation():
    """Run evaluation with reasoning-focused panel."""

    print("\n🧠 Running Reasoning-Focused Evaluation")
    print("=" * 45)

    # Get models
    models_to_evaluate, _ = create_latest_models_evaluation()

    # Get reasoning panel
    reasoning_panel = create_reasoning_panel()

    # Set up dataset
    dataset = CustomDataset(data_source="./test_data/qa_dataset.jsonl")

    # Run evaluation
    evaluator = Evaluator(
        dataset=dataset,
        models=models_to_evaluate,
        scorers=[reasoning_panel],
        output_dir="./results/reasoning_evaluation_latest",
    )

    results = evaluator.run()

    # Display results
    print("\n📊 Reasoning Evaluation Results:")
    for model_name, model_results in results["model_results"].items():
        reasoning_score = model_results["scores"]["panel_judge"]["mean"]
        panel_judge_scores = model_results["scores"]["panel_judge"]
        consensus = panel_judge_scores.get("metadata", {}).get("consensus_level", "N/A")
        print(f"  {model_name}:")
        print(f"    Reasoning Score: {reasoning_score:.3f}")
        if consensus != "N/A":
            print(f"    Consensus Level: {consensus:.3f}")
        else:
            print(f"    Consensus Level: {consensus}")

    return results


def run_comprehensive_evaluation():
    """Run comprehensive evaluation with multiple panels."""

    print("\n🎯 Running Comprehensive Multi-Panel Evaluation")
    print("=" * 55)

    # Get models
    models_to_evaluate, general_panel = create_latest_models_evaluation()
    reasoning_panel = create_reasoning_panel()

    # Set up dataset
    dataset = CustomDataset(data_source="./test_data/qa_dataset.jsonl")

    # Run evaluation with multiple scorers
    evaluator = Evaluator(
        dataset=dataset,
        models=models_to_evaluate,
        scorers=[general_panel, reasoning_panel],
        output_dir="./results/comprehensive_evaluation_latest",
    )

    results = evaluator.run()

    # Display comprehensive results
    print("\n📊 Comprehensive Evaluation Results:")
    for model_name, model_results in results["model_results"].items():
        general_score = model_results["scores"]["panel_judge"]["mean"]
        reasoning_score = model_results["scores"]["panel_judge_1"][
            "mean"
        ]  # Second scorer

        general_scores = model_results["scores"]["panel_judge"]
        reasoning_scores = model_results["scores"]["panel_judge_1"]

        general_consensus = general_scores.get("metadata", {}).get(
            "consensus_level", "N/A"
        )
        reasoning_consensus = reasoning_scores.get("metadata", {}).get(
            "consensus_level", "N/A"
        )

        print(f"  {model_name}:")
        general_consensus_str = (
            f"{general_consensus:.3f}" if general_consensus != "N/A" else "N/A"
        )
        reasoning_consensus_str = (
            f"{reasoning_consensus:.3f}" if reasoning_consensus != "N/A" else "N/A"
        )

        print(
            f"    General Panel Score: {general_score:.3f} (consensus: {general_consensus_str})"
        )
        print(
            f"    Reasoning Panel Score: {reasoning_score:.3f} (consensus: {reasoning_consensus_str})"
        )
        print(f"    Average Score: {(general_score + reasoning_score) / 2:.3f}")

    return results


def main():
    """Main function to run all evaluations."""

    print("🚀 NovaEval: Panel of LLMs with Latest Models")
    print("=" * 70)

    try:
        # Run different types of evaluations
        run_basic_evaluation()

        # Uncomment to run additional evaluations
        # huggingface_results = run_huggingface_evaluation()
        # reasoning_results = run_reasoning_evaluation()
        # comprehensive_results = run_comprehensive_evaluation()

        print("\n✅ All evaluations completed successfully!")
        print("\nResults saved to:")
        print("  - ./results/panel_evaluation_latest/")
        # print("  - ./results/nectar_evaluation_latest/")
        # print("  - ./results/reasoning_evaluation_latest/")
        # print("  - ./results/comprehensive_evaluation_latest/")

    except Exception as e:
        print(f"❌ Error during evaluation: {e!s}")
        raise


if __name__ == "__main__":
    # Example of how to create test data if needed
    import json
    import os

    # Create test data directory if it doesn't exist
    os.makedirs("./test_data", exist_ok=True)

    # Create sample QA dataset if it doesn't exist
    if not os.path.exists("./test_data/qa_dataset.jsonl"):
        sample_data = [
            {
                "input": "What is the capital of France?",
                "expected": "Paris",
                "metadata": {"difficulty": "easy", "category": "geography"},
            },
            {
                "input": "Explain the concept of machine learning in simple terms.",
                "expected": "Machine learning is a subset of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
                "metadata": {"difficulty": "medium", "category": "technology"},
            },
            {
                "input": "What are the main causes of climate change?",
                "expected": "The main causes of climate change include greenhouse gas emissions from burning fossil fuels, deforestation, industrial processes, and agriculture.",
                "metadata": {"difficulty": "medium", "category": "science"},
            },
        ]

        with open("./test_data/qa_dataset.jsonl", "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        print("📝 Created sample test dataset at ./test_data/qa_dataset.jsonl")

    # Run the main evaluation
    main()
