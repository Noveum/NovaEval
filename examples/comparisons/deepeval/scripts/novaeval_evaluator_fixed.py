#!/usr/bin/env python3
"""
NovaEval Evaluator - Using the actual Evaluator class properly
"""

import json
import os
import time

from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer
from novaeval.scorers.accuracy import F1Scorer
from novaeval.scorers.conversational import (
    ConversationCompletenessScorer,
    ConversationRelevancyScorer,
)


def run_novaeval_evaluator():
    """Run evaluation using NovaEval's actual Evaluator class"""

    print("=== NovaEval Evaluator (Proper Implementation) ===")
    start_time = time.time()

    # Setup NovaEval components using their SDK
    print("Setting up NovaEval components...")

    # Create dataset
    dataset = MMLUDataset(subset="elementary_mathematics", num_samples=10, split="test")

    # Create model
    model = OpenAIModel(
        model_name="gpt-4.1-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.0,
        max_tokens=1000,
    )

    # Create scorer
    scorers = [
        AccuracyScorer(),
        F1Scorer(model=model),
        ConversationRelevancyScorer(model),
        ConversationCompletenessScorer(model),
    ]
    print(scorers)

    # Create the actual Evaluator
    evaluator = Evaluator(
        dataset=dataset,
        models=[model],
        scorers=scorers,
        output_dir="./novaeval_evaluator_results",
    )

    print("Running NovaEval Evaluator.run()...")

    # Run the actual evaluator
    results = evaluator.run()

    end_time = time.time()
    total_time = end_time - start_time

    print(f"NovaEval Evaluator completed in {total_time:.2f}s")

    # Save results
    evaluator_results = {
        "framework": "NovaEval Evaluator",
        "evaluation_time": total_time,
        "evaluator_results": results,
        "timestamp": time.time(),
    }

    with open("novaeval_evaluator_results/results.json", "w") as f:
        json.dump(evaluator_results, f, indent=2, default=str)

    print("Results saved to novaeval_evaluator_results.json")

    # Print summary from evaluator results
    if results and "evaluation_results" in results:
        eval_results = results["evaluation_results"]
        print(eval_results)
        if eval_results:
            model_name = next(iter(eval_results.keys()))
            model_results = eval_results[model_name]

            print("\n=== NovaEval Evaluator Results Summary ===")
            print(f"Model: {model_name}")

            if "scorers" in model_results:
                for scorer_name, scorer_data in model_results["scorers"].items():
                    print(f"\n{scorer_name} Results:")
                    for metric, value in scorer_data.items():
                        if isinstance(value, float):
                            if metric == "accuracy":
                                print(f"  {metric}: {value:.1%}")
                            else:
                                print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")

            print(f"\nEvaluation Time: {total_time:.2f}s")
            if "metadata" in results and "dataset" in results["metadata"]:
                num_samples = results["metadata"]["dataset"]["num_samples"]
                print(f"Time per Sample: {total_time/num_samples:.2f}s")

    return evaluator_results


if __name__ == "__main__":
    results = run_novaeval_evaluator()
    print("NovaEval Evaluator complete!")
