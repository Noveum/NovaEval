#!/usr/bin/env python3
"""
DeepEval Final Comparison - Using GEval with optimized configuration for MMLU
"""

import json
import os
import re
import time

from datasets import load_dataset
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def extract_answer_choice(text):
    """Extract A, B, C, or D from model response"""
    # Look for patterns like "Answer: D", "D)", "**D**", etc.
    patterns = [
        r"(?:Answer|answer):\s*([ABCD])",
        r"(?:Answer|answer):\s*([ABCD])\)",
        r"\*\*([ABCD])\*\*",
        r"([ABCD])\)",
        r"(?:^|\s)([ABCD])(?:\s|$)",
        r"option\s*([ABCD])",
        r"choice\s*([ABCD])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # If no pattern found, look for last occurrence of A, B, C, or D
    matches = re.findall(r"[ABCD]", text.upper())
    if matches:
        return matches[-1]

    return "UNKNOWN"


def run_deepeval_final_comparison():
    """Run final DeepEval evaluation with optimized GEval configuration"""

    print("=== DeepEval Final Comparison (Optimized GEval) ===")
    start_time = time.time()

    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", "elementary_mathematics", split="test")
    samples = dataset.select(range(50))

    # Setup DeepEval model
    model = GPTModel(model="gpt-4.1-mini", api_key=os.environ["OPENAI_API_KEY"])

    # Create test cases for DeepEval SDK
    test_cases = []
    print("Creating test cases...")

    for i, sample in enumerate(samples):
        # Format question with choices
        choices = sample["choices"]
        question = f"{sample['question']}\n\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\nAnswer:"

        # Expected answer (convert index to letter)
        expected_answer = ["A", "B", "C", "D"][sample["answer"]]

        # Get actual output from model
        actual_output = model.generate(question)
        if isinstance(actual_output, tuple):
            actual_output = actual_output[0]  # Handle (response, cost) tuple

        # Create LLMTestCase for DeepEval SDK
        test_case = LLMTestCase(
            input=question, actual_output=actual_output, expected_output=expected_answer
        )
        test_cases.append(test_case)

        if (i + 1) % 10 == 0:
            print(f"Created {i+1}/50 test cases")

    # Create EvaluationDataset and add test cases
    eval_dataset = EvaluationDataset()

    # Add test cases to the dataset
    for test_case in test_cases:
        eval_dataset.add_test_case(test_case)

    # Use optimized GEval configuration for MMLU
    correctness_metric = GEval(
        name="MMLU_Correctness",
        criteria="Determine if the actual output contains the same answer choice (A, B, C, or D) as the expected output. Focus on the final answer choice, not the explanation.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.5,  # Lower threshold for more lenient evaluation
        strict_mode=False,  # Non-strict mode
    )

    print("Running DeepEval evaluation with optimized GEval...")

    # Use the actual SDK evaluator pattern: EvaluationDataset.evaluate()
    results = eval_dataset.evaluate(metrics=[correctness_metric])

    end_time = time.time()
    total_time = end_time - start_time

    print(f"DeepEval final comparison completed in {total_time:.2f}s")

    # Manual accuracy calculation for verification
    manual_correct = 0
    for test_case in test_cases:
        extracted = extract_answer_choice(test_case.actual_output)
        if extracted == test_case.expected_output:
            manual_correct += 1

    manual_accuracy = manual_correct / len(test_cases)

    # Save results
    sdk_results = {
        "framework": "DeepEval Final (Optimized GEval)",
        "evaluation_time": total_time,
        "results": results,
        "test_cases_count": len(test_cases),
        "manual_accuracy": manual_accuracy,
        "manual_correct": manual_correct,
        "timestamp": time.time(),
    }

    with open("/home/ubuntu/deepeval_final_results.json", "w") as f:
        json.dump(sdk_results, f, indent=2, default=str)

    print("Results saved to deepeval_final_results.json")

    # Print summary
    if hasattr(results, "test_results") and results.test_results:
        passed_tests = sum(1 for result in results.test_results if result.success)
        total_tests = len(results.test_results)
        geval_accuracy = passed_tests / total_tests if total_tests > 0 else 0

        print("\n=== DeepEval Final Results Summary ===")
        print(f"Total Tests: {total_tests}")
        print(f"GEval Passed Tests: {passed_tests}")
        print(f"GEval Accuracy: {geval_accuracy:.1%}")
        print(f"Manual Accuracy (Verification): {manual_accuracy:.1%}")
        print(f"Evaluation Time: {total_time:.2f}s")
        print(f"Time per Sample: {total_time/total_tests:.2f}s")

        # Compare with previous results
        print("\n=== Comparison with Previous Results ===")
        print("Previous GEval (strict): 48.0%")
        print(f"Optimized GEval: {geval_accuracy:.1%}")
        print(f"Manual verification: {manual_accuracy:.1%}")
        improvement = geval_accuracy - 0.48
        print(f"Improvement: {improvement:+.1%}")

    return sdk_results


if __name__ == "__main__":
    results = run_deepeval_final_comparison()
    print("DeepEval final comparison complete!")
