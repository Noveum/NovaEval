#!/usr/bin/env python3
"""
Simple wrapper to run agent evaluation on N samples.

Usage:
  python examples/run_agent_evaluation.py           # Run on 5 samples (default)
  python examples/run_agent_evaluation.py 10       # Run on 10 samples
  python examples/run_agent_evaluation.py 400      # Run on all 400 samples
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main function to run agent evaluation with configurable sample size."""

    # Get number of samples from command line argument
    num_samples = 5  # default
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
            if num_samples <= 0:
                print("Error: Number of samples must be positive")
                return 1
        except ValueError:
            print("Error: Invalid number of samples. Please provide an integer.")
            return 1

    print(f"🚀 Running agent evaluation on {num_samples} samples...")

    # Temporarily modify the agent_evaluation_clean module
    import examples.agent_evaluation_clean as eval_module

    # Store original values
    # original_num_samples = getattr(eval_module, "DEFAULT_NUM_SAMPLES", 5)
    # original_output_file = getattr(
    #     eval_module, "DEFAULT_OUTPUT_FILE", "agent_scores_with_reasoning.csv"
    # )
    # Update the module's main function to use our parameters
    def custom_main():
        """Main evaluation workflow with custom parameters."""
        print("🚀 Starting Clean Agent Evaluation (Agent Scorers Only)")
        print("=" * 60)

        # Setup
        dataset_file = "swe_agent_dataset.json"
        output_file = f"agent_scores_{num_samples}_samples.csv"

        # Check if dataset exists
        if not os.path.exists(dataset_file):
            print(f"❌ Dataset file '{dataset_file}' not found!")
            print("Please run the SWE dataset creation script first.")
            return 1

        try:
            # Load model and data
            model = eval_module.setup_model()
            agent_samples = eval_module.load_agent_dataset(dataset_file, num_samples)

            # Run evaluation on all samples
            all_results = []
            for i, agent_data in enumerate(agent_samples):
                result = eval_module.evaluate_single_agent_csv(agent_data, model, i)
                all_results.append(result)

            # Create DataFrame and save to CSV
            import pandas as pd

            df = pd.DataFrame(all_results)
            df.to_csv(output_file, index=False)

            print("\n📊 Results Summary:")
            print(f"   Total entries evaluated: {len(all_results)}")
            print(f"   Columns in output: {len(df.columns)}")
            print(f"   Output file: {output_file}")

            # Show column names
            print("\n📋 CSV Columns:")
            for i, col in enumerate(df.columns, 1):
                print(f"   {i:2d}. {col}")

            # Show sample scores (first few columns only)
            print("\n🔍 Sample Results (first 3 rows, scores only):")
            score_columns = [col for col in df.columns if col.endswith("_score")]
            identifier_columns = ["traj_id", "span_id", "step_number"]
            display_columns = (
                identifier_columns + score_columns[:4]
            )  # Show first 4 score columns

            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            print(df[display_columns].head(3).to_string())

            print(f"\n💾 Complete results (scores + reasoning) saved to: {output_file}")
            print("\n✅ Agent evaluation completed successfully!")

            return 0

        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            import traceback

            traceback.print_exc()
            return 1

    # Run the custom main function
    return custom_main()


if __name__ == "__main__":
    exit(main())
