"""
MMLU Re-scoring Script for NovaEval

This script takes the output directory from ollama_mmlu_evaluation.py and re-scores
the results using MultiPatternAccuracyScorer without re-running inference.

Usage:
    python ollama_mmlu_rescoring.py <original_run_directory>

Example:
    python ollama_mmlu_rescoring.py examples/ollama_run_20241201_143022
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add the src directory to the path to import novaeval
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novaeval.scorers.accuracy import MultiPatternAccuracyScorer


def extract_choices_from_input(input_text: str) -> tuple[list[str], Optional[int]]:
    """
    Extract choices from MMLU input text and find the answer index.

    Args:
        input_text: The input text containing the question and choices

    Returns:
        Tuple of (choices_list, answer_index) where answer_index is the index of the correct answer
    """
    # Look for the pattern \n{A,B,C,D}.{spacebar} in the input
    choice_pattern = r"\n([A-D])\.\s*([^\n]+)"
    matches = re.findall(choice_pattern, input_text)

    if not matches:
        return [], None

    choices = []
    for _letter, text in matches:
        # Clean up the choice text
        clean_text = text.strip()
        choices.append(clean_text)

    # Find which choice corresponds to the expected answer
    # This will be determined later when we have the expected answer

    return choices, None


def find_answer_index(expected_answer: str, choices: list[str]) -> Optional[int]:
    """
    Find the index of the expected answer among the choices.

    Args:
        expected_answer: The expected/correct answer text
        choices: List of choice texts

    Returns:
        Index of the correct answer (0-3) or None if not found
    """
    if not choices:
        return None

    # Normalize the expected answer for comparison
    expected_normalized = expected_answer.strip().lower()

    for i, choice in enumerate(choices):
        choice_normalized = choice.strip().lower()
        if expected_normalized == choice_normalized:
            return i

    # If exact match not found, try partial matching
    for i, choice in enumerate(choices):
        choice_normalized = choice.strip().lower()
        if (
            expected_normalized in choice_normalized
            or choice_normalized in expected_normalized
        ):
            return i

    return None


def rescore_detailed_results(
    detailed_csv_path: Path, scorer: MultiPatternAccuracyScorer
) -> pd.DataFrame:
    """
    Re-score the detailed results CSV using the provided scorer.

    Args:
        detailed_csv_path: Path to the detailed_results.csv file
        scorer: The scorer instance to use

    Returns:
        DataFrame with new scores added
    """
    # Read the detailed results
    df = pd.read_csv(detailed_csv_path)

    # Add new scoring columns
    new_scores = []

    for _, row in df.iterrows():
        cur_score = row["score_accuracy"]
        if pd.isna(cur_score):
            cur_score = 0.0
        if abs(cur_score - 1.0) < 1e-6:  # or cur_score >= 0.9999
            new_scores.append(1.0)
            continue
        input_text = row["input"]
        expected_answer = row["expected"]
        prediction = row["prediction"]

        # Extract choices from input
        choices, _ = extract_choices_from_input(input_text)

        # Find answer index
        answer_index = find_answer_index(expected_answer, choices)

        # Create context for the scorer
        context = {"choices": choices, "answer_index": answer_index}

        # Score using the new scorer
        try:
            score = scorer.score(
                prediction=prediction, ground_truth=expected_answer, context=context
            )
            new_scores.append(score)
        except Exception as e:
            print(f"Error scoring sample {row['sample_id']}: {e}")
            new_scores.append(0.0)

    # Add the new scores as a new column
    df[f"score_{scorer.name}"] = new_scores

    return df


def process_mode_directory(
    mode_dir: Path, scorer: MultiPatternAccuracyScorer, output_mode_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process a single mode directory (e.g., mode_low, mode_high).

    Args:
        mode_dir: Path to the input mode directory
        scorer: The scorer instance to use
        output_mode_dir: Path to the output mode directory

    Returns:
        Tuple of (regex_log_df, diff_results_df, rescored_df) for this mode
    """
    # Create output directory
    output_mode_dir.mkdir(parents=True, exist_ok=True)

    regex_log_rows = []
    diff_results_rows = []
    rescored_df = None

    # Process detailed_results.csv if it exists
    detailed_csv = mode_dir / "detailed_results.csv"
    if detailed_csv.exists():
        print(f"  Processing {detailed_csv.name}...")

        # Re-score the detailed results
        rescored_df = rescore_detailed_results(detailed_csv, scorer)

        # Create regex log and diff results for this mode
        for _, row in rescored_df.iterrows():
            # Extract choices from input
            choices, _ = extract_choices_from_input(row["input"])

            # Find answer index by matching expected answer with choices
            answer_index = find_answer_index(row["expected"], choices)

            # Add to regex log
            regex_log_rows.append(
                {
                    "mode": mode_dir.name,
                    "sample_id": row["sample_id"],
                    "input": row["input"],
                    "expected_ans": row["expected"],
                    "extracted_choices": " | ".join(choices) if choices else "",
                    "extracted_answer_index": (
                        answer_index if answer_index is not None else ""
                    ),
                }
            )

            # Check for differences between scorers
            original_score = row.get("score_accuracy", 0.0)
            new_score = row.get(f"score_{scorer.name}", 0.0)

            if original_score != new_score:
                # Extract subset from sample_id (e.g., "high_school_statistics_30" -> "high_school_statistics")
                sample_id = row["sample_id"]
                subset = sample_id.rsplit("_", 1)[0] if "_" in sample_id else "unknown"

                diff_results_rows.append(
                    {
                        "mode": mode_dir.name,
                        "subset": subset,
                        "sample_id": sample_id,
                        "original_score": original_score,
                        "new_score": new_score,
                        "difference": abs(original_score - new_score),
                    }
                )

        # Save to output directory
        output_detailed_csv = output_mode_dir / "detailed_results.csv"
        rescored_df.to_csv(output_detailed_csv, index=False)
        print(f"    Saved rescored results to {output_detailed_csv}")

    # Copy any other files in the mode directory
    for file_path in mode_dir.iterdir():
        if file_path.is_file() and file_path.name != "detailed_results.csv":
            output_file = output_mode_dir / file_path.name
            # For CSV files, we need to add the new score column
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
                # Add the new score column with placeholder values
                # The actual scores will be computed from detailed_results.csv
                df[f"score_{scorer.name}"] = 0.0  # Placeholder
                df.to_csv(output_file, index=False)
            else:
                # Copy other files as-is
                import shutil

                shutil.copy2(file_path, output_file)

    # Create DataFrames
    regex_log_df = pd.DataFrame(regex_log_rows)
    diff_results_df = pd.DataFrame(diff_results_rows)

    return regex_log_df, diff_results_df, rescored_df


def fix_mean_scores_in_csv(csv_file_path: Path) -> pd.DataFrame:
    """
    Fix MEAN row scores in a CSV file by recalculating subset-wise means.

    Args:
        csv_file_path: Path to the CSV file to fix

    Returns:
        DataFrame with corrected MEAN scores
    """
    print(f"    Fixing MEAN scores in {csv_file_path.name}...")

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Group by subset and calculate means for each subset
    subset_means = {}

    for subset in df["subset"].unique():
        if subset == "MEAN":  # Skip if there's a MEAN in subset column
            continue

        # Get all rows for this subset (excluding MEAN rows)
        subset_data = df[(df["subset"] == subset) & (df["sample_id"] != "MEAN")]

        if not subset_data.empty:
            # Calculate mean of score_accuracy for this subset
            # Handle any NaN values by converting to numeric first
            scores = pd.to_numeric(subset_data["score_accuracy"], errors="coerce")
            scores = scores.dropna()  # Remove any NaN values

            if len(scores) > 0:
                mean_score = scores.mean()
                subset_means[subset] = mean_score
                print(f"      {subset}: {len(scores)} samples, mean = {mean_score:.4f}")

    # Update MEAN rows with calculated means
    updated_count = 0
    for idx, row in df.iterrows():
        if row["sample_id"] == "MEAN":
            subset = row["subset"]
            if subset in subset_means:
                old_score = df.at[idx, "score_accuracy"]
                new_score = subset_means[subset]
                df.at[idx, "score_accuracy"] = new_score
                print(
                    f"      Updated MEAN for {subset}: {old_score} -> {new_score:.4f}"
                )
                updated_count += 1

    print(f"      Updated {updated_count} MEAN rows")
    return df


def aggregate_scores_for_summary_csv(
    summary_csv_path: Path,
    mode_detailed_results: dict[str, pd.DataFrame],
    scorer_name: str,
) -> pd.DataFrame:
    """
    Aggregate scores from detailed results to populate summary CSV with actual scores.

    Args:
        summary_csv_path: Path to the summary CSV file
        mode_detailed_results: Dict mapping mode names to their detailed results DataFrames
        scorer_name: Name of the scorer (e.g., 'multi_pattern_accuracy')

    Returns:
        DataFrame with actual scores populated and MEAN values recalculated
    """
    df = pd.read_csv(summary_csv_path)

    # Add the new score column
    df[f"score_{scorer_name}"] = 0.0

    # Create a lookup dictionary from all detailed results
    score_lookup = {}
    for _mode_name, detailed_df in mode_detailed_results.items():
        if detailed_df is not None:
            for _, row in detailed_df.iterrows():
                sample_id = row["sample_id"]
                score = row.get(f"score_{scorer_name}", 0.0)
                score_lookup[sample_id] = score

    # For each row in the summary, look up the score from detailed results
    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        if sample_id in score_lookup:
            df.at[idx, f"score_{scorer_name}"] = score_lookup[sample_id]

    # Recalculate MEAN values for each subset
    subset_means = {}
    for _idx, row in df.iterrows():
        if row["sample_id"] == "MEAN":
            subset = row["subset"]
            # Get all scores for this subset (excluding MEAN rows)
            subset_scores = []
            for _score_idx, score_row in df.iterrows():
                if score_row["subset"] == subset and score_row["sample_id"] != "MEAN":
                    score = score_row[f"score_{scorer_name}"]
                    if pd.notna(score) and score != "":
                        subset_scores.append(float(score))

            if subset_scores:
                new_mean = sum(subset_scores) / len(subset_scores)
                subset_means[subset] = new_mean

    # Update MEAN rows with new calculated values
    for idx, row in df.iterrows():
        if row["sample_id"] == "MEAN":
            subset = row["subset"]
            if subset in subset_means:
                df.at[idx, f"score_{scorer_name}"] = subset_means[subset]

    # Replace score_accuracy with the new scores
    df["score_accuracy"] = df[f"score_{scorer_name}"]

    # Drop the temporary column
    df = df.drop(columns=[f"score_{scorer_name}"])

    return df


def main():
    """Main function to run the re-scoring process."""
    parser = argparse.ArgumentParser(
        description="Re-score MMLU evaluation results using MultiPatternAccuracyScorer"
    )
    parser.add_argument(
        "run_directory",
        help="Path to the original Ollama MMLU evaluation run directory",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_directory)

    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist")
        sys.exit(1)

    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    # Check if this looks like a valid run directory
    mode_dirs = [
        d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("mode_")
    ]
    if not mode_dirs:
        print(
            f"Error: {run_dir} does not appear to be a valid Ollama MMLU run directory"
        )
        print("Expected subdirectories like mode_low, mode_high, etc.")
        sys.exit(1)

    # Create the MultiPatternAccuracyScorer with the specified patterns
    patterns = [
        r"(?:Answer|answer):\s*([A-Za-z0-9]+)",
        r"^([A-D])",
        r"\b([A-D])\b",
        r"answer\s*is\s*([A-D])",
        r"([A-D])\.",
        r"\(([A-D])\)",
    ]

    scorer = MultiPatternAccuracyScorer(patterns=patterns)

    # Create output directory with "multi_pattern_" prefix
    datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"multi_pattern_{run_dir.name}"
    output_dir = run_dir.parent / output_dir_name

    print(f"Original run directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using scorer: {scorer.name}")
    print(f"Patterns: {patterns}")
    print()

    # Process each mode directory
    all_regex_logs = []
    all_diff_results = []
    mode_detailed_results = {}  # Store detailed results for each mode

    for mode_dir in sorted(mode_dirs):
        mode_name = mode_dir.name
        print(f"Processing {mode_name}...")

        output_mode_dir = output_dir / mode_name
        regex_log_df, diff_results_df, rescored_df = process_mode_directory(
            mode_dir, scorer, output_mode_dir
        )

        # Store detailed results for this mode
        mode_detailed_results[mode_name] = rescored_df

        # Collect data from all modes
        if not regex_log_df.empty:
            all_regex_logs.append(regex_log_df)
        if not diff_results_df.empty:
            all_diff_results.append(diff_results_df)

    # Process the summary CSV files in the root directory
    print("\nProcessing summary CSV files...")
    for csv_file in run_dir.glob("*.csv"):
        # Skip mode-specific CSV files (high.csv, low.csv, medium.csv, unspecified.csv)
        # These will be handled separately by copying them with updated scores
        if csv_file.name in ["high.csv", "low.csv", "medium.csv", "unspecified.csv"]:
            print(f"  Will copy {csv_file.name} with updated scores...")
            continue

        print(f"  Processing {csv_file.name}...")

        # Aggregate scores from detailed results
        df_with_scores = aggregate_scores_for_summary_csv(
            csv_file, mode_detailed_results, scorer.name
        )

        # Fix MEAN scores by recalculating subset-wise means
        df_with_scores = fix_mean_scores_in_csv(csv_file)

        # Save to output directory
        output_csv = output_dir / csv_file.name
        df_with_scores.to_csv(output_csv, index=False)
        print(f"    Saved to {output_csv}")

    # Now copy mode-specific CSV files with updated scores
    print("\nCopying mode-specific CSV files with updated scores...")
    for csv_file in run_dir.glob("*.csv"):
        if csv_file.name in ["high.csv", "low.csv", "medium.csv", "unspecified.csv"]:
            print(f"  Processing {csv_file.name}...")

            # Determine which mode this CSV file represents
            mode_name = csv_file.name.replace(
                ".csv", ""
            )  # e.g., "high" from "high.csv"
            corresponding_mode_dir = f"mode_{mode_name}"  # e.g., "mode_high"

            # Read the original CSV
            df = pd.read_csv(csv_file)

            # Add the new score column
            df[f"score_{scorer.name}"] = 0.0

            # Get scores from the corresponding mode's detailed results
            if (
                corresponding_mode_dir in mode_detailed_results
                and mode_detailed_results[corresponding_mode_dir] is not None
            ):
                detailed_df = mode_detailed_results[corresponding_mode_dir]

                # Create a lookup dictionary for this specific mode
                mode_score_lookup = {}
                for _, row in detailed_df.iterrows():
                    sample_id = row["sample_id"]
                    score = row.get(f"score_{scorer.name}", 0.0)
                    mode_score_lookup[sample_id] = score

                # Update scores for each row using only this mode's scores
                for idx, row in df.iterrows():
                    sample_id = row["sample_id"]
                    if sample_id in mode_score_lookup:
                        df.at[idx, f"score_{scorer.name}"] = mode_score_lookup[
                            sample_id
                        ]

            # Replace score_accuracy with the new scores
            df["score_accuracy"] = df[f"score_{scorer.name}"]

            # Drop the temporary column
            df = df.drop(columns=[f"score_{scorer.name}"])

            # Fix MEAN scores by recalculating subset-wise means
            df = fix_mean_scores_in_csv(csv_file)

            # Save to output directory
            output_csv = output_dir / csv_file.name
            df.to_csv(output_csv, index=False)
            print(f"    Saved to {output_csv}")

    # Create regex log CSV
    if all_regex_logs:
        print("\nCreating regex log CSV...")
        combined_regex_log = pd.concat(all_regex_logs, ignore_index=True)
        regex_log_path = output_dir / "regex_log.csv"
        combined_regex_log.to_csv(regex_log_path, index=False)
        print(f"  Saved regex log to: {regex_log_path}")
        print(f"  Total regex log entries: {len(combined_regex_log)}")

    # Create diff results summary CSV
    if all_diff_results:
        print("\nCreating diff results summary CSV...")
        combined_diff_results = pd.concat(all_diff_results, ignore_index=True)

        # Create summary by subject/subset
        if "subset" in combined_diff_results.columns:
            subset_summary = (
                combined_diff_results.groupby("subset")
                .agg({"sample_id": "count", "difference": "sum"})
                .rename(
                    columns={
                        "sample_id": "num_differences",
                        "difference": "total_difference",
                    }
                )
            )
            subset_summary = subset_summary.reset_index()

            subset_summary_path = output_dir / "diff_results_summary.csv"
            subset_summary.to_csv(subset_summary_path, index=False)
            print(f"  Saved subset summary to: {subset_summary_path}")

        # Save full diff results
        diff_results_path = output_dir / "diff_results.csv"
        combined_diff_results.to_csv(diff_results_path, index=False)
        print(f"  Saved full diff results to: {diff_results_path}")
        print(f"  Total differences found: {len(combined_diff_results)}")

    print("\nRe-scoring complete!")
    print(f"Results saved to: {output_dir}")
    print("\nNote: Summary CSVs now contain:")
    print("  - New scores replacing the original score_accuracy column")
    print("  - MEAN values automatically recalculated for each subset")
    print("  - Detailed results with both original and new scores for comparison")
    print("  - All MEAN rows now show correct subset-wise averages")


if __name__ == "__main__":
    main()
