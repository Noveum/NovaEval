#!/usr/bin/env python3
"""
Script to clean GPT-OSS evaluation data by removing score_multi_pattern_accuracy
columns from CSV files and key-value pairs from JSON files.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Union

import pandas as pd


def clean_csv_file(input_path: Path, output_path: Path) -> None:
    """Remove score_multi_pattern_accuracy column from CSV file."""
    try:
        # Read CSV file
        df = pd.read_csv(input_path)

        # Remove the column if it exists
        if "score_multi_pattern_accuracy" in df.columns:
            df = df.drop(columns=["score_multi_pattern_accuracy"])
            print(
                f"  ✓ Removed score_multi_pattern_accuracy column from {input_path.name}"
            )
        else:
            print(
                f"  - score_multi_pattern_accuracy column not found in {input_path.name}"
            )

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save cleaned CSV
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved cleaned CSV to {output_path}")

    except Exception as e:
        print(f"  ✗ Error processing {input_path}: {e}")


def clean_json_file(input_path: Path, output_path: Path) -> None:
    """Remove score_multi_pattern_accuracy key-value pairs from JSON file."""
    try:
        # Read JSON file
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        # Function to recursively remove the key
        def remove_key_recursive(obj: Union[dict, list, Any]) -> Union[dict, list, Any]:
            if isinstance(obj, dict):
                # Remove the key if it exists
                if "score_multi_pattern_accuracy" in obj:
                    del obj["score_multi_pattern_accuracy"]

                # Recursively process all values
                for key, value in obj.items():
                    obj[key] = remove_key_recursive(value)
                return obj
            elif isinstance(obj, list):
                # Recursively process all items in the list
                return [remove_key_recursive(item) for item in obj]
            else:
                # Return primitive values as-is
                return obj

        # Clean the data
        cleaned_data = remove_key_recursive(data)

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save cleaned JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Removed score_multi_pattern_accuracy keys from {input_path.name}")
        print(f"  ✓ Saved cleaned JSON to {output_path}")

    except Exception as e:
        print(f"  ✗ Error processing {input_path}: {e}")


def copy_other_files(input_path: Path, output_path: Path) -> None:
    """Copy non-CSV/JSON files as-is."""
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(input_path, output_path)
        print(f"  ✓ Copied {input_path.name}")

    except Exception as e:
        print(f"  ✗ Error copying {input_path}: {e}")


def process_directory(input_dir: Path, output_dir: Path) -> None:
    """Process all files in a directory recursively."""
    print(f"\nProcessing directory: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all files and subdirectories
    for item in input_dir.iterdir():
        if item.is_file():
            # Determine output path
            relative_path = item.relative_to(input_dir)
            output_path = output_dir / relative_path

            # Process based on file type
            if item.suffix.lower() == ".csv":
                clean_csv_file(item, output_path)
            elif item.suffix.lower() == ".json":
                clean_json_file(item, output_path)
            else:
                # Copy other files as-is
                copy_other_files(item, output_path)

        elif item.is_dir():
            # Recursively process subdirectories
            relative_path = item.relative_to(input_dir)
            output_subdir = output_dir / relative_path
            process_directory(item, output_subdir)


def main():
    """Main function to clean the GPT-OSS data."""
    # Define input and output directories
    input_dir = Path("examples/ollama_run_20250812_180842")
    output_dir = Path("examples/gpt_oss_cleaned_ollama_run_20250812_180842")

    print("GPT-OSS Data Cleaning Script")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Check if input directory exists
    if not input_dir.exists():
        print(f"✗ Error: Input directory {input_dir} does not exist!")
        return

    # Remove output directory if it exists
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Process the directory
    process_directory(input_dir, output_dir)

    print("\n" + "=" * 50)
    print("Cleaning completed!")
    print(f"Cleaned data saved to: {output_dir}")


if __name__ == "__main__":
    main()
