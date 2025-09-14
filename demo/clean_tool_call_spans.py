#!/usr/bin/env python3
"""
Script to clean dataset.json by removing objects that have 'tool.name' in their attributes field.

This script reads the dataset.json file, filters out any objects that contain
the 'tool.name' key in their attributes dictionary, and saves the cleaned data
to dataset_tool_calls_removed.json.
"""

import json
import os
from typing import List, Dict, Any


def clean_dataset(input_file: str, output_file: str) -> None:
    """
    Clean the dataset by removing objects with 'tool.name' in attributes.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file
    """
    print(f"Reading dataset from: {input_file}")
    
    # Read the original dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original dataset contains {len(data)} objects")
    
    # Filter out objects that have 'tool.name' in their attributes
    cleaned_data = []
    removed_count = 0
    
    for obj in data:
        if isinstance(obj, dict) and 'attributes' in obj:
            attributes = obj['attributes']
            if isinstance(attributes, dict) and 'tool.name' in attributes:
                removed_count += 1
                print(f"Removing object with span_id: {obj.get('span_id', 'unknown')} (tool.name: {attributes.get('tool.name')})")
            else:
                cleaned_data.append(obj)
        else:
            # Keep objects that don't have attributes or have malformed structure
            cleaned_data.append(obj)
    
    print(f"Removed {removed_count} objects with 'tool.name' in attributes")
    print(f"Cleaned dataset contains {len(cleaned_data)} objects")
    
    # Save the cleaned dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned dataset saved to: {output_file}")


def main():
    """Main function to run the cleaning process."""
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'dataset.json')
    output_file = os.path.join(script_dir, 'dataset_tool_calls_removed.json')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    try:
        clean_dataset(input_file, output_file)
        print("Dataset cleaning completed successfully!")
    except Exception as e:
        print(f"Error during cleaning process: {e}")


if __name__ == "__main__":
    main()
