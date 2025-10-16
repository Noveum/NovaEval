#!/usr/bin/env python3
"""
Filtering script that removes metadata and container spans from dataset.

This script filters out spans that are not needed for agent evaluation:
- api_selection spans
- reddit_agent_run_1 and reddit_agent_run_2 spans

Usage: python preprocess_filter.py <input_file>
Output: <input_file>_filtered.json
"""

import json
import sys
import os
from typing import Dict, Any


def should_keep_span(span: Dict[str, Any]) -> bool:
    """
    Determine if a span should be kept based on its name.
    Filters out metadata and container spans.
    """
    span_name = span.get('name', '')
    
    # Remove these span types
    excluded_spans = {
        'api_selection',
        'reddit_agent_run_1', 
        'reddit_agent_run_2'
    }
    
    return span_name not in excluded_spans


def convert_tool_output_to_string(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert tool.output.output from JSON array to concatenated string format.
    """
    attributes = span.get('attributes', {})
    tool_output = attributes.get('tool.output.output')
    
    if tool_output and isinstance(tool_output, list):
        # Convert list of objects to concatenated string format
        output_strings = []
        for item in tool_output:
            if isinstance(item, dict):
                # Convert dict to string format like "{'url': '...', 'content': '...'}"
                item_str = str(item).replace("'", "'")  # Ensure single quotes
                output_strings.append(item_str)
            else:
                output_strings.append(str(item))
        
        # Join all items with space
        attributes['tool.output.output'] = ' '.join(output_strings)
        span['attributes'] = attributes
    
    return span


def filter_dataset(input_file: str) -> str:
    """
    Filter the dataset by removing unwanted spans.
    
    Args:
        input_file: Path to input JSON file
        
    Returns:
        Path to output file
    """
    # Read input file
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original dataset: {len(data)} records")
    
    # Filter spans
    print("Filtering spans...")
    filtered_data = [span for span in data if should_keep_span(span)]
    print(f"After filtering: {len(filtered_data)} records")
    
    # Convert tool.output.output from JSON array to string format
    print("Converting tool output format...")
    filtered_data = [convert_tool_output_to_string(span) for span in filtered_data]
    
    # Generate output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_filtered.json"
    
    # Write output file
    print(f"Writing {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Filtering complete! Output: {output_file}")
    return output_file


def main():
    if len(sys.argv) != 2:
        print("Usage: python preprocess_filter.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    try:
        output_file = filter_dataset(input_file)
        print(f"\nSuccess! Created {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
