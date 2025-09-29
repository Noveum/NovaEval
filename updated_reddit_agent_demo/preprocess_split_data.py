#!/usr/bin/env python3
"""
Script to split dataset_filtered_mapped.json by span name into separate files.

Usage:
    python preprocess_split_data.py [input_file] [output_dir]

If no arguments provided, uses:
    - input_file: dataset_filtered_mapped.json
    - output_dir: split_datasets
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def create_name_to_filename_mapping():
    """Create mapping from span names to desired filenames."""
    return {
        'agent.comment_generation': 'agent_comment_gen_dataset.json',
        'agent.query_generation': 'agent_query_gen_dataset.json',
        'email_generation_and_sending': 'email_gen_send_dataset.json',
        'post_validation': 'post_validation_dataset.json',
        'tool:tavily_search_results_json:tavily_search_results_json': 'tavily_search_results_dataset.json',
    }


def sanitize_filename(name):
    """Convert span name to a safe filename."""
    # Replace problematic characters with underscores
    safe_name = name.replace(':', '_').replace('/', '_').replace('\\', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    return f"{safe_name}_dataset.json"


def split_dataset_by_name(input_file, output_dir):
    """
    Split dataset by span name into separate files.
    
    Args:
        input_file (str): Path to input JSON file
        output_dir (str): Path to output directory
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} objects")
    
    # Group data by name
    grouped_data = defaultdict(list)
    for obj in data:
        name = obj.get('name', 'unknown')
        grouped_data[name].append(obj)
    
    print(f"Found {len(grouped_data)} unique span names")
    
    # Create name to filename mapping
    name_mapping = create_name_to_filename_mapping()
    
    # Write separate files for each name
    for name, objects in grouped_data.items():
        # Determine filename
        if name in name_mapping:
            filename = name_mapping[name]
            print(f"Using hardcoded mapping: {name} -> {filename}")
        else:
            filename = sanitize_filename(name)
            print(f"Using sanitized name: {name} -> {filename}")
        
        output_path = os.path.join(output_dir, filename)
        
        # Write the file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(objects, f, indent=2, ensure_ascii=False)
        
        print(f"  Wrote {len(objects)} objects to {output_path}")
    
    print(f"\nSplit complete! Created {len(grouped_data)} files in {output_dir}")


def main():
    """Main function to handle command line arguments."""
    # Check for help
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
        sys.exit(0)
    
    # Default values
    input_file = "dataset_filtered_mapped.json"
    output_dir = "split_datasets"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Split the dataset
    split_dataset_by_name(input_file, output_dir)


if __name__ == "__main__":
    main()
