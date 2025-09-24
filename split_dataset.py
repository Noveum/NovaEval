#!/usr/bin/env python3
"""
Script to split dataset_filtered_mapped.json into separate files based on span names
"""

import json
import os
from collections import defaultdict

def split_dataset_by_name(input_file, output_dir):
    """Split the dataset by span names into separate files."""
    
    # Mapping of span names to output filenames
    name_to_file = {
        'tool:tavily_search_results_json:tavily_search_results_json': 'tavily_search_results_dataset.json',
        'agent.query_generation': 'agent_query_gen_dataset.json',
        'post_validation': 'post_validation_dataset.json',
        'agent.comment_generation': 'agent_comment_gen_dataset.json',
        'email_generation_and_sending': 'email_gen_send_dataset.json'
    }
    
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Group spans by name
        spans_by_name = defaultdict(list)
        for span in data:
            if 'name' in span:
                spans_by_name[span['name']].append(span)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write separate files for each name
        for name, filename in name_to_file.items():
            output_file = os.path.join(output_dir, filename)
            
            if name in spans_by_name:
                spans = spans_by_name[name]
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(spans, f, indent=2, ensure_ascii=False)
                print(f"Created {filename}: {len(spans)} spans")
            else:
                print(f"Warning: No spans found for name '{name}'")
        
        # Summary
        print(f"\nSummary:")
        print(f"Total spans processed: {len(data)}")
        print(f"Output directory: {output_dir}")
        
        # Show counts for each file
        print(f"\nFile breakdown:")
        for name, filename in name_to_file.items():
            if name in spans_by_name:
                count = len(spans_by_name[name])
                print(f"  {filename}: {count} spans")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{input_file}': {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_file = "/home/shivam/Desktop/noveum/NovaEval/reddit_agent_demo/dataset_filtered_mapped.json"
    output_dir = "/home/shivam/Desktop/noveum/NovaEval/reddit_agent_demo/split_datasets"
    
    split_dataset_by_name(input_file, output_dir)
