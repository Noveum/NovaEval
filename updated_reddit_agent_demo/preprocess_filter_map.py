#!/usr/bin/env python3
"""
Preprocessing script that filters and maps dataset spans.

This script replicates the preprocessing steps:
1. Filter out metadata and container spans
2. Add standardized fields for agent evaluation

Usage: python preprocess_filter_map.py <input_file>
Output: <input_file>_filtered_mapped.json
"""

import json
import sys
import os
from typing import Dict, Any, List


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


def add_agent_task_and_response(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add agent_task and agent_response fields to agent spans.
    """
    attributes = span.get('attributes', {})
    
    # Create agent_task from API information
    api_title = attributes.get('api.title', '')
    api_source = attributes.get('api.source', '')
    api_url = attributes.get('api.url', '')
    
    if api_title and api_source and api_url:
        agent_task = f"{api_title} {api_source} {api_url}"
        attributes['agent_task'] = agent_task
    else:
        attributes['agent_task'] = None
    
    # Create agent_response from events
    events = span.get('events', [])
    if events:
        # Convert events to JSON string for agent_response
        attributes['agent_response'] = json.dumps(events)
    else:
        attributes['agent_response'] = None
    
    span['attributes'] = attributes
    return span


def add_tool_response(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add tool_response field to tool/validation spans.
    """
    attributes = span.get('attributes', {})
    
    # For post_validation spans, create tool_response from validation data
    if span.get('name') == 'post_validation':
        validation_data = {}
        for key, value in attributes.items():
            if key.startswith('post_validation.'):
                validation_data[key] = value
        
        if validation_data:
            attributes['tool_response'] = json.dumps(validation_data)
        else:
            attributes['tool_response'] = None
    
    # For email_generation spans, create tool_response from email data
    elif span.get('name') == 'email_generation_and_sending':
        email_data = {}
        for key, value in attributes.items():
            if key.startswith('email_generation.'):
                email_data[key] = value
        
        if email_data:
            attributes['tool_response'] = json.dumps(email_data)
        else:
            attributes['tool_response'] = None
    
    span['attributes'] = attributes
    return span


def process_span(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single span by adding appropriate mapped fields.
    """
    span_name = span.get('name', '')
    
    # Add agent fields for agent spans
    if span_name in ['agent.query_generation', 'agent.comment_generation']:
        span = add_agent_task_and_response(span)
    
    # Add tool response for validation/email spans
    elif span_name in ['post_validation', 'email_generation_and_sending']:
        span = add_tool_response(span)
    
    # Tool call spans (tool:tavily_search_results_json) don't need changes
    
    return span


def preprocess_dataset(input_file: str) -> str:
    """
    Preprocess the dataset by filtering and mapping spans.
    
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
    
    # Map spans
    print("Mapping spans...")
    mapped_data = [process_span(span) for span in filtered_data]
    
    # Generate output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_filtered_mapped.json"
    
    # Write output file
    print(f"Writing {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(mapped_data, f, indent=2)
    
    print(f"Preprocessing complete! Output: {output_file}")
    return output_file


def main():
    if len(sys.argv) != 2:
        print("Usage: python preprocess_filter_map.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    try:
        output_file = preprocess_dataset(input_file)
        print(f"\nSuccess! Created {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


