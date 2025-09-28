#!/usr/bin/env python3
"""
Mapping script that adds standardized fields to dataset spans.

This script adds evaluation-ready fields to different span types:
- Agent spans get agent_task and agent_response fields
- Tool/validation spans get tool_response field

Usage: python preprocess_map.py <input_file>
Output: <input_file>_mapped.json
"""

import json
import sys
import os
from typing import Dict, Any


def add_agent_comment_generation_fields(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add agent_task and agent_response fields for agent.comment_generation spans.
    
    agent_task -> agent operation is + (agent.operation) +\n + api_title - (api_title) + (events[0].attributes.subreddit) + (events[0].attributes.post_title)
    agent_response -> (events[0].attributes.comment)
    """
    attributes = span.get('attributes', {})
    events = span.get('events', [])
    
    # Build agent_task
    agent_operation = attributes.get('agent.operation', '')
    api_title = attributes.get('api_title', '')
    
    agent_task_parts = [f"agent operation is {agent_operation}"]
    if api_title:
        agent_task_parts.append(f"api_title - {api_title}")
    
    # Add subreddit and post_title from events[0].attributes
    if events and len(events) > 0:
        event_attrs = events[0].get('attributes', {})
        subreddit = event_attrs.get('subreddit', '')
        post_title = event_attrs.get('post_title', '')
        
        if subreddit:
            agent_task_parts.append(f"({subreddit})")
        if post_title:
            agent_task_parts.append(f"({post_title})")
    
    agent_task = '\n'.join(agent_task_parts)
    attributes['agent_task'] = agent_task
    
    # Build agent_response from events[0].attributes.comment
    if events and len(events) > 0:
        event_attrs = events[0].get('attributes', {})
        comment = event_attrs.get('comment', '')
        attributes['agent_response'] = comment
    else:
        attributes['agent_response'] = ''
    
    span['attributes'] = attributes
    return span


def add_agent_query_generation_fields(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add agent_task and agent_response fields for agent.query_generation spans.
    
    agent_task -> api title - (api.title) + api description - (api.description) 
    agent_response -> (concatenate query_generation.queries, it has a list of strings)
    """
    attributes = span.get('attributes', {})
    
    # Build agent_task
    api_title = attributes.get('api.title', '')
    api_source = attributes.get('api.source', '')
    api_url = attributes.get('api.url', '')
    api_description = attributes.get('api.description', 'not available')  # Default to 'not available' if not found
    
    agent_task_parts = []
    if api_title:
        agent_task_parts.append(f"api title - {api_title}")
    if api_source:
        agent_task_parts.append(f"api source - {api_source}")
    if api_url:
        agent_task_parts.append(f"api url - {api_url}")
    agent_task_parts.append(f"api description - {api_description}")
    
    attributes['agent_task'] = '\n'.join(agent_task_parts)
    
    # Build agent_response from query_generation.queries
    queries = attributes.get('query_generation.queries', [])
    if isinstance(queries, list) and queries:
        # Concatenate all queries with newlines
        agent_response = '\n'.join(queries)
    elif isinstance(queries, list) and not queries:
        # If list is empty, use stringified version of events
        events = span.get('events', [])
        agent_response = json.dumps(events)
    else:
        agent_response = str(queries) if queries else ''
    
    attributes['agent_response'] = agent_response
    
    span['attributes'] = attributes
    return span


def add_email_generation_fields(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add tool_response field for email_generation_and_sending spans.
    
    tool_response -> (string value of events key's value, it is a list of jsons)
    """
    attributes = span.get('attributes', {})
    events = span.get('events', [])
    
    # Convert events list to JSON string
    if events:
        attributes['tool_response'] = json.dumps(events)
    else:
        attributes['tool_response'] = '[]'
    
    span['attributes'] = attributes
    return span


def add_post_validation_fields(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add tool_response field for post_validation spans.
    
    tool_response -> (string value of events key's value, it is a list of jsons)
    """
    attributes = span.get('attributes', {})
    events = span.get('events', [])
    
    # Convert events list to JSON string
    if events:
        attributes['tool_response'] = json.dumps(events)
    else:
        attributes['tool_response'] = '[]'
    
    span['attributes'] = attributes
    return span


def process_span(span: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single span by adding appropriate mapped fields.
    """
    span_name = span.get('name', '')
    
    # Process different span types
    if span_name == 'agent.comment_generation':
        span = add_agent_comment_generation_fields(span)
    elif span_name == 'agent.query_generation':
        span = add_agent_query_generation_fields(span)
    elif span_name == 'email_generation_and_sending':
        span = add_email_generation_fields(span)
    elif span_name == 'post_validation':
        span = add_post_validation_fields(span)
    # Tool call spans (tool:tavily_search_results_json) don't need changes
    
    return span


def map_dataset(input_file: str) -> str:
    """
    Map the dataset by adding standardized fields to spans.
    
    Args:
        input_file: Path to input JSON file
        
    Returns:
        Path to output file
    """
    # Read input file
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Input dataset: {len(data)} records")
    
    # Map spans
    print("Mapping spans...")
    mapped_data = [process_span(span) for span in data]
    
    # Generate output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_mapped.json"
    
    # Write output file
    print(f"Writing {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(mapped_data, f, indent=2)
    
    print(f"Mapping complete! Output: {output_file}")
    return output_file


def main():
    if len(sys.argv) != 2:
        print("Usage: python preprocess_map.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    try:
        output_file = map_dataset(input_file)
        print(f"\nSuccess! Created {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()