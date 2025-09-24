#!/usr/bin/env python3
"""
Manual mapping script to add new fields to span attributes based on span names.
"""

import json
import sys
from typing import Dict, Any, List


def process_comment_generation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process agent.comment_generation spans."""
    attributes = item.get('attributes', {})
    events = item.get('events', [])
    
    # Build agent_task: api_title + agent.operation + post titles from events
    api_title = attributes.get('api_title', '')
    agent_operation = attributes.get('agent.operation', '')
    
    post_titles = []
    for event in events:
        if 'attributes' in event and 'post_title' in event['attributes']:
            post_titles.append(f"post title is {event['attributes']['post_title']}")
    
    agent_task = f"{api_title} {agent_operation} {' '.join(post_titles)}"
    
    # Build agent_response: comments from events
    comments = []
    for event in events:
        if 'attributes' in event and 'comment' in event['attributes']:
            comments.append(f"comment is {event['attributes']['comment']}")
    
    agent_response = ' '.join(comments)
    
    # Add new fields to attributes
    attributes['agent_task'] = agent_task
    attributes['agent_response'] = agent_response
    
    return item


def process_query_generation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process agent.query_generation spans."""
    attributes = item.get('attributes', {})
    events = item.get('events', [])
    
    # Build agent_task: api.title + api.source + api.url
    api_title = attributes.get('api.title', '')
    api_source = attributes.get('api.source', '')
    api_url = attributes.get('api.url', '')
    
    agent_task = f"{api_title} {api_source} {api_url}"
    
    # Build agent_response: join query_generation.queries or use events if queries is empty
    queries = attributes.get('query_generation.queries', [])
    if isinstance(queries, list) and len(queries) > 0:
        agent_response = ' '.join(queries)
    else:
        # If queries is empty, use JSON string of events
        agent_response = json.dumps(events)
    
    # Add new fields to attributes
    attributes['agent_task'] = agent_task
    attributes['agent_response'] = agent_response
    
    return item


def process_email_generation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process email_generation_and_sending spans."""
    attributes = item.get('attributes', {})
    events = item.get('events', [])
    
    # Build tool_response: JSON string of events
    tool_response = json.dumps(events)
    
    # Add new field to attributes
    attributes['tool_response'] = tool_response
    
    return item


def process_post_validation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process post_validation spans."""
    attributes = item.get('attributes', {})
    events = item.get('events', [])
    
    # Build tool_response: JSON string of events
    tool_response = json.dumps(events)
    
    # Add new field to attributes
    attributes['tool_response'] = tool_response
    
    return item


def process_dataset(input_file: str, output_file: str) -> None:
    """Process the entire dataset and apply mappings."""
    print(f"Loading dataset from {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} items...")
    
    processed_count = 0
    span_counts = {
        'agent.comment_generation': 0,
        'agent.query_generation': 0,
        'email_generation_and_sending': 0,
        'post_validation': 0
    }
    
    for item in data:
        span_name = item.get('name', '')
        
        if span_name == 'agent.comment_generation':
            item = process_comment_generation(item)
            span_counts['agent.comment_generation'] += 1
        elif span_name == 'agent.query_generation':
            item = process_query_generation(item)
            span_counts['agent.query_generation'] += 1
        elif span_name == 'email_generation_and_sending':
            item = process_email_generation(item)
            span_counts['email_generation_and_sending'] += 1
        elif span_name == 'post_validation':
            item = process_post_validation(item)
            span_counts['post_validation'] += 1
        
        processed_count += 1
        if processed_count % 500 == 0:
            print(f"Processed {processed_count} items...")
    
    print(f"\nProcessing complete!")
    print("Span counts processed:")
    for span_name, count in span_counts.items():
        print(f"  {span_name}: {count}")
    
    print(f"\nSaving processed dataset to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Done!")


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python manual_mapping.py <input_file> <output_file>")
        print("Example: python manual_mapping.py dataset_filtered.json dataset_filtered_mapped.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        process_dataset(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
