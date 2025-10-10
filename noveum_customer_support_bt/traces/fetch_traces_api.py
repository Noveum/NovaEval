#!/usr/bin/env python3
"""
Script to fetch traces from Noveum API and save them to a traces directory.
Supports batch fetching with pagination for large numbers of traces.
"""

import os
import json
import requests
import argparse
import shutil
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Get API key and project from environment
api_key = os.getenv('NOVEUM_API_KEY')
project = 'noveum-ai-agent-rag-websearch'

if not api_key:
    print('Error: NOVEUM_API_KEY environment variable not found')
    exit(1)

# Common headers
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

def clean_and_create_traces_dir():
    """Clean existing traces directory and create a new one"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    traces_dir = os.path.join(script_dir, 'traces')
    
    if os.path.exists(traces_dir):
        print(f"Cleaning existing traces directory: {traces_dir}")
        shutil.rmtree(traces_dir)
    
    os.makedirs(traces_dir)
    print(f"Created traces directory: {traces_dir}")
    return traces_dir

def fetch_traces_batch(size: int, from_offset: int = 0) -> Dict[str, Any]:
    """Fetch a batch of traces from the API"""
    traces_url = 'https://api.noveum.ai/api/v1/traces'
    params = {
        'project': project,
        'size': size,
        'from': from_offset,
        'includeSpans': True
    }
    
    print(f"Fetching traces: size={size}, from={from_offset}")
    
    try:
        response = requests.get(traces_url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        print(f"Successfully fetched {len(data.get('traces', []))} traces")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f'Error fetching traces: {e}')
        if hasattr(e, 'response') and e.response is not None:
            print(f'Response status: {e.response.status_code}')
            print(f'Response text: {e.response.text}')
        return None

def save_traces_batch(traces_dir: str, batch_data: Dict[str, Any], batch_number: int):
    """Save a batch of traces to a JSON file"""
    filename = f"traces_batch_{batch_number:03d}.json"
    filepath = os.path.join(traces_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    print(f"Saved batch {batch_number} to: {filepath}")
    return filepath

def main():
    global project
    
    parser = argparse.ArgumentParser(description='Fetch traces from Noveum API and save to traces directory')
    parser.add_argument('count', type=int, help='Number of traces to fetch')
    parser.add_argument('--project', type=str, default=project, help=f'Project name (default: {project})')
    
    args = parser.parse_args()
    
    # Update project if specified
    project = args.project
    
    print(f"Fetching {args.count} traces for project: {project}")
    
    # Clean and create traces directory
    traces_dir = clean_and_create_traces_dir()
    
    # Calculate number of batches needed
    max_per_batch = 100
    num_batches = (args.count + max_per_batch - 1) // max_per_batch  # Ceiling division
    
    print(f"Will fetch in {num_batches} batch(es) of up to {max_per_batch} traces each")
    
    total_fetched = 0
    batch_number = 1
    
    for batch in range(num_batches):
        # Calculate size and from_offset for this batch
        remaining_traces = args.count - total_fetched
        current_size = min(max_per_batch, remaining_traces)
        current_from = batch * max_per_batch
        
        print(f"\n--- Batch {batch_number}/{num_batches} ---")
        
        # Fetch this batch
        batch_data = fetch_traces_batch(current_size, current_from)
        
        if batch_data is None:
            print(f"Failed to fetch batch {batch_number}")
            break
        
        # Save this batch
        save_traces_batch(traces_dir, batch_data, batch_number)
        
        # Update counters
        traces_in_batch = len(batch_data.get('traces', []))
        total_fetched += traces_in_batch
        
        print(f"Batch {batch_number} complete: {traces_in_batch} traces")
        print(f"Total fetched so far: {total_fetched}/{args.count}")
        
        # Check if we've fetched enough traces
        if total_fetched >= args.count:
            print(f"Reached target of {args.count} traces")
            break
        
        # Check if there are more traces available
        pagination = batch_data.get('pagination', {})
        has_more = pagination.get('has_more', False)
        
        if not has_more:
            print("No more traces available from API")
            break
        
        batch_number += 1
    
    print(f"\n=== Summary ===")
    print(f"Total traces fetched: {total_fetched}")
    print(f"Batches created: {batch_number}")
    print(f"Traces directory: {traces_dir}")
    
    # List all created files
    files = [f for f in os.listdir(traces_dir) if f.endswith('.json')]
    files.sort()
    print(f"Created files: {', '.join(files)}")

if __name__ == "__main__":
    main()
