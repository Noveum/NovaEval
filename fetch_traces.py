#!/usr/bin/env python3
"""
Script to fetch traces and status from Noveum API
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key and project from environment
api_key = os.getenv('NOVEUM_API_KEY')
project = os.getenv('NOVEUM_PROJECT', 'chat-bot')
trace_id = os.getenv('NOVEUM_TRACE_ID', 'trace_123')

if not api_key:
    print('Error: NOVEUM_API_KEY environment variable not found')
    exit(1)

# Common headers
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

def make_request(url, description):
    """Make a request and print the response"""
    print(f"\n{description}")
    print("=" * 50)
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        print(f"Status Code: {response.status_code}")
        print("Response:")
        data = response.json()
        print(data)
        
    except requests.exceptions.RequestException as e:
        print(f'Error making request: {e}')
        if hasattr(e, 'response') and e.response is not None:
            print(f'Response status: {e.response.status_code}')
            print(f'Response text: {e.response.text}')

# 1. Get specific trace
trace_url = f'https://api.noveum.ai/api/v1/traces/{trace_id}'
make_request(trace_url, f"Fetching trace: {trace_id}")

# 2. Get API status
status_url = 'https://api.noveum.ai/api/v1/status'
make_request(status_url, "Fetching API status")

# 3. Get traces list (original functionality)
traces_url = 'https://api.noveum.ai/api/v1/traces'
params = {
    'project': project,
    'limit': 2
}

print(f"\nFetching traces list for project: {project}")
print("=" * 50)
print(f"URL: {traces_url}")
print(f"Parameters: {params}")

try:
    response = requests.get(traces_url, headers=headers, params=params)
    response.raise_for_status()
    
    print(f"Status Code: {response.status_code}")
    print("Response:")
    data = response.json()
    print(data)
    
    # Write the response to a file
    output_file = 'traces_response.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResponse written to: {output_file}")
    
    # Read and validate the JSON file
    print(f"\nReading and validating JSON from: {output_file}")
    print("=" * 50)
    
    try:
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        
        print("✅ Valid JSON file!")
        print(f"Data type: {type(loaded_data)}")
        
        if isinstance(loaded_data, dict):
            print(f"Top-level keys: {list(loaded_data.keys())}")
            print(f"Number of top-level keys: {len(loaded_data.keys())}")
            
            # Show some details about each key
            for key, value in loaded_data.items():
                print(f"  - '{key}': {type(value)}")
                if isinstance(value, (list, dict)):
                    print(f"    Length/size: {len(value)}")
        elif isinstance(loaded_data, list):
            print(f"JSON is a list with {len(loaded_data)} items")
            if loaded_data:
                print(f"First item type: {type(loaded_data[0])}")
                if isinstance(loaded_data[0], dict):
                    print(f"First item keys: {list(loaded_data[0].keys())}")
        else:
            print(f"JSON contains: {loaded_data}")
            
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in file: {e}")
    except Exception as e:
        print(f"❌ Error reading/validating JSON: {e}")
    
except requests.exceptions.RequestException as e:
    print(f'Error making request: {e}')
    if hasattr(e, 'response') and e.response is not None:
        print(f'Response status: {e.response.status_code}')
        print(f'Response text: {e.response.text}')
