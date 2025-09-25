#!/usr/bin/env python3
"""
Script to upload dataset items to Noveum API.
Reads a JSON file containing a list of dataset items and uploads them via POST request.
"""

import os
import json
import requests
import argparse
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Default dataset JSON path
dataset_json = 'processed_agent_dataset.json'

# Get API credentials from environment
api_key = os.getenv('NOVEUM_API_KEY')
org_slug = os.getenv('NOVEUM_ORG_SLUG')
dataset_slug = os.getenv('NOVEUM_DATASET_SLUG')

def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = {
        'NOVEUM_API_KEY': api_key,
        'NOVEUM_ORG_SLUG': org_slug,
        'NOVEUM_DATASET_SLUG': dataset_slug
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment")
        return False
    
    return True

def load_dataset_items(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset items from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Error: JSON file should contain a list of objects, got {type(data)}")
            return []
        
        print(f"Loaded {len(data)} items from {file_path}")
        return data
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {file_path}: {e}")
        return []
    except (OSError, IOError) as e:
        print(f"Error loading dataset: {e}")
        return []

def upload_dataset_items(items: List[Dict[str, Any]], version: str, item_type: str = "conversation") -> bool:
    """Upload dataset items to Noveum API"""
    if not items:
        print("No items to upload")
        return False
    
    # Schema keys from schema.tsx - these will be surfaced at the same level as content
    schema_keys = {
        "item_id", "dataset_id", "item_key", "item_hash", "organization_id", 
        "organization_slug", "dataset_slug", "item_version", "deleted_at_version", 
        "deleted_at_date", "item_type", "schema_version", "source_trace_id", 
        "source_span_id", "content", "metadata", "agent_name", "agent_role", 
        "agent_task", "agent_response", "system_prompt", "user_id", "session_id", 
        "turn_id", "ground_truth", "expected_tool_call", "tools_available", 
        "tool_calls", "tool_call_results", "parameters_passed", "retrieval_query", 
        "retrieved_context", "exit_status", "agent_exit", "trace_data", 
        "conversation_id", "speaker", "message", "conversation_context", 
        "input_text", "output_text", "expected_output", "evaluation_context", 
        "criteria", "quality_score", "validation_status", "validation_errors", 
        "tags", "custom_attributes", "created_at", "updated_at"
    }
    
    # Transform items to the required format
    transformed_items = []
    for item in items:
        # Create a copy of the item to avoid modifying the original
        item['item_type'] = item_type
        item_copy = item.copy()
        
        # Start with base structure
        transformed_item = {
            "item_key": item.get("turn_id", ""),
            "item_type": item_type,  # Use the provided item_type
            "metadata": {}  # Empty metadata as specified
        }
        
        # Surface schema keys at the same level as content
        for key in schema_keys:
            if key in item_copy:
                value = item_copy.pop(key)
                # Handle special cases for required fields
                if key == "item_type" and (not value or value == ""):
                    # Keep our default value if the item's value is empty
                    continue
                elif key == "metadata":
                    # Ensure metadata is always an object
                    if isinstance(value, dict):
                        transformed_item[key] = value
                    elif isinstance(value, str) and value.strip():
                        try:
                            transformed_item[key] = json.loads(value)
                        except json.JSONDecodeError:
                            transformed_item[key] = {}
                    else:
                        transformed_item[key] = {}
                elif key == "content":
                    # Handle content field - always ensure it's an object
                    if isinstance(value, str) and value.strip():
                        try:
                            transformed_item[key] = json.loads(value)
                        except json.JSONDecodeError:
                            transformed_item[key] = {}
                    elif isinstance(value, dict):
                        transformed_item[key] = value
                    else:
                        transformed_item[key] = {}
                else:
                    transformed_item[key] = value
        
        # Always put the entire original item in content field
        transformed_item["content"] = item
        transformed_items.append(transformed_item)
    
    # Construct API URL
    api_url = f"https://noveum.ai/api/v1/organizations/{org_slug}/datasets/{dataset_slug}/items"
    
    # Prepare headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Cookie': f'apiKeyCookie={api_key}'
    }
    
    # Prepare request data
    request_data = {
        "version": version,
        "items": transformed_items
    }
    
    print(f"Uploading {len(items)} items to: {api_url}")
    print(f"Organization: {org_slug}")
    print(f"Dataset: {dataset_slug}")
    print(f"Version: {version}")
    
    try:
        response = requests.post(api_url, headers=headers, json=request_data, timeout=30)
        response.raise_for_status()
        
        print(f"Successfully uploaded {len(items)} items")
        print(f"Response status: {response.status_code}")
        
        # Print response content if available
        try:
            response_data = response.json()
            print(f"Response data: {json.dumps(response_data, indent=2)}")
        except json.JSONDecodeError:
            print(f"Response text: {response.text}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error uploading dataset items: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return False

def main():
    # Generate default version as simple semantic version
    default_version = "1.0.0"
    default_item_type = "conversation"
    
    parser = argparse.ArgumentParser(description='Upload dataset items to Noveum API')
    parser.add_argument('--dataset-json', type=str, default=dataset_json,
                       help=f'Path to JSON file containing dataset items (default: {dataset_json})')
    parser.add_argument('--version', type=str, default=default_version,
                       help=f'Version string for the dataset (default: {default_version})')
    parser.add_argument('--item-type', type=str, default=default_item_type,
                       help=f'Item type for the dataset items (default: {default_item_type})')
    
    args = parser.parse_args()
    
    # Validate environment variables
    if not validate_environment():
        return 1
    
    # Load dataset items
    items = load_dataset_items(args.dataset_json)
    if not items:
        return 1
    
    # Upload items
    success = upload_dataset_items(items, args.version, args.item_type)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
