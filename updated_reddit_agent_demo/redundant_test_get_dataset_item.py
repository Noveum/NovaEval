#!/usr/bin/env python3
"""
Script to get a specific dataset item from Noveum API by item key.
Fetches a single item using its unique item key.
"""

import os
import json
import requests
import argparse
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# Get API credentials from environment
api_key = os.getenv('NOVEUM_API_KEY')
org_slug = os.getenv('NOVEUM_ORG_SLUG')
dataset_slug = os.getenv('NOVEUM_DATASET_SLUG')

# Default item key
default_item_key = "eb4e2099-e0d7-4bce-80c8-66a37e75c4f5"

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

def get_dataset_item(item_key: str) -> Optional[Dict[str, Any]]:
    """Get a specific dataset item from Noveum API by item key"""
    
    # Construct API URL
    api_url = f"https://noveum.ai/api/v1/organizations/{org_slug}/datasets/{dataset_slug}/items/{item_key}"
    
    # Prepare headers
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    print(f"Fetching dataset item from: {api_url}")
    print(f"Organization: {org_slug}")
    print(f"Dataset: {dataset_slug}")
    print(f"Item Key: {item_key}")
    
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print("Successfully fetched dataset item")
        print(f"Response status: {response.status_code}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching dataset item: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Get a specific dataset item from Noveum API by item key')
    parser.add_argument('--item-key', type=str, default=default_item_key,
                       help=f'Item key to fetch (default: {default_item_key})')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty print the JSON response')
    
    args = parser.parse_args()
    
    # Validate environment variables
    if not validate_environment():
        return 1
    
    # Fetch dataset item
    data = get_dataset_item(args.item_key)
    
    if data is None:
        return 1
    
    # Print the response
    if args.pretty:
        print("\nResponse data:")
        print(json.dumps(data, indent=2))
    else:
        print(f"\nResponse data: {json.dumps(data)}")
    
    return 0

if __name__ == "__main__":
    exit(main())
