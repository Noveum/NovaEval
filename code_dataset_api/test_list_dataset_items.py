#!/usr/bin/env python3
"""
Script to list dataset items from Noveum API.
Fetches items from a specific dataset with pagination support.
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

def list_dataset_items(version: str = "", limit: int = 1, offset: int = 0, item_type: str = "any") -> Optional[Dict[str, Any]]:
    """List dataset items from Noveum API"""
    
    # Construct API URL
    api_url = f"https://noveum.ai/api/v1/organizations/{org_slug}/datasets/{dataset_slug}/items"
    
    # Prepare headers
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    # Prepare query parameters
    params = {
        'version': version,
        'limit': limit,
        'offset': offset,
        'item_type': item_type
    }
    
    print(f"Fetching dataset items from: {api_url}")
    print(f"Organization: {org_slug}")
    print(f"Dataset: {dataset_slug}")
    print(f"Parameters: version='{version}', limit={limit}, offset={offset}, item_type='{item_type}'")
    
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print("Successfully fetched dataset items")
        print(f"Response status: {response.status_code}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching dataset items: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description='List dataset items from Noveum API')
    parser.add_argument('--version', type=str, default="",
                       help='Version to filter by (default: empty string for all versions)')
    parser.add_argument('--limit', type=int, default=50,
                       help='Number of items to fetch (default: 50)')
    parser.add_argument('--offset', type=int, default=0,
                       help='Number of items to skip (default: 0)')
    parser.add_argument('--item-type', type=str, default="",
                       help='Item type to filter by (default: empty string for all types)')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty print the JSON response')
    parser.add_argument('--output', type=str, default="dataset_items_response.json",
                       help='Output file to save the JSON response (default: dataset_items_response.json)')
    
    args = parser.parse_args()
    
    # Validate environment variables
    if not validate_environment():
        return 1
    
    # Fetch dataset items
    data = list_dataset_items(
        version=args.version,
        limit=args.limit,
        offset=args.offset,
        item_type=args.item_type
    )
    
    if data is None:
        return 1
    
    # Save response to file
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\nResponse saved to: {args.output}")
    except (OSError, IOError) as e:
        print(f"Error saving response to file: {e}")
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
