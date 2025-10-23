#!/usr/bin/env python3
"""
Script to fetch dataset items from Noveum API and create api_data.json file.
This file is needed for the upload_scores.py script.
"""

import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# Get API credentials from environment
api_key = os.getenv('NOVEUM_API_KEY')
org_slug = os.getenv('NOVEUM_ORG_SLUG')
dataset_slug = os.getenv('NOVEUM_DATASET_SLUG')
latest_version = os.getenv('LATEST_VERSION')
beta_env = os.getenv('BETA', 'false').lower() == 'true'

def fetch_dataset_items() -> Optional[Dict[str, Any]]:
    """Fetch dataset items from Noveum API"""
    
    # Construct API URL based on BETA environment variable
    if beta_env:
        api_url = f"https://noveum.ai/api/v1/datasets/{dataset_slug}/items?organizationSlug={org_slug}&version={latest_version}"
    else:
        api_url = f"https://noveum.ai/api/v1/organizations/{org_slug}/datasets/{dataset_slug}/items?version={latest_version}"
    
    # Prepare headers
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Cookie': f'apiKeyCookie={api_key}'
    }
    
    print(f"Fetching dataset items from: {api_url}")
    print(f"Organization: {org_slug}")
    print(f"Dataset: {dataset_slug}")
    print(f"Version: {latest_version}")
    
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"Successfully fetched {len(data.get('items', []))} items")
        print(f"Response status: {response.status_code}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching dataset items: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def main():
    # Validate environment variables
    required_vars = {
        'NOVEUM_API_KEY': api_key,
        'NOVEUM_ORG_SLUG': org_slug,
        'NOVEUM_DATASET_SLUG': dataset_slug,
        'LATEST_VERSION': latest_version
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        return 1
    
    # Fetch dataset items
    data = fetch_dataset_items()
    
    if data is None:
        return 1
    
    # Save response to api_data.json
    try:
        with open('api_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\nDataset items saved to: api_data.json")
        print(f"Total items: {len(data.get('items', []))}")
    except (OSError, IOError) as e:
        print(f"Error saving dataset items: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
