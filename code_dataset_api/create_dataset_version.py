#!/usr/bin/env python3
"""
Script to create a new dataset version in Noveum API.
Creates a new version for the specified dataset.
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
latest_version = os.getenv('LATEST_VERSION')
beta_env = os.getenv('BETA', 'false').lower() == 'true'

def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = {
        'NOVEUM_API_KEY': api_key,
        'NOVEUM_ORG_SLUG': org_slug,
        'NOVEUM_DATASET_SLUG': dataset_slug,
        'LATEST_VERSION': latest_version
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment")
        return False
    
    return True

def create_dataset_version(version: str) -> Optional[Dict[str, Any]]:
    """Create a new dataset version in Noveum API"""
    
    # Construct API URL based on BETA environment variable
    if beta_env:
        api_url = f"https://noveum.ai/api/v1/datasets/{dataset_slug}/versions?organizationSlug={org_slug}"
    else:
        api_url = f"https://noveum.ai/api/v1/organizations/{org_slug}/datasets/{dataset_slug}/versions"
    
    # Prepare headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Cookie': f'apiKeyCookie={api_key}'
    }
    
    # Prepare request data
    request_data = {
        "version": version
    }
    
    print(f"Creating dataset version at: {api_url}")
    print(f"Organization: {org_slug}")
    print(f"Dataset: {dataset_slug}")
    print(f"Version: {version}")
    
    try:
        response = requests.post(api_url, headers=headers, json=request_data, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print("Successfully created dataset version")
        print(f"Response status: {response.status_code}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error creating dataset version: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Create a new dataset version in Noveum API')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty print the JSON response')
    parser.add_argument('--output', type=str, default="dataset_version_response.json",
                       help='Output file to save the JSON response (default: dataset_version_response.json)')
    
    args = parser.parse_args()
    
    # Validate environment variables
    if not validate_environment():
        return 1
    
    # Create dataset version
    data = create_dataset_version(version=latest_version)
    
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
