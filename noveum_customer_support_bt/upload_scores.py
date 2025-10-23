#!/usr/bin/env python3
"""
Script to upload scorer results to Noveum API.
Reads scores and reasonings from a CSV file and uploads them via API.
"""

import os
import json
import argparse
import csv
from typing import Dict, List, Optional
import requests
from dotenv import load_dotenv


def load_api_data(api_data_path: str) -> Dict[str, str]:
    """
    Load api_data.json and create a mapping from item_key to item_id.
    
    Args:
        api_data_path: Path to the api_data.json file
        
    Returns:
        Dictionary mapping item_key to item_id
    """
    print(f"Loading API data from {api_data_path}...")
    with open(api_data_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping from item_key to item_id
    key_to_id = {}
    items = data.get('items', [])
    
    for item in items:
        item_key = item.get('item_key')
        item_id = item.get('item_id')
        if item_key and item_id:
            key_to_id[item_key] = item_id
    
    print(f"Loaded {len(key_to_id)} item mappings")
    return key_to_id


def read_csv_data(
    csv_path: str,
    item_key_col: str,
    score_col: str,
    reasoning_col: str
) -> List[Dict]:
    """
    Read CSV file and extract relevant columns.
    
    Args:
        csv_path: Path to the CSV file
        item_key_col: Column name for item keys
        score_col: Column name for scores
        reasoning_col: Column name for reasonings
        
    Returns:
        List of dictionaries with item_key, score, and reasoning
    """
    print(f"Reading CSV from {csv_path}...")
    results = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Verify columns exist
        if item_key_col not in reader.fieldnames:
            raise ValueError(f"Column '{item_key_col}' not found in CSV. Available columns: {reader.fieldnames}")
        if score_col not in reader.fieldnames:
            raise ValueError(f"Column '{score_col}' not found in CSV. Available columns: {reader.fieldnames}")
        if reasoning_col not in reader.fieldnames:
            raise ValueError(f"Column '{reasoning_col}' not found in CSV. Available columns: {reader.fieldnames}")
        
        for row in reader:
            item_key = row[item_key_col]
            score = row[score_col]
            reasoning = row[reasoning_col]
            
            # Skip empty rows
            if not item_key or not score:
                continue
            
            results.append({
                'item_key': item_key,
                'score': float(score),
                'reasoning': reasoning
            })
    
    print(f"Read {len(results)} rows from CSV")
    return results


def create_batch_payload(
    csv_data: List[Dict],
    key_to_id: Dict[str, str],
    org_slug: str,
    project: str,
    environment: str,
    dataset_slug: str,
    dataset_version: str,
    scorer_id: str = "custom_scorer",
    scorer_version: str = "1.0.0"
) -> List[Dict]:
    """
    Create the batch payload for API submission.
    
    Args:
        csv_data: List of dictionaries with item_key, score, and reasoning
        key_to_id: Mapping from item_key to item_id
        org_slug: Organization slug
        project: Project name
        environment: Environment name
        dataset_slug: Dataset slug
        dataset_version: Dataset version
        scorer_id: Scorer ID (default: "custom_scorer")
        scorer_version: Scorer version (default: "1.0.0")
        
    Returns:
        List of result objects ready for API submission
    """
    results = []
    skipped = []
    
    for row in csv_data:
        item_key = row['item_key']
        
        # Find corresponding item_id
        item_id = key_to_id.get(item_key)
        
        if not item_id:
            skipped.append(item_key)
            continue
        
        result = {
            "organizationSlug": org_slug,
            "project": project,
            "environment": environment,
            "datasetSlug": dataset_slug,
            "datasetVersion": dataset_version,
            "itemId": item_id,
            "scorerId": scorer_id,
            "scorerVersion": scorer_version,
            "score": row['score'],
            "passed": row['score'] > 0.5,  # Default threshold, can be adjusted
            "metadata": {
                "details": row['reasoning']
            },
            "executionTimeMs": 0.0
        }
        
        results.append(result)
    
    if skipped:
        print(f"Warning: Skipped {len(skipped)} rows with missing item_id mappings")
        print(f"First few skipped keys: {skipped[:5]}")
    
    print(f"Created {len(results)} results for upload")
    return results


def upload_results(
    results: List[Dict],
    api_key: str,
    org_slug: str,
    batch_size: int = 100
) -> None:
    """
    Upload results to the API in batches.
    
    Args:
        results: List of result objects
        api_key: API key for authentication
        org_slug: Organization slug
        batch_size: Number of results per batch (default: 100)
    """
    api_url = f"https://beta.noveum.ai/api/v1/scorers/results/batch?organizationSlug={org_slug}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Split results into batches
    total = len(results)
    batches = [results[i:i + batch_size] for i in range(0, total, batch_size)]
    
    print(f"\nUploading {total} results in {len(batches)} batches...")
    
    for i, batch in enumerate(batches, 1):
        payload = {"results": batch}
        
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                print(f"✓ Batch {i}/{len(batches)} uploaded successfully ({len(batch)} results)")
            else:
                print(f"✗ Batch {i}/{len(batches)} failed: {response.status_code}")
                print(f"  Response: {response.text}")
        
        except Exception as e:
            print(f"✗ Batch {i}/{len(batches)} error: {str(e)}")
    
    print("\nUpload complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Upload scorer results from CSV to Noveum API"
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file containing scores and reasonings"
    )
    parser.add_argument(
        "--item-key-col",
        required=True,
        help="Column name for item keys"
    )
    parser.add_argument(
        "--score-col",
        required=True,
        help="Column name for scores"
    )
    parser.add_argument(
        "--reasoning-col",
        required=True,
        help="Column name for reasonings"
    )
    parser.add_argument(
        "--api-data",
        default="api_data.json",
        help="Path to api_data.json file (default: api_data.json)"
    )
    parser.add_argument(
        "--scorer-id",
        default="custom_scorer",
        help="Scorer ID (default: custom_scorer)"
    )
    parser.add_argument(
        "--scorer-version",
        default="1.0.0",
        help="Scorer version (default: 1.0.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of results per batch (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare data but don't upload to API"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    print("Loaded environment variables")
    
    # Get required environment variables
    required_vars = {
        'NOVEUM_PROJECT': os.getenv('NOVEUM_PROJECT'),
        'NOVEUM_ENVIRONMENT': os.getenv('NOVEUM_ENVIRONMENT'),
        'NOVEUM_API_KEY': os.getenv('NOVEUM_API_KEY'),
        'NOVEUM_ORG_SLUG': os.getenv('NOVEUM_ORG_SLUG'),
        'NOVEUM_DATASET_SLUG': os.getenv('NOVEUM_DATASET_SLUG'),
        'LATEST_VERSION': os.getenv('LATEST_VERSION')
    }
    
    # Check for missing variables
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set them in your .env file or environment:")
        for var in missing_vars:
            print(f"  {var}=<value>")
        return 1
    
    # Load API data
    key_to_id = load_api_data(args.api_data)
    
    # Read CSV data
    csv_data = read_csv_data(
        args.csv_file,
        args.item_key_col,
        args.score_col,
        args.reasoning_col
    )
    
    # Create batch payload
    results = create_batch_payload(
        csv_data=csv_data,
        key_to_id=key_to_id,
        org_slug=required_vars['NOVEUM_ORG_SLUG'],
        project=required_vars['NOVEUM_PROJECT'],
        environment=required_vars['NOVEUM_ENVIRONMENT'],
        dataset_slug=required_vars['NOVEUM_DATASET_SLUG'],
        dataset_version=required_vars['LATEST_VERSION'],
        scorer_id=args.scorer_id,
        scorer_version=args.scorer_version
    )
    
    if not results:
        print("Error: No valid results to upload")
        return 1
    
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        print(f"Would upload {len(results)} results")
        print("\nSample result:")
        print(json.dumps(results[0], indent=2))
        return 0
    
    # Upload results
    upload_results(
        results=results,
        api_key=required_vars['NOVEUM_API_KEY'],
        org_slug=required_vars['NOVEUM_ORG_SLUG'],
        batch_size=args.batch_size
    )
    
    return 0


if __name__ == "__main__":
    exit(main())

