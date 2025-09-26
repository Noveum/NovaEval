#!/usr/bin/env python3
"""
Script to download the BAAI/JudgeLM-100K dataset from Hugging Face.

This script downloads the BAAI/JudgeLM-100K dataset to a specified directory
using the Hugging Face datasets library.
"""

import os
import argparse
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
except ImportError:
    print("Please install required packages: pip install datasets huggingface-hub")
    exit(1)


def download_geval_dataset(
    output_dir: str = "/mnt/drive2/",
    dataset_name: str = "BAAI/JudgeLM-100K",
    cache_dir: Optional[str] = None,
    use_auth_token: Optional[str] = None
) -> None:
    """
    Download the BAAI/JudgeLM-100K dataset from Hugging Face.

    Args:
        output_dir: Directory to save the dataset (default: /mnt/drive2/)
        dataset_name: Name of the HuggingFace dataset (default: BAAI/JudgeLM-100K)
        cache_dir: Optional cache directory for Hugging Face datasets
        use_auth_token: Optional Hugging Face authentication token
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {dataset_name} to {output_dir}")
    print("This may take a while depending on dataset size and internet connection...")

    try:
        # Method 1: Using datasets library to load and save
        print("\n=== Method 1: Loading dataset using datasets library ===")
        
        dataset = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token
        )
        
        # Save the dataset in different formats
        dataset_output_dir = output_path / "JudgeLM-100K"
        dataset_output_dir.mkdir(exist_ok=True)
        
        print(f"Dataset info: {dataset}")
        
        # Save as JSON
        json_dir = dataset_output_dir / "json"
        json_dir.mkdir(exist_ok=True)
        
        for split_name, split_data in dataset.items():
            json_file = json_dir / f"{split_name}.json"
            print(f"Saving {split_name} split to {json_file}")
            split_data.to_json(str(json_file))
        
        # Save as Parquet (more efficient for large datasets)
        parquet_dir = dataset_output_dir / "parquet"
        parquet_dir.mkdir(exist_ok=True)
        
        for split_name, split_data in dataset.items():
            parquet_file = parquet_dir / f"{split_name}.parquet"
            print(f"Saving {split_name} split to {parquet_file}")
            split_data.to_parquet(str(parquet_file))
        
        print(f"✅ Dataset successfully downloaded and saved to {dataset_output_dir}")
        
        # Print dataset statistics
        print("\n=== Dataset Statistics ===")
        total_samples = 0
        for split_name, split_data in dataset.items():
            num_samples = len(split_data)
            total_samples += num_samples
            print(f"{split_name}: {num_samples} samples")
            
            # Print column names
            if hasattr(split_data, 'column_names'):
                print(f"  Columns: {split_data.column_names}")
        
        print(f"Total samples: {total_samples}")
        
        # Optional: Method 2 - Raw file download using snapshot_download
        print(f"\n=== Method 2: Raw repository download ===")
        raw_output_dir = dataset_output_dir / "raw"
        
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=str(raw_output_dir),
            use_auth_token=use_auth_token
        )
        
        print(f"✅ Raw repository files downloaded to {raw_output_dir}")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print(f"Make sure you have internet connection and sufficient disk space in {output_dir}")
        
        # Check if it's an authentication issue
        if "authentication" in str(e).lower() or "token" in str(e).lower():
            print("\n💡 If this is a private dataset, you may need to:")
            print("1. Set up Hugging Face authentication: huggingface-cli login")
            print("2. Or pass --use-auth-token with your HF token")
        
        raise


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download BAAI/JudgeLM-100K dataset from Hugging Face"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/mnt/drive2/",
        help="Output directory to save the dataset (default: /mnt/drive2/)"
    )
    
    parser.add_argument(
        "--dataset-name",
        default="BAAI/JudgeLM-100K",
        help="Name of the HuggingFace dataset (default: BAAI/JudgeLM-100K)"
    )
    
    parser.add_argument(
        "--cache-dir",
        help="Cache directory for Hugging Face datasets"
    )
    
    parser.add_argument(
        "--use-auth-token",
        help="Hugging Face authentication token"
    )
    
    args = parser.parse_args()
    
    # Check if output directory is writable
    output_path = Path(args.output_dir)
    if output_path.exists() and not os.access(output_path, os.W_OK):
        print(f"❌ Error: Output directory {args.output_dir} is not writable")
        print("Please check permissions or use a different directory")
        exit(1)
    
    print("=== BAAI/JudgeLM-100K Dataset Downloader ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output directory: {args.output_dir}")
    if args.cache_dir:
        print(f"Cache directory: {args.cache_dir}")
    
    try:
        download_geval_dataset(
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            cache_dir=args.cache_dir,
            use_auth_token=args.use_auth_token
        )
        print("\n🎉 Download completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Download failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
