import os
from datetime import datetime
import pandas as pd
import boto3
from typing import Optional, Union
from pathlib import Path

def save_dataset(
    df: pd.DataFrame, 
    prefix: str = "all-tones",
    data_dir: str = "~/data",
    upload_to_s3: bool = False,
    bucket_name: Optional[str] = "",
    s3_prefix: str = "eval/tones",
    append: bool = True  # New parameter to control append behavior
) -> str:
    """
    Save a dataset with timestamp and optionally upload to S3
    
    Args:
        df: DataFrame to save
        prefix: Prefix for the filename (default: "all-tones")
        data_dir: Local directory to save data (default: "~/data")
        upload_to_s3: Whether to upload to S3 (default: False)
        bucket_name: S3 bucket name
        s3_prefix: Prefix for S3 key (default: "eval/tones")
        append: Whether to append to existing file if found (default: True)
    
    Returns:
        str: Path to the saved file
    """
    # Expand home directory and create if needed
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Try to find existing file if append is True
    existing_df = None
    if append:
        try:
            existing_df = load_latest_dataset(data_dir, prefix=prefix)
            print(f"Found existing dataset with {len(existing_df)} rows")
            # Combine with new data
            df = pd.concat([existing_df, df], ignore_index=True)
            print(f"Combined dataset has {len(df)} rows")
        except FileNotFoundError:
            print("No existing dataset found, creating new file")
    
    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}-{timestamp}.csv"
    filepath = os.path.join(data_dir, filename)
    
    # Save locally
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")
    
    # Upload to S3 if requested
    if upload_to_s3:
        if not bucket_name:
            raise ValueError("bucket_name must be provided when upload_to_s3 is True")
        
        s3 = boto3.client('s3')
        s3_key = f"{s3_prefix}/{filename}"
        s3.upload_file(Filename=filepath, Bucket=bucket_name, Key=s3_key)
        print(f"Uploaded to s3://{bucket_name}/{s3_key}")
    
    return filepath

def load_latest_dataset(
    data_dir: str = "~/data",
    prefix: Optional[str] = "all-tones",
    n_latest: int = 1
) -> Union[pd.DataFrame, list[pd.DataFrame]]:
    """
    Load the latest dataset(s) from the data directory
    
    Args:
        data_dir: Directory containing the data files (default: "~/data")
        prefix: Optional prefix to filter files (default: None)
        n_latest: Number of latest files to load (default: 1)
    
    Returns:
        Union[pd.DataFrame, list[pd.DataFrame]]: Single DataFrame if n_latest=1, 
        otherwise list of DataFrames sorted by timestamp (newest first)
    
    Raises:
        FileNotFoundError: If no matching files are found
    """
    # Expand home directory
    data_dir = os.path.expanduser(data_dir)
    
    # List and filter files
    files = []
    for f in os.listdir(data_dir):
        if not f.endswith('.csv'):
            continue
        if prefix and not f.startswith(prefix):
            continue
        files.append(f)
    
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}" + 
            (f" with prefix '{prefix}'" if prefix else "")
        )
    
    # Sort files by name (which includes timestamp)
    files = sorted(files, reverse=True)
    
    # Load the requested number of files
    files = files[:n_latest]
    dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in files]
    
    # Return single DataFrame if n_latest=1, otherwise return list
    return dfs[0] if n_latest == 1 else dfs 