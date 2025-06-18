import os
from datetime import datetime, UTC
import pandas as pd
import boto3
import tempfile
from typing import Optional, List, Tuple
from urllib.parse import urlparse
import uuid

s3_client = boto3.client("s3")


def write_dataset(df: pd.DataFrame, data_dir: str, file_prefix: str, format: str):
    # Ensure 'uuid' column exists
    if "uuid" not in df.columns:
        df["uuid"] = None  # or np.nan

    # Fill missing or empty UUIDs
    df["uuid"] = df["uuid"].apply(
        lambda x: x if pd.notna(x) and x != "" else str(uuid.uuid4())
    )

    if is_s3_path(data_dir):
        bucket, prefix = parse_s3_path(data_dir)
        write_dataset_s3(df, bucket, prefix, file_prefix, format)
    else:
        write_dataset_local(df, data_dir, file_prefix, format)


def write_dataset_s3(
    df: pd.DataFrame, bucket: str, prefix: str, file_prefix: str, format: str
):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "temp.csv")
        df.to_csv(temp_file, index=False)
        key = os.path.join(prefix, add_timestamp_to_file_prefix(file_prefix, format))
        print(f"Writing dataset to s3://{bucket}/{key}")
        s3_client.upload_file(temp_file, bucket, key)


def write_dataset_local(
    df: pd.DataFrame, data_dir: str, file_prefix: str, format: str
) -> str:
    # Expand home directory and create if needed
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(
        data_dir, add_timestamp_to_file_prefix(file_prefix, format)
    )
    df.to_csv(output_path, index=False)
    print(f"Saved locally to {output_path}")


def add_timestamp_to_file_prefix(file_prefix, format):
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{file_prefix}-{timestamp}.{format.lower()}"


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path."""
    return path.startswith("s3://")


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse an S3 path into bucket and prefix."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def list_s3_files(bucket: str, prefix: str, suffix: str = ".csv") -> List[str]:
    """List all files in an S3 bucket with a specific prefix and suffix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        return []

    files = [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(suffix)]
    return files


def load_latest_dataset_from_s3(bucket: str, prefix: str) -> pd.DataFrame:
    """Load the latest dataset from an S3 bucket."""
    files = list_s3_files(bucket, prefix)

    if not files:
        raise FileNotFoundError(f"No CSV files found in s3://{bucket}/{prefix}")

    # Sort files by name (which should include timestamp)
    latest_file = sorted(files, reverse=True)[0]

    # Download and load the file
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
        s3_client.download_file(bucket, latest_file, temp_file.name)
        return pd.read_csv(temp_file.name)


def load_latest_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load the latest dataset from either local filesystem or S3.

    Args:
        data_dir: Path to data directory or S3 URI (s3://bucket/prefix)

    Returns:
        DataFrame containing the dataset
    """
    if is_s3_path(data_dir):
        bucket, prefix = parse_s3_path(data_dir)
        return load_latest_dataset_from_s3(bucket, prefix)
    else:

        # Local file handling
        data_dir = os.path.expanduser(data_dir)

        files = []
        for f in os.listdir(data_dir):
            if not f.endswith(".csv"):
                continue
            files.append(f)

        if not files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        file_path = sorted(files, reverse=True)[0]
        print(f"Loading {file_path}")
        return pd.read_csv(os.path.join(data_dir, file_path))


def latest_file_name(data_dir: str) -> str:
    if is_s3_path(data_dir):
        bucket, prefix = parse_s3_path(data_dir)
        files = list_s3_files(bucket, prefix)
        if not files:
            raise FileNotFoundError(f"No CSV files found in s3://{bucket}/{prefix}")

        # Sort files by name (which should include timestamp)
        latest_file = sorted(files, reverse=True)[0]
        return latest_file

    data_dir = os.path.expanduser(data_dir)

    files = []
    for f in os.listdir(data_dir):
        if not f.endswith(".csv"):
            continue
        files.append(f)

    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    latest_file = sorted(files, reverse=True)[0]

    return latest_file
