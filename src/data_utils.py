import os
from datetime import datetime
import pandas as pd
import boto3
import tempfile

"""
Notes eetsala:

- Appending to latest file can cause duplicated lines, and it's unclear 
when we want to append and when not in the final workflow. I would propose that
we don't do it here. That would simplify the save function a lot, and give it a single 
responsibility. We can think about the append logic later and how we would like to do that.
Maybe we don't need to append to the same file, but instead assume that all the files are disjoint.
Then we could read multiple files instead of one in case we want to.
That way we wouldn't need to deal with duplicates.
- Do we need to append timestamps here or do we assume that they already come with timestamps?
"""

def write_dataset_to_s3(
    df: pd.DataFrame,
    bucket: str,
    key_prefix: str,
    format: str
) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "temp.jsonl")
        df.to_json(
            temp_file, 
            orient="records", 
            lines=bool(format=="jsonl")
        )
        s3_client = boto3.client('s3')
        key = add_timestamp_to_file_prefix(key_prefix, format)
        print(f"Writing dataset to bucket {bucket} and key {key}.")
        s3_client.upload_file(temp_file, bucket, key)
    return f"s3://{bucket}/{key}"

def write_dataset_local(
    df: pd.DataFrame,
    data_dir: str,
    file_prefix: str
) -> str:
    # Expand home directory and create if needed
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(
        data_dir, 
        add_timestamp_to_file_prefix(file_prefix, "csv")
    )
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return output_path

def add_timestamp_to_file_prefix(file_prefix, format):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{file_prefix}-{timestamp}.{format.lower()}"

def load_latest_dataset(
    data_dir: str,
) -> pd.DataFrame:
    data_dir = os.path.expanduser(data_dir)
    
    files = []
    for f in os.listdir(data_dir):
        if not f.endswith('.csv'):
            continue
        files.append(f)
    
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    file_path = sorted(files, reverse=True)[0]
    return pd.read_csv(os.path.join(data_dir, file_path))
