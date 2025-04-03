import boto3
from dynaconf import Dynaconf
from src.data_generator import generate_dataset, PROMPT_MAP
from src.data_utils import save_dataset
import os
import argparse
from typing import List
import pandas as pd

def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description='WRAVAL - Writing Assistant Evaluation Tool')
    
    # Main action argument
    parser.add_argument('action', choices=['generate'],
                      help='Action to perform (currently only "generate" is supported)')
    
    # Dataset type argument
    parser.add_argument('--type', '-t', choices=list(PROMPT_MAP.keys()) + ['all'],
                      default='all',
                      help='Type of dataset to generate (default: all)')
    
    # Model argument
    parser.add_argument('--model', '-m', default='haiku-3',
                      help='Model to use for generation (default: haiku-3)')
    
    # S3 upload argument
    parser.add_argument('--upload-s3', action='store_true',
                      help='Upload generated datasets to S3')
    
    return parser

def generate_all_datasets(settings: Dynaconf, 
                         bedrock_client: boto3.client,
                         model_name: str,
                         upload_to_s3: bool = False) -> None:
    """Generate all available dataset types"""
    all_data = []
    
    for dataset_type in PROMPT_MAP.keys():
        print(f"Generating {dataset_type}...")
        raw_output, df = generate_dataset(
            settings.model,
            bedrock_client,
            dataset_type
        )
        
        # Add metadata with new column names
        df['tone'] = dataset_type.replace('_sentences', '')
        df['synthetic_model'] = model_name
        all_data.append(df)
        
        # Save raw output for reference
        raw_filename = f"{dataset_type}_raw.txt"
        data_dir = os.path.expanduser('~/data')
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, raw_filename), 'w') as f:
            f.write(raw_output)
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save with append behavior
    save_dataset(
        combined_df, 
        upload_to_s3=upload_to_s3,
        append=True
    )

def generate_specific_dataset(settings: Dynaconf, 
                            bedrock_client: boto3.client, 
                            dataset_type: str,
                            model_name: str,
                            upload_to_s3: bool = False) -> None:
    """Generate a specific dataset type"""
    print(f"Generating {dataset_type}...")
    raw_output, df = generate_dataset(
        settings.model,
        bedrock_client,
        dataset_type
    )
    
    # Add metadata with new column names
    df['tone'] = dataset_type.replace('_sentences', '')
    df['synthetic_model'] = model_name
    
    # Save the dataset with all-tones prefix
    save_dataset(
        df, 
        prefix="all-tones",
        upload_to_s3=upload_to_s3,
        append=True
    )
    
    # Save raw output for reference
    raw_filename = f"{dataset_type}_raw.txt"
    data_dir = os.path.expanduser('~/data')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, raw_filename), 'w') as f:
        f.write(raw_output)

def main():
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Load settings using Dynaconf with specified model
    settings = Dynaconf(
        settings_files=["settings.toml"], 
        env=args.model,
        environments=True
    )
    
    # Initialize bedrock client
    bedrock_client = boto3.client(
        service_name='bedrock-runtime', 
        region_name=settings.region
    )
    
    # Handle different actions
    if args.action == 'generate':
        if args.type == 'all':
            generate_all_datasets(settings, bedrock_client, args.model, args.upload_s3)
        else:
            generate_specific_dataset(settings, bedrock_client, args.type, args.model, args.upload_s3)

if __name__ == "__main__":
    main() 