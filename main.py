import boto3
from dynaconf import Dynaconf
from src.data_generator import generate_dataset, PROMPT_MAP
from src.data_utils import save_dataset, load_latest_dataset
import os
import argparse
from typing import List
import pandas as pd
from src.completion import (
    batch_get_completions, 
    invoke_sagemaker_endpoint
)
from src.format import format_prompt_as_xml, format_prompt
from src.prompt_tones import master_sys_prompt
from tqdm import tqdm
import json
import time
from importlib import import_module
from transformers import AutoTokenizer

def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description='WRAVAL - Writing Assistant Evaluation Tool')
    
    # Main action argument
    parser.add_argument('action', choices=['generate', 'inference'],
                      help='Action to perform (generate data or run inference)')
    
    # Dataset type argument
    parser.add_argument('--type', '-t', choices=list(PROMPT_MAP.keys()) + ['all'],
                      default='all',
                      help='Type of dataset to generate (default: all)')
    
    # Model argument
    parser.add_argument('--model', '-m', default='haiku-3',
                      help='Model to use (default: haiku-3)')
    
    # AWS account argument
    parser.add_argument('--aws-account', required=False,
                      help='AWS account number for Bedrock ARN')
    
    # S3 upload argument
    parser.add_argument('--upload-s3', action='store_true',
                      help='Upload generated datasets to S3')
    
    # Endpoint type argument
    parser.add_argument('--endpoint-type', choices=['bedrock', 'sagemaker'], default='bedrock',
                      help='Type of endpoint to use (default: bedrock)')
    
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

def get_job_error_details(bedrock, job_arn):
    """Get detailed error information for a failed job"""
    try:
        response = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        failure_reason = response.get('failureReason', 'No failure reason provided')
        error_details = {
            'status': response.get('status'),
            'failure_reason': failure_reason,
            'creation_time': response.get('creationTime'),
            'completion_time': response.get('completionTime'),
            'input_config': response.get('inputDataConfig'),
            'output_config': response.get('outputDataConfig')
        }
        return error_details
    except Exception as e:
        return f"Error getting job details: {str(e)}"

def run_inference(settings: Dynaconf,
                 client: boto3.client,
                 model_name: str,
                 upload_to_s3: bool = False,
                 endpoint_type: str = 'bedrock') -> None:
    """Run inference on sentences using the specified model"""
    try:
        df = load_latest_dataset()
        print(f"Loaded dataset with {len(df)} rows")
    except FileNotFoundError:
        print("No dataset found. Please generate data first.")
        return

    # Initialize new columns if they don't exist
    if 'rewrite' not in df.columns:
        df['rewrite'] = None
    if 'inference_model' not in df.columns:
        df['inference_model'] = None

    # Get unique tones
    tones = df['tone'].unique()
    print(f"Found tones: {tones}")
    
    # Process each tone
    for tone in tones:
        print(f'''
        ---------------------
        {tone}
        ---------------------
        ''')
        
        # Get tone-specific prompt

        prompts_module = import_module(f"prompts.prompt_tones")

        tone_prompt = getattr(prompts_module, f"{tone.capitalize()}Prompt")
        
        # Get unique inputs for this tone
        queries = df[df['tone'] == tone]['synthetic_data'].unique()
        
        # Format prompts
        
        print(f"Processing {len(queries)} unique inputs for tone: {tone}")
        
        # Run batch inference using appropriate method
        n = len(queries)
        if endpoint_type == 'bedrock':
            prompts = [format_prompt_as_xml(text, tone_prompt()) for text in tqdm(queries)]
            print(f"Sample prompt:\n{prompts[0]}")
            outputs = batch_get_completions(
                settings.model, 
                client, 
                prompts, 
                [master_sys_prompt] * n
            )
        else:  # sagemaker
            tokenizer = AutoTokenizer.from_pretrained(settings.hf_name, trust_remote_code=True)
            prompts = [format_prompt(text, tone_prompt(), tokenizer) for text in tqdm(queries)]
            print(f"Sample prompt:\n{prompts[0]}")
            outputs = [invoke_sagemaker_endpoint(
                {"inputs": prompt}
                # endpoint_name=settings.model
            ) for prompt in tqdm(prompts)]
        
        # Update DataFrame with results
        for query, output in zip(queries, outputs):
            mask = (df['synthetic_data'] == query) & (df['tone'] == tone)
            cleaned_output = output.strip().strip('"')
            df.loc[mask, 'rewrite'] = cleaned_output
            df.loc[mask, 'inference_model'] = model_name
    
    # Save updated dataset
    save_dataset(
        df,
        prefix='all-tones',
        upload_to_s3=upload_to_s3,
        append=False
    )

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Load settings using Dynaconf with specified model
    settings = Dynaconf(
        settings_files=["settings.toml"], 
        env=args.model,
        environments=True
    )
    
    if args.action == 'generate':
        if not args.aws_account:
            parser.error("--aws-account is required for generate action")
        # Replace AWS account placeholder in model ARN
        settings.model = settings.model.format(aws_account=args.aws_account)
        
        # Initialize bedrock client
        bedrock_client = boto3.client(
            service_name='bedrock-runtime', 
            region_name=settings.region
        )
        
        if args.type == 'all':
            generate_all_datasets(settings, bedrock_client, args.model, args.upload_s3)
        else:
            generate_specific_dataset(settings, bedrock_client, args.type, args.model, args.upload_s3)
            
    elif args.action == 'inference':
        if args.endpoint_type == 'bedrock':
            if not args.aws_account:
                parser.error("--aws-account is required for bedrock endpoint")
            # Replace AWS account placeholder in model ARN
            settings.model = settings.model.format(aws_account=args.aws_account)
            client = boto3.client(
                service_name='bedrock-runtime', 
                region_name=settings.region
            )
        else:  # sagemaker
            settings.model = args.model  # Use model name directly as endpoint name
            client = None  # Not needed for SageMaker endpoint
            
        run_inference(settings, client, args.model, args.upload_s3, endpoint_type=args.endpoint_type)

if __name__ == "__main__":
    main() 
