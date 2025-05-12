#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
from typing import List, Dict, Optional
from dynaconf import Dynaconf
from .data_utils import write_dataset_local, write_dataset_to_s3, load_latest_dataset
from .prompts_judge import generate_input_prompt, generate_system_prompt, get_rubric, rewrite_prompt
from .completion import batch_get_bedrock_completions
import re
import boto3

def extract_score(text: str) -> Optional[int]:
    """Extract score from text using regex pattern.
    
    Args:
        text: String containing score in format <score>N</score>
        
    Returns:
        Extracted score as integer or None if no score found
    """
    match = re.search(r"<score>(\d+)</score>", text)
    return int(match.group(1)) if match else None

def validate_dataset(d: pd.DataFrame) -> bool:
    """Validate required columns exist in dataset.
    
    Args:
        d: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = {"synthetic_data", "rewrite", "tone"}
    if not all(col in d.columns for col in required_columns):
        print(f"Missing required columns. Required: {required_columns}")
        return False
    return True

def process_tone_data(
    settings: Dynaconf, 
    d: pd.DataFrame,
    tone: str,
    model_name: str,
    client: boto3.client,
    tone_rubrics: Dict
) -> pd.DataFrame:
    """Process data for a specific tone and generate scores.
    
    Args:
        d: Input DataFrame
        tone: Current tone being processed
        model_name: Name of the model to use
        client: Boto3 client
        tone_rubrics: Dictionary of rubrics for the tone
        
    Returns:
        Processed DataFrame with scores
    """
    dmt = d[d.tone == tone].copy()
    rubrics = list(tone_rubrics.keys())
    
    # Generate prompts
    user_prompts = []
    sys_prompts = []

    for q, a in zip(dmt["synthetic_data"], dmt["rewrite"]):
        for rubric in rubrics:
            user_prompts.append(generate_input_prompt(q, a, tone))
            sys_prompts.append(generate_system_prompt(tone_rubrics[rubric]))
    
    # Get completions
    # import pdb
    # pdb.set_trace()
    completions = batch_get_bedrock_completions(
        settings,
        user_prompts, 
        sys_prompts
    )
    
    rubrics = [r.lower() for r in rubrics]

    # Process scores
    for i, rubric in enumerate(rubrics):
        dmt[rubric] = completions[i::len(rubrics)]
        dmt[f'{rubric}_score'] = dmt[rubric].apply(extract_score)
    
    # Move all score columns to the right
    score_columns = [f'{r}_score' for r in rubrics]
    other_columns = [col for col in dmt.columns if col not in score_columns]
    dmt = dmt[other_columns + score_columns]
    
    dmt['overall_score'] = dmt[score_columns].mean(axis=1)
    return dmt

def judge(
    settings: Dynaconf,
    client: boto3.client,
    model_name: str,
    upload_s3: bool,
    endpoint_type: str = "bedrock"
) -> None:
    """Judge rewrites using specified model and rubrics.
    
    Args:
        settings: Dynaconf settings object
        client: Boto3 client
        model_name: Name of the model to use
        upload_s3: Whether to upload results to S3
        data_dir: Directory containing input data
        endpoint_type: Type of endpoint to use
    """
    try:
        d = load_latest_dataset(settings.data_dir)
        print(f"Loaded dataset with {len(d)} rows")
    except FileNotFoundError:
        print("No dataset found. Please generate data first.")
        return
        
    if not validate_dataset(d):
        return
        
    tones = d["tone"].unique()
    print(f"Found tones: {tones}")
    
    for tone in tones:
        print(f"\n{'='*20}\n{tone}\n{'='*20}")
        
        tone_rubrics = get_rubric(tone.upper())
        dmt = process_tone_data(settings, d, tone, model_name, client, tone_rubrics)
        
        # Update main dataframe
        mask = (d.tone == tone)
        d.loc[mask, dmt.columns] = dmt.values
    
    # Save results
    write_dataset_local(d, "./data", "all-tones")
    if upload_s3:
        write_dataset_to_s3(d, settings.s3_bucket, "inference/all", "csv")

def rewrite_judge(
    model_id: str,
    bedrock_client: boto3.client,
    queries: List[str],
    answers: List[str]
) -> pd.DataFrame:
    """Judge rewrites for a set of query-answer pairs.
    
    Args:
        model_id: Model identifier
        bedrock_client: Boto3 client for Bedrock
        queries: List of input queries
        answers: List of corresponding answers
        
    Returns:
        DataFrame with input, output, and scores
    """
    d = pd.DataFrame({'input': queries, 'output': answers})
    prompts = [rewrite_prompt(q, a) for q, a in zip(queries, answers)]
    d['rewrite_score'] = batch_get_bedrock_completions(
        model_id, 
        bedrock_client,
        prompts,
        max_concurrent=len(prompts)
    )
    return d
