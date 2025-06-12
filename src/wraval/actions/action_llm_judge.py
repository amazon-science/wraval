#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
from typing import List, Dict, Any, Optional
from itertools import product
from dynaconf import Dynaconf
from .data_utils import write_dataset, load_latest_dataset
from .completion import batch_get_bedrock_completions
import re
import boto3

# Import prompt functions based on settings
def get_prompt_functions(settings: Dynaconf):
    """Get the appropriate prompt functions based on settings."""
    if settings.custom_prompts:
        from wraval.custom_prompts.prompts_judge import (
            generate_input_prompt,
            generate_system_prompt,
            get_rubric
        )
    else:
        from .prompts_judge import (
            generate_input_prompt,
            generate_system_prompt,
            get_rubric
        )
    return generate_input_prompt, generate_system_prompt, get_rubric

def extract_score(text: str) -> Optional[int]:
    """Extract score from text using regex pattern.
    
    Args:
        text: String containing score in format <score>N</score>
        
    Returns:
        Extracted score as integer or None if no score found
    """
    match = re.search(r"<score>(\d+)</score>", text)
    return int(match.group(1)) if match else None

def validate_dataset(results: pd.DataFrame) -> bool:
    """Validate required columns exist in dataset.
    
    Args:
        d: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = {"synthetic_data", "rewrite", "tone"}
    if not all(col in results.columns for col in required_columns):
        print(f"Missing required columns. Required: {required_columns}")
        return False
    return True

def process_tone_data(
    settings: Dynaconf, 
    results: pd.DataFrame,
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
    # Get the appropriate prompt functions
    generate_input_prompt, generate_system_prompt, _ = get_prompt_functions(settings)

    temp_results = results.copy()
    rubrics = list(tone_rubrics.keys())
    
    # Generate prompts
    user_prompts = []
    sys_prompts = []

    for q, a in zip(temp_results["synthetic_data"], temp_results["rewrite"]):
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
        temp_results[rubric] = completions[i::len(rubrics)]
        temp_results[f'{rubric}_score'] = temp_results[rubric].apply(extract_score)
    
    # Move all score columns to the right
    score_columns = [f'{r}_score' for r in rubrics]
    other_columns = [col for col in temp_results.columns if col not in score_columns]
    temp_results = temp_results[other_columns + score_columns]
    
    temp_results['overall_score'] = temp_results[score_columns].mean(axis=1)
    return temp_results

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
        results = load_latest_dataset(settings.data_dir)
        print(f"Loaded dataset with {len(results)} rows")
    except FileNotFoundError:
        print("No dataset found. Please generate data first.")
        return
        
    if not validate_dataset(results):
        return
        
    tones = results["tone"].unique()
    inf_models = results["inference_model"].unique()
    print(f"Found tones: {tones}")
    print(f"Found inference_models: {inf_models}")

    if settings.type != "all":
        tones = [settings.type]
    
    # Get the appropriate prompt functions
    _, _, get_rubric = get_prompt_functions(settings)
    
    # Process each tone-model combination that needs scoring
    for tone, inf_model in product(tones, inf_models):
        mask = (results.inference_model == inf_model) & (results.tone == tone)
        # check if any score is missing for this inference model and this tone
        if 'overall_score' not in results.columns:
            results['overall_score'] = None
        if not results[mask].overall_score.isna().any():
            continue
            
        print(f"\n{'='*20}\n{tone} tone\nfor inference model {inf_model}\n{'='*20}")
        
        tone_rubrics = get_rubric(tone.upper())
        temp_results = process_tone_data(settings, results[mask], tone, model_name, client, tone_rubrics)
        results.loc[mask, temp_results.columns] = temp_results.values
    
    # Save results
    write_dataset(results, settings.data_dir, "all", "csv")

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
    results = pd.DataFrame({'input': queries, 'output': answers})
    prompts = [rewrite_prompt(q, a) for q, a in zip(queries, answers)]
    results['rewrite_score'] = batch_get_bedrock_completions(
        model_id, 
        bedrock_client,
        prompts,
        max_concurrent=len(prompts)
    )
    return results
