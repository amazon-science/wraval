#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from typing import Optional, Tuple
import pandas as pd
from src.completion import get_completion
from src.prompt_tones import Tone, get_prompt
from src.data_utils import write_dataset_local, write_dataset_to_s3
from dynaconf import Dynaconf
from src.prompt_tones import master_sys_prompt, get_prompt, get_all_tones, Tone
import os
import boto3

from src.data_generation_prompts import *

# Map dataset types to their corresponding prompts
PROMPT_MAP = {
    "witty": WITTY_SENTENCES_PROMPT,
    "professional": PROFESSIONAL_SENTENCES_PROMPT,
    "summary": PARAGRAPH_SUMMARY_PROMPT,
    "casual": CASUAL_SENTENCES_PROMPT,
    "elaborate": ELABORATE_SENTENCES_PROMPT,
    "shorten": SHORTEN_SENTENCES_PROMPT,
    "improve": IMPROVE_SENTENCES_PROMPT,
    "keypoints": KEYPOINTS_SENTENCES_PROMPT,
    "proofread": PROOFREAD_SENTENCES_PROMPT,
    "emojify": EMOJIFY_SENTENCES_PROMPT
}


def generate_dataset(model: str, 
                    bedrock_client: Optional[boto3.client] = None,
                    dataset_type: str = "witty") -> Tuple[str, pd.DataFrame]:
    """
    Generate dataset based on the specified type
    
    Args:
        model: The model to use for generation
        bedrock_client: Optional bedrock client for AWS models
        dataset_type: Type of dataset to generate. Must be one of:
            - witty
            - professional
            - summary
            - casual
            - elaborate
            - shorten
            - improve
            - keypoints
            - proofread
            - emojify
    
    Returns:
        Tuple[str, pd.DataFrame]: Raw output string and processed DataFrame
        
    Raises:
        ValueError: If dataset_type is not recognized
    """
    if dataset_type not in PROMPT_MAP:
        valid_types = "\n- ".join(PROMPT_MAP.keys())
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}\n"
            f"Must be one of:\n- {valid_types}"
        )
    
    prompt = PROMPT_MAP[dataset_type]
    print(prompt)
    raw_output = get_completion(model, bedrock_client, prompt) #[0]["text"]
    print(raw_output)
    df = process_raw_output(raw_output, dataset_type)
    
    return raw_output, df 


def process_raw_output(output: str, tone: Tone) -> pd.DataFrame:
    """Process raw LLM output into a pandas DataFrame"""
    if tone == Tone.SUMMARIZE:
        # Split on newlines and filter empty lines
        pairs = [
            line.strip().split("|||")
            for line in output.split("\n")
            if "|||" in line and line.strip()
        ]
        return pd.DataFrame(pairs, columns=["synthetic_data", "summary"])
    else:
        # Split on newlines and filter empty lines
        sentences = [line.strip() for line in output.split("\n") if line.strip()]
        return pd.DataFrame(sentences, columns=["synthetic_data"])

def generate_all_datasets(
    settings: Dynaconf, bedrock_client: boto3.client, model_name: str, upload_s3: bool
) -> None:
    all_data = []

    for dataset_type in get_all_tones():
        print(f"Generating {dataset_type}...")
        raw_output, df = generate_dataset(
            settings.model, bedrock_client, dataset_type
        )

        df["tone"] = dataset_type.replace("_sentences", "")
        df["synthetic_model"] = model_name
        all_data.append(df)
        raw_filename = f"{dataset_type}_raw.txt"
        data_dir = os.path.expanduser("~/data")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, raw_filename), "w") as f:
            f.write(raw_output)

    combined_df = pd.concat(all_data, ignore_index=True)

    write_dataset_local(combined_df, "~/data", "all-tones")
    if upload_s3:
        write_dataset_to_s3(df, settings.s3_bucket, "generate/all", "csv")


def generate_specific_dataset(
    settings: Dynaconf,
    bedrock_client: boto3.client,
    dataset_type: str,
    model_name: str,
    upload_s3: bool,
) -> None:
    print(f"Generating {dataset_type}...")
    raw_output, df = generate_dataset(
        settings.model, bedrock_client, dataset_type
    )

    df["tone"] = dataset_type
    df["synthetic_model"] = model_name

    write_dataset_local(df, "~/data", "all-tones")
    if upload_s3:
        write_dataset_to_s3(df, settings.s3_bucket, f"generate/{dataset_type}", "csv")

    # Save raw output for reference
    raw_filename = f"{dataset_type}_raw.txt"
    data_dir = os.path.expanduser("~/data")
    os.makedirs(data_dir, exist_ok=True)
    print(raw_output)
    with open(os.path.join(data_dir, raw_filename), "w") as f:
        f.write(raw_output)