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
from src.model_router import route_completion

# Map dataset types to their corresponding prompts
PROMPT_MAP = {
    "witty": WITTY_SENTENCES_PROMPT,
    "professional": PROFESSIONAL_SENTENCES_PROMPT,
    "summarize": PARAGRAPH_SUMMARY_PROMPT,
    "casual": CASUAL_SENTENCES_PROMPT,
    "elaborate": ELABORATE_SENTENCES_PROMPT,
    "shorten": SHORTEN_SENTENCES_PROMPT,
    "improve": IMPROVE_SENTENCES_PROMPT,
    "keypoints": KEYPOINTS_SENTENCES_PROMPT,
    "proofread": PROOFREAD_SENTENCES_PROMPT,
    "emojify": EMOJIFY_SENTENCES_PROMPT
}


def generate_dataset(settings,
                    dataset_type: str = "witty") -> Tuple[str, pd.DataFrame]:
    """
    Generate dataset based on the specified type
    
    Args:
        model: The model to use for generation
        bedrock_client: Optional bedrock client for AWS models
        dataset_type: Type of dataset to generate. Must be one of:
            - witty
            - professional
            - summarize
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
    raw_output = route_completion(settings, prompt)[0]
    d = process_raw_output(raw_output, dataset_type)
    
    return d


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

def generate_tone_data(
    settings: Dynaconf,
    model_name: str,
    upload_s3: bool,
    tone = None
) -> None:
    d = []

    if tone is None:
        tones = get_all_tones()
    else:
        tones = [tone]

    for tone in tones:
        print(f"Generating {tone}...")
        t = generate_dataset(settings, tone)
        t["tone"] = tone
        t["synthetic_model"] = model_name
        d.append(t)
        data_dir = os.path.expanduser("~/data")
        os.makedirs(data_dir, exist_ok=True)

    combined = pd.concat(d, ignore_index=True)

    write_dataset_local(combined, "~/data", "all-tones")
    if upload_s3:
        write_dataset_to_s3(df, settings.s3_bucket, "generate/all", "csv")
