#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
from src.data_utils import write_dataset_local, write_dataset_to_s3
from dynaconf import Dynaconf
from src.prompt_tones import get_all_tones, Tone
import os

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


def generate_dataset(
    settings,
    dataset_type: str = "witty"
) -> pd.DataFrame:
    if dataset_type not in PROMPT_MAP:
        valid_types = "\n- ".join(PROMPT_MAP.keys())
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}\n"
            f"Must be one of:\n- {valid_types}"
        )
    raw_output = route_completion(
        settings,
        [PROMPT_MAP[dataset_type]]
    )
    return process_raw_output(raw_output[0], Tone(dataset_type))

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
    tone: str
) -> None:
    datasets = []
    tones = [tone]
    if tone == "all":
        tones = get_all_tones()

    for tone in tones:
        print(f"Generating {tone}...")
        dataset = generate_dataset(settings, tone)
        dataset["tone"] = tone
        dataset["synthetic_model"] = model_name
        datasets.append(dataset)
        data_dir = os.path.expanduser(settings.data_dir)
        os.makedirs(data_dir, exist_ok=True)

    combined = pd.concat(datasets, ignore_index=True)

    write_dataset_local(combined, settings.data_dir, "all-tones")
    if upload_s3:
        write_dataset_to_s3(combined, settings.s3_bucket, "generate/all", "csv")
