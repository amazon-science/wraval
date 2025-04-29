#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import boto3
from dynaconf import Dynaconf
from src.data_utils import write_dataset_local, write_dataset_to_s3, load_latest_dataset
from src.completion import batch_get_completions, invoke_sagemaker_endpoint
from src.format import format_prompt_as_xml, format_prompt
import os
from transformers import AutoTokenizer
import pandas as pd
from src.completion import batch_get_completions
from src.format import format_prompt_as_xml
from src.prompt_tones import master_sys_prompt, get_prompt, get_all_tones, Tone
from tqdm import tqdm

def run_inference(
    settings: Dynaconf,
    client: boto3.client,
    model_name: str,
    upload_s3: bool,
    data_dir: str,
    endpoint_type: str = "bedrock"
) -> None:
    """Run inference on sentences using the specified model"""
    try:
        df = load_latest_dataset(data_dir)
        print(f"Loaded dataset with {len(df)} rows")
    except FileNotFoundError:
        print("No dataset found. Please generate data first.")
        return

    if "rewrite" not in df.columns:
        df["rewrite"] = None
    if "inference_model" not in df.columns:
        df["inference_model"] = None

    tones = df["tone"].unique()
    print(f"Found tones: {tones}")

    for tone in tones:
        print(
            f"""
        ---------------------
        {tone}
        ---------------------
        """
        )

xb        tone_prompt = get_prompt(Tone(tone))

        queries = df[df["tone"] == tone]["synthetic_data"].unique()

        print(f"Processing {len(queries)} unique inputs for tone: {tone}")

        n = len(queries)
        if endpoint_type == "bedrock":
            prompts = [format_prompt(text, tone_prompt, type="bedrock") for text in tqdm(queries)]
            print(f"Sample prompt:\n{prompts[0]}")
            outputs = batch_get_completions(
                settings.model, client, prompts, [master_sys_prompt] * n
            )
        else:  # sagemaker
            tokenizer = AutoTokenizer.from_pretrained(
                settings.hf_name, trust_remote_code=True
            )
            prompts = [
                format_prompt(text, tone_prompt, tokenizer, type="hf") for text in tqdm(queries)
            ]
            print(f"Sample prompt:\n{prompts[0]}")
            outputs = [
                invoke_sagemaker_endpoint(
                    {"inputs": prompt}
                    # endpoint_name=settings.model
                )
                for prompt in tqdm(prompts)
            ]

        for query, output in zip(queries, outputs):
            mask = (df["synthetic_data"] == query) & (df["tone"] == tone)
            cleaned_output = output.strip().strip('"')
            df.loc[mask, "rewrite"] = cleaned_output
            df.loc[mask, "inference_model"] = model_name

    write_dataset_local(df, "~/data", "all-tones")
    if upload_s3:
        write_dataset_to_s3(df, settings.s3_bucket, "inference/all", "csv")
