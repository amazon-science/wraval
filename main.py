#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from dynaconf import Dynaconf
from src.data_utils import write_dataset_local, write_dataset_to_s3, load_latest_dataset
import argparse
from src.format import format_prompt_as_xml, format_prompt
from src.prompt_tones import master_sys_prompt
from tqdm import tqdm
import os
import pandas as pd
from src.format import format_prompt_as_xml
from src.prompt_tones import master_sys_prompt, get_prompt, get_all_tones, Tone
from tqdm import tqdm
from src.action_generate import generate_tone_data
from src.action_inference import run_inference
from src.action_llm_judge import judge
from src.aws_utils import get_current_aws_account_id

def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="WRAVAL - Writing Assistant Evaluation Tool"
    )

    parser.add_argument(
        "action",
        choices=[
            "generate",
            "inference",
            "llm_judge",
            "human_judge_upload",
            "human_judge_parsing"
            ],
        help="Action to perform (generate data or run inference)",
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=get_all_tones() + ["all"],
        default="all",
        help="Type of dataset to generate (default: all)",
    )

    parser.add_argument(
        "--model", "-m", default="haiku-3", help="Model to use (default: haiku-3)"
    )

    parser.add_argument(
        "--number-of-samples", "-n", default=100, help="Number of samples to generate (default:100)"
    )

    parser.add_argument(
        "--aws-account", required=False, help="AWS account number for Bedrock ARN"
    )

    parser.add_argument(
        "--upload-s3", default=False, help="Upload generated datasets to S3"
    )

    parser.add_argument(
        "--data-dir", default="~/data", help="Where the data files are stored"
    )
    
    parser.add_argument(
        "--endpoint-type",
        choices=["bedrock", "sagemaker", "ollama"],
        default="bedrock",
        help="Type of endpoint to use (default: bedrock)",
    )

    parser.add_argument(
        "--local-tokenizer-path", required=False, help="Allow for a local path to a tokenizer."
    )
    
    return parser

def main():
    parser = setup_argparse()
    args = parser.parse_args()
        
    settings = Dynaconf(
        settings_files=["settings.toml"], env=args.model, environments=True
    )

    settings.endpoint_type = args.endpoint_type

    if args.local_tokenizer_path:
        settings.local_tokenizer_path = args.local_tokenizer_path    

    if args.endpoint_type == "bedrock":
        if args.aws_account is None:
            settings.aws_account = get_current_aws_account_id() 
        else:
            settings.aws_account = args.aws_account
        settings.model = settings.model.format(aws_account=settings.aws_account)    

    if args.action == "generate":

        if args.type == "all":
            generate_tone_data(settings,
                               args.model,
                               args.upload_s3)
        else:
            generate_tone_data(settings,
                               args.model,
                               args.upload_s3,
                               args.type)

    elif args.action == "inference":

        run_inference(
            settings,
            args.model,
            args.upload_s3,
            args.data_dir
        )

    elif args.action == "llm_judge":        
        if args.endpoint_type == "bedrock":
            judge_model = settings.model.format(aws_account=settings.aws_account)
            client = boto3.client(
                service_name="bedrock-runtime", region_name=settings.region
            )
        else:  # sagemaker
            judge_model = args.model  # Use model name directly as endpoint name
            client = None  # Not needed for SageMaker endpoint

        judge(
            settings,
            client,
            judge_model,
            args.upload_s3,
            args.data_dir,
            args.endpoint_type
        )
        
if __name__ == "__main__":
    main()
