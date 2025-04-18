import boto3
from dynaconf import Dynaconf
from src.data_utils import write_dataset_local, write_dataset_to_s3, load_latest_dataset
import argparse
from src.completion import batch_get_completions, invoke_sagemaker_endpoint
from src.format import format_prompt_as_xml, format_prompt
from src.prompt_tones import master_sys_prompt
from tqdm import tqdm
import os
from transformers import AutoTokenizer
import pandas as pd
from src.completion import batch_get_completions
from src.format import format_prompt_as_xml
from src.prompt_tones import master_sys_prompt, get_prompt, get_all_tones, Tone
from tqdm import tqdm
from src.action_generate import generate_all_datasets, generate_specific_dataset
from src.action_inference import run_inference


def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="WRAVAL - Writing Assistant Evaluation Tool"
    )

    parser.add_argument(
        "action",
        choices=["generate", "inference", "human_eval_upload", "human_eval_parsing"],
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
        "--aws-account", required=True, help="AWS account number for Bedrock ARN"
    )

    parser.add_argument(
        "--upload-s3", action="store_true", help="Upload generated datasets to S3"
    )

    parser.add_argument(
        "--endpoint-type",
        choices=["bedrock", "sagemaker"],
        default="bedrock",
        help="Type of endpoint to use (default: bedrock)",
    )

    parser.add_argument(
        "--data-dir", default="~/data", help="Where the data files are stored"
    )

    return parser

def main():
    parser = setup_argparse()
    args = parser.parse_args()

    settings = Dynaconf(
        settings_files=["settings.toml"], env=args.model, environments=True
    )

    settings.model = settings.model.format(aws_account=args.aws_account)

    bedrock_client = boto3.client(
        service_name="bedrock-runtime", region_name=settings.region
    )

    if args.action == "generate":
        if not args.aws_account:
            parser.error("--aws-account is required for generate action")
        settings.model = settings.model.format(aws_account=args.aws_account)

        bedrock_client = boto3.client(
            service_name="bedrock-runtime", region_name=settings.region
        )

        if args.type == "all":
            generate_all_datasets(settings, bedrock_client, args.model, args.upload_s3)
        else:
            generate_specific_dataset(
                settings, bedrock_client, args.type, args.model, args.upload_s3
            )

    elif args.action == "inference":
        if args.endpoint_type == "bedrock":
            if not args.aws_account:
                parser.error("--aws-account is required for bedrock endpoint")
            settings.model = settings.model.format(aws_account=args.aws_account)
            client = boto3.client(
                service_name="bedrock-runtime", region_name=settings.region
            )
        else:  # sagemaker
            settings.model = args.model  # Use model name directly as endpoint name
            client = None  # Not needed for SageMaker endpoint

        run_inference(
            settings,
            client,
            args.model,
            args.upload_s3,
            args.data_dir,
            args.endpoint_type
        )


if __name__ == "__main__":
    main()
