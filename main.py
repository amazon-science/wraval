#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from dynaconf import Dynaconf
import argparse
import boto3
from src.prompt_tones import get_all_tones
from src.action_generate import generate_tone_data
from src.action_inference import run_inference
from src.action_llm_judge import judge
from src.aws_utils import get_current_aws_account_id

def parse_args() -> argparse.Namespace:
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
        "--upload-s3", default=False, help="Upload generated datasets to S3"
    )

    parser.add_argument(
        "--local-tokenizer-path", required=False, help="Allow for a local path to a tokenizer."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    settings = Dynaconf(
        settings_files=["settings.toml"],
        env=f"default,{args.model}",
        environments=True
    )

    settings.aws_account = get_current_aws_account_id()

    if args.local_tokenizer_path:
        settings.local_tokenizer_path = args.local_tokenizer_path

    if settings.endpoint_type == "bedrock":
        settings.model = settings.model.format(aws_account=settings.aws_account)

    if args.action == "generate":
        generate_tone_data(
            settings,
            args.model,
            args.upload_s3,
            args.type
        )

    elif args.action == "inference":
        run_inference(
            settings,
            args.model,
            args.upload_s3,
            settings.data_dir
        )

    elif args.action == "llm_judge":        
        if settings.endpoint_type == "bedrock":
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
            settings.data_dir,
            args.endpoint_type
        )
    else:
        raise ValueError(f"Invalid action {args.action}")
        
if __name__ == "__main__":
    main()
