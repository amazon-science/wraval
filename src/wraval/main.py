#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from dynaconf import Dynaconf
import argparse
import boto3
from wraval.actions.prompt_tones import get_all_tones
from wraval.actions.action_generate import generate_tone_data
from wraval.actions.action_inference import run_inference
from wraval.actions.action_llm_judge import judge
from wraval.actions.aws_utils import get_current_aws_account_id
import os

def get_settings(args):
    settings = Dynaconf(
        settings_files=[os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'config',
            'settings.toml'
        )
        ],
        env=f"default,{args.model}",
        environments=True
    )
    if settings.endpoint_type in ("bedrock", "sagemaker"):
        settings.aws_account = get_current_aws_account_id()
    if args.local_tokenizer_path:
        settings.local_tokenizer_path = args.local_tokenizer_path
    if settings.endpoint_type == "bedrock":
        settings.model = settings.model.format(aws_account=settings.aws_account)
    return settings

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

def handle_generate(args, settings):
    generate_tone_data(settings, args.model, args.upload_s3, args.type)

def handle_inference(args, settings):
    run_inference(settings, args.model, args.upload_s3, settings.data_dir)

def handle_judge(args, settings):
    if args.endpoint_type == "bedrock":
        judge_model = settings.model.format(aws_account=settings.aws_account)
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.region
        )
    else:  # sagemaker
        judge_model = args.model
        client = None

    judge(
        settings,
        client,
        judge_model,
        args.upload_s3,
        settings.data_dir,
        args.endpoint_type
    )

def main():
    args = parse_args()
    settings = get_settings(args)

    match args.action:
        case "generate":
            handle_generate(args, settings)
        case "inference":
            handle_inference(args, settings)
        case "llm_judge":
            handle_judge(args, settings)
        case _:
            raise ValueError(f"Unknown action: {args.action}")
        
if __name__ == "__main__":
    main()
