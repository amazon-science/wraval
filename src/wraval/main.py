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
from wraval.actions.action_results import show_results
from wraval.actions.action_deploy import deploy
import os


def get_settings(args):
    settings = Dynaconf(
        settings_files=[
            os.path.join(
                os.path.dirname(__file__), "..", "..", "config", "settings.toml"
            )
        ],
        env=f"default,{args.model}",
        environments=True,
    )
    if settings.endpoint_type in ("bedrock", "sagemaker"):
        settings.aws_account = get_current_aws_account_id()
    if args.local_tokenizer_path:
        settings.local_tokenizer_path = args.local_tokenizer_path
    ## add the AWS account you are logged into, if necessary.
    settings.model = settings.model.format(aws_account=settings.aws_account)
    settings.data_dir = settings.data_dir.format(aws_account=settings.aws_account)
    settings.deploy_bucket_name = settings.deploy_bucket_name.format(aws_account=settings.aws_account)
    settings.sagemaker_execution_role_arn = settings.sagemaker_execution_role_arn.format(aws_account=settings.aws_account)

    if args.custom_prompts:
        settings.custom_prompts = True
    else:
        settings.custom_prompts = False
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
            "human_judge_parsing",
            "show_results",
            "deploy"
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
        "--local-tokenizer-path",
        required=False,
        help="Allow for a local path to a tokenizer.",
    )

    parser.add_argument(
        "--custom-prompts", default=False, help="Load custom prompts from a prompt folder"
    )

    parser.add_argument(
        "--cleanup_endpoints", action='store_true'
    )

    return parser.parse_args()


def handle_generate(args, settings):
    generate_tone_data(settings, args.model, args.upload_s3, args.type)


def handle_inference(args, settings):
    run_inference(settings, args.model, args.upload_s3, settings.data_dir)


def handle_judge(args, settings):
    if settings.endpoint_type == "bedrock":
        judge_model = settings.model
        client = boto3.client(
            service_name="bedrock-runtime", region_name=settings.region
        )
    else:  # sagemaker
        judge_model = args.model
        client = None

    judge(
        settings,
        client,
        judge_model,
        args.upload_s3,
        settings.endpoint_type,
    )

def handle_show_results(args, settings):
    show_results(settings, args.type)

def handle_deploy(settings):
    deploy(settings)

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
        case "show_results":
            handle_show_results(args, settings)
        case "deploy":
            handle_deploy(settings)
        case _:
            raise ValueError(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main()
