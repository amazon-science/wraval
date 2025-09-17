#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
# Import config first to suppress warnings
from wraval.aws_config import *  # This will run the config code before any other imports

from dynaconf import Dynaconf
import boto3
from wraval.actions.prompt_tones import get_all_tones
from wraval.actions.action_generate import generate_tone_data
from wraval.actions.action_inference import run_inference
from wraval.actions.action_llm_judge import judge
from wraval.actions.aws_utils import get_current_aws_account_id
from wraval.actions.action_results import get_results
from wraval.actions.action_deploy import deploy
from wraval.actions.action_examples import get_examples
from wraval.actions.action_human_judge_upload import upload_human_judge
import os
import typer
from typing import Optional
from enum import Enum

app = typer.Typer(help="WRAVAL - Writing Assistant Evaluation Tool")


# Create an enum for tone types
class ToneType(str, Enum):
    ALL = "all"
    WITTY = "witty"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ELABORATE = "elaborate"
    SHORTEN = "shorten"
    IMPROVE = "improve"
    KEYPOINTS = "keypoints"
    PROOFREAD = "proofread"
    EMOJIFY = "emojify"
    SUMMARIZE = "summarize"


def get_settings(
    model: str = "haiku-3",
    tone: ToneType = ToneType.ALL,
    custom_prompts: bool = False,
    local_tokenizer_path: Optional[str] = None,
) -> Dynaconf:
    """Get settings with the specified configuration."""
    settings = Dynaconf(
        settings_files=[
            os.path.join(
                os.path.dirname(__file__), "..", "..", "config", "settings.toml"
            )
        ],
        env=f"default,{model}",
        environments=True,
        case_sensitive=False,
    )

    if settings.endpoint_type in ("bedrock", "sagemaker"):
        settings.aws_account = get_current_aws_account_id()

    if local_tokenizer_path:
        settings.local_tokenizer_path = local_tokenizer_path

    # Format settings with AWS account
    settings.model = settings.model.format(aws_account=settings.aws_account)
    settings.data_dir = settings.data_dir.format(aws_account=settings.aws_account)
    settings.deploy_bucket_name = settings.deploy_bucket_name.format(
        aws_account=settings.aws_account
    )
    settings.sagemaker_execution_role_arn = (
        settings.sagemaker_execution_role_arn.format(aws_account=settings.aws_account)
    )
    settings.type = tone
    settings.custom_prompts = custom_prompts

    return settings


@app.command()
def generate(
    model: str = typer.Option("haiku-3", "--model", "-m", help="Model to use"),
    tone: ToneType = typer.Option(
        ToneType.ALL, "--type", "-t", help="Type of dataset to generate"
    ),
    upload_s3: bool = typer.Option(
        False, "--upload-s3", help="Upload generated datasets to S3"
    ),
    custom_prompts: bool = typer.Option(
        False, "--custom-prompts", help="Load custom prompts from a prompt folder"
    ),
    local_tokenizer_path: Optional[str] = typer.Option(
        None, "--local-tokenizer-path", help="Path to local tokenizer"
    ),
):
    """Generate tone data using the specified model."""
    settings = get_settings(model, tone, custom_prompts, local_tokenizer_path)
    generate_tone_data(settings, model, upload_s3, tone)


@app.command()
def inference(
    model: str = typer.Option("haiku-3", "--model", "-m", help="Model to use"),
    tone: ToneType = typer.Option(
        ToneType.ALL, "--type", "-t", help="Type of dataset to process"
    ),
    upload_s3: bool = typer.Option(False, "--upload-s3", help="Upload results to S3"),
    custom_prompts: bool = typer.Option(
        False, "--custom-prompts", help="Load custom prompts from a prompt folder"
    ),
    local_tokenizer_path: Optional[str] = typer.Option(
        None, "--local-tokenizer-path", help="Path to local tokenizer"
    ),
):
    """Run inference on the dataset using the specified model."""
    settings = get_settings(model, tone, custom_prompts, local_tokenizer_path)
    run_inference(settings, model, upload_s3, settings.data_dir)


@app.command()
def llm_judge(
    model: str = typer.Option(
        "haiku-3", "--model", "-m", help="Model to use for judging"
    ),
    tone: ToneType = typer.Option(
        ToneType.ALL, "--type", "-t", help="Type of dataset to judge"
    ),
    upload_s3: bool = typer.Option(False, "--upload-s3", help="Upload results to S3"),
    custom_prompts: bool = typer.Option(
        False, "--custom-prompts", help="Load custom prompts from a prompt folder"
    ),
    local_tokenizer_path: Optional[str] = typer.Option(
        None, "--local-tokenizer-path", help="Path to local tokenizer"
    ),
):
    """Judge the dataset using LLM evaluation."""
    settings = get_settings(model, tone, custom_prompts, local_tokenizer_path)

    if settings.endpoint_type == "bedrock":
        judge_model = settings.model
        client = boto3.client(
            service_name="bedrock-runtime", region_name=settings.region
        )
    else:  # sagemaker
        judge_model = model
        client = None

    judge(
        settings,
        client,
        judge_model,
        upload_s3,
        settings.endpoint_type,
    )


@app.command()
def show_results(
    tone: ToneType = typer.Option(
        ToneType.ALL, "--type", "-t", help="Type of dataset to show examples for"
    )
):
    """Show results for the dataset."""
    settings = get_settings()
    get_results(settings, tone)


@app.command()
def show_examples(
    model: str = typer.Option("haiku-3", "--model", "-m", help="Model to use"),
    tone: ToneType = typer.Option(
        ToneType.ALL, "--type", "-t", help="Type of dataset to show examples for"
    ),
    n_examples: int = typer.Option(
        10,
        "--n-examples",
        "-n",
        help="Number of examples to show per tone-model combination",
    ),
    custom_prompts: bool = typer.Option(
        False, "--custom-prompts", help="Load custom prompts from a prompt folder"
    ),
    local_tokenizer_path: Optional[str] = typer.Option(
        None, "--local-tokenizer-path", help="Path to local tokenizer"
    ),
):
    """Show examples from the dataset."""
    settings = get_settings(model, tone, custom_prompts, local_tokenizer_path)
    get_examples(settings, tone, n_examples)


@app.command()
def human_judge_upload(
    tone: ToneType = typer.Option(
        ToneType.ALL, "--type", "-t", help="Type of dataset to show examples for"
    ),
    n_samples: int = typer.Option(
        10,
        "--n-samples",
        "-n",
        help="Total number of samples to sample for human evaluation.",
    ),
    synthetic_model: str = typer.Option(
        "haiku-3", "--synthetic-model", "-sm", help="Synthetic model to sample from."
    ),
    inference_models: Optional[str] = typer.Option(
        None,
        "--inference-models",
        "-im",
        help="Comma-separated list of inference models (e.g., 'model1,model2,model3')",
    ),
):
    """Upload human judgment dataset and create a smaller sampled version."""
    settings = get_settings()
    settings.tone = tone
    settings.n_samples = n_samples
    settings.synthetic_model = synthetic_model

    # Parse comma-separated inference models
    if inference_models:
        settings.inference_models = [
            model.strip() for model in inference_models.split(",")
        ]
    elif not hasattr(settings, "inference_models"):
        settings.inference_models = ["haiku-3"]  # fallback default

    upload_human_judge(settings)


@app.command()
def deploy(
    model: str = typer.Option("haiku-3", "--model", "-m", help="Model to deploy"),
    cleanup_endpoints: bool = typer.Option(
        False,
        "--cleanup-endpoints",
        help="Cleanup endpoints before deploying new models",
    ),
    custom_prompts: bool = typer.Option(
        False, "--custom-prompts", help="Load custom prompts from a prompt folder"
    ),
    local_tokenizer_path: Optional[str] = typer.Option(
        None, "--local-tokenizer-path", help="Path to local tokenizer"
    ),
):
    """Deploy the model to the specified endpoint."""
    settings = get_settings(model, ToneType.ALL, custom_prompts, local_tokenizer_path)
    if cleanup_endpoints:
        settings.cleanup_endpoints = True
    deploy(settings)


def main():
    app()


if __name__ == "__main__":
    main()
