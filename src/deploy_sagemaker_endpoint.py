from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sagemaker.huggingface import HuggingFaceModel
import torch
import tarfile
import boto3
import json

MODEL_DIRECTORY = "model_artifacts"

def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--model_name", type=str, required=True, choices=(
        "Qwen/Qwen2.5-1.5B-Instruct", 
        "microsoft/Phi-3.5-mini-instruct"
        )
    )
    arg_parser.add_argument("--bucket_name", type=str, required=True)
    arg_parser.add_argument("--bucket_prefix", type=str, required=True)
    arg_parser.add_argument("--sagemaker_execution_role_arn", type=str, required=True)
    return arg_parser.parse_args()

def load_artifacts(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.save_pretrained(MODEL_DIRECTORY)
    tokenizer.save_pretrained(MODEL_DIRECTORY)

def write_model_to_s3(args, model_name):
    tar_file_name = f"{model_name}.tar.gz"
    
    with tarfile.open(tar_file_name, "w:gz") as tar:
        tar.add(MODEL_DIRECTORY, arcname=".")
    
    s3_model_path = f"{args.bucket_prefix}/{tar_file_name}"
    s3_client = boto3.client("s3")
    s3_client.upload_file(tar_file_name, args.bucket_name, s3_model_path)
        
    s3_uri = f"s3://{args.bucket_name}/{s3_model_path}"
    print(f"\nModel uploaded to: {s3_uri}")
    return s3_uri

def deploy_endpoint(s3_uri, role, endpoint_name):
    env = {
        "HF_TASK": "text-generation",
        "HF_HUB_OFFLINE": "1",
        "HF_MODEL_QUANTIZE": "bitsandbytes"
    }

    huggingface_model = HuggingFaceModel(
        model_data=s3_uri,
        role=role,
        transformers_version="4.48.0",
        pytorch_version="2.3.0",
        py_version="py311",
        env=env
    )

    return huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.2xlarge",
        endpoint_name=endpoint_name
    )

def validate_deployment(predictor):
    try:
        sagemaker_runtime_client = boto3.client("sagemaker-runtime")
        input_string = json.dumps({"inputs": "Hello, my dog is a little"})
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=predictor.endpoint_name,
            Body=input_string.encode("utf-8"),
            ContentType="application/json"
        )
        print("Test response:", response["Body"].readlines())
    except Exception as e:
        print(f"Validation failed: {e}")
        raise e

def deploy():
    args = parse_args()
    load_artifacts(args)
    sanitized_model_name = args.model_name.split('/')[1].replace('.', '-')
    s3_uri = write_model_to_s3(args, sanitized_model_name)
    predictor = deploy_endpoint(
        s3_uri, 
        args.sagemaker_execution_role_arn, 
        sanitized_model_name
    )
    validate_deployment(predictor)

if __name__ == "__main__":
    deploy()
