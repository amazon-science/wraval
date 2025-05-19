from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sagemaker.huggingface import HuggingFaceModel
import torch
import tarfile
import boto3
import json

def cleanup_endpoints(endpoint_name):

    sagemaker_client = boto3.client("sagemaker", region_name='us-east-1')

    endpoints = sagemaker_client.list_endpoints()['Endpoints']
    endpoints_configs = sagemaker_client.list_endpoint_configs()['EndpointConfigs']

    endpoints_names = [e['EndpointName'] for e in endpoints]
    endpoints_configs_names = [e['EndpointConfigName'] for e in endpoints_configs]

    if endpoint_name in endpoints_names:
        sagemaker_client.delete_endpoint(EndpointConfigName=endpoint_name)
    if endpoint_name in endpoints_configs_names:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)

def load_artifacts(settings):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(settings.hf_name)

    model.save_pretrained(MODEL_DIRECTORY)
    tokenizer.save_pretrained(MODEL_DIRECTORY)

def write_model_to_s3(settings, model_name):
    tar_file_name = f"{model_name}.tar.gz"
    
    with tarfile.open(tar_file_name, "w:gz") as tar:
        tar.add(MODEL_DIRECTORY, arcname=".")
    
    s3_model_path = f"{settings.deploy_bucket_prefix}/{tar_file_name}"
    s3_client = boto3.client("s3")
    s3_client.upload_file(tar_file_name, settings.deploy_bucket_name, s3_model_path)
        
    s3_uri = f"s3://{settings.deploy_bucket_name}/{s3_model_path}"
    print(f"Model uploaded to: {s3_uri}")
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

def deploy(settings, cleanup_endpoints=False):
    sanitized_model_name = settings.hf_name.split('/')[1].replace('.', '-')
    if cleanup_endpoints:
        cleanup_endpoints(sanitized_model_name)
    load_artifacts(settings)
    s3_uri = write_model_to_s3(settings, sanitized_model_name)
    predictor = deploy_endpoint(
        s3_uri,
        settings.sagemaker_execution_role_arn, 
        sanitized_model_name
    )
    validate_deployment(predictor)