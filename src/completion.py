#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import time
import random
from botocore.exceptions import ClientError
from src.prompt_tones import Prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import boto3
import base64
import sys
import re
# Function to extract last assistant response from each entry


def extract_last_assistant_response(data):
    matches = re.findall(r"<\|assistant\|>(.*?)<\|end\|>", data, re.DOTALL)
    # matches = re.findall(r"<assistant>(.*?)</assistant>", data, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return data

def get_completion(modelId, bedrock_client, prompt, system_prompt=None, max_retries=3, initial_delay=1):
    for attempt in range(max_retries):
        try:
            inference_config = {
                "temperature": 0.0,
                "maxTokens": 3000
            }
            
            # Handle prompt as string
            prompt_text = prompt if isinstance(prompt, str) else prompt.sys_prompt
            
            messages = [{"role": "user", "content": [{"text": prompt_text}]}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            if 'nova' not in modelId:
                additional_model_fields = {
                    "top_p": 1,
                }
                converse_api_params = {
                    "modelId": modelId,
                    "messages": messages,
                    "inferenceConfig": inference_config,
                    "additionalModelRequestFields": additional_model_fields
                }
            else:
                converse_api_params = {
                    "modelId": modelId,
                    "messages": messages,
                    "inferenceConfig": inference_config
                }        
            
            response = bedrock_client.converse(**converse_api_params)
            return response['output']['message']['content'][0]['text']

        except ClientError as err:
            message = err.response['Error']['Message']
            if "Too many requests" in message:
                if attempt < max_retries - 1:
                    wait_time = (initial_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"Rate limited, waiting {wait_time:.2f} seconds before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
            print(f"A client error occurred: {message}")
            raise

def batch_get_completions(modelId, bedrock_client, prompts, system_prompts=None, max_concurrent=5):
    if system_prompts is None:
        system_prompts = [None] * len(prompts)
    elif len(system_prompts) != len(prompts):
        raise ValueError("The number of system prompts must match the number of prompts")

    results = [None] * len(prompts)
    
    def process_prompt(index, prompt, system_prompt):
        try:
            response = get_completion(modelId, bedrock_client, prompt, system_prompt)
            return index, response
        except Exception as e:
            return index, f"Error processing prompt {index}: {str(e)}"

    # Add rate limiting at batch level
    completed_requests = 0
    batch_size = max_concurrent

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(
                    process_prompt, 
                    i, 
                    prompts[i], 
                    system_prompts[i]
                ) for i in range(batch_start, batch_end)
            ]
            
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result
                completed_requests += 1
        
        # Add delay between batches if not the last batch
        if batch_end < len(prompts):
            time.sleep(2)  # Adjust this delay as needed

    return results

# Add this new function for future use
def bedrock_batch_inference(modelId, bedrock_client, prompts, system_prompt, aws_account, tone):
    """
    Future implementation of Bedrock's official batch inference API.
    Currently stored for later use.
    
    Args:
        modelId: The Bedrock model ID
        bedrock_client: Boto3 Bedrock client
        prompts: List of prompts to process
        system_prompt: System prompt to use
        aws_account: AWS account number
        tone: Current tone being processed
    """
    data_dir = os.path.expanduser('~/data')
    input_file = os.path.join(data_dir, 'batch_input.jsonl')
    
    # Create JSONL records
    with open(input_file, 'w') as f:
        for idx, prompt in enumerate(prompts):
            record = {
                "recordId": f"RECORD{idx:08d}",
                "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                }
            }
            f.write(json.dumps(record) + '\n')
    
    # Configure batch job
    input_config = {
        "s3InputDataConfig": {
            "s3Uri": f"s3://llm-finetune-us-east-1-{aws_account}/batch_input/{tone}.jsonl"
        }
    }
    
    output_config = {
        "s3OutputDataConfig": {
            "s3Uri": f"s3://llm-finetune-us-east-1-{aws_account}/batch_output/{tone}/"
        }
    }
    
    # Create batch inference job
    bedrock = boto3.client(service_name="bedrock")
    response = bedrock.create_model_invocation_job(
        roleArn=f"arn:aws:iam::{aws_account}:role/BedrockBatchInferenceRole",
        modelId=modelId,
        jobName=f"tone-transform-{tone}-{int(time.time())}",
        inputDataConfig=input_config,
        outputDataConfig=output_config
    )
    
    return response.get('jobArn')

def get_job_status(bedrock_client, job_arn):
    """Helper function to get batch job status"""
    return bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)['status']

def get_job_error_details(bedrock_client, job_arn):
    """Helper function to get error details for failed jobs"""
    response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
    return {
        'status': response.get('status'),
        'failure_reason': response.get('failureReason', 'No failure reason provided'),
        'creation_time': response.get('creationTime'),
        'completion_time': response.get('completionTime'),
        'input_config': response.get('inputDataConfig'),
        'output_config': response.get('outputDataConfig')
    }

def invoke_sagemaker_endpoint(payload, endpoint_name="Phi-3-5-mini-instruct", region="us-east-1"):
    try:
        sagemaker_runtime_client = boto3.client("sagemaker-runtime", region_name=region)
        input_string = json.dumps(payload)
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=input_string.encode("utf-8"),
            ContentType="application/json"
        )
        json_output = response["Body"].readlines()
        plain_output = '\n'.join(json.loads(json_output[0]))
        last_assistant = extract_last_assistant_response(plain_output)
        print("Test response:", last_assistant)
        return last_assistant
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

