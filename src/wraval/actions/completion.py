#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import time
import random
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import boto3
import re
import requests
import uuid


# Function to extract last assistant response from each entry
def extract_last_assistant_response(data):

    if r"<\|assistant\|>" in data: # phi
        assistant_part = data.split(r"<\|assistant\|>")[-1]
        response = response.replace(r"<\|end\|>", "").strip()
        return response
        
    if r"<|im_start|>assistant" in data: # qwen
        assistant_part = data.split(r"<|im_start|>assistant")[-1]
        
        # Remove the thinking part if it exists
        if r"<think>" in assistant_part:
            # Extract everything after </think>
            response = assistant_part.split(r"</think>")[-1]
        else:
            response = assistant_part
        response = response.replace(r"<|im_end|>", "").strip()
        return response
    
    return data

def get_bedrock_completion(settings, prompt, system_prompt=None):
    bedrock_client = boto3.client(
        service_name="bedrock-runtime", region_name=settings.region
    )
    max_retries = 2
    initial_delay = 1
    for attempt in range(max_retries):
        try:
            inference_config = {"temperature": 0.0, "maxTokens": 3000}

            converse_api_params = {
                "modelId": settings.model,
                "inferenceConfig": inference_config,
            }

            # Handle prompt as string
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": [{"text": prompt}]}]
            else:
                messages = prompt

            converse_api_params.update({"messages": messages})

            if "nova" not in settings.model:
                converse_api_params.update(
                    {"additionalModelRequestFields": {"top_p": 1}}
                )

            if isinstance(system_prompt, str) and len(system_prompt) > 0:
                converse_api_params.update({"system": [{"text": system_prompt}]})
                # converse_api_params["messages"] = [{"role": "assistant", "content": [{"text": system_prompt}]}] + converse_api_params["messages"]

            response = bedrock_client.converse(**converse_api_params)
            return response["output"]["message"]["content"][0]["text"]

        except ClientError as err:
            message = err.response["Error"]["Message"]
            if "Too many requests" in message:
                if attempt < max_retries - 1:
                    wait_time = (initial_delay * (2**attempt)) + random.uniform(0, 1)
                    print(
                        f"Rate limited, waiting {wait_time:.2f} seconds before retry {attempt + 1}"
                    )
                    time.sleep(wait_time)
                    continue
            print(f"A client error occurred: {message}")
            raise


def batch_get_bedrock_completions(
    settings, prompts, system_prompts=None, max_concurrent=100
):
    if system_prompts is None:
        system_prompts = [None] * len(prompts)
    elif len(system_prompts) != len(prompts):
        raise ValueError(
            "The number of system prompts must match the number of prompts"
        )

    results = [None] * len(prompts)

    # Add rate limiting at batch level
    completed_requests = 0
    batch_size = max_concurrent

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(
                    process_prompt, i, prompts[i], system_prompts[i], settings
                )
                for i in range(batch_start, batch_end)
            ]

            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result
                completed_requests += 1

        # Add delay between batches if not the last batch
        if batch_end < len(prompts):
            time.sleep(2)  # Adjust this delay as needed

    return results


def process_prompt(index, prompt, system_prompt, settings):
    try:
        response = get_bedrock_completion(settings, prompt, system_prompt)
        return index, response
    except Exception as e:
        return index, f"Error processing prompt {index}: {str(e)}"


# Add this new function for future use
def bedrock_batch_inference(
    model_id, bedrock_client, prompts, system_prompt, aws_account, tone
):
    data_dir = os.path.expanduser("./data")
    input_file = os.path.join(data_dir, "batch_input.jsonl")

    # Create JSONL records
    with open(input_file, "w") as f:
        for idx, prompt in enumerate(prompts):
            record = {
                "recordId": f"RECORD{idx:08d}",
                "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    ],
                },
            }
            f.write(json.dumps(record) + "\n")

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
        modelId=model_id,
        jobName=f"tone-transform-{tone}-{int(time.time())}",
        inputDataConfig=input_config,
        outputDataConfig=output_config,
    )
    return response.get("jobArn")


def get_job_status(bedrock_client, job_arn):
    """Helper function to get batch job status"""
    return bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)["status"]


def get_job_error_details(bedrock_client, job_arn):
    """Helper function to get error details for failed jobs"""
    response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
    return {
        "status": response.get("status"),
        "failure_reason": response.get("failureReason", "No failure reason provided"),
        "creation_time": response.get("creationTime"),
        "completion_time": response.get("completionTime"),
        "input_config": response.get("inputDataConfig"),
        "output_config": response.get("outputDataConfig"),
    }


def invoke_sagemaker_endpoint(
    payload, endpoint_name="Phi-3-5-mini-instruct", region="us-east-1"
):
    try:
        sagemaker_runtime_client = boto3.client("sagemaker-runtime", region_name=region)
        input_string = json.dumps(payload)
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=input_string.encode("utf-8"),
            ContentType="application/json",
        )
        json_output = response["Body"].readlines()
        plain_output = "\n".join(json.loads(json_output[0]))
        last_assistant = extract_last_assistant_response(plain_output)
        print("Test response:", last_assistant)
        return last_assistant
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e


def invoke_ollama_endpoint(payload, endpoint_name, url="127.0.0.1:11434"):
    request_body = {"prompt": payload, "model": endpoint_name}

    api_url = f"http://{url}/api/generate"

    response = requests.post(api_url, json=request_body)

    r = response.text.split("\n")
    lines = []
    for r in r:
        if r != "":
            lines.append(json.loads(r))

    return "".join([l["response"] for l in lines])


def batch_invoke_sagemaker_endpoint(
    payloads,
    endpoint_name,
    region="us-east-1",
    s3_bucket=None,
    s3_input_prefix="/eval/async/input/",
    poll_interval_seconds=10,
    timeout_seconds=600,
):
    """
    Invoke a SageMaker async endpoint for a batch of payloads.

    - payloads: list of JSON-serializable objects (each is one request)
    - endpoint_name: name of the async SageMaker endpoint
    - region: AWS region
    - s3_bucket: S3 bucket to upload inputs (required)
    - s3_input_prefix: S3 prefix for input uploads
    - poll_interval_seconds: interval between checks for output readiness
    - timeout_seconds: max time to wait for each result

    Returns list of raw results (strings) in the same order as payloads.
    """
    if s3_bucket is None:
        raise ValueError("s3_bucket is required for async invocations")
    if not isinstance(s3_bucket, str) or not s3_bucket.strip():
        raise ValueError(
            "s3_bucket must be a non-empty string (e.g., 'my-bucket-name'), got: "
            f"{type(s3_bucket).__name__}"
        )

    sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    # Normalize prefix
    input_prefix = s3_input_prefix.lstrip("/")

    input_locations = []
    output_locations = []
    inference_ids = []

    # 1) Upload all payloads and invoke async endpoint
    for idx, payload in enumerate(payloads):
        print(f"Submitting {idx + 1}/{len(payloads)} to async endpoint '{endpoint_name}'...")
        request_id = str(uuid.uuid4())[:8]
        input_key = f"{input_prefix}batch-{request_id}-{idx}.json"

        # Ensure payload is in expected format for the model container
        if isinstance(payload, str):
            payload_to_upload = {"inputs": payload}
        elif isinstance(payload, list) and all(isinstance(p, str) for p in payload):
            payload_to_upload = {"inputs": payload}
        elif isinstance(payload, dict):
            payload_to_upload = payload
        else:
            # Fallback: wrap unknown types under inputs
            payload_to_upload = {"inputs": payload}

        s3_client.put_object(
            Bucket=s3_bucket,
            Key=input_key,
            Body=json.dumps(payload_to_upload),
            ContentType="application/json",
        )

        input_location = f"s3://{s3_bucket}/{input_key}"
        input_locations.append(input_location)

        response = sagemaker_runtime.invoke_endpoint_async(
            EndpointName=endpoint_name,
            InputLocation=input_location,
            ContentType="application/json",
            InvocationTimeoutSeconds=3600,
        )

        output_locations.append(response["OutputLocation"])  # s3 uri
        inference_ids.append(response.get("InferenceId"))
        print(f"Submitted {idx + 1}/{len(payloads)}. Output will be written to {response['OutputLocation']}")

    # 2) Poll for each output and download results
    results = []
    for i, output_location in enumerate(output_locations):
        start_time = time.time()

        # Parse s3 uri and derive expected result key: <prefix>/<InferenceId>.out
        uri = output_location.replace("s3://", "")
        bucket, key = uri.split("/", 1)
        inference_id = inference_ids[i]
        expected_key = f"{key.rstrip('/')}/{inference_id}.out" if isinstance(inference_id, str) and inference_id else key
        if expected_key != key:
            print(f"Polling for result object s3://{bucket}/{expected_key}")

        while True:
            try:
                # First, check expected result key (InferenceId.out)
                s3_client.head_object(Bucket=bucket, Key=expected_key)
                break
            except Exception:
                if time.time() - start_time > timeout_seconds:
                    print(f"Timed out waiting for result {i + 1}/{len(output_locations)} after {timeout_seconds}s")
                    results.append(None)
                    break
                elapsed = int(time.time() - start_time)
                print(f"Waiting for result {i + 1}/{len(output_locations)}... {elapsed}s elapsed")

                # Try to detect async failure artifact: async-endpoint-failures/.../<InferenceId>-error.out
                if isinstance(inference_id, str) and inference_id:
                    try:
                        candidates = s3_client.list_objects_v2(
                            Bucket=bucket,
                            Prefix="async-endpoint-failures/",
                            MaxKeys=1000,
                        )
                        for obj in candidates.get("Contents", []):
                            k = obj.get("Key", "")
                            if k.endswith(f"{inference_id}-error.out"):
                                err_obj = s3_client.get_object(Bucket=bucket, Key=k)
                                err_text = err_obj["Body"].read().decode("utf-8", errors="replace")
                                print(f"Error for request {i + 1}/{len(output_locations)} (InferenceId={inference_id}):\n{err_text}")
                                results.append(None)
                                # Stop waiting for this one
                                elapsed = int(time.time() - start_time)
                                print(f"Marking request {i + 1} as failed after {elapsed}s due to async failure artifact: s3://{bucket}/{k}")
                                # Break out of the polling loop
                                raise StopIteration
                    except StopIteration:
                        break
                    except Exception:
                        # Ignore listing errors silently and keep polling
                        pass
                time.sleep(poll_interval_seconds)

        if len(results) == 0 or results[-1] is not None:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            result_body = obj["Body"].read().decode("utf-8")
            results.append(result_body)
            total = int(time.time() - start_time)
            print(f"Result ready for {i + 1}/{len(output_locations)} after {total}s")

    return results
