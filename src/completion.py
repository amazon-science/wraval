#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import time
import random
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_completion(modelId, bedrock_client, prompt, system_prompt=None, max_retries=3, initial_delay=1):
    for attempt in range(max_retries):
        try:
            inference_config = {
                "temperature": 0.0,
                "maxTokens": 3000
            }
            
            if 'nova' not in modelId:
                additional_model_fields = {
                    "top_p": 1,
                }
                converse_api_params = {
                    "modelId": modelId,
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": inference_config,
                    "additionalModelRequestFields": additional_model_fields
                }
            else:
                converse_api_params = {
                    "modelId": modelId,
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": inference_config
                }        
            
            if system_prompt:
                converse_api_params["system"] = [{"text": system_prompt}]

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
