from src.completion import batch_get_completions, get_completion, invoke_sagemaker_endpoint, invoke_ollama_endpoint
from src.format import format_prompt_as_xml, format_prompt
import boto3
from transformers import AutoTokenizer
from tqdm import tqdm

def route_completion(settings, queries, master_sys_prompt=None):

    if isinstance(queries, str):
        queries = [queries]

    if settings.endpoint_type == "bedrock":
        n = len(queries)
        bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=settings.region)
        prompts = [format_prompt(text, type="bedrock") for text in queries]
        m = settings.model.format(aws_account=settings.aws_account)
        if len(prompts) == 1:
            outputs = get_completion(settings, prompts[0])
        else:
            outputs = batch_get_completions(settings, prompts, [master_sys_prompt] * n)

    else: # local
        client = None  # Not needed for SageMaker endpoint
        if settings.local_tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(
                settings.local_tokenizer_path
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                settings.hf_name, trust_remote_code=True
            )
        prompts = [
            format_prompt(text, tokenizer=tokenizer, type="hf") for text in queries
        ]

        print(f"Sample prompt:\n{prompts[0]}")
        
        if settings.endpoint_type == "sagemaker":
        
            outputs = [
                invoke_sagemaker_endpoint({"inputs": prompt})
                for prompt in tqdm(prompts)
            ]
    
        elif settings.endpoint_type == "ollama":
        
            outputs = [
                invoke_ollama_endpoint(prompt)
                for prompt in tqdm(prompts)
            ]

    return outputs
