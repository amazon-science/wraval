from .completion import (
    batch_get_bedrock_completions,
    invoke_sagemaker_endpoint,
    invoke_ollama_endpoint,
    batch_invoke_sagemaker_endpoint,
)
from .format import format_prompt
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from typing import List
from dynaconf.base import LazySettings
from abc import ABC, abstractmethod


class ModelRouter(ABC):
    def __init__(self, master_sys_prompt):
        self.master_sys_prompt = master_sys_prompt

    @abstractmethod
    def get_completion(self, queries: List[str]) -> List[str]:
        pass


class HuggingFaceModelRouter(ModelRouter, ABC):
    def __init__(self, master_sys_prompt, settings):
        super().__init__(master_sys_prompt)
        self.tokenizer = get_tokenizer(settings)


class OllamaRouter(HuggingFaceModelRouter):
    def __init__(self, master_sys_prompt, settings):
        super().__init__(master_sys_prompt, settings)
        self.model_name = settings.model

    def get_completion(self, queries: List[str]) -> List[str]:
        prompts = [
            format_prompt(text, self.master_sys_prompt, self.tokenizer, type="hf")
            for text in queries
        ]
        return [
            invoke_ollama_endpoint(prompt, self.model_name) for prompt in tqdm(prompts)
        ]


class SageMakerRouter(HuggingFaceModelRouter):
    def __init__(self, master_sys_prompt, settings):
        super().__init__(master_sys_prompt, settings)
        self.model_name = settings.model
        self.region = settings.region
        self.thinking = None
        if settings.exists('thinking'):
            self.thinking = settings.thinking
        self.async_config = False
        if settings.exists('asynchronous'):
            self.async_config = settings.asynchronous
        self.deploy_bucket_name = settings.deploy_bucket_name

    def get_completion(self, queries: List[str]) -> List[str]:
        prompts = [
            format_prompt(text, self.master_sys_prompt, self.tokenizer, "hf", self.thinking)
            for text in queries
        ]
        if self.async_config:
            return batch_invoke_sagemaker_endpoint(prompts, 
                                                   self.model_name, 
                                                   self.region, 
                                                   self.deploy_bucket_name)
        return [
            invoke_sagemaker_endpoint({"inputs": prompt}, 
                                      self.model_name, 
                                      self.region) 
                                      for prompt in tqdm(prompts)
        ]


class BedrockModelRouter(ModelRouter):
    def __init__(self, master_sys_prompt, settings):
        super().__init__(master_sys_prompt)
        self.settings = settings

    def get_completion(self, queries: List[str]) -> List[str]:
        prompts = [
            format_prompt(text, self.master_sys_prompt, type="bedrock")
            for text in queries
        ]

        return batch_get_bedrock_completions(
            self.settings, prompts, [self.master_sys_prompt] * len(prompts)
        )


def route_completion(
    settings: LazySettings, queries: List[str], master_sys_prompt=None
) -> List[str]:
    return get_router(settings, master_sys_prompt).get_completion(queries)


def get_router(settings: LazySettings, master_sys_prompt: str) -> ModelRouter:
    match settings.endpoint_type:
        case "bedrock":
            return BedrockModelRouter(master_sys_prompt, settings)
        case "sagemaker":
            return SageMakerRouter(master_sys_prompt, settings)
        case "ollama":
            return OllamaRouter(master_sys_prompt, settings)
        case _:
            raise ValueError(f"Unknown endpoint type: {settings.endpoint_type}")


def get_tokenizer(settings: LazySettings):
    if settings.get("local_tokenizer_path"):
        return AutoTokenizer.from_pretrained(settings.local_tokenizer_path)
    return AutoTokenizer.from_pretrained(settings.hf_name, trust_remote_code=True)
