from .completion import (
    batch_get_bedrock_completions,
    invoke_sagemaker_endpoint,
    invoke_ollama_endpoint,
)
from .format import format_prompt
from .dspy_provider import build_dspy_llm
from .dspy_programs import build_dspy_program
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

    def get_completion(self, queries: List[str]) -> List[str]:
        prompts = [
            format_prompt(text, self.master_sys_prompt, self.tokenizer, type="hf")
            for text in queries
        ]
        return [
            invoke_sagemaker_endpoint({"inputs": prompt}, self.model_name) for prompt in tqdm(prompts)
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


class DSpyRouter(ModelRouter):
    def __init__(self, master_sys_prompt, settings):
        super().__init__(master_sys_prompt)
        self.settings = settings
        try:
            import dspy
        except Exception as e:  # pragma: no cover
            raise RuntimeError("dspy is not installed; please add dspy-ai to requirements") from e

        self._dspy = dspy
        self.llm = build_dspy_llm(settings)
        self.program = build_dspy_program(settings, self.llm)
        # Use new settings API per upstream guidance
        if hasattr(self._dspy, "settings") and hasattr(self._dspy.settings, "configure"):
            self._dspy.settings.configure(lm=self.llm)
        else:
            # Fallback for older versions
            self._dspy.configure(llm=self.llm)
        # Minimal visibility into DSPy configuration
        try:
            provider = getattr(settings, "dspy_provider", "unknown")
            model = getattr(settings, "dspy_model", getattr(settings, "model", "unknown"))
            print(f"[DSPy] Configured LM via dspy.LM -> provider={provider} model={model}")
        except Exception:
            pass

    def get_completion(self, queries: List[str]) -> List[str]:
        print(f"[DSPy] Executing DSPy program on {len(queries)} queries")
        prompts = [
            format_prompt(text, self.master_sys_prompt, type="bedrock")
            for text in queries
        ]
        # For DSPy, we want a single string, not structured bedrock messages.
        normalized_prompts = []
        for p in prompts:
            if isinstance(p, list):
                # collapse bedrock-style messages into text
                chunks = []
                for msg in p:
                    if isinstance(msg, dict) and "content" in msg and msg["content"]:
                        for block in msg["content"]:
                            if isinstance(block, dict) and "text" in block:
                                chunks.append(block["text"])
                merged = "\n\n".join(chunks)
                if self.master_sys_prompt:
                    merged = f"{self.master_sys_prompt}\n\n{merged}"
                normalized_prompts.append(merged)
            else:
                text = str(p)
                if self.master_sys_prompt:
                    text = f"{self.master_sys_prompt}\n\n{text}"
                normalized_prompts.append(text)

        outputs: List[str] = []
        for p in normalized_prompts:
            result = self.program(p)
            if hasattr(result, "output"):
                outputs.append(result.output)
            else:
                outputs.append(str(result))
        return outputs


def route_completion(
    settings: LazySettings, queries: List[str], master_sys_prompt=None
) -> List[str]:
    return get_router(settings, master_sys_prompt).get_completion(queries)


def get_router(settings: LazySettings, master_sys_prompt: str) -> ModelRouter:
    match settings.endpoint_type:
        case "dspy":
            return DSpyRouter(master_sys_prompt, settings)
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
