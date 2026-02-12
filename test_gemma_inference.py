#!/usr/bin/env python
"""Quick test script for Gemma model inference."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3n-E4B-it-litert-preview"

def load_model(model_id: str):
    """Load model without quantization (Gemma 3n has issues with bitsandbytes)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def predict(model, tokenizer, prompt: str) -> str:
    """Run single prediction."""
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = load_model(MODEL_ID)
    
    prompt = "Make this text witty: My flight lands at 4pm"
    print(f"\nPrompt: {prompt}")
    
    response = predict(model, tokenizer, prompt)
    print(f"\nResponse: {response}")
