from typing import Optional, Tuple
import boto3
import pandas as pd
from src.completion import get_completion
from prompts.data_generation_prompts import (
    WITTY_SENTENCES_PROMPT,
    PROFESSIONAL_SENTENCES_PROMPT,
    PARAGRAPH_SUMMARY_PROMPT,
    CASUAL_SENTENCES_PROMPT,
    ELABORATE_SENTENCES_PROMPT,
    SHORTEN_SENTENCES_PROMPT,
    IMPROVE_SENTENCES_PROMPT,
    KEYPOINTS_SENTENCES_PROMPT,
    PROOFREAD_SENTENCES_PROMPT,
    EMOJIFY_SENTENCES_PROMPT
)

# Map dataset types to their corresponding prompts
PROMPT_MAP = {
    "witty_sentences": WITTY_SENTENCES_PROMPT,
    "professional_sentences": PROFESSIONAL_SENTENCES_PROMPT,
    "paragraph_summary": PARAGRAPH_SUMMARY_PROMPT,
    "casual_sentences": CASUAL_SENTENCES_PROMPT,
    "elaborate_sentences": ELABORATE_SENTENCES_PROMPT,
    "shorten_sentences": SHORTEN_SENTENCES_PROMPT,
    "improve_sentences": IMPROVE_SENTENCES_PROMPT,
    "keypoints_sentences": KEYPOINTS_SENTENCES_PROMPT,
    "proofread_sentences": PROOFREAD_SENTENCES_PROMPT,
    "emojify_sentences": EMOJIFY_SENTENCES_PROMPT
}

def process_raw_output(output: str, dataset_type: str) -> pd.DataFrame:
    """Process raw LLM output into a pandas DataFrame"""
    if dataset_type == "paragraph_summary":
        # Split on newlines and filter empty lines
        pairs = [line.strip().split('|||') for line in output.split('\n') 
                if '|||' in line and line.strip()]
        return pd.DataFrame(pairs, columns=['synthetic_data', 'summary'])
    else:
        # Split on newlines and filter empty lines
        sentences = [line.strip() for line in output.split('\n') 
                    if line.strip()]
        return pd.DataFrame(sentences, columns=['synthetic_data'])

def generate_dataset(model: str, 
                    bedrock_client: Optional[boto3.client] = None,
                    dataset_type: str = "witty_sentences") -> Tuple[str, pd.DataFrame]:
    """
    Generate dataset based on the specified type
    
    Args:
        model: The model to use for generation
        bedrock_client: Optional bedrock client for AWS models
        dataset_type: Type of dataset to generate. Must be one of:
            - witty_sentences
            - professional_sentences
            - paragraph_summary
            - casual_sentences
            - elaborate_sentences
            - shorten_sentences
            - improve_sentences
            - keypoints_sentences
            - proofread_sentences
            - emojify_sentences
    
    Returns:
        Tuple[str, pd.DataFrame]: Raw output string and processed DataFrame
        
    Raises:
        ValueError: If dataset_type is not recognized
    """
    if dataset_type not in PROMPT_MAP:
        valid_types = "\n- ".join(PROMPT_MAP.keys())
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}\n"
            f"Must be one of:\n- {valid_types}"
        )
    
    prompt = PROMPT_MAP[dataset_type]
    raw_output = get_completion(model, bedrock_client, prompt)
    df = process_raw_output(raw_output, dataset_type)
    
    return raw_output, df 