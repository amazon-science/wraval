from typing import Optional, Tuple
import boto3
import pandas as pd
from src.completion import get_completion
from src.prompt_tones import Tone, get_prompt

def generate_dataset(
    model: str, 
    tone: Tone,
    bedrock_client: Optional[boto3.client] = None
) -> Tuple[str, pd.DataFrame]:
    prompt = get_prompt(tone)
    raw_output = get_completion(model, bedrock_client, prompt)
    df = process_raw_output(raw_output, tone)
    return raw_output, df 

def process_raw_output(output: str, tone: Tone) -> pd.DataFrame:
    """Process raw LLM output into a pandas DataFrame"""
    if tone == Tone.SUMMARIZE:
        # Split on newlines and filter empty lines
        pairs = [line.strip().split('|||') for line in output.split('\n') 
                if '|||' in line and line.strip()]
        return pd.DataFrame(pairs, columns=['synthetic_data', 'summary'])
    else:
        # Split on newlines and filter empty lines
        sentences = [line.strip() for line in output.split('\n') 
                    if line.strip()]
        return pd.DataFrame(sentences, columns=['synthetic_data'])
