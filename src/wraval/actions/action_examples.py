#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from dynaconf import Dynaconf
from wraval.actions.data_utils import load_latest_dataset
import pandas as pd
from typing import Optional


def get_examples(settings: Dynaconf, tone: Optional[str] = None, n_examples: int = 3) -> None:
    """
    Load the latest dataset and display examples grouped by tone and model.
    
    Args:
        settings: Dynaconf settings object with data_dir setting
        tone: Optional tone to filter by
        n_examples: Number of examples to show per tone-model combination
    """
    try:
        # Use settings.data_dir which could be either local path or S3 URI
        data_location = settings.data_dir
        print(f"Loading data from: {data_location}")
        d = load_latest_dataset(data_location)
        
        if tone and tone != "all":
            if tone not in d['tone'].unique():
                print(f"Error: Tone '{tone}' not found in dataset.")
                print(f"Available tones: {', '.join(d['tone'].unique())}")
                return
            d = d[d['tone'] == tone]
            print(f"\nExamples for Tone: {tone}")
        else:
            print("\nExamples by Tone and Model:")
        
        # Get unique combinations of tone and inference_model
        combinations = d[['tone', 'inference_model']].drop_duplicates()
        
        for _, (tone, model) in combinations.iterrows():
            print("\n" + "=" * 80)
            print(f"Tone: {tone} | Model: {model}")
            print("=" * 80)
            
            # Get examples for this tone-model combination
            examples = d[(d['tone'] == tone) & (d['inference_model'] == model)]
            
            # Sample n_examples if we have more
            if len(examples) > n_examples:
                examples = examples.sample(n=n_examples, random_state=42)
            
            # Display each example
            for idx, row in examples.iterrows():
                print(f"\nExample {idx + 1}:")
                print(f"Original: {row['synthetic_data']}")
                print(f"Rewrite:  {row['rewrite']}")
                print(f"Score:    {row['overall_score']:.2f}")
                print("-" * 40)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate and judge data first.")
    except KeyError as e:
        print(f"Error: Missing required column {e}. Please ensure the dataset has been properly judged.")
    except Exception as e:
        print(f"Unexpected error: {e}") 