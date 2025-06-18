#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from dynaconf import Dynaconf
from wraval.actions.data_utils import load_latest_dataset
import pandas as pd
from typing import Optional


def normalize_scores(d: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize scores from 1-3 scale to 0-100 scale.
    
    Args:
        d: DataFrame containing scores on a 1-3 scale
        
    Returns:
        DataFrame with scores normalized to 0-100 scale
    """
    return 100 * (d - 1) / 2


def get_results(settings: Dynaconf, tone: Optional[str] = None) -> None:
    """
    Load the latest dataset and display normalized results table grouped by tone.
    
    Args:
        settings: Dynaconf settings object with data_dir setting
        tone: Optional tone to filter by
    """
    try:
        # Configure pandas to show all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
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
            print(f"\nResults Table for Tone: {tone}")
        else:
            print("\nResults Table by Tone:")
        
        print("=" * 50)
        
        # Group by model, inference model, and tone, calculate mean of overall_score
        grouped = d.groupby(['inference_model', 'tone'])['overall_score'].mean()
        
        # Normalize scores to 0-100 scale
        normalized = normalize_scores(grouped)
        
        # Display results rounded to 2 decimal places
        print(normalized.round(2))
        print("=" * 50)
        print("\nNote: Scores are normalized to 0-100 scale (0=poor, 100=excellent)")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate and judge data first.")
    except KeyError as e:
        print(f"Error: Missing required column {e}. Please ensure the dataset has been properly judged.")
    except Exception as e:
        print(f"Unexpected error: {e}") 