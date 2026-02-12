#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from dynaconf import Dynaconf
from wraval.actions.data_utils import load_latest_dataset
import pandas as pd
from typing import Optional


def get_examples(
    settings: Dynaconf, tone: Optional[str] = None, n_examples: int = 3, random_seed: int = 42
) -> None:
    """
    Load the latest dataset and display examples grouped by tone and model.

    Args:
        settings: Dynaconf settings object with data_dir setting
        tone: Optional tone to filter by
        n_examples: Number of examples to show per tone-model combination
        random_seed: Random seed for sampling (default: 42)
    """
    try:
        # Use settings.data_dir which could be either local path or S3 URI
        data_location = settings.data_dir
        print(f"Loading data from: {data_location}")
        d = load_latest_dataset(data_location)

        # Convert enum to string value if needed
        tone_str = tone.value if hasattr(tone, "value") else tone

        if tone_str and tone_str != "all":
            if tone_str not in d["tone"].unique():
                print(f"Error: Tone '{tone_str}' not found in dataset.")
                print(f"Available tones: {', '.join(d['tone'].unique())}")
                return
            d = d[d["tone"] == tone_str]
            print(f"\nExamples for Tone: {tone_str}")
        else:
            print("\nExamples by Tone and Model:")

        # Check for language confidence columns (e.g., EN, DE) and binary (is_EN, is_DE)
        lang_cols = [col for col in d.columns if col.isupper() and len(col) == 2]
        is_lang_cols = [col for col in d.columns if col.startswith("is_") and len(col) == 5]

        # Check if prompt_commit column exists
        group_cols = ["tone", "inference_model"]
        if "prompt_commit" in d.columns:
            d["prompt_commit"] = d["prompt_commit"].fillna("no_commit")
            group_cols.append("prompt_commit")

        # Get unique combinations
        combinations = d[group_cols].drop_duplicates()

        for _, row in combinations.iterrows():
            tone = row["tone"]
            model = row["inference_model"]
            commit = row.get("prompt_commit", None)
            
            print("\n" + "=" * 80)
            if commit:
                print(f"Tone: {tone} | Model: {model} | Commit: {commit}")
            else:
                print(f"Tone: {tone} | Model: {model}")
            print("=" * 80)

            # Get examples for this combination
            mask = (d["tone"] == tone) & (d["inference_model"] == model)
            if commit:
                mask = mask & (d["prompt_commit"] == commit)
            examples = d[mask]

            # Sample n_examples if we have more
            if len(examples) > n_examples:
                examples = examples.sample(n=n_examples, random_state=random_seed)

            # Display each example
            for i, (idx, row) in enumerate(examples.iterrows(), 1):
                print(f"\nExample {i}:")
                print(f"Original: {row['synthetic_data']}")
                print(f"Rewrite:  {row['rewrite']}")
                score_str = f"Score: {row['overall_score']:.2f}"
                # Add language confidence if available
                for col in lang_cols:
                    if col in row and pd.notna(row[col]):
                        score_str += f" | {col}: {row[col]*100:.1f}%"
                # Add binary detection if available
                for col in is_lang_cols:
                    if col in row and pd.notna(row[col]):
                        label = "✓" if row[col] == 1 else "✗"
                        score_str += f" | {col}: {label}"
                print(score_str)
                print("-" * 40)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate and judge data first.")
    except KeyError as e:
        print(
            f"Error: Missing required column {e}. Please ensure the dataset has been properly judged."
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
