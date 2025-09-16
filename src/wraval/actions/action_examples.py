#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from dynaconf import Dynaconf
from wraval.actions.data_utils import load_latest_dataset
import pandas as pd
from typing import Optional


def get_examples(
    settings: Dynaconf, tone: Optional[str] = None, n_examples: int = 3
) -> None:
    """
    Load the latest dataset and display examples grouped by tone and model.
    """
    try:
        data_location = settings.data_dir
        print(f"Loading data from: {data_location}")
        d = load_latest_dataset(data_location)

        # Columns required for grouping/selection
        required_group_cols = ["tone", "inference_model"]
        missing_group = [c for c in required_group_cols if c not in d.columns]
        if missing_group:
            print(f"Warning: Missing columns {missing_group}. Showing raw examples instead.")
            # Show a few raw examples without grouping
            examples = d.head(n_examples)
            for idx, row in examples.iterrows():
                original = row.get("synthetic_data", "<missing synthetic_data>")
                rewrite = row.get("rewrite", "<missing rewrite>")
                score_val = row.get("overall_score", None)
                if pd.notna(score_val):
                    try:
                        score = f"{float(score_val):.2f}"
                    except Exception:
                        score = str(score_val)
                else:
                    score = "N/A"
                print(f"\nExample {idx + 1}:")
                print(f"Original: {original}")
                print(f"Rewrite:  {rewrite}")
                print(f"Score:    {score}")
                print("-" * 40)
            return

        if tone and tone != "all":
            if tone not in d["tone"].unique():
                print(f"Error: Tone '{tone}' not found in dataset.")
                print(f"Available tones: {', '.join(d['tone'].unique())}")
                return
            d = d[d["tone"] == tone]
            print(f"\nExamples for Tone: {tone}")
        else:
            print("\nExamples by Tone and Model:")

        combinations = d[["tone", "inference_model"]].drop_duplicates()
        for _, (tone_val, model) in combinations.iterrows():
            print("\n" + "=" * 80)
            print(f"Tone: {tone_val} | Model: {model}")
            print("=" * 80)

            examples = d[(d["tone"] == tone_val) & (d["inference_model"] == model)]
            if len(examples) > n_examples:
                examples = examples.sample(n=n_examples, random_state=42)

            for idx, row in examples.iterrows():
                original = row.get("synthetic_data", "<missing synthetic_data>")
                rewrite = row.get("rewrite", "<missing rewrite>")
                score_val = row.get("overall_score", None)
                if pd.notna(score_val):
                    try:
                        score = f"{float(score_val):.2f}"
                    except Exception:
                        score = str(score_val)
                else:
                    score = "N/A"
                print(f"\nExample {idx + 1}:")
                print(f"Original: {original}")
                print(f"Rewrite:  {rewrite}")
                print(f"Score:    {score}")
                print("-" * 40)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate and judge data first.")
    except Exception as e:
        # Catch-all that logs and keeps the program alive
        print(f"Unexpected error: {e}")
