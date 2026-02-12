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


def get_results(settings: Dynaconf, tone: Optional[str] = None, export_xlsx: bool = False) -> None:
    """
    Load the latest dataset and display normalized results table grouped by tone.

    Args:
        settings: Dynaconf settings object with data_dir setting
        tone: Optional tone to filter by
    """
    try:
        # Configure pandas to show all rows and columns
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

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
            print(f"\nResults Table for Tone: {tone_str}")
        else:
            print("\nResults Table by Tone:")

        print("=" * 50)

        # Check for language confidence columns (e.g., EN, DE) and binary (is_EN, is_DE)
        lang_cols = [col for col in d.columns if col.isupper() and len(col) == 2]
        is_lang_cols = [col for col in d.columns if col.startswith("is_") and len(col) == 5]

        # Compute net_score if language detection exists (score * is_lang)
        if is_lang_cols:
            # Use first is_lang column (e.g., is_EN)
            is_col = is_lang_cols[0]
            d["net_score"] = d["overall_score"] * d[is_col]
            d.loc[d["net_score"] == 0, "net_score"] = 1  # Avoid 0 scores

        # Count non-empty rewrites
        d["has_rewrite"] = d["rewrite"].notna() & (d["rewrite"] != "")

        # Build columns to aggregate
        agg_cols = {"overall_score": "mean", "has_rewrite": "sum"}
        if is_lang_cols:
            agg_cols["net_score"] = "mean"
        for col in lang_cols:
            agg_cols[col] = "mean"
        for col in is_lang_cols:
            agg_cols[col] = "mean"

        # Check if prompt_commit column exists
        group_cols = ["inference_model", "tone"]
        if "prompt_commit" in d.columns:
            # Fill NaN values with "no_commit" to include rows without commit hash
            d["prompt_commit"] = d["prompt_commit"].fillna("no_commit")
            group_cols.append("prompt_commit")

        # Group by model, tone, and optionally commit hash
        grouped = d.groupby(group_cols, dropna=False).agg(agg_cols)
        grouped = grouped.rename(columns={"has_rewrite": "n_samples"})
        grouped["n_samples"] = grouped["n_samples"].astype(int)

        # Normalize overall_score to 0-100 scale
        grouped["overall_score"] = normalize_scores(grouped["overall_score"])
        if "net_score" in grouped.columns:
            grouped["net_score"] = normalize_scores(grouped["net_score"])

        # Convert language confidence to percentage
        for col in lang_cols:
            grouped[col] = grouped[col] * 100
        for col in is_lang_cols:
            grouped[col] = grouped[col] * 100

        # Display results rounded to 2 decimal places
        print(grouped.round(2))
        print("=" * 50)
        print("\nNote: overall_score normalized to 0-100 (0=poor, 100=excellent)")
        if lang_cols:
            print(f"      {', '.join(lang_cols)} = language confidence %")
        if is_lang_cols:
            print(f"      {', '.join(is_lang_cols)} = % detected as that language")

        # Export to xlsx and open if requested
        if export_xlsx:
            import subprocess
            from datetime import datetime
            import os

            results_dir = os.path.join("data", "results_summary")
            os.makedirs(results_dir, exist_ok=True)

            filename = os.path.join(
                results_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )

            # Use net_score if available, otherwise overall_score
            score_col = "net_score" if "net_score" in grouped.columns else "overall_score"

            # Reset index for pivoting
            reset_df = grouped.reset_index()
            
            # Filter out rows with NaN inference_model or tone
            reset_df = reset_df.dropna(subset=["inference_model", "tone"])
            
            if len(reset_df) == 0:
                print("No valid data to export")
                return
            
            # Check if prompt_commit exists in the data
            has_commit = "prompt_commit" in reset_df.columns
            
            if has_commit:
                # Pivot with commit: create multi-index with model+commit as rows
                reset_df["model_commit"] = reset_df["inference_model"] + " (" + reset_df["prompt_commit"] + ")"
                pivot = reset_df.pivot(
                    index="model_commit", columns="tone", values=score_col
                )
            else:
                # Pivot: tones as columns, models as rows
                pivot = reset_df.pivot(
                    index="inference_model", columns="tone", values=score_col
                )

            # Rename and reorder columns
            tone_order = [
                "emojify", "shorten", "summarize", "professional", "witty",
                "casual", "elaborate", "improve", "keypoints", "proofread"
            ]
            tone_rename = {
                "emojify": "Emojify",
                "shorten": "Shorten/summarize",
                "summarize": "Shorten/summarize",
                "professional": "Professional",
                "witty": "Witty",
                "casual": "Casual",
                "elaborate": "Elaborate",
                "improve": "Improve",
                "keypoints": "Keypoints",
                "proofread": "Proofread",
            }

            # Rename columns - handle NaN safely
            pivot.columns = [
                tone_rename.get(col, col.capitalize() if isinstance(col, str) else str(col)) 
                for col in pivot.columns
            ]

            # Reorder columns (only include existing ones)
            ordered_cols = ["Emojify", "Shorten/summarize", "Professional", "Witty",
                           "Casual", "Elaborate", "Improve", "Keypoints", "Proofread"]
            existing_cols = [c for c in ordered_cols if c in pivot.columns]
            pivot = pivot[existing_cols]

            # Round tone columns to 1 decimal
            tone_cols = [c for c in pivot.columns]
            pivot[tone_cols] = pivot[tone_cols].round(1)

            # Add average column (2 decimals)
            pivot["Average"] = pivot[tone_cols].mean(axis=1).round(2)

            # Convert to string with dot decimal separator
            for col in tone_cols:
                pivot[col] = pivot[col].apply(lambda x: f"{x:.1f}")
            pivot["Average"] = pivot["Average"].apply(lambda x: f"{x:.2f}")

            pivot.to_excel(filename)
            print(f"\nExported to: {filename}")
            subprocess.run(["open", filename])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate and judge data first.")
    except KeyError as e:
        print(
            f"Error: Missing required column {e}. Please ensure the dataset has been properly judged."
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
