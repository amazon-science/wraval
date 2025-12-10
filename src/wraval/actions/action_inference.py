#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
from dynaconf import Dynaconf
from .data_utils import write_dataset, load_latest_dataset
from .model_router import route_completion


def extract_language_from_path(data_dir: str) -> str:
    """Extract language code from data_dir path.
    
    Examples:
        's3://.../eval/sync/de/tones/' -> 'de'
        's3://.../eval/sync/jp/tones/' -> 'jp'
        './data/' -> None
    """
    import re
    # Look for pattern like /de/, /jp/, /en_us/ in path
    match = re.search(r'/([a-z]{2}(?:_[a-z]{2})?)/tones', data_dir)
    return match.group(1) if match else None


def get_prompt_functions(settings: Dynaconf):
    """Get the appropriate prompt functions based on settings."""
    from .prompt_loader import get_prompt, Tone, get_commit_hash
    return get_prompt, Tone, get_commit_hash


def run_inference(
    settings: Dynaconf, model_name: str, upload_s3: bool, data_dir: str
) -> None:
    """Run inference on sentences using the specified model"""
    get_prompt, Tone, get_commit_hash = get_prompt_functions(settings)
    results = load_latest_dataset(data_dir)

    # Extract language from data_dir path
    language = extract_language_from_path(data_dir)
    
    # Get commit hash from prompt metadata
    commit_hash = get_commit_hash(language=language, custom_prompts=settings.custom_prompts)

    no_rewrite = False

    if "rewrite" not in results.columns:
        if "inference_model" not in results.columns:
            no_rewrite = True
            results["rewrite"] = None
            results["inference_model"] = None
            if commit_hash:
                results["prompt_commit"] = None

    tones = results["tone"].unique()
    print(f"Found tones: {tones}")

    if settings.type != "all":
        tones = [settings.type]

    for tone in tones:
        print(
            f"""
        ---------------------
        {tone}
        ---------------------
        """
        )

        tone_prompt = get_prompt(Tone(tone), language=language, custom_prompts=settings.custom_prompts)

        queries = results[results["tone"] == tone]["synthetic_data"].unique()

        print(f"Processing {len(queries)} inputs for tone: {tone}")

        outputs = route_completion(settings, queries, tone_prompt)
        
        cleaned_output = [o.strip().strip('"') for o in outputs]
        
        if no_rewrite:
            mask = results["tone"] == tone
            results.loc[mask, "rewrite"] = cleaned_output
            results.loc[mask, "inference_model"] = model_name
            if commit_hash:
                results.loc[mask, "prompt_commit"] = commit_hash
        else:
            new_results = pd.DataFrame(
                {"synthetic_data": results[results["tone"] == tone]["synthetic_data"].unique()}
            )
            new_results["tone"] = tone
            new_results["rewrite"] = cleaned_output
            new_results["inference_model"] = model_name
            if commit_hash:
                new_results["prompt_commit"] = commit_hash
            results = pd.concat([results, new_results], ignore_index=True)

    write_dataset(results, data_dir, "all", "csv")
