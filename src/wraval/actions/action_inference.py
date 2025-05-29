#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
from dynaconf import Dynaconf
from .data_utils import write_dataset, load_latest_dataset
from .prompt_tones import get_prompt, Tone
from .model_router import route_completion

def run_inference(
    settings: Dynaconf,
    model_name: str,
    upload_s3: bool,
    data_dir: str
) -> None:
    """Run inference on sentences using the specified model"""
    results = load_latest_dataset(data_dir)

    if "rewrite" not in results.columns:
        results["rewrite"] = None
    if "inference_model" not in results.columns:
        results["inference_model"] = None

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

        tone_prompt = get_prompt(Tone(tone))

        queries = results[results["tone"] == tone]["synthetic_data"].unique()

        print(f"Processing {len(queries)} unique inputs for tone: {tone}")

        outputs = route_completion(settings, queries, tone_prompt)

        cleaned_output = [o.strip().strip('"') for o in outputs]
        new_results = pd.DataFrame({"synthetic_data" : queries, "tone" : tone})
        new_results["rewrite"] = cleaned_output
        new_results["inference_model"] = model_name

        results = pd.concat([results, new_results], ignore_index=True)

    write_dataset(results, data_dir, "all", "csv")