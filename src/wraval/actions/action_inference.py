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
    d = load_latest_dataset(data_dir)

    if "rewrite" not in d.columns:
        d["rewrite"] = None
    if "inference_model" not in d.columns:
        d["inference_model"] = None

    tones = d["tone"].unique()
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

        queries = d[d["tone"] == tone]["synthetic_data"].unique()

        print(f"Processing {len(queries)} unique inputs for tone: {tone}")

        outputs = route_completion(settings, queries, tone_prompt)

        cleaned_output = [o.strip().strip('"') for o in outputs]
        new = pd.DataFrame({"synthetic_data" : queries, "tone" : tone})
        new["rewrite"] = cleaned_output
        new["inference_model"] = model_name

        d = pd.concat([d, new], ignore_index=True)

    write_dataset(d, data_dir, "all", "csv")