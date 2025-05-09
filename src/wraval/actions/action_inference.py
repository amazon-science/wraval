#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
from dynaconf import Dynaconf
from .data_utils import write_dataset_local, write_dataset_to_s3, load_latest_dataset
from .prompt_tones import get_prompt, Tone
from .model_router import route_completion

def run_inference(
    settings: Dynaconf,
    model_name: str,
    upload_s3: bool,
    data_dir: str
) -> None:
    """Run inference on sentences using the specified model"""
    try:
        d = load_latest_dataset(data_dir)
        print(f"Loaded dataset with {len(d)} rows")
    except FileNotFoundError:
        print("No dataset found. Please generate data first.")
        return

    if "rewrite" not in d.columns:
        d["rewrite"] = None
    if "inference_model" not in d.columns:
        d["inference_model"] = None

    tones = d["tone"].unique()
    print(f"Found tones: {tones}")

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

        for query, output in zip(queries, outputs):
            mask = (d["synthetic_data"] == query) & (d["tone"] == tone)
            cleaned_output = output.strip().strip('"')
            d.loc[mask, "rewrite"] = cleaned_output
            d.loc[mask, "inference_model"] = model_name

    write_dataset_local(d, "./data", "all-tones")
    if upload_s3:
        write_dataset_to_s3(d, settings.s3_bucket, "inference/all", "csv")
