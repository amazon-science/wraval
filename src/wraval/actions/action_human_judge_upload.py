#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
from .data_utils import write_dataset, load_latest_dataset, latest_file_name
import os

def upload_human_judge(settings):

    results = load_latest_dataset(settings.data_dir)
    results.rename(columns={
        "synthetic_data": "original", 
        "rewrite": "gen"}, 
        inplace=True)

    if results.shape[0] < settings.n_samples:
        print(f"Requested {settings.n_samples} samples, but only {results.shape[0]} available. The entire dataset will be used for human evaluation.")
        write_dataset(results, settings.data_dir, "all", "jsonl")
        return

    unique_tones = results['tone'].nunique()
    unique_models = results['inference_model'].nunique()
    samples_per_group = settings.n_samples // (unique_tones * unique_models)

    grouped = results.groupby(['tone', 'inference_model'])

    sampled_results = grouped.apply(
        lambda x: x.sample(n=min(len(x), samples_per_group), random_state=42)
    ).reset_index(drop=True)

    leftover_samples = settings.n_samples - len(sampled_results)
    if leftover_samples > 0:
        remaining_data = results[~results.index.isin(sampled_results.index)]
        additional_samples = remaining_data.sample(n=leftover_samples, random_state=42)
        sampled_results = pd.concat([sampled_results, additional_samples])

    
    file_name = latest_file_name(settings.data_dir).replace('.csv', '.jsonl')

    path = os.path.join(settings.data_dir, file_name)

    sampled_results[['original', 'gen']].to_json(
        path, orient='records', lines=True
    )