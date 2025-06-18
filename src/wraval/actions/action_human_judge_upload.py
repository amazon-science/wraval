#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
from .data_utils import write_dataset, load_latest_dataset, latest_file_name
import os
import numpy as np

def upload_human_judge(settings):

    results = load_latest_dataset(settings.data_dir)
    
    # Filter by inference models if specified
    if hasattr(settings, 'inference_models') and settings.inference_models:
        results = results[results['inference_model'].isin(settings.inference_models)]
    
    results.rename(columns={
        "synthetic_data": "original", 
        "rewrite": "gen"}, 
        inplace=True)

    if results.shape[0] < settings.n_samples:
        print(f"Requested {settings.n_samples} samples, but only {results.shape[0]} available. The entire dataset will be used for human evaluation.")
        write_dataset(results, settings.data_dir, "all", "jsonl")
        return

    # Get group sizes to understand the distribution
    grouped = results.groupby(['tone', 'inference_model'])
    group_sizes = grouped.size()
    
    print(f"Available samples per tone-model combination:")
    print(group_sizes)
    print(f"\nTotal groups: {len(group_sizes)}")
    print(f"Requested samples: {settings.n_samples}")
    
    # Calculate proportional sampling
    total_available = group_sizes.sum()
    if total_available < settings.n_samples:
        print(f"Warning: Only {total_available} samples available, but {settings.n_samples} requested.")
        settings.n_samples = total_available
    
    # Calculate proportional samples per group
    proportional_samples = (group_sizes / total_available * settings.n_samples).round().astype(int)
    
    # Ensure we don't exceed available samples in each group
    proportional_samples = proportional_samples.clip(upper=group_sizes)
    
    # Adjust to reach exactly n_samples
    current_total = proportional_samples.sum()
    if current_total < settings.n_samples:
        # Add samples to groups that have more available
        remaining = settings.n_samples - current_total
        available_for_additional = group_sizes - proportional_samples
        additional_samples = available_for_additional.nlargest(remaining).index
        
        for idx in additional_samples:
            if proportional_samples[idx] < group_sizes[idx]:
                proportional_samples[idx] += 1
                remaining -= 1
                if remaining == 0:
                    break
    
    print(f"\nSampling plan:")
    for (tone, model), samples in proportional_samples.items():
        print(f"  {tone} + {model}: {samples} samples")
    
    # Perform the sampling
    sampled_results = []
    for (tone, model), n_samples in proportional_samples.items():
        if n_samples > 0:
            group_data = results[(results['tone'] == tone) & (results['inference_model'] == model)]
            sampled_group = group_data.sample(n=min(n_samples, len(group_data)), random_state=42)
            sampled_results.append(sampled_group)
    
    if sampled_results:
        sampled_results = pd.concat(sampled_results, ignore_index=True)
    else:
        sampled_results = pd.DataFrame()
    
    print(f"\nFinal sample size: {len(sampled_results)}")
    
    # Create output file
    file_name = latest_file_name(settings.data_dir).replace('.csv', '.jsonl').split('/')[-1]
    path = os.path.join(settings.data_dir, file_name)
    
    sampled_results['source'] = sampled_results['original']
    
    sampled_results[['source', 'original', 'gen', 'uuid', 'tone', 'inference_model']].to_json(
        path, orient='records', lines=True
    )
    
    print(f"Sampled dataset saved to: {path}")