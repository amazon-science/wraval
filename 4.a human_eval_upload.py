#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import itertools
import numpy as np

### formatting all csv files to the AWS Groundtruth accepted format (manifest.jsonl)

tone_types = ['casual', 'elaborate', 'emojify', 'improve', 'keypoints', 'professional', 'proofread', 'shorten', 'witty']
model_types = ['haiku', 'qwen', 'phi']

dataset_pieces = []

for tone in tone_types:
    for model in model_types:
        csv_data = pd.read_csv('data/tones/' + tone + '_' + model + '.csv')
        subset = csv_data[['input', 'output']]
        subset['tone'] = tone
        subset['model'] = model
        dataset_pieces.append(subset)

combined_dataset = pd.concat(dataset_pieces, axis=0, ignore_index=True)

combined_dataset.rename(columns={'input': 'source', 'output': 'gen'}, inplace=True)
combined_dataset['original'] = combined_dataset['source']

print(combined_dataset[:4].to_markdown())

combined_dataset.to_json('s3://slm-benchmarking/tones/annotate/manifest.jsonl', orient="records", lines=True)

###### Gather the entire dataset into one CSV

for tone in tone_types:
    for model in model_types:                
        csv_data = pd.read_csv('data/tones/' + tone + '_' + model + '.csv')
        csv_data['tone'] = tone
        csv_data['model'] = model
        dataset_pieces.append(csv_data)

combined_dataset = pd.concat(dataset_pieces, axis=0, ignore_index=True)

combined_dataset.to_csv('data/tones/all.csv')


###### Sample 100 rewrites with hierarchical sampling.

# Number of samples per group
total_samples = 100
unique_tones = combined_dataset['tone'].nunique()
unique_models = combined_dataset['model'].nunique()
samples_per_group = total_samples // (unique_tones * unique_models)

# Group by 'tone' and 'model'
grouped = combined_dataset.groupby(['tone', 'model'])

# Perform sampling
sampled_dataset = grouped.apply(lambda x: x.sample(n=min(len(x), samples_per_group), random_state=42)).reset_index(drop=True)

# If there are leftover samples to reach exactly 100, sample them randomly from the remaining data
leftover_samples = total_samples - len(sampled_dataset)
if leftover_samples > 0:
    remaining_data = combined_dataset[~combined_dataset.index.isin(sampled_dataset.index)]
    additional_samples = remaining_data.sample(n=leftover_samples, random_state=42)
    sampled_dataset = pd.concat([sampled_dataset, additional_samples])

# Reset index for final dataframe
sampled_dataset = sampled_dataset.reset_index(drop=True)

sampled_dataset[['tone', 'model']]
sampled_dataset.columns

sampled_dataset.to_json('s3://slm-benchmarking/tones/annotate/manifest_small.jsonl', orient="records", lines=True)

sampled_dataset.to_csv('data/tones/all_small.csv')