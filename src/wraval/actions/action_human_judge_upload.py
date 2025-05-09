#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import os
from argparse import ArgumentParser
from src.data_utils import write_dataset_local, write_dataset_to_s3

OUTPUT_DIR = "data"
TONE = "tone"
SYNTHETIC_MODEL = "synthetic_model"

def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset_path", type=str, required=True)
    arg_parser.add_argument("--bucket_name", type=str)
    arg_parser.add_argument("--dataset_name", type=str, default="all-tones")
    arg_parser.add_argument("--n_samples", type=int, default=100)
    return arg_parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = pd.read_csv(args.dataset_path)
    dataset.rename(columns={"synthetic_data": "source", "rewrite": "gen"}, inplace=True)
    dataset["original"] = dataset["source"]

    write_dataset_to_s3(dataset, args.bucket_name, f"tones/annotate/{args.dataset_name}", "jsonl")

    if dataset.shape[0] < args.n_samples:
        raise ValueError(f"Requested {args.n_samples} samples, but only {dataset.shape[0]} available.")

    unique_tones = dataset[TONE].nunique()
    unique_models = dataset[SYNTHETIC_MODEL].nunique()
    samples_per_group = args.n_samples // (unique_tones * unique_models)

    grouped = dataset.groupby([TONE, SYNTHETIC_MODEL])

    sampled_dataset = grouped.apply(
        lambda x: x.sample(n=min(len(x), samples_per_group), random_state=42)
    ).reset_index(drop=True)

    leftover_samples = args.n_samples - len(sampled_dataset)
    if leftover_samples > 0:
        remaining_data = dataset[~dataset.index.isin(sampled_dataset.index)]
        additional_samples = remaining_data.sample(n=leftover_samples, random_state=42)
        sampled_dataset = pd.concat([sampled_dataset, additional_samples])

    sampled_dataset = sampled_dataset.reset_index(drop=True)
    write_dataset_to_s3(sampled_dataset, args.bucket_name, f"tones/annotate/{args.dataset_name}_small", "jsonl")
    write_dataset_local(sampled_dataset, OUTPUT_DIR, args.dataset_name)

if __name__ == "__main__":
    main()