#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#

import pandas as pd
import boto3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import textwrap
import os
from argparse import ArgumentParser

def get_human_vs_judge_dataset(args):
    s3 = boto3.client('s3')

    # Create a reusable Paginator
    # see https://boto3.amazonaws.com/v1/documentation/api/latest/guide/paginators.html
    paginator = s3.get_paginator('list_objects_v2')

    # Create a PageIterator from the Paginator
    page_iterator = paginator.paginate(Bucket=args.bucket_name, Prefix=args.bucket_prefix)
    o = [obj for obj in page_iterator]
    files = [obj['Key'] for obj in o[0]['Contents'] if obj['Key'].endswith('.json')]
    records = []
    for file in files:
        response = s3.get_object(Bucket=args.bucket_name, Key=file)
        content = pd.read_json(response['Body'])

        for answer in content['answers']:
            # change this with the new design -> 0,1,2,3
            grading = next((int(k) for k, v in answer['answerContent'].get('prefer', {}).items() if v), 0)
            
            records.append({
                'sample': int(file.split('/')[-2]),
                'grading': int(grading),
                'worker': answer['workerId'].split('.')[-1],
            })

    # Convert to DataFrame
    data = pd.DataFrame(records)
    results = data.groupby('sample').agg(grading=('grading', 'mean'), count=('grading', 'count'))

    # all data
    folder_path = os.path.expanduser(args.dataset_path)
    files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    last_file = files[-2]
    all_data = pd.read_csv(os.path.join(folder_path, last_file))

    rewrite = pd.concat(records, axis = 0)
    # TODO: Where to find this rewrite_score? Are we still assuming one file per tone?
    rewrite.rewrite_score = rewrite.rewrite_score.astype(int)
    rewrite['key'] = rewrite[['tone', 'model','input']].astype(str).agg('_'.join, axis=1)
    rewrite['key'] = rewrite['key'].astype(str)


    subset = pd.read_csv('~/data/all_small.csv')

    subset['human_rating'] = ''

    subset.loc[results.index, 'human_rating'] = results['grading'].values

    subset.rename({'source': 'input', 'gen': 'output'}, axis = 1, inplace = True)
    subset.drop('original', axis = 1, inplace = True)
    subset.drop('Unnamed: 0', axis = 1, inplace = True)

    merged = pd.merge(all_data, subset,
                        on=['input', 'output', 'tone', 'model'],
                        how='inner')

    wrapped_data = merged.map(lambda x: wrap_text(x, width=30))

    human_vs_judge = wrapped_data.loc[:, ['model', 'tone', 'input', 'output', 'overall_score', 'human_rating']]

    return human_vs_judge

def wrap_text(text, width=30):
    if isinstance(text, str):
        return "\n".join(textwrap.wrap(text, width))
    return text

def plot(human_vs_judge: pd.DataFrame) -> None:
    human_vs_judge['human_rating'] = pd.to_numeric(human_vs_judge['human_rating'], errors="coerce")

    slope, intercept = np.polyfit(human_vs_judge['overall_score'], human_vs_judge['human_rating'], 1)
    regression_line = slope * human_vs_judge['overall_score'] + intercept

    # Add jittering to avoid overlap
    human_vs_judge['jitter_overall_score'] = human_vs_judge['overall_score'] + np.random.normal(0, 0.02, size=len(human_vs_judge['overall_score']))
    human_vs_judge['jitter_human_rating'] = human_vs_judge['human_rating'] + np.random.normal(0, 0.02, size=len(human_vs_judge['human_rating']))

    fig = px.scatter(human_vs_judge,
                    x="jitter_overall_score", y="jitter_human_rating",
                    hover_data={col: True for col in human_vs_judge.columns if 'jitter' not in col},
                    color="model")
    # Add regression line
    fig.add_trace(
        go.Scatter(
            x=human_vs_judge['overall_score'],
            y=regression_line,
            mode="lines",
            name="Regression Line",
            line=dict(color="red", width=2)
        )
    )
    # Update layout for Economist style
    fig.update_layout(
        title_font=dict(size=24, family="Georgia, Times New Roman, serif"),
        font=dict(size=14, family="Georgia, Times New Roman, serif"),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False,
                title="LLM Judge Score"),
        yaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False,
                title="Human Score"),
        margin=dict(l=60, r=40, t=80, b=60),
        title_x=0.5  # Center the title
    )
    # Update marker styles
    fig.update_traces(
        marker=dict(size=10, opacity=0.8, line=dict(width=1, color="black"))
    )
    fig.write_html("data/plots/humanVjudge_madrid.html")

def parse_args():
    arg_parser = ArgumentParser()
    # bucket_name = 'slm-benchmarking'
    # path = f'tones/annotate/slm-writing-assist-seattle-small/annotations/worker-response/iteration-1'
    arg_parser.add_argument('--bucket_name', type=str, required=True)
    arg_parser.add_argument('--bucket_prefix', type=str, required=True)
    return arg_parser.parse_args()

def main():
    args = parse_args()
    human_vs_judge = get_human_vs_judge_dataset(args)
    human_vs_judge.to_csv('data/humanVjudge_seattle.csv')
    plot(human_vs_judge)

if __name__ == "__main__":
    main()