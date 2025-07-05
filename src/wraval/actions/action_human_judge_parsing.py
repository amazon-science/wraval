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
from argparse import ArgumentParser
import boto3
from .data_utils import write_dataset, load_latest_dataset, latest_file_name
from .data_utils import parse_s3_path
import s3fs
import json
from tqdm import tqdm
from tabulate import tabulate  # for markdown-to-HTML conversion
import os

def parse_groundtruth_output(s3_path):
    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem(anon=False)  # Use AWS credentials
    
    records = []
    
    # Read directly from S3
    with fs.open(s3_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                
                # Extract the base data
                parsed_record = {
                    'source': record.get('source'),
                    'original': record.get('original'),
                    'generated': record.get('gen'),
                    'uuid': record.get('uuid'),
                    'tone': record.get('tone')
                }
                
                # Get the annotation data
                template_key = 'slm-writing-assist-mturk-phi-4-4b-go-500-clone-metadata'
                annotation_data = record.get(template_key, {})
                
                # Merge annotation data with base data
                parsed_record.update(annotation_data)
                
                records.append(parsed_record)
                
            except json.JSONDecodeError:
                print(f"Couldn't parse line: {line}")
                continue
    
    return pd.DataFrame(records)

def get_worker_response(worker_ref):
    if pd.isna(worker_ref):
        return None
    
    fs = s3fs.S3FileSystem(anon=False)
    try:
        with fs.open(worker_ref, 'r') as f:
            record = json.loads(f.read())
            answers = []
            for a in record['answers']:
                r = a['answerContent']['prefer']
                answers.append([key for key, value in r.items() if value][0])
            return answers
    except Exception as e:
        print(f"Error processing {worker_ref}: {e}")
        return None


def wrap_text(text, width=30):
    if isinstance(text, str):
        return "\n".join(textwrap.wrap(text, width))
    return text

def plot_and_table(human_vs_judge: pd.DataFrame, merged: pd.DataFrame, output_path: str) -> None:
    # Clean and jitter
    human_vs_judge["human_rating"] = pd.to_numeric(human_vs_judge["human_rating"], errors="coerce")
    slope, intercept = np.polyfit(human_vs_judge["overall_score"], human_vs_judge["human_rating"], 1)
    regression_line = slope * human_vs_judge["overall_score"] + intercept
    human_vs_judge["jitter_overall_score"] = human_vs_judge["overall_score"] + np.random.normal(0, 0.02, len(human_vs_judge))
    human_vs_judge["jitter_human_rating"] = human_vs_judge["human_rating"] + np.random.normal(0, 0.02, len(human_vs_judge))

    fig = px.scatter(
        human_vs_judge,
        x="jitter_overall_score",
        y="jitter_human_rating",
        hover_data={col: True for col in human_vs_judge.columns if "jitter" not in col},
        color="inference_model",
    )
    fig.add_trace(go.Scatter(
        x=human_vs_judge["overall_score"],
        y=regression_line,
        mode="lines",
        name="Regression Line",
        line=dict(color="red", width=2),
    ))
    fig.update_layout(
        title="Human vs LLM Judge Score",
        title_font=dict(size=24, family="Georgia, Times New Roman, serif"),
        font=dict(size=14, family="Georgia, Times New Roman, serif"),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False, title="LLM Judge Score"),
        yaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False, title="Human Score"),
        margin=dict(l=60, r=40, t=80, b=60),
        title_x=0.5,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="black")))

    # Convert plot to HTML string
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Create markdown tables
    annotated = merged[~merged.human_rating.isna()]
    grouped = annotated.groupby(['tone', 'inference_model'])

    table1_md = grouped['human_rating'].count().reset_index(name='count')
    table1_html = tabulate(table1_md, headers='keys', tablefmt='html')

    sampled = grouped.sample(n=3, random_state=1)
    table2_md = sampled[['tone', 'inference_model', 'synthetic_data', 'rewrite', 'human_rating']].applymap(wrap_text)
    table2_html = tabulate(table2_md, headers='keys', tablefmt='html')

    # Combine everything into one HTML file
    full_html = f"""
    <html>
    <head>
        <title>Human vs Judge Report</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Georgia, serif; margin: 40px; }}
            h2 {{ margin-top: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 40px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; word-wrap: break-word; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Human vs LLM Judge Report</h1>
        {plot_html}
        <h2>Table 1: Count of Ratings per Tone and Model</h2>
        {table1_html}
        <h2>Table 2: Sampled Examples (3 per Tone/Model)</h2>
        {table2_html}
    </body>
    </html>
    """

    with open(os.path.expanduser(output_path), "w", encoding="utf-8") as f:
        f.write(full_html)

def parse_human_judgements():
    return

def temp():        
    tqdm.pandas()

    single_manifest = 's3://llm-finetune-us-east-1-797954477170/human_eval/tones/slm-writing-assist-mturk-phi-4-4b-go-500-clone/manifests/output/output.manifest'
    annotations = parse_groundtruth_output(single_manifest)
    results = load_latest_dataset('s3://llm-finetune-us-east-1-797954477170/eval/tones/')

    annotations['worker_answers'] = annotations['worker-response-ref'].progress_apply(get_worker_response)

    annotations.rename(columns={'worker_answers': 'human_rating'}, inplace=True)

    merged = pd.merge(results, single_df[['uuid', 'human_rating']], on='uuid', how='left')

    write_dataset(merged)

    annotated = merged[~merged.human_rating.isna()]

    sampled = grouped.sample(n=3, random_state=1)

    annotated_exploded = annotated.explode('human_rating')
    plot_and_table(
        annotated_exploded[['tone', 'inference_model', 'synthetic_data', 'rewrite', 'human_rating', 'overall_score']],
        merged,
        '~/Desktop/humanVjudge_combined.html'
    )


    merged.to_csv(os.path.expanduser('~/Desktop/phi-4-human-eval.csv'), index=False)
