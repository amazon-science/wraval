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

bucket_name = 'slm-benchmarking'
# path = 'tones/annotate/slm-writing-assist-evaluation-clone/annotations/worker-response/iteration-1'
path = 'tones/annotate/slm-writing-assist-madrid-clone/annotations/worker-response/iteration-1'

s3 = boto3.client('s3')

# Create a reusable Paginator
# see https://boto3.amazonaws.com/v1/documentation/api/latest/guide/paginators.html
paginator = s3.get_paginator('list_objects_v2')

# Create a PageIterator from the Paginator
page_iterator = paginator.paginate(Bucket=bucket_name)

files = [obj['Contents']['Key'] for obj in page_iterator if obj['Key'].endswith('.json')]

records = []

for file in files:
    # Load JSON file directly into a DataFrame
    content = pd.read_json(f's3://{bucket_name}/{file}')

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

all_data = pd.read_csv('data/tones/all.csv')
all_data['human_rating'] = ''
all_data.loc[results.index, 'human_rating'] = results['grading'].values


def wrap_text(text, width=30):
    if isinstance(text, str):
        return "\n".join(textwrap.wrap(text, width))
    return text

wrapped_data = all_data.map(lambda x: wrap_text(x, width=30))

human_vs_judge = wrapped_data.loc[results.index, ['model', 'tone', 'input', 'output', 'overall_score', 'human_rating']]

print(human_vs_judge.to_markdown())

human_vs_judge.to_csv('data/tones/humanVjudge_madrid.csv')


############## plot the judge VS human ratings

human_vs_judge['human_rating'] = pd.to_numeric(human_vs_judge['human_rating'], errors="coerce")


slope, intercept = np.polyfit(human_vs_judge['overall_score'], human_vs_judge['human_rating'], 1)
regression_line = slope * human_vs_judge['overall_score'] + intercept

fig = px.scatter(human_vs_judge, x="overall_score", y="human_rating", trendline="ols",
                 title="Scatter Plot with Regression Line",
                 labels={"X": "Judge Score", "Y": "Human Score"})

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

fig.show()

fig.write_html("data/plots/humanVjudge_madrid.html")
