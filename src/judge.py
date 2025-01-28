#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
from prompts.prompts import generate_input_prompt, generate_system_prompt, get_rubric
from src.completion import get_completion, batch_get_completions
import re

def extract_score(i):
    # Extract score using a regular expression
    match = re.search(r"<score>(\d+)</score>", i)
    if match:
        score = int(match.group(1))  # Convert to integer
        return(score)
    else:
        print("No score found.")

def judge(modelId, bedrock_client, q, a, tone='PROFESSIONAL'):
    """
    Given a query, judge the rewrite with a particular tone.
    
    :param modelId: The model in Transformers format
    :param bedrock_client: AWS model client
    :param q: query
    :param a: answer (rewrite)
    :param tone: any of 'casual', 'elaborate', 'emojify', 'improve', 'keypoints', 
    'professional', 'proofread', 'shorten', 'witty'.
    """    
    tone_rubrics = get_rubric(tone)
    rubrics = tone_rubrics.keys()

    d = pd.DataFrame({'input': q, 'output': a})

    for rubric in rubrics:
        d[rubric] = ''

    j = 0
    sys = []
    user = []
    for i,o in zip(q, a):
        for rubric in rubrics:
            sys.append(generate_system_prompt(tone_rubrics[rubric]))
            user.append(generate_input_prompt(i, o, tone))
        j = j + 1

    r = batch_get_completions(modelId, bedrock_client, user, sys, max_concurrent=len(user))

    i = 0
    for j in range(j):
        for rubric in rubrics:
            d.loc[j, rubric] = r[i]
            i = i + 1

    d = d.assign(**{f'{metric}_score': 
                    d[metric].apply(lambda x: extract_score(x)) for metric in rubrics}
                )

    d['overall_score'] = d[[f'{metric}_score' for metric in rubrics]].mean(axis=1)

    return(d)