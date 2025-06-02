# WRAVAL – WRiting Assist eVALuation

WRAVAL helps in evaluating LLMs for writing assistant tasks like summarization, professional tone, witty tone, etc.

## Quick start

```bash
uv pip install .
wraval generate
```

> Disclaimer: the deploy action requires a machine that supports bitsandbytes and CUDA. See below.

## Step by step

### 1. Start by generating evaluation data for each of the writing assistant tasks (a.k.a. tones)

```bash
# By default generates all tone types. A specific tone and model can be specified.
wraval generate --type witty --model nova-lite
```

### 2. You can then use Bedrock hosted models or self-hosted models, to play the role of a writing assistant.

```bash
# Bedrock hosted models on all tones
wraval inference --model nova-lite --endpoint-type bedrock
# Self-hosted Sagemaker models on all tones
wraval inference --model {MODEL_NAME} --endpoint-type sagemaker
```

> Note: `MODEL_NAME` uses the proposed mapping in `settings.toml`.

### 3. You can use an LLM-as-a-judge to evaluate these models

```bash
# By default generates all tone types. A specific tone and model can be specified.
wraval llm_judge --model nova-lite
```

### 4. Finally you can make a human-as-a-judge setup with a Sagemaker Groundtruth task

```bash
# By default generates all tone types. A specific tone and model can be specified.
wraval human_judge
```

> Note: ideally different models are used for each step, to avoid bias.

### 5. Deploy a Sagemaker Endpoint to be used by the steps above.

> Use a machine with CUDA support

```bash
uv pip install ".[gpu]"
wraval deploy -m ...
```

An additional notebook is provided to benchmark models on translation tasks on open datasets [here](Haiku_translate.ipynb).

## Motivation

With the popularity of large language models (LLMs), the focus of Language Model (LM) evaluation strongly shifted to problem solving or reasoning tasks, thus targeting a form of general intelligence. Small Language Models (SLMs) – defined here as LMs under 10B parameters – score low on these forms of LM evaluation, sometimes 3-4 times lower than Large Language Models (LLMs). We show that the performance of many of the most popular representative uses for LLMs in industrial settings, including tone change (e.g., funny, serious, professional), are not accurately reflected by these metrics. This paper proposes an evaluation framework that highlights SLMs' strengths on non-reasoning tasks that do not have a predefined evaluation dataset. We contribute with data generation, prompt-tuning, LLM-as-a-judge; and show how this framework helps highlight the potential of finetuning for a set of specific tasks. Our framework helps practitioners benchmark SLMs or LLMs on tasks they are good at and reinforces their usefulness in edge and private computing.


## Data

Data is saved to CSV files with timestamps and can optionally be uploaded to S3.
By default, generated data is saved to `./data/all-tones-{timestamp}.csv`

## Available Tone Types

- `witty`: Factual sentences to be made witty
- `professional`: Casual sentences to be made professional
- `casual`: Formal sentences to be made casual
- `elaborate`: Simple sentences to be made detailed
- `shorten`: Wordy sentences to be made concise
- `improve`: Poorly written sentences to be improved
- `keypoints`: Detailed paragraphs for key point extraction
- `proofread`: Sentences with errors to be corrected
- `emojify`: Plain sentences to be enhanced with emojis
- `paragraph_summary`: Paragraph-summary pairs

Feel free to add your own for your own purposes in the prompt files.

## Notebook quick start

You can use the [CloudFormation yaml](src/cloudformation.yml) to start a Sagemaker notebook with the permissions to call Bedrock models (make sure you enable the Bedrock models in your AWS console beforehand).

## ToDo

- [ ] make everything a one-liner
    - [x] 1.pynb
    - [x] 2.pynb
    - [x] 3.pynb
    - [ ] 4a.pynb:
        - use functions in data_utils
        - read the tones and models from the last csv retrieved via data_utils
        - refactor into different functions (1) csv->manifest.jsonl (2) optional hierarchical sampling function
    - [ ] 4b.pynb:
        - use functions in data_utils
        - read the tones and models from the last csv retrieved via data_utils
        - refactor into different functions (1) get output data on AWS (2) merge it back into the csv and save (3) plot the LLM judge results VS the human feedback
- [x] run Qwen and Phi as standalone sagemaker endpoints.
- [x] requirements.txt
- [x] data
- [x] 1. data generation -> prompt library
- [x] 2.b. LLM -> implement this in a modular way in in format_prompt_as_xml
- [x] merge generate_all_datasets and generate_specific_datasets
- [x] add a model_router.py
- [x] uv
- [x] from main.py to setup.py
- [ ] transfer args to settings
- [ ] batch processing for Bedrock
- [ ] batch processing for Sagemaker endpoint
- [ ] better sagemaker inference output parsing

## How to Cite This Repository

```bibtex
@misc{wraval,
    author = {Gabriel Benedict, Matthew Butler, Naved Merchant, Eetu Salama-Laine},
    title = {{WRAVAL – WRiting Assist eVALuation}},
    howpublished = {\url{https://github.com/amazon-science/wraval}},
    year = {2025}
}
```
