# WRAVAL – WRiting Assist eVALuation

WRAVAL helps in evaluating LLMs for writing assistant tasks like summarization, professional tone, witty tone, etc.


> With the popularity of large language models (LLMs), the focus of Language Model (LM) evaluation strongly shifted to problem solving or reasoning tasks, thus targeting a form of general intelligence. Small Language Models (SLMs) – defined here as LMs under 10B parameters – score low on these forms of LM evaluation, sometimes 3-4 times lower than Large Language Models (LLMs). We show that the performance of many of the most popular representative uses for LLMs in industrial settings, including tone change (e.g., funny, serious, professional), are not accurately reflected by these metrics. This paper proposes an evaluation framework that highlights SLMs' strengths on non-reasoning tasks that do not have a predefined evaluation dataset. We contribute with data generation, prompt-tuning, LLM-as-a-judge; and show how this framework helps highlight the potential of finetuning for a set of specific tasks. Our framework helps practitioners benchmark SLMs or LLMs on tasks they are good at and reinforces their usefulness in edge and private computing.

## Quick start

```bash
pip install -r requirements.txt
python main.py generate
```

## Structure

1. Start by generating evaluation data for each of the writing assistant tasks (a.k.a. tones) [here](1. data_generation.ipynb)
2. You can then use Bedrock hosted models ([here](2.b. Haiku_tones.ipynb)) or self-hosted models ([here](2.a. SLM_tones.ipynb)), to play the role of a writing assistant.
3. You can use an LLM-as-a-judge to evaluate these models ([here](3. judge_eval.ipynb))
4. Finally you can setup a Sagemaker Groundtruth tasks [here](4 human_eval.py)

An additional notebook is provided to benchmark models on translation tasks on open datasets [here](Haiku_translate.ipynb).

## Data Generation

Generate synthetic data for tone transformation using various LLMs. Data is saved to CSV files with timestamps and can optionally be uploaded to S3.

### Basic Usage

```bash
# By default generates all tone types. A specific tone and model can be specified.
python main.py generate --type witty_sentences --model nova-lite
```

### Available Tone Types
- `witty_sentences`: Factual sentences to be made witty
- `professional_sentences`: Casual sentences to be made professional
- `casual_sentences`: Formal sentences to be made casual
- `elaborate_sentences`: Simple sentences to be made detailed
- `shorten_sentences`: Wordy sentences to be made concise
- `improve_sentences`: Poorly written sentences to be improved
- `keypoints_sentences`: Detailed paragraphs for key point extraction
- `proofread_sentences`: Sentences with errors to be corrected
- `emojify_sentences`: Plain sentences to be enhanced with emojis
- `paragraph_summary`: Paragraph-summary pairs

### Output
- Generated data is saved to `~/data/all-tones-{timestamp}.csv`
- Raw outputs are saved to `~/data/{tone_type}_raw.txt`

## Notebook quick start

You can use the [CloudFormation yaml](src/cloudformation.yml) to start a Sagemaker notebook with the permissions to call Bedrock models (make sure you enable the Bedrock models in your AWS console beforehand).

## ToDo



- [ ] make everything a one-liner
    - [x] 1.pynb
    - [ ] 2.pynb
    - [ ] 3.pynb
    - [ ] 4a.pynb: 
        - use functions in data_utils
        - read the tones and models from the last csv retrieved via data_utils
        - refactor into different functions (1) csv->manifest.jsonl (2) optional hierarchical sampling function
    - [ ] 4b.pynb:     
        - use functions in data_utils
        - read the tones and models from the last csv retrieved via data_utils
        - refactor into different functions (1) get output data on AWS (2) merge it back into the csv and save (3) plot the LLM judge results VS the human feedback
- [ ] run Qwen and Phi as standalone sagemaker endpoints.
- [x] requirements.txt. 
- [x] data
- [x] 1. data generation -> prompt library
- [ ] 2.b. LLM -> implement this in a modular way in in format_prompt_as_xml
- [ ] merge generate_all_datasets and generate_specific_datasets
- [ ] batch processing for Bedrock
- [ ] batch processing for Sagemaker endpoint
- [ ] uv?
- [ ] from main.py to setup.py