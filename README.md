# WRAVAL – WRiting Assist eVALuation

WRAVAL helps in evaluating LLMs for writing assistant tasks like summarization, professional tone, witty tone, etc.


> With the popularity of large language models (LLMs), the focus of Language Model (LM) evaluation strongly shifted to problem solving or reasoning tasks, thus targeting a form of general intelligence. Small Language Models (SLMs) – defined here as LMs under 10B parameters – score low on these forms of LM evaluation, sometimes 3-4 times lower than Large Language Models (LLMs). We show that the performance of many of the most popular representative uses for LLMs in industrial settings, including tone change (e.g., funny, serious, professional), are not accurately reflected by these metrics. This paper proposes an evaluation framework that highlights SLMs' strengths on non-reasoning tasks that do not have a predefined evaluation dataset. We contribute with data generation, prompt-tuning, LLM-as-a-judge; and show how this framework helps highlight the potential of finetuning for a set of specific tasks. Our framework helps practitioners benchmark SLMs or LLMs on tasks they are good at and reinforces their usefulness in edge and private computing.

## Quick start

You can use the [CloudFormation yaml](src/cloudformation.yml) to start a Sagemaker notebook with the permissions to call Bedrock models (make sure you enable the Bedrock models in your AWS console beforehand).

## Structure

1. Start by generating evaluation data for each of the writing assistant tasks (a.k.a. tones) [here](1. data_generation.ipynb)
2. You can then use Bedrock hosted models ([here](2.b. Haiku_tones.ipynb)) or self-hosted models ([here](2.a. SLM_tones.ipynb)), to play the role of a writing assistant.
3. You can use an LLM-as-a-judge to evaluate these models ([here](3. judge_eval.ipynb))
4. Finally you can setup a Sagemaker Groundtruth tasks [here](4 human_eval.py)

An additional notebook is provided to benchmark models on translation tasks on open datasets [here](Haiku_translate.ipynb).

## ToDo

- [ ] run Qwen and Phi as standalone sagemaker endpoints.
- [x] requirements.txt. (uv?)
- [ ] data
- [ ] 1. data generation -> prompt library
- [ ] 2.b. LLM -> implement this in a modular way in in format_prompt_as_xml
