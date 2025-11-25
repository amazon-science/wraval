# Project Structure

## Directory Layout

```
wraval/
├── config/
│   └── settings.toml              # Model and AWS configuration
├── data/                          # Generated datasets (timestamped CSVs)
│   ├── clean/                     # Cleaned/processed datasets
│   ├── old/                       # Archived datasets
│   └── unique_queries/            # Deduplicated queries
├── src/wraval/                    # Main package source
│   ├── __init__.py
│   ├── main.py                    # CLI entry point (Typer app)
│   ├── aws_config.py              # AWS configuration and warning suppression
│   ├── testing.py                 # Testing utilities
│   ├── actions/                   # Core action modules
│   │   ├── action_generate.py    # Data generation logic
│   │   ├── action_inference.py   # Model inference execution
│   │   ├── action_llm_judge.py   # LLM-as-a-judge evaluation
│   │   ├── action_deploy.py      # SageMaker deployment
│   │   ├── action_results.py     # Results visualization
│   │   ├── action_examples.py    # Example display
│   │   ├── action_human_judge_upload.py  # Human eval setup
│   │   ├── action_human_judge_parsing.py # Human eval parsing
│   │   ├── aws_utils.py          # AWS helper functions
│   │   ├── completion.py         # Model completion wrappers
│   │   ├── data_utils.py         # Data manipulation utilities
│   │   ├── format.py             # Prompt formatting
│   │   ├── model_router.py       # Model endpoint routing
│   │   ├── prompt_tones.py       # Tone definitions and prompts
│   │   ├── prompts_judge.py      # Judge evaluation prompts
│   │   ├── data_generation_prompts.py  # Data gen prompts
│   │   ├── read_random_lines.py  # Sampling utilities
│   │   ├── cloudformation.yml    # CloudFormation templates
│   │   ├── cloudformation_BedrockBatchInference.yml
│   │   └── groundtruth_eval_template.html  # Human eval UI
│   ├── custom_prompts/           # Custom prompt templates
│   │   ├── data_generation_prompts.py
│   │   ├── prompt_tones.py
│   │   ├── prompts_judge.py
│   │   ├── tone_prompts.py
│   │   └── s3_transfer.sh        # S3 sync script
│   └── model_artifacts/          # SageMaker deployment artifacts
│       └── code/
│           ├── inference.py      # SageMaker inference handler
│           └── requirements.txt  # Model deployment deps
├── resources/                     # Documentation and presentations
├── build/                         # Build artifacts
├── .ipynb_checkpoints/           # Jupyter notebook checkpoints
├── pyproject.toml                # Package configuration
├── setup.py                      # Setup script
├── requirements.txt              # Pinned dependencies
├── LICENSE-2.0.txt               # Apache 2.0 license
├── NOTICE.txt                    # Copyright notice
└── README.md                     # Project documentation
```

## Module Organization

### Entry Point
- **main.py**: CLI application using Typer with commands for each workflow step

### Actions Module (`src/wraval/actions/`)
Core functionality organized by workflow step:
- **Generation**: `action_generate.py` - Creates synthetic datasets
- **Inference**: `action_inference.py` - Runs models on datasets
- **Evaluation**: `action_llm_judge.py` - Automated evaluation
- **Deployment**: `action_deploy.py` - SageMaker endpoint management
- **Human Eval**: `action_human_judge_*.py` - Human evaluation workflows
- **Utilities**: Supporting modules for AWS, data, prompts, formatting

### Custom Prompts (`src/wraval/custom_prompts/`)
User-customizable prompt templates that override defaults when `--custom-prompts` flag is used.

### Model Artifacts (`src/wraval/model_artifacts/`)
SageMaker-specific deployment code:
- `inference.py`: Custom inference handler for deployed models
- `requirements.txt`: Runtime dependencies for deployed models

## Configuration Files

### settings.toml
Environment-based configuration with model profiles:
- `[default]`: Base settings (region, buckets, roles)
- `[model-name]`: Model-specific configs (endpoint type, HF model name)
- Supports string interpolation for AWS account/region

### pyproject.toml
Package metadata and dependencies:
- Main dependencies in `dependencies` array
- Optional GPU dependencies in `[project.optional-dependencies]`
- Entry point: `wraval` command → `wraval.main:main`

## Data Flow

1. **Generation**: `wraval generate` → `data/all-{timestamp}.csv`
2. **Inference**: Reads latest CSV → adds model outputs → saves updated CSV
3. **Evaluation**: Reads CSV with outputs → adds judge scores → saves updated CSV
4. **Human Eval**: Samples from CSV → uploads to S3 → creates SageMaker Ground Truth job

## File Naming Conventions

- **Datasets**: `all-{YYYYMMDD_HHMMSS}.csv` (timestamped)
- **Actions**: `action_{verb}.py` (e.g., `action_generate.py`)
- **Utilities**: `{noun}_utils.py` (e.g., `aws_utils.py`, `data_utils.py`)
- **Prompts**: `{type}_prompts.py` or `prompt_{type}.py`

## Import Patterns

- Actions import from sibling modules: `from wraval.actions.{module} import {function}`
- Main imports actions: `from wraval.actions.action_{name} import {function}`
- Config loaded via dynaconf: `Dynaconf(settings_files=[...])`
- AWS config imported first to suppress warnings: `from wraval.aws_config import *`

## Key Architectural Patterns

1. **CLI-driven**: All functionality exposed through Typer commands
2. **Configuration-based**: Model behavior controlled via settings.toml profiles
3. **Stateless actions**: Each action reads/writes CSV files independently
4. **Pluggable prompts**: Custom prompts override defaults when specified
5. **Multi-endpoint**: Unified interface for Bedrock, SageMaker, Ollama
