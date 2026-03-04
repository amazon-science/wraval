# Requirements Document

## Introduction

WRAVAL Webapp is a web-based interface for the existing WRAVAL (WRiting Assist eVALuation) CLI tool. The webapp provides annotators and ML practitioners with a browser-based UI to edit prompts, trigger model inference and LLM-as-a-judge evaluation, and visualize evaluation data — all without needing to use the command line. The webapp reuses the existing Python backend logic, wrapping CLI actions with a FastAPI server and a lightweight frontend.

## Glossary

- **Webapp**: The web application consisting of a FastAPI backend and a browser-based frontend
- **Prompt_Editor**: The UI component that allows users to view and edit tone prompt templates
- **Tone**: A writing style transformation type (e.g., witty, professional, casual, elaborate, shorten, improve, keypoints, proofread, emojify, summarize)
- **Prompt**: A system prompt and optional few-shot examples that instruct a model how to perform a tone transformation
- **Inference_Runner**: The backend component that executes model inference on datasets
- **Judge_Runner**: The backend component that executes LLM-as-a-judge evaluation on inference results
- **Data_Table**: The UI component that displays evaluation datasets in a rich tabular format
- **Model_Profile**: A named configuration in settings.toml defining a model's endpoint type, model ID, and other parameters
- **Dataset**: A timestamped CSV file containing synthetic data, rewrites, scores, and metadata
- **Settings**: The dynaconf-managed configuration loaded from config/settings.toml

## Requirements

### Requirement 1: Prompt Viewing and Editing

**User Story:** As an annotator, I want to view and edit tone prompt templates through a web UI, so that I can iterate on prompts without modifying Python source files.

#### Acceptance Criteria

1. WHEN the Webapp loads the Prompt_Editor page, THE Prompt_Editor SHALL display a list of all available Tone names
2. WHEN a user selects a Tone, THE Prompt_Editor SHALL display the system prompt text and few-shot examples for that Tone
3. WHEN a user modifies a system prompt or few-shot examples and clicks save, THE Webapp SHALL persist the updated Prompt to the custom_prompts directory on disk
4. WHEN a user saves a Prompt, THE Webapp SHALL validate that the system prompt text is non-empty before persisting
5. IF a save operation fails due to a file system error, THEN THE Webapp SHALL display an error message describing the failure

### Requirement 2: Model Inference Execution

**User Story:** As an ML practitioner, I want to trigger model inference from the web UI, so that I can run evaluations without switching to the terminal.

#### Acceptance Criteria

1. WHEN the Webapp loads the inference page, THE Webapp SHALL display a list of available Model_Profile names from Settings
2. WHEN the Webapp loads the inference page, THE Webapp SHALL display a list of available Tone options including an "all" option
3. WHEN a user selects a Model_Profile and Tone and clicks the run inference button, THE Inference_Runner SHALL execute inference using the selected model and tone parameters
4. WHILE the Inference_Runner is executing, THE Webapp SHALL display a status indicator showing that inference is in progress
5. WHEN the Inference_Runner completes successfully, THE Webapp SHALL display a success message with the number of processed items
6. IF the Inference_Runner encounters an error, THEN THE Webapp SHALL display the error message to the user

### Requirement 3: LLM Judge Evaluation Execution

**User Story:** As an ML practitioner, I want to trigger LLM-as-a-judge evaluation from the web UI, so that I can score model outputs without using the CLI.

#### Acceptance Criteria

1. WHEN a user selects a judge Model_Profile and Tone and clicks the run judge button, THE Judge_Runner SHALL execute LLM-as-a-judge evaluation using the selected parameters
2. WHILE the Judge_Runner is executing, THE Webapp SHALL display a status indicator showing that evaluation is in progress
3. WHEN the Judge_Runner completes successfully, THE Webapp SHALL display a success message with the number of evaluated items
4. IF the Judge_Runner encounters an error, THEN THE Webapp SHALL display the error message to the user

### Requirement 4: Data Table Visualization

**User Story:** As an ML practitioner, I want to view evaluation data in a rich table format in the browser, so that I can compare model outputs and scores across tones and models.

#### Acceptance Criteria

1. WHEN the Webapp loads the Data_Table page, THE Data_Table SHALL load and display the latest Dataset from the configured data directory
2. WHEN displaying data, THE Data_Table SHALL show columns for input text, rewrite text, tone, inference model, and overall score
3. WHEN a user selects a Tone filter, THE Data_Table SHALL display only rows matching the selected Tone
4. WHEN a user selects an inference model filter, THE Data_Table SHALL display only rows matching the selected model
5. THE Data_Table SHALL support pagination to handle large Datasets without degrading browser performance
6. WHEN no Dataset is found, THE Data_Table SHALL display a message indicating that no data is available

### Requirement 5: Model and Tone Configuration Display

**User Story:** As an ML practitioner, I want to see available models and their configurations, so that I can make informed choices when running inference or evaluation.

#### Acceptance Criteria

1. WHEN the Webapp starts, THE Webapp SHALL parse Settings from config/settings.toml and expose available Model_Profile names through the API
2. THE Webapp SHALL expose the list of supported Tone values through the API
3. WHEN a Model_Profile is selected, THE Webapp SHALL display the endpoint type for that model

### Requirement 6: Job Status Tracking

**User Story:** As an ML practitioner, I want to track the status of running inference and judge jobs, so that I can know when results are ready.

#### Acceptance Criteria

1. WHEN an inference or judge job is started, THE Webapp SHALL assign a unique job identifier and return the identifier to the user
2. WHILE a job is running, THE Webapp SHALL allow the user to query the job status using the job identifier
3. WHEN a job completes, THE Webapp SHALL update the job status to reflect completion or failure
4. THE Webapp SHALL allow only one inference job and one judge job to run concurrently to prevent resource conflicts

### Requirement 7: API Design

**User Story:** As a developer, I want a well-structured REST API, so that the frontend can communicate with the backend reliably.

#### Acceptance Criteria

1. THE Webapp SHALL expose a REST API using FastAPI with JSON request and response bodies
2. THE Webapp SHALL serve the frontend static files from the same server process
3. WHEN an API request contains invalid parameters, THE Webapp SHALL return an appropriate HTTP error status code with a descriptive error message
4. THE Webapp SHALL provide an OpenAPI documentation endpoint at /docs
