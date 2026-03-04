# Implementation Plan: WRAVAL Webapp

## Overview

Build a FastAPI webapp that wraps the existing WRAVAL CLI tool. The backend reuses `get_settings()`, `run_inference()`, and `judge()` directly. The frontend is vanilla HTML/JS served as static files. Prompts are edited in-place in `custom_prompts/prompt_tones.py` with git versioning.

## Tasks

- [x] 1. Set up FastAPI application skeleton and project structure
  - [x] 1.1 Create `src/wraval/webapp/` package with `__init__.py` and `app.py`
    - Create FastAPI app instance with title "WRAVAL Webapp"
    - Add placeholder router includes for prompts, inference, judge, data, config, jobs
    - Mount static files directory at "/" with `html=True`
    - Add a `webapp` CLI command to `src/wraval/main.py` that starts uvicorn
    - _Requirements: 7.1, 7.2_
  - [x] 1.2 Create `src/wraval/webapp/routers/` package with empty router files
    - Create `__init__.py`, `prompts.py`, `inference.py`, `judge.py`, `data.py`, `config.py`, `jobs.py`
    - Each file defines an `APIRouter` with appropriate prefix tags
    - _Requirements: 7.1_
  - [x] 1.3 Create `src/wraval/webapp/static/` with minimal `index.html`, `style.css`, `app.js`
    - `index.html`: tab navigation (Prompts, Inference, Judge, Data), content containers
    - `style.css`: basic layout styles
    - `app.js`: tab switching logic, empty placeholder functions for each tab
    - _Requirements: 7.2_
  - [x] 1.4 Add `fastapi`, `uvicorn`, and `httpx` to `pyproject.toml` dependencies
    - Add `fastapi` and `uvicorn[standard]` to main dependencies
    - Add `httpx` and `hypothesis` to a new `[project.optional-dependencies] dev` section
    - _Requirements: 7.1_

- [ ] 2. Implement Config API and Job Manager
  - [x] 2.1 Implement config router (`src/wraval/webapp/routers/config.py`)
    - `GET /api/config/models`: parse `config/settings.toml` using `tomllib`/`tomli`, return model profile names and endpoint types (sections other than "default")
    - `GET /api/config/tones`: return all Tone enum values plus "all"
    - _Requirements: 2.1, 2.2, 5.1, 5.2, 5.3_
  - [-] 2.2 Implement Job Manager (`src/wraval/webapp/jobs.py`)
    - Implement `JobStatus`, `JobType`, `Job` dataclass, and `JobManager` class
    - `start_job()`: acquire type-specific lock, spawn background thread, return Job
    - `get_job()`: return job by ID
    - Enforce one-per-type concurrency with threading.Lock
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [~] 2.3 Implement jobs router (`src/wraval/webapp/routers/jobs.py`)
    - `GET /api/jobs/{job_id}`: return job status from JobManager
    - Return 404 if job_id not found
    - _Requirements: 6.2_
  - [ ]* 2.4 Write property tests for Job Manager
    - **Property 10: Job IDs are unique**
    - **Validates: Requirements 6.1**
  - [ ]* 2.5 Write property tests for Job Manager concurrency
    - **Property 11: Job status reflects outcome**
    - **Property 12: Concurrent job type rejection**
    - **Validates: Requirements 6.3, 6.4**
  - [ ]* 2.6 Write property test for config tones completeness
    - **Property 3: Tones API completeness**
    - **Validates: Requirements 1.1, 2.2, 5.2**

- [~] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement Prompt API with git versioning
  - [~] 4.1 Implement prompt reading logic in prompts router (`src/wraval/webapp/routers/prompts.py`)
    - `GET /api/prompts/tones`: list all tone names
    - `GET /api/prompts/{tone}`: import and instantiate the Prompt class for the tone, return sys_prompt and examples
    - Use `wraval.custom_prompts.prompt_tones.get_prompt` and `Tone` enum
    - Return 404 for invalid tone names
    - _Requirements: 1.1, 1.2_
  - [~] 4.2 Implement prompt writing logic with git commit
    - `PUT /api/prompts/{tone}`: validate non-empty sys_prompt, regenerate the prompt class in `custom_prompts/prompt_tones.py`, git add + git commit
    - Write a helper function to parse and regenerate the Python file, updating only the target tone's class
    - Run `subprocess.run(["git", "add", ...])` and `subprocess.run(["git", "commit", "-m", ...])` after file write
    - Validate sys_prompt is non-empty (Pydantic `Field(min_length=1)`)
    - _Requirements: 1.3, 1.4, 1.5_
  - [ ]* 4.3 Write property test for prompt round-trip
    - **Property 1: Prompt save/load round-trip**
    - **Validates: Requirements 1.3**
  - [ ]* 4.4 Write property test for empty prompt rejection
    - **Property 2: Empty/whitespace prompt rejection**
    - **Validates: Requirements 1.4**
  - [ ]* 4.5 Write property test for prompt retrieval
    - **Property 4: Prompt retrieval returns valid data for all tones**
    - **Validates: Requirements 1.2**

- [ ] 5. Implement Inference and Judge APIs
  - [~] 5.1 Implement inference router (`src/wraval/webapp/routers/inference.py`)
    - `POST /api/inference/run`: accept `{model, tone}`, validate model exists in settings, call `get_settings(model, tone, custom_prompts=True)` then `run_inference(settings, model, False, settings.data_dir)` via JobManager
    - Return job ID and status
    - _Requirements: 2.3, 2.4, 2.5, 2.6_
  - [~] 5.2 Implement judge router (`src/wraval/webapp/routers/judge.py`)
    - `POST /api/judge/run`: accept `{model, tone}`, validate model exists, call `get_settings()` then set up boto3 client and call `judge()` via JobManager — same logic as `llm_judge` CLI command
    - Return job ID and status
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 6. Implement Data API
  - [~] 6.1 Implement data router (`src/wraval/webapp/routers/data.py`)
    - `GET /api/data/latest`: call `get_settings()` then `load_latest_dataset(settings.data_dir)`
    - Apply optional tone and model query param filters
    - Implement pagination with `page` and `page_size` params
    - Return `DataResponse` with rows, total count, available tones, available models
    - Handle missing dataset gracefully (return empty rows with message)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [ ]* 6.2 Write property tests for data filtering and pagination
    - **Property 7: Tone filter returns only matching rows**
    - **Property 8: Model filter returns only matching rows**
    - **Property 9: Pagination respects page size**
    - **Validates: Requirements 4.3, 4.4, 4.5**

- [~] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Build frontend UI
  - [~] 8.1 Implement Prompts tab in frontend
    - Tone selector dropdown populated from `GET /api/prompts/tones`
    - System prompt textarea and examples editor (add/remove pairs) populated from `GET /api/prompts/{tone}`
    - Save button calls `PUT /api/prompts/{tone}`, shows success/error message with git commit info
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [~] 8.2 Implement Inference tab in frontend
    - Model selector populated from `GET /api/config/models`
    - Tone selector populated from `GET /api/config/tones`
    - Run button calls `POST /api/inference/run`, then polls `GET /api/jobs/{id}` every 3 seconds
    - Display running/completed/failed status with error message if applicable
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  - [~] 8.3 Implement Judge tab in frontend
    - Same layout as Inference tab but calls `POST /api/judge/run`
    - Model selector, tone selector, run button, status polling
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [~] 8.4 Implement Data tab in frontend
    - Load data from `GET /api/data/latest` on tab activation
    - Tone and model filter dropdowns populated from response metadata
    - Paginated HTML table showing synthetic_data, rewrite, tone, inference_model, overall_score
    - Previous/Next page buttons
    - "No data available" message when dataset is empty
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 9. Wire everything together and add validation
  - [~] 9.1 Register all routers in `app.py` and verify end-to-end flow
    - Ensure all router imports and prefix mounts are correct
    - Verify static file serving works for the frontend
    - Verify `/docs` OpenAPI endpoint is accessible
    - _Requirements: 7.1, 7.2, 7.4_
  - [~] 9.2 Add input validation error handling
    - Ensure FastAPI validation errors return proper 4xx JSON responses
    - Add custom exception handlers for RuntimeError (job conflicts → 409), ValueError (bad params → 400), FileNotFoundError (missing data → 404)
    - _Requirements: 7.3_
  - [ ]* 9.3 Write property test for invalid API requests
    - **Property 13: Invalid API requests return error responses**
    - **Validates: Requirements 7.3**

- [~] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The webapp always uses `custom_prompts=True` so inference/judge use edited prompts
- Inference and judge call the exact same functions as the CLI — no logic duplication
- Property tests use `hypothesis` with minimum 100 iterations each
- Unit tests use `pytest` with FastAPI `TestClient` (via `httpx`)
