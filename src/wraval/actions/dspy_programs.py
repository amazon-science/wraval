from typing import Any, List, Dict
import os
import configparser
from pathlib import Path
from .data_utils import load_latest_dataset
import pandas as pd
import dspy
import boto3
from .completion import batch_get_bedrock_completions
from .action_llm_judge import extract_score, get_prompt_functions
from .prompt_tones import Tone, get_prompt


def _ensure_env_creds_from_shared_config() -> None:
    """Populate AWS_* env vars from ~/.aws/credentials [default] if missing."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return
    creds_path = Path.home() / ".aws" / "credentials"
    if not creds_path.exists():
        return
    parser = configparser.ConfigParser()
    try:
        parser.read(creds_path)
        if "default" not in parser:
            return
        section = parser["default"]
        access_key = section.get("aws_access_key_id")
        secret_key = section.get("aws_secret_access_key")
        session_token = section.get("aws_session_token") or section.get("aws_security_token")
        if access_key and secret_key:
            os.environ.setdefault("AWS_ACCESS_KEY_ID", access_key)
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret_key)
            if session_token:
                os.environ.setdefault("AWS_SESSION_TOKEN", session_token)
    except Exception:
        return


def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None, settings=None, client=None, tone="PROFESSIONAL", debug=False):
    """Evaluate prediction using judge logic from action_llm_judge.py"""
    
    # Get judge components
    generate_input_prompt, generate_system_prompt, get_rubric = get_prompt_functions(settings)
    tone_rubrics = get_rubric(tone.upper())
    
    # Extract input and output
    input_text = example.input if hasattr(example, 'input') else str(example)
    output_text = prediction.output if hasattr(prediction, 'output') else str(prediction)
    
    if debug:
        print(f"\n[DSPy DEBUG] Teacher Evaluation:")
        print(f"Tone: {tone}, Input: {input_text}, Output: {output_text}")
    
    # Generate prompts and get completions
    user_prompts = []
    sys_prompts = []
    rubrics = list(tone_rubrics.keys())
    
    for rubric in rubrics:
        user_prompts.append(generate_input_prompt(input_text, output_text, tone))
        sys_prompts.append(generate_system_prompt(tone_rubrics[rubric]))
    
    completions = batch_get_bedrock_completions(settings, user_prompts, sys_prompts)
    
    # Extract scores
    scores = []
    for rubric, completion in zip(rubrics, completions):
        score = extract_score(completion)
        scores.append(score if score is not None else 2)  # Default to middle score
    
    # Calculate normalized score (1-3 -> 0-1)
    overall_score = sum(scores) / len(scores) if scores else 2
    normalized_score = (overall_score - 1) / 2
    normalized_score = max(0.0, min(1.0, normalized_score))
    
    feedback = "; ".join([f"{r}: {s}/3" for r, s in zip(rubrics, scores)])
    
    if debug:
        print(f"Scores: {scores}, Normalized: {normalized_score}, Feedback: {feedback}")
    
    return dspy.Prediction(score=normalized_score, feedback=feedback)


class ToneRewriteProgram(dspy.Module):
    """DSPy program that uses tone-specific prompts for rewriting."""

    def __init__(self, default_tone="PROFESSIONAL"):
        super().__init__()
        self.default_tone = default_tone.upper()
        self.predict = dspy.Predict("input -> output")

    def get_tone_instruction(self, tone):
        """Get the instruction for a specific tone."""
        tone_enum = Tone(tone.lower())
        tone_prompt = get_prompt(tone_enum)
        return tone_prompt.sys_prompt

    def forward(self, input: str):
        instruction = self.get_tone_instruction(self.default_tone)
        contextualized_input = f"{instruction}\n\nText to rewrite: {input}"
        return self.predict(input=contextualized_input)
    
    def forward_with_debug(self, input: str, tone: str = None, debug=False):
        current_tone = tone or self.default_tone
        instruction = self.get_tone_instruction(current_tone)
        contextualized_input = f"{instruction}\n\nText to rewrite: {input}"
        
        if debug:
            print(f"\n[DSPy DEBUG] Student Prompt:")
            print(f"Tone: {current_tone}")
            print(f"Instruction: {instruction}")
            print(f"Input: {contextualized_input}")
        
        result = self.predict(input=contextualized_input)
        
        if debug:
            print(f"Output: {result.output}")
        
        return result


def build_dspy_program(settings: Any, llm):
    import dspy
    
    # Ensure Bedrock auth is available
    _ensure_env_creds_from_shared_config()

    # For non-GEPA, just return basic program
    if str(getattr(settings, "dspy_compile", "")).lower() != "gepa":
        program = ToneRewriteProgram()
        print(f"[DSPy] Built basic program")
        return program

    # GEPA compilation
    from dspy import GEPA
    
    # Load dataset
    df = load_latest_dataset(getattr(settings, "data_dir", "./data"))
    max_rows = int(getattr(settings, "dspy_devset_size", 10))
    
    # Create examples
    ds = []
    tones_found = set()
    
    for _, row in df.head(max_rows).iterrows():
        tone = row.get("tone", "PROFESSIONAL")
        tones_found.add(tone.upper())
        ds.append(dspy.Example(input=str(row["synthetic_data"]), tone=tone).with_inputs("input"))
    
    # Create program with detected tone
    default_tone = list(tones_found)[0] if tones_found else "PROFESSIONAL"
    program = ToneRewriteProgram(default_tone)
    
    print(f"[DSPy] Found tones: {tones_found}")
    print(f"[DSPy] Using default tone: {default_tone}")
    
    # Setup teacher model and client
    client = boto3.client('bedrock-runtime')
    teacher_model = getattr(settings, 'dspy_teacher_model', None)
    teacher_lm = dspy.LM(teacher_model) if teacher_model else llm
    
    # Create metric function
    evaluation_counter = {"count": 0}
    
    def simple_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        tone = getattr(gold, 'tone', 'PROFESSIONAL')
        
        # Use teacher model settings if available
        if teacher_model:
            import copy
            judge_settings = copy.deepcopy(settings)
            judge_settings.model = teacher_model
        else:
            judge_settings = settings
        
        debug = evaluation_counter["count"] < 2
        evaluation_counter["count"] += 1
        
        result = metric_with_feedback(
            gold, pred, trace, pred_name, pred_trace, 
            settings=judge_settings, client=client, tone=tone, debug=debug
        )
        
        return float(result.score)

    # Get GEPA parameters
    gepa_iterations = int(getattr(settings, 'dspy_gepa_iterations', 3))
    gepa_minibatch_size = int(getattr(settings, 'dspy_gepa_minibatch_size', 2))
    gepa_num_threads = int(getattr(settings, 'dspy_gepa_num_threads', 1))
    
    # Create and run GEPA
    optimizer = GEPA(
        metric=simple_metric,
        auto="light",
        num_threads=gepa_num_threads,
        track_stats=False,
        reflection_minibatch_size=gepa_minibatch_size,
        reflection_lm=teacher_lm
    )

    # Configure and show settings
    dspy.configure(lm=llm)
    print(f"[DSPy] Student: {settings.model}")
    print(f"[DSPy] Teacher: {teacher_model or settings.model}")
    print(f"[DSPy] Dataset: {len(ds)}, Iterations: {gepa_iterations}, Threads: {gepa_num_threads}")
    print(f"[DSPy] Estimated calls: ~{gepa_iterations * gepa_num_threads * len(ds)}")

    # Run sample
    if ds:
        print(f"\n[DSPy] Sample run:")
        sample = ds[0]
        student_result = program.forward_with_debug(sample.input, getattr(sample, 'tone', default_tone), debug=True)
        teacher_result = metric_with_feedback(sample, student_result, settings=judge_settings if teacher_model else settings, client=client, tone=getattr(sample, 'tone', default_tone), debug=True)
        print(f"Score: {teacher_result.score}")

    print(f"\n[DSPy] Starting optimization...")
    optimized_program = optimizer.compile(program, trainset=ds)
    print(f"[DSPy] Optimization completed!")

    return optimized_program or program