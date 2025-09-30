import os
from typing import Any
import boto3
import configparser

try:
    import dspy
except Exception:  # pragma: no cover
    dspy = None  # Soft import; raise helpful error at runtime


def ensure_dspy_installed():
    if dspy is None:
        raise RuntimeError(
            "dspy is not installed. Please add `dspy-ai` to requirements and pip install."
        )


def build_dspy_llm(settings: Any):
    """Return a configured dspy LM using the unified LM API.

    Settings:
      - dspy_provider: {openai, anthropic, bedrock, openai_compat}
      - dspy_model: provider-specific model identifier (for bedrock, the raw modelId)
      - dspy_temperature, dspy_max_tokens, dspy_base_url (optional)
    """
    ensure_dspy_installed()

    provider = getattr(settings, "dspy_provider", None) or "openai"
    model = getattr(settings, "dspy_model", None) or getattr(settings, "model", None)
    temperature = float(getattr(settings, "dspy_temperature", 0.2))
    max_tokens = int(getattr(settings, "dspy_max_tokens", 2048))
    base_url = getattr(settings, "dspy_base_url", None)
    # Ensure region and credentials are available for AWS SDK default chain
    region = getattr(settings, "region", None)
    if region and not os.environ.get("AWS_REGION") and not os.environ.get("AWS_DEFAULT_REGION"):
        os.environ["AWS_REGION"] = str(region)
        os.environ["AWS_DEFAULT_REGION"] = str(region)
    # If no explicit env creds, try to lift from current boto3 session (e.g., SSO)
    if provider == "bedrock":
        # Attempt to populate env vars from ~/.aws/credentials (default profile) if missing
        _ensure_env_creds_from_shared_config()
        have_env_creds = os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not have_env_creds:
            try:
                session = boto3.Session()
                creds = session.get_credentials()
                if creds is not None:
                    frozen = creds.get_frozen_credentials()
                    os.environ.setdefault("AWS_ACCESS_KEY_ID", frozen.access_key)
                    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", frozen.secret_key)
                    if getattr(frozen, "token", None):
                        os.environ.setdefault("AWS_SESSION_TOKEN", frozen.token)
            except Exception:
                pass

    # Compose unified model string for LM
    if provider == "bedrock":
        # Expect raw Bedrock modelId; prefix with bedrock/
        model_str = f"bedrock/{model}"
        lm = dspy.LM(model=model_str, max_tokens=max_tokens, temperature=temperature)
        return lm

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        model_str = f"openai/{model}"
        return dspy.LM(model=model_str, max_tokens=max_tokens, temperature=temperature)

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        model_str = f"anthropic/{model}"
        return dspy.LM(model=model_str, max_tokens=max_tokens, temperature=temperature)

    if provider == "openai_compat":
        if not base_url:
            raise RuntimeError("dspy_base_url must be set for openai_compat provider")
        # OpenAI-compatible servers via base_url
        return dspy.LM(model=model, base_url=base_url, max_tokens=max_tokens, temperature=temperature)

    raise ValueError(f"Unsupported dspy_provider: {provider}")


def _ensure_env_creds_from_shared_config(profile_name: str | None = None) -> None:
    """Populate AWS_* env vars from ~/.aws/credentials if not already set.

    Prefers the provided profile_name; otherwise uses AWS_PROFILE env var; otherwise 'default'.
    Does nothing if keys are already present in the environment or if the file/profile is missing.
    """
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return

    profile = profile_name or os.environ.get("AWS_PROFILE") or "default"
    cred_path = os.path.expanduser("~/.aws/credentials")
    if not os.path.exists(cred_path):
        return

    parser = configparser.ConfigParser()
    try:
        parser.read(cred_path)
        if profile not in parser:
            return
        section = parser[profile]
        access_key = section.get("aws_access_key_id")
        secret_key = section.get("aws_secret_access_key")
        session_token = section.get("aws_session_token") or section.get("aws_security_token")

        if access_key and secret_key:
            os.environ.setdefault("AWS_ACCESS_KEY_ID", access_key)
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret_key)
            if session_token:
                os.environ.setdefault("AWS_SESSION_TOKEN", session_token)
    except Exception:
        # Silent no-op on parse errors
        return


