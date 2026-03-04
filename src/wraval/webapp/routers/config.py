"""Model and tone configuration router."""

import os
from typing import List

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from fastapi import APIRouter
from pydantic import BaseModel

from wraval.actions.prompt_tones import Tone

router = APIRouter(tags=["config"])

SETTINGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "config", "settings.toml"
)


class ModelInfo(BaseModel):
    name: str
    endpoint_type: str


@router.get("/models", response_model=List[ModelInfo])
def get_models():
    """Return available model profile names and their endpoint types."""
    resolved = os.path.normpath(SETTINGS_PATH)
    with open(resolved, "rb") as f:
        data = tomllib.load(f)

    models: List[ModelInfo] = []
    for section, values in data.items():
        if section == "default":
            continue
        if isinstance(values, dict):
            endpoint_type = values.get("endpoint_type", "unknown")
            models.append(ModelInfo(name=section, endpoint_type=endpoint_type))
    return models


@router.get("/tones", response_model=List[str])
def get_tones():
    """Return all supported tone values plus 'all'."""
    tones = [tone.value for tone in Tone]
    tones.append("all")
    return tones
