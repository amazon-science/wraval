"""Tests for the config router endpoints."""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from wraval.actions.prompt_tones import Tone
from wraval.webapp.app import app
from wraval.webapp.routers import config as config_module

client = TestClient(app)


@pytest.fixture(autouse=True)
def patch_settings_path(tmp_path, monkeypatch):
    """Point the config router at a temporary settings.toml for isolation."""
    toml_content = b"""
[default]
region = "us-east-1"
endpoint_type = "bedrock"

[test-model-bedrock]
model = "some-bedrock-model"
endpoint_type = "bedrock"

[test-model-sagemaker]
model = "some-sagemaker-model"
endpoint_type = "sagemaker"
"""
    toml_file = tmp_path / "settings.toml"
    toml_file.write_bytes(toml_content)
    monkeypatch.setattr(config_module, "SETTINGS_PATH", str(toml_file))


class TestGetModels:
    def test_returns_model_profiles(self):
        resp = client.get("/api/config/models")
        assert resp.status_code == 200
        models = resp.json()
        names = [m["name"] for m in models]
        assert "test-model-bedrock" in names
        assert "test-model-sagemaker" in names

    def test_excludes_default_section(self):
        resp = client.get("/api/config/models")
        names = [m["name"] for m in resp.json()]
        assert "default" not in names

    def test_includes_endpoint_type(self):
        resp = client.get("/api/config/models")
        models = {m["name"]: m["endpoint_type"] for m in resp.json()}
        assert models["test-model-bedrock"] == "bedrock"
        assert models["test-model-sagemaker"] == "sagemaker"

    def test_model_info_schema(self):
        resp = client.get("/api/config/models")
        for model in resp.json():
            assert "name" in model
            assert "endpoint_type" in model


class TestGetTones:
    def test_returns_all_tone_values(self):
        resp = client.get("/api/config/tones")
        assert resp.status_code == 200
        tones = resp.json()
        for tone in Tone:
            assert tone.value in tones

    def test_includes_all_option(self):
        resp = client.get("/api/config/tones")
        assert "all" in resp.json()

    def test_tone_count(self):
        resp = client.get("/api/config/tones")
        tones = resp.json()
        # All Tone enum values + "all"
        assert len(tones) == len(Tone) + 1
