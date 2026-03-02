import pytest
from pydantic import ValidationError

from configs.settings import Settings
from services.annotator.backend import auth
from src.models.schemas import AcquireTaskRequest


def test_security_settings_rejects_weak_jwt_secret():
    settings = Settings(
        JWT_SECRET_KEY="weak",
        CORS_ALLOW_ORIGINS="http://localhost:8000",
    )
    with pytest.raises(RuntimeError):
        settings.validate_security_settings()


def test_security_settings_rejects_wildcard_cors():
    settings = Settings(
        JWT_SECRET_KEY="a" * 32,
        CORS_ALLOW_ORIGINS="*",
    )
    with pytest.raises(RuntimeError):
        settings.validate_security_settings()


def test_security_settings_accepts_strong_values():
    settings = Settings(
        JWT_SECRET_KEY="b" * 48,
        CORS_ALLOW_ORIGINS="http://localhost:8000,http://127.0.0.1:8000",
    )
    settings.validate_security_settings()
    assert settings.cors_allow_origins == ["http://localhost:8000", "http://127.0.0.1:8000"]


def test_annotator_token_requires_strong_secret(monkeypatch):
    monkeypatch.setattr(auth, "_SECRET_CACHE", None)
    monkeypatch.setenv("JWT_SECRET_KEY", "short")
    with pytest.raises(RuntimeError):
        auth.generate_token("tester")


def test_annotator_token_roundtrip_with_valid_secret(monkeypatch):
    monkeypatch.setattr(auth, "_SECRET_CACHE", None)
    monkeypatch.setenv("JWT_SECRET_KEY", "c" * 40)
    token = auth.generate_token("tester")
    assert isinstance(token, str)
    assert auth.verify_token(token) == "tester"


def test_acquire_request_requires_non_empty_credentials():
    with pytest.raises(ValidationError):
        AcquireTaskRequest(
            source="root.a.b",
            host="127.0.0.1",
            port="6667",
            user=" ",
            password=" ",
            point_name="FI_1.PV",
            target_points=5000,
        )
