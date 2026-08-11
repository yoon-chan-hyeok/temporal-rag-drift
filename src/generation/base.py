"""Generation backend abstractions and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Shared generation parameters."""

    backend: str
    model_name: str
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 128
    n_samples: int = 16
    request_timeout: int = 120
    max_retries: int = 5
    retry_min_seconds: float = 1.0
    retry_max_seconds: float = 30.0
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    extra: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "GenerationConfig":
        """Parse config values from a mapping."""
        known = {
            "backend",
            "model_name",
            "temperature",
            "top_p",
            "max_new_tokens",
            "n_samples",
            "request_timeout",
            "max_retries",
            "retry_min_seconds",
            "retry_max_seconds",
            "base_url",
            "api_key_env",
        }
        extra = {key: value for key, value in data.items() if key not in known}
        return cls(
            backend=str(data.get("backend", "openai_compatible")),
            model_name=str(data.get("model_name", "")),
            temperature=float(data.get("temperature", 0.8)),
            top_p=float(data.get("top_p", 0.95)),
            max_new_tokens=int(data.get("max_new_tokens", 128)),
            n_samples=int(data.get("n_samples", 16)),
            request_timeout=int(data.get("request_timeout", 120)),
            max_retries=int(data.get("max_retries", 5)),
            retry_min_seconds=float(data.get("retry_min_seconds", 1)),
            retry_max_seconds=float(data.get("retry_max_seconds", 30)),
            base_url=data.get("base_url"),
            api_key_env=str(data.get("api_key_env", "OPENAI_API_KEY")),
            extra=extra,
        )


class BaseGenerator(ABC):
    """Abstract text generation backend."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, question: str, context: str, sample_seed: int | None = None) -> str:
        """Generate one answer."""


def build_generator(config: GenerationConfig) -> BaseGenerator:
    """Instantiate a generation backend by name."""
    backend = config.backend.lower()
    if backend == "openai_compatible":
        from src.generation.openai_compatible import OpenAICompatibleGenerator

        return OpenAICompatibleGenerator(config)
    if backend == "local_hf":
        from src.generation.local_hf import LocalHFGenerator

        return LocalHFGenerator(config)
    if backend == "mock":
        return MockGenerator(config)
    raise ValueError(f"Unsupported generation backend: {config.backend}")


class MockGenerator(BaseGenerator):
    """A deterministic local backend for pipeline smoke tests."""

    def generate(self, question: str, context: str, sample_seed: int | None = None) -> str:
        prefix = context.strip().splitlines()[-1] if context.strip() else "insufficient context"
        return f"Mock answer for: {question} | {prefix[:160]}"
