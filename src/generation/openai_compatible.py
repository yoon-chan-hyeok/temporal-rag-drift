"""OpenAI-compatible chat completion backend with retries."""

from __future__ import annotations

import logging
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.base import BaseGenerator, GenerationConfig
from src.generation.prompting import build_chat_messages

LOGGER = logging.getLogger(__name__)


class OpenAICompatibleGenerator(BaseGenerator):
    """Generate answers with OpenAI-compatible chat-completions APIs."""

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use backend=openai_compatible") from exc

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing API key environment variable: {config.api_key_env}")
        kwargs = {"api_key": api_key, "timeout": config.request_timeout}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.prompt_mode = str((config.extra or {}).get("prompt_mode", "grounded"))
        self.system_prompt_override = (config.extra or {}).get("system_prompt_override")

    def generate(self, question: str, context: str, sample_seed: int | None = None) -> str:
        """Generate one chat completion, retrying transient failures."""

        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.config.retry_min_seconds,
                max=self.config.retry_max_seconds,
            ),
            reraise=True,
        )
        def _call() -> str:
            LOGGER.debug("Calling OpenAI-compatible backend for model=%s", self.config.model_name)
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=build_chat_messages(
                    question,
                    context,
                    prompt_mode=self.prompt_mode,
                    system_prompt_override=str(self.system_prompt_override) if self.system_prompt_override else None,
                ),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_new_tokens,
                seed=sample_seed,
            )
            content = response.choices[0].message.content or ""
            return content.strip()

        return _call()
