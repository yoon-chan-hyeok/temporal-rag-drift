from __future__ import annotations

import unittest

from src.generation.base import GenerationConfig
from src.pipeline.sample_responses_async_openai import completion_request_kwargs


class CompletionRequestKwargsTest(unittest.TestCase):
    def test_legacy_model_uses_max_tokens(self) -> None:
        config = GenerationConfig.from_mapping(
            {
                "backend": "openai_compatible",
                "model_name": "gpt-4o-mini-2024-07-18",
                "max_new_tokens": 72,
                "temperature": 0.8,
                "top_p": 0.95,
            }
        )
        kwargs = completion_request_kwargs(
            config,
            [{"role": "user", "content": "question"}],
            42,
        )
        self.assertEqual(kwargs["max_tokens"], 72)
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertEqual(kwargs["seed"], 42)

    def test_luna_uses_completion_tokens_and_reasoning_effort(self) -> None:
        config = GenerationConfig.from_mapping(
            {
                "backend": "openai_compatible",
                "model_name": "gpt-5.6-luna",
                "max_new_tokens": 72,
                "temperature": 0.8,
                "top_p": 0.95,
                "use_max_completion_tokens": True,
                "reasoning_effort": "none",
            }
        )
        kwargs = completion_request_kwargs(
            config,
            [{"role": "user", "content": "question"}],
            99,
        )
        self.assertEqual(kwargs["max_completion_tokens"], 72)
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["seed"], 99)


if __name__ == "__main__":
    unittest.main()
