from __future__ import annotations

import unittest

from src.data.load_dataset import DatasetRecord
from src.generation.base import GenerationConfig
from src.pipeline.sample_responses import _prompt_question


class TemporalPromptTimeTest(unittest.TestCase):
    def test_clark_time_keys_are_rendered_for_stale_and_current(self) -> None:
        record = DatasetRecord(
            id="clark-test",
            question="Who holds the position?",
            gold_answer="New",
            current_docs=["new"],
            stale_docs=["old"],
            stale_answer="Old",
            metadata={
                "time_x": "2023-07-31",
                "time_y": "2023-11-21",
            },
        )
        config = GenerationConfig.from_mapping(
            {
                "backend": "local_hf",
                "model_name": "test",
                "use_condition_time_prefix": True,
                "condition_time_prefix_template": "As of {time}: {question}",
            }
        )

        self.assertEqual(
            _prompt_question(record, "stale_only", config),
            "As of 2023-07-31: Who holds the position?",
        )
        self.assertEqual(
            _prompt_question(record, "current_only", config),
            "As of 2023-11-21: Who holds the position?",
        )


if __name__ == "__main__":
    unittest.main()
