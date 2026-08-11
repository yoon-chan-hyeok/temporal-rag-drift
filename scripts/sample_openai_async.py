"""CLI for resumable asynchronous OpenAI sampling."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.sample_responses_async_openai import run_sampling_async


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--requests-per-minute", type=int, default=None)
    parser.add_argument("--tokens-per-minute", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        run_sampling_async(
            args.config,
            args.run_dir,
            concurrency=args.concurrency,
            requests_per_minute=args.requests_per_minute,
            tokens_per_minute=args.tokens_per_minute,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
