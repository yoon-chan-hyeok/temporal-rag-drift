"""CLI: compute metrics from sampled responses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.compute_metrics import run_compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_compute_metrics(args.run_dir)


if __name__ == "__main__":
    main()
