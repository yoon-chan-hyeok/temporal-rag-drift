"""CLI: sample responses for all configured interventions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.sample_responses import run_sampling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run_dir", default=None, help="Existing/new run dir for checkpoint resume.")
    args = parser.parse_args()
    run_dir = run_sampling(args.config, run_dir=args.run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
