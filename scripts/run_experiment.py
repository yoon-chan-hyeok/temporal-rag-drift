"""CLI: run sampling, metrics, stats, and figures end to end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.compute_metrics import run_compute_metrics
from src.pipeline.make_figures import run_make_figures
from src.pipeline.run_stats import run_stats
from src.pipeline.sample_responses import run_sampling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run_dir", default=None, help="Optional run dir for checkpoint resume.")
    parser.add_argument("--skip_sampling", action="store_true", help="Reuse an existing run_dir without new generation.")
    args = parser.parse_args()

    if args.skip_sampling:
        if not args.run_dir:
            raise SystemExit("--skip_sampling requires --run_dir")
        run_dir = Path(args.run_dir)
    else:
        run_dir = run_sampling(args.config, run_dir=args.run_dir)
    run_compute_metrics(run_dir)
    run_stats(run_dir)
    run_make_figures(run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
