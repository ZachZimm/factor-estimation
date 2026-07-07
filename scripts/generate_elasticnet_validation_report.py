#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging

from ff5_predictor.elasticnet_validation_report import run_elasticnet_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a standalone ElasticNet fold-validation HTML report.")
    parser.add_argument("--run-dir", help="Completed elasticnet_time_series_validation_v1 run directory. Defaults to newest.")
    parser.add_argument("--output-dir", help="Directory for report artifacts. Defaults to <run-dir>/analysis/fold_validation_report.")
    parser.add_argument("--title", help="HTML report title.")
    parser.add_argument("--open-browser", action="store_true", help="Open the generated report in a browser.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    result = run_elasticnet_validation_report(
        args.run_dir,
        output_dir=args.output_dir,
        title=args.title,
        open_browser=args.open_browser,
    )
    print(result.html_path)


if __name__ == "__main__":
    main()
