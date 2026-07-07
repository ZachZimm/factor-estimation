#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging

from ff5_predictor.performance_analysis import run_performance_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a standalone model performance analysis report.")
    parser.add_argument("--manifest", help="Architecture comparison manifest.tsv. Defaults to newest completed manifest.")
    parser.add_argument("--output-dir", help="Directory for report artifacts. Defaults to <manifest-dir>/performance_analysis.")
    parser.add_argument("--title", help="HTML report title.")
    parser.add_argument("--open-browser", action="store_true", help="Open the generated report in a browser.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    result = run_performance_analysis(
        args.manifest,
        output_dir=args.output_dir,
        title=args.title,
        open_browser=args.open_browser,
    )
    print(result.html_path)


if __name__ == "__main__":
    main()
