"""
Command‑line entry point for ModelTest.

This script loads configuration from environment variables, executes
all defined tests (jailbreak, bias and injection) and writes a
comprehensive HTML report. Users can specify the output location of
the report via the `--output` command‑line argument. If no output
argument is provided, the report will be saved to `report.html` in
the current working directory.

Example usage:
    python run_tests.py --output my_report.html

See the README for details on how to set environment variables to
control model endpoints, API keys and other parameters.
"""

import argparse
import os

from src.config import Config
from src.tester import run_all_tests
from src.report_generator import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model evaluation tests and generate an HTML report."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report.html",
        help="Path to save the generated HTML report.",
    )
    args = parser.parse_args()

    # Instantiate configuration from environment variables
    config = Config()

    # Execute all tests
    results = run_all_tests(config)

    # Generate the report
    output_path = args.output
    generate_report(results, output_path)

    print(f"Report generated at {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
