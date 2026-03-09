#!/usr/bin/env python3
"""CLI tool for generating benchmark reports."""
import argparse
import sys
from pathlib import Path

import httpx


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate markdown report from TensorRT BenchLab run"
    )
    parser.add_argument("--run-id", required=True, help="Run ID to generate report for")
    parser.add_argument(
        "--controller-url",
        default="http://localhost:8000",
        help="Controller URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        help="Output file path (default: {run_id}.md)",
    )
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print report to stdout instead of saving",
    )

    args = parser.parse_args()

    # Fetch report from controller
    url = f"{args.controller_url}/runs/{args.run_id}/report.md"

    print(f"Fetching report from {url}...", file=sys.stderr)

    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        markdown = data["markdown"]

        if args.print:
            # Print to stdout
            print(markdown)
        else:
            # Save to file
            output_path = args.out or Path(f"{args.run_id}.md")
            output_path.write_text(markdown)
            print(f"✅ Report saved to {output_path}", file=sys.stderr)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"❌ Error: Run {args.run_id} not found", file=sys.stderr)
        else:
            print(f"❌ HTTP Error: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"❌ Connection Error: {e}", file=sys.stderr)
        print(
            f"   Make sure the controller is running at {args.controller_url}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
