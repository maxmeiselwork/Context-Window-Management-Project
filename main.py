"""
main.py

"""

import argparse
from pipeline import run


DEFAULT_JSON = "data/conversations.json"


def main():
    parser = argparse.ArgumentParser(
        description="Run the two-path memory compression experiment on a Claude conversation export."
    )
    parser.add_argument(
        "--json",
        type    = str,
        default = DEFAULT_JSON,
        help    = f"Path to the conversations JSON file. Defaults to: {DEFAULT_JSON}"
    )
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  MEMORY MANAGEMENT EXPERIMENT")
    print("  Comparing: Baseline vs BART-large-CNN  ->  GPT-4o-mini")
    print("=" * 72)
    print(f"  Source: {args.json}")
    print("=" * 72 + "\n")

    run(json_path=args.json)


if __name__ == "__main__":
    main()
