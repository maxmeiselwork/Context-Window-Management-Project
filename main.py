"""
main.py

Entry point for the memory management experiment.

"""

import argparse
from pipeline import run


DEFAULT_PDF = "data/The-Odyssey-of-Homer---Lattimore-Richmond-lv1hj.pdf"


def main():
    parser = argparse.ArgumentParser(
        description="Run the two-path summarization experiment on a PDF."
    )
    parser.add_argument(
        "--pdf",
        type    = str,
        default = DEFAULT_PDF,
        help    = f"Path to the PDF file to summarize. Defaults to: {DEFAULT_PDF}"
    )
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  MEMORY MANAGEMENT EXPERIMENT")
    print("  Comparing: Baseline vs Long-T5  ->  GPT-4o-mini")
    print("=" * 72)
    print(f"  PDF:    {args.pdf}")
    print(f"  Prompt: Summarize the text. Include all major episodes and stories.")
    print("=" * 72 + "\n")

    run(pdf_path=args.pdf)


if __name__ == "__main__":
    main()
