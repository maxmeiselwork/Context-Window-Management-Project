"""
pipeline.py

"""

import time
import traceback

from pdf_loader    import load_pdf
from chunker       import chunk_text
from baseline      import run_baseline
from t5_summarizer import summarize_chunks as t5_summarize
from openai_client import summarize_with_openai, TokenUsage
from evaluate      import evaluate



def _failed_summary(path_name: str, error: Exception) -> str:
    return f"[{path_name} FAILED: {type(error).__name__}: {error}]"


def _zero_usage() -> TokenUsage:
    return TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)



# Main pipeline

def run(pdf_path: str):
    """
    Run the full two-path experiment on a single PDF and print the report.
    """

    # Step 1 — Load PDF

    print("\n[pipeline] === Step 1: Loading PDF ===")
    raw_text       = load_pdf(pdf_path)
    raw_word_count = len(raw_text.split())
    print(f"[pipeline] Document loaded — {raw_word_count:,} words total.\n")

    # Step 2 — Chunk the raw text (used by the T5 path)
    print("[pipeline] === Step 2: Chunking text ===")
    chunks = chunk_text(raw_text)
    print(f"[pipeline] {len(chunks)} chunks ready.\n")

    # Step 3 — Path A: Baseline (raw text -> GPT-4o-mini, no chunking)
    print("[pipeline] === Step 3: Path A — Baseline ===")
    baseline_start = time.time()
    try:
        baseline_summary, baseline_usage = run_baseline(raw_text)
    except Exception as e:
        print(f"[pipeline] Baseline FAILED (expected — input exceeds context window):")
        traceback.print_exc()
        baseline_summary = _failed_summary("Baseline", e)
        baseline_usage   = _zero_usage()
    baseline_elapsed = time.time() - baseline_start
    print(f"[pipeline] Baseline finished in {baseline_elapsed:.1f}s\n")

    # Step 4 — Path B: T5 local summarization -> GPT-4o-mini
    print("[pipeline] === Step 4: Path B — T5 + GPT-4o-mini ===")
    t5_start = time.time()
    try:
        # T5 summarizes all chunks locally into one condensed string
        t5_local_summary = t5_summarize(chunks)

        # Pass the condensed T5 output to GPT-4o-mini for a final polished summary
        t5_summary, t5_usage = summarize_with_openai(
            t5_local_summary, label="T5 (chunked)"
        )
    except Exception as e:
        print(f"[pipeline] T5 path FAILED:")
        traceback.print_exc()
        t5_summary = _failed_summary("T5", e)
        t5_usage   = _zero_usage()
    t5_elapsed = time.time() - t5_start
    print(f"[pipeline] T5 path finished in {t5_elapsed:.1f}s\n")

    # Step 5 — Evaluate and print comparison report + generate charts
    print("[pipeline] === Step 5: Evaluation ===")
    evaluate(
        raw_word_count   = raw_word_count,
        baseline_summary = baseline_summary,
        baseline_usage   = baseline_usage,
        baseline_elapsed = baseline_elapsed,
        t5_summary       = t5_summary,
        t5_usage         = t5_usage,
        t5_elapsed       = t5_elapsed,
    )
