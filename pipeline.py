"""
pipeline.py

"""

import time
import traceback

from pdf_loader      import load_pdf
from chunker         import chunk_text
from baseline        import run_baseline
from bart_summarizer import summarize_chunks as bart_summarize
from openai_client   import summarize_with_openai, TokenUsage
from evaluate        import evaluate



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
    print(f"[pipeline] Document loaded — {raw_word_count:,} words total.")

    raw_text_for_bart = raw_text
    print()

    # Step 2 — Chunk the capped text for the BART path
    print("[pipeline] === Step 2: Chunking text ===")
    chunks = chunk_text(raw_text_for_bart)
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

    # Step 4 — Path B: BART local summarization -> GPT-4o-mini
    print("[pipeline] === Step 4: Path B — BART + GPT-4o-mini ===")
    bart_start = time.time()
    try:
        # BART summarizes all chunks locally into one condensed string
        bart_local_summary = bart_summarize(chunks)

        # Pass the condensed BART output to GPT-4o-mini for a final polished summary
        bart_summary, bart_usage = summarize_with_openai(
            bart_local_summary, label="BART (chunked)"
        )
    except Exception as e:
        print(f"[pipeline] BART path FAILED:")
        traceback.print_exc()
        bart_summary = _failed_summary("BART", e)
        bart_usage   = _zero_usage()
    bart_elapsed = time.time() - bart_start
    print(f"[pipeline] BART path finished in {bart_elapsed:.1f}s\n")

    # Step 5 — Evaluate and print comparison report + generate charts
    print("[pipeline] === Step 5: Evaluation ===")
    evaluate(
        raw_word_count   = raw_word_count,
        baseline_summary = baseline_summary,
        baseline_usage   = baseline_usage,
        baseline_elapsed = baseline_elapsed,
        bart_summary     = bart_summary,
        bart_usage       = bart_usage,
        bart_elapsed     = bart_elapsed,
    )
