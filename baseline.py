"""
baseline.py

The naive baseline: send as much raw PDF text as the context window allows
directly to GPT-4o-mini — no chunking, no local summarization.

Because the Odyssey is ~202k tokens and GPT-4o-mini's context window is
128k tokens, the text is truncated to MAX_BASELINE_WORDS before sending.
This means the baseline only sees roughly 60% of the document — the
truncation is the key finding that motivates the T5 chunking approach.
"""

from openai_client import summarize_with_openai, TokenUsage, CONTEXT_LIMIT


# At ~1.3 tokens/word, 90,000 words ≈ 117,000 tokens — safely under the
# 128k limit once the system prompt and output tokens are included.
MAX_BASELINE_WORDS = 90_000


def run_baseline(raw_text: str) -> tuple:
    print("[baseline] Running baseline — sending raw text directly to GPT-4o-mini ...")
    print("[baseline] No chunking, no local summarization, no memory management.")

    words       = raw_text.split()
    total_words = len(words)

    if total_words > MAX_BASELINE_WORDS:
        truncated_text = " ".join(words[:MAX_BASELINE_WORDS])
        pct_seen       = (MAX_BASELINE_WORDS / total_words) * 100
        print(f"[baseline] Document truncated: sending first {MAX_BASELINE_WORDS:,} of "
              f"{total_words:,} words ({pct_seen:.1f}% of the document).")
        print(f"[baseline] The remaining {total_words - MAX_BASELINE_WORDS:,} words "
              f"are cut off — the model never sees them.")
    else:
        truncated_text = raw_text
        print("[baseline] Document fits within context window — no truncation needed.")

    summary, usage = summarize_with_openai(truncated_text, label="baseline (truncated)")

    print(f"[baseline] Complete. Summary length: {len(summary.split())} words.")

    return summary, usage
