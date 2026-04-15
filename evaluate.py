"""
evaluate.py
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from openai_client import TokenUsage


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Use non-interactive backend so charts save without needing a display.
matplotlib.use("Agg")

COLOURS = ["#6c757d", "#0d9488"]
LABELS  = ["Baseline\n(GPT-4o-mini)", "BART-large-CNN\n+ GPT-4o-mini"]


# Metric computation

def _compute_metrics(label, raw_word_count, summary, usage, elapsed) -> dict:
    summary_words = len(summary.split())

    # What fraction of the original document does the summary represent?
    compression_ratio = summary_words / raw_word_count if raw_word_count > 0 else 0.0

    # Input tokens spent per word of useful output — lower is more efficient
    tokens_per_output_word = (
        usage.input_tokens / summary_words if summary_words > 0 else float("inf")
    )

    return {
        "label"                 : label,
        "input_tokens"          : usage.input_tokens,
        "output_tokens"         : usage.output_tokens,
        "total_tokens"          : usage.total_tokens,
        "elapsed_seconds"       : elapsed,
        "summary_words"         : summary_words,
        "compression_ratio"     : compression_ratio,
        "tokens_per_output_word": tokens_per_output_word,
        "summary"               : summary,
    }


# Console report

def _print_divider():
    print("-" * 72)


def _print_table(results: list):
    print("\n" + "=" * 72)
    print("  RESULTS COMPARISON")
    print("=" * 72)
    print(f"  {'Metric':<30} {'Baseline':>18} {'BART':>18}")
    _print_divider()

    metrics = [
        ("Input tokens",          "input_tokens",           ",d"),
        ("Output tokens",         "output_tokens",          ",d"),
        ("Total tokens",          "total_tokens",           ",d"),
        ("Elapsed (seconds)",     "elapsed_seconds",        ".1f"),
        ("Summary words",         "summary_words",          ",d"),
        ("Compression ratio",     "compression_ratio",      ".4f"),
        ("Tokens / output word",  "tokens_per_output_word", ",.1f"),
    ]

    for display_name, key, fmt in metrics:
        row = f"  {display_name:<30}"
        for r in results:
            row += f"  {format(r[key], fmt):>18}"
        print(row)

    _print_divider()

    # Token savings row
    baseline_input = results[0]["input_tokens"]
    print(f"  {'Token savings vs baseline':<30}", end="")
    for r in results:
        savings = baseline_input - r["input_tokens"]
        print(f"  {f'{savings:+,}':>18}", end="")
    print()

    _print_divider()


def _print_summaries(results: list):
    print("\n" + "=" * 72)
    print("  SUMMARIES")
    print("=" * 72)

    for r in results:
        print(f"\n  [{r['label'].upper()}]")
        _print_divider()
        words = r["summary"].split()
        line  = "  "
        for word in words:
            if len(line) + len(word) + 1 > 72:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
        print()


# Chart helpers

def _save(fig: plt.Figure, filename: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[evaluate] Chart saved → {path}")


def _bar_chart(title, ylabel, values, filename, fmt=".0f", note=""):
    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(LABELS, values, color=COLOURS, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            format(val, fmt),
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)

    if note:
        ax.set_xlabel(note, fontsize=9, color="#555555")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, filename)


# Individual chart generators

def _chart_token_usage(results: list):
    fig, ax = plt.subplots(figsize=(8, 5))

    x      = range(len(LABELS))
    width  = 0.25
    groups = [
        ("Input tokens",  "input_tokens",  "#0d9488"),
        ("Output tokens", "output_tokens", "#f59e0b"),
        ("Total tokens",  "total_tokens",  "#6c757d"),
    ]

    for i, (group_label, key, colour) in enumerate(groups):
        offsets = [xi + (i - 1) * width for xi in x]
        vals    = [r[key] for r in results]
        bars    = ax.bar(offsets, vals, width=width, label=group_label,
                         color=colour, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:,}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_title("Token Usage by Path", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Tokens", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS, fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "token_usage.png")


def _chart_speed(results):
    _bar_chart(
        title    = "End-to-End Processing Time",
        ylabel   = "Seconds",
        values   = [r["elapsed_seconds"] for r in results],
        filename = "speed.png",
        fmt      = ".1f",
        note     = "Lower is faster"
    )


def _chart_compression(results):
    _bar_chart(
        title    = "Compression Ratio",
        ylabel   = "Summary words / Source words",
        values   = [r["compression_ratio"] for r in results],
        filename = "compression.png",
        fmt      = ".4f",
        note     = "Lower = more compression"
    )


def _chart_efficiency(results):
    _bar_chart(
        title    = "Token Efficiency",
        ylabel   = "Input tokens per output word",
        values   = [r["tokens_per_output_word"] for r in results],
        filename = "efficiency.png",
        fmt      = ",.1f",
        note     = "Lower = more efficient use of GPT-4o-mini's context"
    )


def _chart_summary_words(results):
    _bar_chart(
        title    = "Final Summary Length",
        ylabel   = "Words",
        values   = [r["summary_words"] for r in results],
        filename = "summary_words.png",
        fmt      = ",d",
    )


# Main entry point

def evaluate(
    raw_word_count:   int,
    baseline_summary: str,
    baseline_usage:   TokenUsage,
    baseline_elapsed: float,
    bart_summary:     str,
    bart_usage:       TokenUsage,
    bart_elapsed:     float,
):
    results = [
        _compute_metrics("Baseline", raw_word_count, baseline_summary, baseline_usage, baseline_elapsed),
        _compute_metrics("BART",     raw_word_count, bart_summary,     bart_usage,     bart_elapsed),
    ]

    _print_table(results)
    _print_summaries(results)

    print("\n[evaluate] Generating charts ...")
    _chart_token_usage(results)
    _chart_speed(results)
    _chart_compression(results)
    _chart_efficiency(results)
    _chart_summary_words(results)

    print("\n" + "=" * 72)
    print(f"  Evaluation complete. Charts saved to: {RESULTS_DIR}")
    print("=" * 72 + "\n")
