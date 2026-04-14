"""
openai_client.py

Handles all calls to the OpenAI API and tracks token usage for each request.

Model: gpt-4o-mini — 128k token context window.
"""

import os
import re
from openai import OpenAI
from dataclasses import dataclass


MODEL_ID          = "gpt-4o-mini"
CONTEXT_LIMIT     = 128_000
MAX_OUTPUT_TOKENS = 4096

SYSTEM_PROMPT = (
    "Summarize the following text. Include all major episodes and stories. "
    "Write in plain prose with no Markdown formatting, bullet points, or bold text."
)


@dataclass
class TokenUsage:
    input_tokens:  int
    output_tokens: int
    total_tokens:  int

    def __str__(self):
        return (
            f"Tokens used — input: {self.input_tokens:,}  "
            f"output: {self.output_tokens:,}  "
            f"total: {self.total_tokens:,}  "
            f"(context limit: {CONTEXT_LIMIT:,})"
        )


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Add it to Colab Secrets as 'OPENAI_API_KEY'."
        )
    return OpenAI(api_key=api_key)


def summarize_with_openai(text: str, label: str = "input") -> tuple:
    client = _get_client()

    # Pre-flight token estimate — ~1.3 tokens per word
    word_count       = len(text.split())
    estimated_tokens = int(word_count * 1.3)

    print(f"[openai_client] '{label}' input — "
          f"{word_count:,} words, ~{estimated_tokens:,} estimated tokens "
          f"(limit: {CONTEXT_LIMIT:,})")

    if estimated_tokens > CONTEXT_LIMIT:
        print(f"  [openai_client] WARNING: estimated tokens ({estimated_tokens:,}) "
              f"EXCEED context window ({CONTEXT_LIMIT:,}). "
              f"Attempting anyway — expect an error or truncated output.")

    try:
        response = client.chat.completions.create(
            model      = MODEL_ID,
            max_tokens = MAX_OUTPUT_TOKENS,
            messages   = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text}
            ]
        )
    except Exception as e:
        raise RuntimeError(
            f"OpenAI rejected the request for '{label}'. "
            f"Input was ~{estimated_tokens:,} tokens (limit: {CONTEXT_LIMIT:,}). "
            f"Original error: {e}"
        ) from e

    # Strip any stray Markdown bold markers GPT might add
    summary = re.sub(r'\*+', '', response.choices[0].message.content)

    usage = TokenUsage(
        input_tokens  = response.usage.prompt_tokens,
        output_tokens = response.usage.completion_tokens,
        total_tokens  = response.usage.total_tokens,
    )

    print(f"[openai_client] '{label}' complete. {usage}")

    return summary, usage
