"""
json_loader.py

Loads a Claude conversation export (conversations.json) and converts it
into a single readable text string for the summarization pipeline.

Claude exports conversations as an array of conversation objects, each
with a list of chat_messages containing sender and text fields.
"""

import json
import os
from datetime import datetime


def load_json(filepath: str) -> str:

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found at path: '{filepath}'")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both a single conversation object and an array of conversations
    if isinstance(data, dict):
        conversations = [data]
    else:
        conversations = data

    blocks = []

    for convo in conversations:
        title    = convo.get("name") or convo.get("title") or "Untitled"
        created  = convo.get("created_at", "")

        # Parse date for readable header
        try:
            date_str = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            date_str = created[:10] if created else "unknown date"

        header = f"=== CONVERSATION: {title} ({date_str}) ==="
        lines  = [header]

        messages = convo.get("chat_messages") or convo.get("messages") or []

        for msg in messages:
            sender = msg.get("sender", "unknown").capitalize()
            text   = (msg.get("text") or "").strip()
            if text:
                lines.append(f"{sender}: {text}")

        if len(lines) > 1:  # only include conversations that had messages
            blocks.append("\n".join(lines))

    if not blocks:
        raise ValueError(
            f"No conversation messages could be extracted from '{filepath}'. "
            "Check that the file is a valid Claude conversations export."
        )

    full_text = "\n\n".join(blocks)

    print(f"[json_loader] Loaded '{os.path.basename(filepath)}' — "
          f"{len(conversations)} conversation(s), {len(full_text.split()):,} words extracted.")

    return full_text
