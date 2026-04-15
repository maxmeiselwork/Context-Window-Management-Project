"""
bart_summarizer.py

Summarizes text chunks using BART-large-CNN (facebook/bart-large-cnn),
running on GPU via Hugging Face transformers

"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME        = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS  = 1024   # BART-large-CNN hard limit
MAX_OUTPUT_TOKENS = 256
MIN_OUTPUT_TOKENS = 60


def load_model(model_name: str = MODEL_NAME):
    print(f"[bart_summarizer] Loading '{model_name}' ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype = torch.float16,
        device_map  = "auto"
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"[bart_summarizer] Model loaded on {device}.")

    return model, tokenizer


def _summarize_single(text: str, model, tokenizer) -> str:
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors = "pt",
        max_length     = MAX_INPUT_TOKENS,
        truncation     = True,
        padding        = False,
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask       = inputs["attention_mask"],
            max_length           = MAX_OUTPUT_TOKENS,
            min_length           = MIN_OUTPUT_TOKENS,
            num_beams            = 4,
            length_penalty       = 2.0,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


def summarize_chunks(chunks: list, model_name: str = MODEL_NAME) -> str:
    if not chunks:
        raise ValueError("chunks list is empty — nothing to summarize.")

    model, tokenizer = load_model(model_name)

    print(f"[bart_summarizer] Summarizing {len(chunks)} chunk(s) ...")

    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"  [bart_summarizer] Chunk {i}/{len(chunks)} ...")
        summary = _summarize_single(chunk, model, tokenizer)
        chunk_summaries.append(summary)

    # Combined summaries go straight to GPT-4o-mini for final synthesis
    combined = " ".join(chunk_summaries)
    print(f"[bart_summarizer] Done. Combined summaries: {len(combined.split())} words "
          f"-> passing to GPT-4o-mini for final summary.")

    return combined
