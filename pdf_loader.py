"""
pdf_loader.py

Responsible for loading a PDF file from disk and extracting all of its text
content into a single raw string

"""

import os
import pdfplumber


def load_pdf(filepath: str) -> str:

    #  Validate that the file actually exists before trying to open it 
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found at path: '{filepath}'")

    page_texts = []  # Accumulate text from each page here

    with pdfplumber.open(filepath) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):

            # extract_text() returns None if a page has no extractable text
            text = page.extract_text()

            if text:
                # Strip per-page leading/trailing whitespace before storing
                page_texts.append(text.strip())
            else:
                # Image-only or blank pages are skipped without crashing
                print(f"  [pdf_loader] Page {page_number} had no extractable text — skipping.")

    # if a pdf doesnt have any extractable text
    if not page_texts:
        raise ValueError(
            f"No text could be extracted from '{filepath}'. "
            "The PDF may contain only scanned images."
        )

    # Join all pages with a single newline between them, then strip the whole block
    full_text = "\n".join(page_texts).strip()

    print(f"[pdf_loader] Loaded '{os.path.basename(filepath)}' — "
          f"{len(page_texts)} pages, {len(full_text):,} characters extracted.")

    return full_text
