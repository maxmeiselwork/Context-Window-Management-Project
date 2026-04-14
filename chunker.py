"""
chunker.py

"""
CHUNK_SIZE   = 700  # Words per chunk (~910 tokens) — sized to fit within
                    # BART-large-CNN's 1,024 token hard limit.
OVERLAP_SIZE =  50  # Words carried over from the previous chunk



def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> list:

    #  Input validation 
    if not text or not text.strip():
        raise ValueError("Input text is empty — nothing to chunk.")

    if overlap >= chunk_size:
        raise ValueError(
            f"Overlap ({overlap}) must be smaller than chunk_size ({chunk_size}). "
            "An overlap >= chunk_size would cause an infinite loop."
        )

    #  Tokenize by whitespace 
    words = text.split()

    chunks    = []   # Accumulate finished chunks here
    start_idx = 0   # Index into `words` where the current chunk begins

    #  Sliding window over the word list 
    while start_idx < len(words):

        # Slice out the words for this chunk
        end_idx     = start_idx + chunk_size
        chunk_words = words[start_idx:end_idx]

        # Rejoin words into a readable string and store
        chunks.append(" ".join(chunk_words))

        # Advance the window by (chunk_size - overlap) so the next chunk
        start_idx += chunk_size - overlap

    print(f"[chunker] Split text into {len(chunks)} chunks "
          f"(chunk_size={chunk_size} words, overlap={overlap} words).")

    return chunks
