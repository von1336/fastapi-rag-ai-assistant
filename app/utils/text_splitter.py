import re


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into chunks by paragraphs first, then by sentences if paragraph is too large.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: list[str] = []

    paragraphs = re.split(r"\n\s*\n", text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(sentence) > chunk_size:
                        for i in range(0, len(sentence), chunk_size - overlap):
                            chunk = sentence[i : i + chunk_size]
                            if chunk:
                                chunks.append(chunk)
                        current_chunk = ""
                    else:
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) >= overlap else current_chunk
                        current_chunk = f"{overlap_text} {sentence}".strip()
            if current_chunk:
                chunks.append(current_chunk)

    return [c for c in chunks if c.strip()]
