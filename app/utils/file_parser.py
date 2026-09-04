import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(
    filepath: str | Path,
    content_type: str,
    max_pdf_pages: int = 100,
    max_extracted_chars: int = 1_000_000,
) -> str:
    """
    Extract text from file. Supports .txt, .md, .csv, .pdf.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix in (".txt", ".md", ".csv"):
        return _read_text_file(path, max_extracted_chars)

    if suffix == ".pdf":
        return _extract_pdf(path, max_pdf_pages, max_extracted_chars)

    raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .md, .csv, .pdf")


def _read_text_file(path: Path, max_extracted_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
        if len(text) > max_extracted_chars:
            raise ValueError("Extracted text is too large")
        return text
    except Exception as e:
        logger.exception("Failed to read text file %s: %s", path, e)
        raise


def _extract_pdf(path: Path, max_pdf_pages: int, max_extracted_chars: int) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        if len(reader.pages) > max_pdf_pages:
            raise ValueError("PDF exceeds maximum page limit")
        parts = []
        total_chars = 0
        for page in reader.pages:
            text = page.extract_text()
            if text:
                total_chars += len(text)
                if total_chars > max_extracted_chars:
                    raise ValueError("Extracted text is too large")
                parts.append(text)
        return "\n\n".join(parts) if parts else ""
    except ImportError:
        raise ImportError(
            "PDF support requires pypdf. Install with: pip install pypdf"
        )
    except Exception as e:
        logger.exception("Failed to extract PDF %s: %s", path, e)
        raise
