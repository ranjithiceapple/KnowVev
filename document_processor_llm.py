# document_processor_llm.py

from datetime import datetime
from typing import List
from document_parser_router import parse_to_markdown
from document_processor import PageMetadata, DocumentMetadata, ExtractionResult
import re

def extract_text_from_document(file_path: str, **kwargs) -> ExtractionResult:
    md_text = parse_to_markdown(file_path)

    # Split markdown into logical pages/sections
    sections = re.split(r"\n#{1,6}\s+", md_text)

    pages: List[PageMetadata] = []
    total_chars = 0
    total_words = 0

    for i, section in enumerate(sections, start=1):
        text = section.strip()
        if not text:
            continue

        char_count = len(text)
        word_count = len(text.split())

        pages.append(
            PageMetadata(
                page_number=i,
                text=text,
                char_count=char_count,
                word_count=word_count,
            )
        )

        total_chars += char_count
        total_words += word_count

    metadata = DocumentMetadata(
        file_name=file_path.split("/")[-1],
        file_type="markdown",
        total_pages=len(pages),
        total_chars=total_chars,
        total_words=total_words,
        extraction_date=datetime.utcnow().isoformat(),
        pages=pages,
    )

    return ExtractionResult(
        text=md_text,
        metadata=metadata,
        pages=pages
    )
