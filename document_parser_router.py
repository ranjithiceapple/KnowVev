# document_parser_router.py

from pdf2llm import convert_pdf_hybrid
from ppt2llm import convert_pptx_to_markdown
from docx2llm import convert_docx_to_markdown
from pathlib import Path

def parse_to_markdown(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return convert_pdf_hybrid(file_path)

    elif ext in [".pptx", ".ppt"]:
        return convert_pptx_to_markdown(file_path)

    elif ext in [".docx", ".doc"]:
        return convert_docx_to_markdown(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
