import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import fitz  # PyMuPDF
from docx import Document
import io
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class PageMetadata:
    """Metadata for a single page."""
    page_number: int
    text: str
    char_count: int
    word_count: int
    urls: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    has_toc: bool = False
    sections: List[str] = field(default_factory=list)  # Section titles if present

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DocumentMetadata:
    """Complete document metadata."""
    file_name: str
    file_type: str
    total_pages: int
    total_chars: int
    total_words: int
    extraction_date: str
    pages: List[PageMetadata] = field(default_factory=list)
    all_urls: List[str] = field(default_factory=list)
    all_emails: List[str] = field(default_factory=list)
    all_headings: List[str] = field(default_factory=list)
    has_toc: bool = False
    toc_page_numbers: List[int] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)  # Document sections
    total_sections: int = 0  # Number of sections found
    has_section_breaks: bool = False  # Whether document has section breaks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'file_name': self.file_name,
            'file_type': self.file_type,
            'total_pages': self.total_pages,
            'total_chars': self.total_chars,
            'total_words': self.total_words,
            'extraction_date': self.extraction_date,
            'all_urls': self.all_urls,
            'all_emails': self.all_emails,
            'all_headings': self.all_headings,
            'has_toc': self.has_toc,
            'toc_page_numbers': self.toc_page_numbers,
            'sections': self.sections,
            'total_sections': self.total_sections,
            'has_section_breaks': self.has_section_breaks,
            'pages': [page.to_dict() for page in self.pages]
        }


@dataclass
class ExtractionResult:
    """Complete extraction result with text and metadata."""
    text: str
    metadata: DocumentMetadata
    pages: List[PageMetadata]
    
    def get_text_by_page(self, page_number: int) -> Optional[str]:
        """Get text for a specific page."""
        for page in self.pages:
            if page.page_number == page_number:
                return page.text
        return None
    
    def get_pages_range(self, start: int, end: int) -> str:
        """Get combined text from page range."""
        texts = []
        for page in self.pages:
            if start <= page.page_number <= end:
                texts.append(page.text)
        return "\n\n".join(texts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'metadata': self.metadata.to_dict(),
            'pages': [page.to_dict() for page in self.pages]
        }


class TextPatternDetector:
    """Detect various patterns in text for metadata extraction."""
    
    def __init__(self):
        # URL pattern
        self.url_pattern = re.compile(
            r'(?:https?|ftp)://[^\s]+|www\.[^\s]+',
            re.IGNORECASE
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Heading patterns (all caps, title case, numbered sections)
        self.heading_patterns = [
            re.compile(r'^[A-Z][A-Z\s]{5,}$', re.MULTILINE),  # ALL CAPS HEADINGS
            re.compile(r'^(?:Chapter|Section|Part)\s+\d+[:\s]', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\d+\.\s+[A-Z][a-zA-Z\s]{3,}$', re.MULTILINE),  # 1. Heading Text
            re.compile(r'^\d+\.\d+\s+[A-Z][a-zA-Z\s]{3,}$', re.MULTILINE),  # 1.1 Heading Text
        ]
        
        # TOC patterns
        self.toc_patterns = [
            re.compile(r'(?i)table\s+of\s+contents'),
            re.compile(r'\.{3,}'),  # Dot leaders
            re.compile(r'^.{10,}\.{3,}\s*\d+\s*$', re.MULTILINE),  # Title ... 123
            re.compile(r'^\d+\.\d+\s+.+\s+\d+\s*$', re.MULTILINE),  # 1.2 Title 45
        ]
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text."""
        urls = list(set(self.url_pattern.findall(text)))
        logger.debug(f"Pattern Detection: Found {len(urls)} unique URLs")
        return urls
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract all email addresses from text."""
        emails = list(set(self.email_pattern.findall(text)))
        logger.debug(f"Pattern Detection: Found {len(emails)} unique email addresses")
        return emails
    
    def extract_headings(self, text: str) -> List[str]:
        """Extract potential headings from text."""
        headings = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue

            # Check against heading patterns
            for pattern in self.heading_patterns:
                if pattern.match(line):
                    headings.append(line)
                    logger.debug(f"Pattern Detection: Detected heading: '{line[:50]}...'")
                    break

        unique_headings = list(set(headings))
        logger.debug(f"Pattern Detection: Found {len(unique_headings)} unique headings")
        return unique_headings
    
    def detect_toc(self, text: str) -> bool:
        """Detect if text contains Table of Contents."""
        for i, pattern in enumerate(self.toc_patterns):
            if pattern.search(text):
                logger.debug(f"Pattern Detection: TOC detected (pattern {i+1}/{len(self.toc_patterns)})")
                return True
        logger.debug("Pattern Detection: No TOC detected")
        return False


def extract_text_from_document(
    file_path: Union[str, Path, io.BytesIO],
    file_type: Optional[str] = None,
    encoding: str = 'utf-8',
    extract_metadata: bool = True,
    preserve_page_structure: bool = True
) -> Union[str, ExtractionResult]:
    """
    Extract text content from various document formats with optional metadata.
    
    Args:
        file_path: Path to the document file or BytesIO object
        file_type: Explicit file type ('pdf', 'docx', 'txt'). 
                  If None, inferred from file extension
        encoding: Text encoding for TXT files (default: 'utf-8')
        extract_metadata: If True, return ExtractionResult with metadata
        preserve_page_structure: If True, maintain page-level separation
    
    Returns:
        str or ExtractionResult: Extracted text or complete result with metadata
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If unsupported file type is provided
        Exception: For any processing errors during extraction
        
    Example:
        >>> # Simple text extraction
        >>> text = extract_text_from_document('report.pdf', extract_metadata=False)
        >>> 
        >>> # Full extraction with metadata
        >>> result = extract_text_from_document('report.pdf', extract_metadata=True)
        >>> print(result.metadata.total_pages)
        >>> print(result.pages[0].urls)
    """
    try:
        file_name = "buffer"
        
        # Determine file type
        if isinstance(file_path, (str, Path)):
            file_path = Path(file_path)
            file_name = file_path.name
            
            # Validate file exists
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"Document not found: {file_path}")
            
            # Infer file type from extension if not provided
            if file_type is None:
                file_type = file_path.suffix.lower().lstrip('.')
                logger.debug(f"Inferred file type: {file_type} from {file_path.name}")
            
            logger.info(f"Processing document: {file_path.name} (type: {file_type})")
        else:
            # Handle BytesIO objects
            if file_type is None:
                logger.error("file_type must be specified for BytesIO objects")
                raise ValueError("file_type parameter is required for BytesIO objects")
            logger.info(f"Processing document from buffer (type: {file_type})")
        
        # Process based on file type
        if file_type == 'pdf':
            if extract_metadata:
                result = _extract_from_pdf_with_metadata(file_path, file_name)
            else:
                text = _extract_from_pdf(file_path)
                result = text
            
        elif file_type == 'docx':
            if extract_metadata:
                result = _extract_from_docx_with_metadata(file_path, file_name)
            else:
                text = _extract_from_docx(file_path)
                result = text
            
        elif file_type == 'txt':
            if extract_metadata:
                result = _extract_from_txt_with_metadata(file_path, file_name, encoding)
            else:
                text = _extract_from_txt(file_path, encoding)
                result = text
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(
                f"Unsupported file type: {file_type}. "
                f"Supported types: pdf, docx, txt"
            )
        
        # Log extraction statistics
        if isinstance(result, str):
            char_count = len(result)
            word_count = len(result.split())
            logger.info(
                f"Successfully extracted text: {char_count} characters, "
                f"{word_count} words"
            )
        else:
            logger.info(
                f"Successfully extracted document: {result.metadata.total_pages} pages, "
                f"{result.metadata.total_chars} characters, "
                f"{result.metadata.total_words} words"
            )
        
        return result
        
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from document: {str(e)}") from e


def _extract_from_pdf(file_path: Union[Path, io.BytesIO]) -> str:
    """Extract text from PDF using PyMuPDF (simple mode)."""
    try:
        logger.debug("Starting PDF text extraction")
        
        if isinstance(file_path, io.BytesIO):
            doc = fitz.open(stream=file_path, filetype="pdf")
        else:
            doc = fitz.open(file_path)
        
        text_parts = []
        
        for page_num in range(len(doc)):
            logger.debug(f"Extracting text from page {page_num + 1}/{len(doc)}")
            page = doc[page_num]
            text_parts.append(page.get_text())
        
        doc.close()
        
        full_text = "\n\n".join(text_parts)
        logger.debug(f"PDF extraction complete: {len(doc)} pages processed")
        
        return full_text
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from PDF: {str(e)}") from e


def _extract_from_pdf_with_metadata(
    file_path: Union[Path, io.BytesIO],
    file_name: str
) -> ExtractionResult:
    """Extract text from PDF with full metadata."""
    try:
        logger.debug("Starting PDF text extraction with metadata")
        
        if isinstance(file_path, io.BytesIO):
            doc = fitz.open(stream=file_path, filetype="pdf")
        else:
            doc = fitz.open(file_path)
        
        detector = TextPatternDetector()
        pages_metadata = []
        all_text_parts = []
        all_urls = set()
        all_emails = set()
        all_headings = set()
        toc_pages = []
        
        for page_num in range(len(doc)):
            logger.debug(f"PDF Extraction: Processing page {page_num + 1}/{len(doc)}")
            page = doc[page_num]
            page_text = page.get_text()

            char_count = len(page_text)
            word_count = len(page_text.split())
            logger.debug(f"PDF Extraction: Page {page_num + 1} - {char_count} chars, {word_count} words")

            # Extract metadata for this page
            urls = detector.extract_urls(page_text)
            emails = detector.extract_emails(page_text)
            headings = detector.extract_headings(page_text)
            has_toc = detector.detect_toc(page_text)

            if has_toc:
                toc_pages.append(page_num + 1)
                logger.debug(f"PDF Extraction: Page {page_num + 1} contains Table of Contents")

            if urls:
                logger.debug(f"PDF Extraction: Page {page_num + 1} - Found URLs: {urls[:3]}{'...' if len(urls) > 3 else ''}")
            if emails:
                logger.debug(f"PDF Extraction: Page {page_num + 1} - Found emails: {emails[:3]}{'...' if len(emails) > 3 else ''}")
            if headings:
                logger.debug(f"PDF Extraction: Page {page_num + 1} - Found {len(headings)} headings")

            # Update global collections
            all_urls.update(urls)
            all_emails.update(emails)
            all_headings.update(headings)

            # Create page metadata
            page_meta = PageMetadata(
                page_number=page_num + 1,
                text=page_text,
                char_count=char_count,
                word_count=word_count,
                urls=urls,
                emails=emails,
                headings=headings,
                has_toc=has_toc,
                sections=[]  # PDF pages don't have section breaks within a page
            )

            pages_metadata.append(page_meta)
            all_text_parts.append(page_text)
        
        doc.close()
        
        # Combine all text
        full_text = "\n\n".join(all_text_parts)
        
        # Create document metadata
        doc_metadata = DocumentMetadata(
            file_name=file_name,
            file_type='pdf',
            total_pages=len(pages_metadata),
            total_chars=len(full_text),
            total_words=len(full_text.split()),
            extraction_date=datetime.now().isoformat(),
            pages=pages_metadata,
            all_urls=list(all_urls),
            all_emails=list(all_emails),
            all_headings=list(all_headings),
            has_toc=len(toc_pages) > 0,
            toc_page_numbers=toc_pages,
            sections=[],  # PDF doesn't have section breaks like DOCX
            total_sections=0,
            has_section_breaks=False
        )
        
        logger.debug(
            f"PDF Extraction Complete: {len(pages_metadata)} pages, "
            f"{len(all_urls)} URLs, {len(all_emails)} emails, {len(all_headings)} headings, "
            f"TOC pages: {toc_pages if toc_pages else 'None'}"
        )
        
        return ExtractionResult(
            text=full_text,
            metadata=doc_metadata,
            pages=pages_metadata
        )
        
    except Exception as e:
        logger.error(f"PDF extraction with metadata failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from PDF: {str(e)}") from e


def _extract_from_docx(file_path: Union[Path, io.BytesIO]) -> str:
    """Extract text from DOCX using python-docx (simple mode)."""
    try:
        logger.debug("Starting DOCX text extraction")

        if isinstance(file_path, io.BytesIO):
            doc = Document(file_path)
        else:
            doc = Document(str(file_path))

        text_parts = []
        current_section_text = []
        section_count = 0

        # Extract text from paragraphs with section awareness
        for para in doc.paragraphs:
            if not para.text.strip():
                continue

            # Check for section break by examining paragraph properties
            # Section breaks in DOCX are indicated by paragraph's section property changes
            if para._element.pPr is not None:
                sect_pr = para._element.pPr.find('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr')
                if sect_pr is not None and current_section_text:
                    # Found a section break
                    section_count += 1
                    text_parts.extend(current_section_text)
                    text_parts.append("\n--- SECTION BREAK ---\n")
                    current_section_text = []

            current_section_text.append(para.text)

        # Add remaining section text
        if current_section_text:
            text_parts.extend(current_section_text)

        # Extract text from tables
        if doc.tables:
            text_parts.append("\n--- TABLES ---\n")
            for table_idx, table in enumerate(doc.tables):
                text_parts.append(f"\n[Table {table_idx + 1}]")
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(" | ".join(row_text))

        full_text = "\n".join(text_parts)
        logger.debug(
            f"DOCX extraction complete: {len(doc.paragraphs)} paragraphs, "
            f"{len(doc.tables)} tables, {section_count} section breaks"
        )

        return full_text

    except Exception as e:
        logger.error(f"DOCX extraction failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from DOCX: {str(e)}") from e


def _extract_from_docx_with_metadata(
    file_path: Union[Path, io.BytesIO],
    file_name: str
) -> ExtractionResult:
    """Extract text from DOCX with metadata (treats as single page with section awareness)."""
    try:
        logger.debug("Starting DOCX text extraction with metadata")

        if isinstance(file_path, io.BytesIO):
            doc = Document(file_path)
        else:
            doc = Document(str(file_path))

        detector = TextPatternDetector()

        # Track sections and their content
        sections = []
        current_section_text = []
        section_titles = []
        section_count = 0
        has_section_breaks = False

        # Extract text from paragraphs with enhanced metadata
        text_parts = []
        for para in doc.paragraphs:
            if not para.text.strip():
                continue

            # Check for section break
            if para._element.pPr is not None:
                sect_pr = para._element.pPr.find('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr')
                if sect_pr is not None and current_section_text:
                    # Found a section break
                    section_count += 1
                    has_section_breaks = True
                    section_content = "\n".join(current_section_text)
                    sections.append(section_content)
                    text_parts.extend(current_section_text)
                    text_parts.append("\n--- SECTION BREAK ---\n")
                    current_section_text = []

            # Detect if this paragraph is a heading
            if para.style and para.style.name.startswith('Heading'):
                section_titles.append(para.text.strip())

            current_section_text.append(para.text)

        # Add remaining section text
        if current_section_text:
            section_content = "\n".join(current_section_text)
            sections.append(section_content)
            text_parts.extend(current_section_text)

        # Extract text from tables with better formatting
        table_texts = []
        if doc.tables:
            text_parts.append("\n--- TABLES ---\n")
            for table_idx, table in enumerate(doc.tables):
                text_parts.append(f"\n[Table {table_idx + 1}]")
                table_content = []
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        formatted_row = " | ".join(row_text)
                        text_parts.append(formatted_row)
                        table_content.append(formatted_row)
                table_texts.append("\n".join(table_content))

        full_text = "\n".join(text_parts)

        # Extract metadata
        urls = detector.extract_urls(full_text)
        emails = detector.extract_emails(full_text)
        headings = detector.extract_headings(full_text)

        # Combine detected headings with DOCX heading styles
        all_headings = list(set(headings + section_titles))

        has_toc = detector.detect_toc(full_text)

        # Create single page metadata with section information
        page_meta = PageMetadata(
            page_number=1,
            text=full_text,
            char_count=len(full_text),
            word_count=len(full_text.split()),
            urls=urls,
            emails=emails,
            headings=all_headings,
            has_toc=has_toc,
            sections=section_titles
        )

        # Create document metadata with section information
        doc_metadata = DocumentMetadata(
            file_name=file_name,
            file_type='docx',
            total_pages=1,
            total_chars=len(full_text),
            total_words=len(full_text.split()),
            extraction_date=datetime.now().isoformat(),
            pages=[page_meta],
            all_urls=urls,
            all_emails=emails,
            all_headings=all_headings,
            has_toc=has_toc,
            toc_page_numbers=[1] if has_toc else [],
            sections=section_titles,
            total_sections=max(len(sections), section_count + 1),
            has_section_breaks=has_section_breaks
        )

        logger.debug(
            f"DOCX extraction complete: {len(doc.paragraphs)} paragraphs, "
            f"{len(doc.tables)} tables, {len(urls)} URLs, "
            f"{doc_metadata.total_sections} sections, "
            f"{'with' if has_section_breaks else 'without'} section breaks"
        )

        return ExtractionResult(
            text=full_text,
            metadata=doc_metadata,
            pages=[page_meta]
        )

    except Exception as e:
        logger.error(f"DOCX extraction with metadata failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from DOCX: {str(e)}") from e


def _extract_from_txt(file_path: Union[Path, io.BytesIO], encoding: str) -> str:
    """Extract text from TXT file using native Python (simple mode)."""
    try:
        logger.debug(f"Starting TXT text extraction with encoding: {encoding}")
        
        if isinstance(file_path, io.BytesIO):
            text = file_path.read().decode(encoding)
        else:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
        
        logger.debug("TXT extraction complete")
        return text
        
    except UnicodeDecodeError as e:
        logger.error(
            f"Encoding error with {encoding}: {str(e)}. "
            f"Try a different encoding."
        )
        raise Exception(
            f"Failed to decode text file with encoding '{encoding}': {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"TXT extraction failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from TXT: {str(e)}") from e


def _extract_from_txt_with_metadata(
    file_path: Union[Path, io.BytesIO],
    file_name: str,
    encoding: str
) -> ExtractionResult:
    """Extract text from TXT with metadata (treats as single page)."""
    try:
        logger.debug(f"Starting TXT text extraction with metadata, encoding: {encoding}")
        
        if isinstance(file_path, io.BytesIO):
            text = file_path.read().decode(encoding)
        else:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
        
        detector = TextPatternDetector()
        
        # Extract metadata
        urls = detector.extract_urls(text)
        emails = detector.extract_emails(text)
        headings = detector.extract_headings(text)
        has_toc = detector.detect_toc(text)
        
        # Create single page metadata
        page_meta = PageMetadata(
            page_number=1,
            text=text,
            char_count=len(text),
            word_count=len(text.split()),
            urls=urls,
            emails=emails,
            headings=headings,
            has_toc=has_toc,
            sections=[]  # TXT files don't have explicit sections
        )

        # Create document metadata
        doc_metadata = DocumentMetadata(
            file_name=file_name,
            file_type='txt',
            total_pages=1,
            total_chars=len(text),
            total_words=len(text.split()),
            extraction_date=datetime.now().isoformat(),
            pages=[page_meta],
            all_urls=urls,
            all_emails=emails,
            all_headings=headings,
            has_toc=has_toc,
            toc_page_numbers=[1] if has_toc else [],
            sections=[],
            total_sections=0,
            has_section_breaks=False
        )
        
        logger.debug(f"TXT extraction complete: {len(urls)} URLs, {len(headings)} headings")
        
        return ExtractionResult(
            text=text,
            metadata=doc_metadata,
            pages=[page_meta]
        )
        
    except Exception as e:
        logger.error(f"TXT extraction with metadata failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from TXT: {str(e)}") from e


# Example usage
if __name__ == "__main__":
    # Configure logging for demonstration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("EXAMPLE 1: Simple text extraction")
    print("=" * 70)
    try:
        text = extract_text_from_document('example.pdf', extract_metadata=False)
        print(f"Extracted {len(text)} characters")
        print(text[:500] + "...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Full extraction with metadata (PDF)")
    print("=" * 70)
    try:
        result = extract_text_from_document('example.pdf', extract_metadata=True)

        print(f"\nDocument: {result.metadata.file_name}")
        print(f"Type: {result.metadata.file_type}")
        print(f"Total Pages: {result.metadata.total_pages}")
        print(f"Total Characters: {result.metadata.total_chars}")
        print(f"Total Words: {result.metadata.total_words}")
        print(f"URLs Found: {len(result.metadata.all_urls)}")
        print(f"Emails Found: {len(result.metadata.all_emails)}")
        print(f"Headings Found: {len(result.metadata.all_headings)}")
        print(f"Has TOC: {result.metadata.has_toc}")

        if result.metadata.all_urls:
            print(f"\nURLs: {result.metadata.all_urls[:3]}")

        if result.metadata.all_headings:
            print(f"\nHeadings: {result.metadata.all_headings[:5]}")

        # Access specific page
        print(f"\nPage 1 has {result.pages[0].word_count} words")
        print(f"Page 1 URLs: {result.pages[0].urls}")

        # Get text by page range
        if result.metadata.total_pages >= 2:
            page_range_text = result.get_pages_range(1, 2)
            print(f"\nPages 1-2 combined: {len(page_range_text)} characters")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("EXAMPLE 3: DOCX extraction with section awareness")
    print("=" * 70)
    try:
        result = extract_text_from_document('example.docx', extract_metadata=True)

        print(f"\nDocument: {result.metadata.file_name}")
        print(f"Type: {result.metadata.file_type}")
        print(f"Total Sections: {result.metadata.total_sections}")
        print(f"Has Section Breaks: {result.metadata.has_section_breaks}")
        print(f"Total Characters: {result.metadata.total_chars}")
        print(f"Total Words: {result.metadata.total_words}")
        print(f"URLs Found: {len(result.metadata.all_urls)}")
        print(f"Headings Found: {len(result.metadata.all_headings)}")

        if result.metadata.sections:
            print(f"\nSection Titles Found:")
            for idx, section_title in enumerate(result.metadata.sections[:5], 1):
                print(f"  {idx}. {section_title}")

        if result.metadata.all_headings:
            print(f"\nAll Headings: {result.metadata.all_headings[:5]}")

    except Exception as e:
        print(f"Error: {e}")