"""
Enterprise Chunking Pipeline

A production-ready chunking system that preserves document structure,
applies semantic windowing, and generates comprehensive metadata.

Features:
- Page-level chunking with source tracking
- Section-aware splitting using document hierarchy
- Semantic windowing with overlap
- Token-aware chunking
- Rich metadata for each chunk
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from datetime import datetime
import time
from logger_config import get_logger

logger = get_logger(__name__)


class BoundaryType(Enum):
    """Types of chunk boundaries."""
    PAGE = "page"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    BULLET_LIST = "bullet_list"


class ChunkType(Enum):
    """
    Types of chunks for hybrid (OpenSearch + vector) search.

    Different chunk types serve different retrieval purposes:
    - CONTENT: Standard content chunks for semantic search
    - HEADING: Heading-only chunks for keyword/faceted search
    - CLAUSE: Single-sentence/clause chunks for precise retrieval
    - METADATA: Document metadata chunks for filtering
    - SUMMARY: Section/document summary chunks for high-level matching
    """
    CONTENT = "content"           # Standard content chunks
    HEADING = "heading"           # Heading-only chunks (titles, section names)
    CLAUSE = "clause"             # Single sentence/clause chunks
    METADATA = "metadata"         # Document metadata chunks
    SUMMARY = "summary"           # Summary chunks (section/document level)


@dataclass
class ChunkMetadata:
    """
    Comprehensive metadata for each chunk.
    Includes source tracking, position, hierarchy, and content metrics.
    """
    # Document identification
    doc_id: str
    file_name: str
    chunk_id: str  # Unique identifier for this chunk

    # Page mapping (critical for source reference)
    page_number_start: int
    page_number_end: int

    # Section hierarchy (breadcrumb navigation)
    section_title: Optional[str] = None  # Clean section name (primary field)
    section_title_raw: Optional[str] = None  # Raw section name as appears in document
    heading_path: List[str] = field(default_factory=list)  # Full hierarchy: ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]
    heading_level: Optional[int] = None  # Heading level: 1=H1, 2=H2, 3=H3, 4=H4, 5=H5, 6=H6
    parent_section: Optional[str] = None  # Direct parent section name

    # Chunk positioning
    chunk_index: int = 0  # 0-based index
    total_chunks: int = 0

    # Content metrics
    chunk_char_len: int = 0
    chunk_word_count: int = 0
    chunk_token_count: Optional[int] = None  # If tokenizer is provided

    # Boundary information
    boundary_type: str = BoundaryType.PARAGRAPH.value
    has_overlap: bool = False
    overlap_with_previous: int = 0  # Characters overlapping with previous chunk
    overlap_with_next: int = 0  # Characters overlapping with next chunk

    # Chunk type for hybrid search
    chunk_type: str = ChunkType.CONTENT.value  # content, heading, clause, metadata, summary
    parent_chunk_id: Optional[str] = None  # Reference to parent content chunk (for clause/heading chunks)

    # Content
    normalized_text: str = ""  # Actual chunk content (normalized)
    original_page_text: Optional[str] = None  # For debugging

    # Additional context
    contains_tables: bool = False
    contains_code: bool = False
    contains_bullets: bool = False
    urls_in_chunk: List[str] = field(default_factory=list)

    # Project scoping (for multi-tenancy)
    project_id: Optional[str] = None
    schema_version: Optional[str] = None

    # Document-level metadata (for citations)
    document_title: Optional[str] = None  # Clean title for citations

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ChunkingConfig:
    """Configuration for the chunking pipeline."""
    # Chunk size constraints
    max_chunk_size: int = 1000  # Maximum characters per chunk
    min_chunk_size: int = 100  # Minimum characters per chunk
    target_chunk_size: int = 500  # Target size (will try to get close to this)

    # Semantic windowing
    enable_overlap: bool = True
    overlap_size: int = 100  # Characters to overlap between chunks
    overlap_strategy: str = "sentence"  # "sentence" or "token" or "character"

    # Boundary preservation
    respect_page_boundaries: bool = True  # Never merge across pages
    respect_section_boundaries: bool = True  # Prefer splitting on sections
    respect_paragraph_boundaries: bool = True  # Prefer splitting on paragraphs

    # Heading-aware chunking (NEW)
    use_structured_headings: bool = True  # Use font/style-based heading detection from extraction
    prefer_structured_over_regex: bool = True  # Prefer structured headings over regex patterns

    # Special element handling
    keep_tables_intact: bool = True
    keep_code_blocks_intact: bool = True
    keep_bullet_lists_intact: bool = True

    # Token-aware chunking (requires tokenizer)
    token_aware: bool = False
    max_tokens: Optional[int] = None
    tokenizer: Optional[Any] = None  # Pass tiktoken or transformers tokenizer

    # Metadata options
    include_original_page_text: bool = False  # For debugging (increases memory)
    extract_urls: bool = True

    # Hybrid search chunk types (OpenSearch + Vector)
    # These generate additional specialized chunks alongside content chunks
    generate_heading_chunks: bool = False    # Generate heading-only chunks for keyword search
    generate_clause_chunks: bool = False     # Generate single-sentence/clause chunks for precise retrieval
    generate_metadata_chunks: bool = False   # Generate document metadata chunks for filtering
    generate_summary_chunks: bool = False    # Generate summary chunks for high-level matching

    # Clause chunking options
    clause_min_length: int = 20              # Minimum clause length in characters
    clause_max_length: int = 300             # Maximum clause length in characters
    clause_overlap_sentences: int = 1        # Number of adjacent sentences for context

    # Summary options (requires external summarizer or extracts first N sentences)
    summary_sentences: int = 3               # Number of sentences to extract for section summaries
    generate_document_summary: bool = True   # Generate a document-level summary chunk
    generate_section_summaries: bool = True  # Generate section-level summary chunks


class BoundaryDetector:
    """
    Detects various document boundaries for intelligent chunking.
    """

    def __init__(self):
        # Section markers - patterns return (pattern, extractor_function, level)
        # LEVELS: 1 (H1), 2 (H2), 3 (H3), 4 (H4), 5 (H5), 6 (H6)
        self.section_patterns = [
            # MARKDOWN HEADINGS (Most Common) - Proper hierarchy
            (re.compile(r'^#\s+(.+)$', re.MULTILINE),
             lambda m: m.group(1).strip(), 1),  # H1
            (re.compile(r'^##\s+(.+)$', re.MULTILINE),
             lambda m: m.group(1).strip(), 2),  # H2
            (re.compile(r'^###\s+(.+)$', re.MULTILINE),
             lambda m: m.group(1).strip(), 3),  # H3
            (re.compile(r'^####\s+(.+)$', re.MULTILINE),
             lambda m: m.group(1).strip(), 4),  # H4
            (re.compile(r'^#####\s+(.+)$', re.MULTILINE),
             lambda m: m.group(1).strip(), 5),  # H5
            (re.compile(r'^######\s+(.+)$', re.MULTILINE),
             lambda m: m.group(1).strip(), 6),  # H6

            # UNDERLINED HEADINGS (reStructuredText style)
            (re.compile(r'^(.+)\n[=]{3,}$', re.MULTILINE),
             lambda m: m.group(1).strip(), 1),  # H1 (= underline)
            (re.compile(r'^(.+)\n[-]{3,}$', re.MULTILINE),
             lambda m: m.group(1).strip(), 2),  # H2 (- underline)

            # CHAPTER/SECTION/PART MARKERS
            (re.compile(r'^((?:Chapter|CHAPTER)\s+\d+[:.]\s*.*)$', re.MULTILINE),
             lambda m: m.group(1).strip(), 1),  # Chapter = H1
            (re.compile(r'^((?:Section|SECTION)\s+\d+(?:\.\d+)?[:.]\s*.*)$', re.MULTILINE),
             lambda m: m.group(1).strip(), 2),  # Section = H2
            (re.compile(r'^((?:Part|PART)\s+\d+[:.]\s*.*)$', re.MULTILINE),
             lambda m: m.group(1).strip(), 1),  # Part = H1

            # NUMBERED HEADINGS (Detect level by dots)
            (re.compile(r'^(\d+[\.)]\s+.+)$', re.MULTILINE),  # "1. Title"
             lambda m: m.group(1).strip(), 1),  # Level 1
            (re.compile(r'^(\d+\.\d+[\.)]\s+.+)$', re.MULTILINE),  # "1.1 Title"
             lambda m: m.group(1).strip(), 2),  # Level 2
            (re.compile(r'^(\d+\.\d+\.\d+[\.)]\s+.+)$', re.MULTILINE),  # "1.1.1 Title"
             lambda m: m.group(1).strip(), 3),  # Level 3
            (re.compile(r'^(\d+\.\d+\.\d+\.\d+[\.)]\s+.+)$', re.MULTILINE),  # "1.1.1.1 Title"
             lambda m: m.group(1).strip(), 4),  # Level 4

            # ALL CAPS HEADINGS (Conservative - treat as H1)
            (re.compile(r'^([A-Z][A-Z\s]{10,})$', re.MULTILINE),
             lambda m: m.group(1).strip(), 1),  # Must be 11+ chars of ALL CAPS
        ]

        # Page markers
        self.page_marker_pattern = re.compile(r'<<<PAGE_(\d+)>>>')

        # Table markers
        self.table_patterns = [
            re.compile(r'^\|.+\|$', re.MULTILINE),
            re.compile(r'^--- TABLES ---$', re.MULTILINE),
            re.compile(r'^\[Table \d+\]$', re.MULTILINE),
        ]

        # Code block markers
        self.code_patterns = [
            re.compile(r'```[\s\S]*?```'),
            re.compile(r'^(?: {4}|\t).+$', re.MULTILINE),
        ]

        # Bullet list markers
        self.bullet_pattern = re.compile(r'^[\s]*[•·∙●○◦▪▫■□\*\-\+]\s+', re.MULTILINE)

        # URL pattern
        self.url_pattern = re.compile(r'https?://[^\s]+')

    def find_page_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """
        Find page boundary positions.

        Returns:
            List of (position, page_number) tuples
        """
        boundaries = []
        for match in self.page_marker_pattern.finditer(text):
            page_num = int(match.group(1))
            boundaries.append((match.start(), page_num))
        return boundaries

    def _is_valid_section_heading(self, title: str, match_text: str, surrounding_context: str = "") -> bool:
        """
        STRICT VALIDATION: Determine if extracted text is a true section heading.

        POSITIVE RULES (must pass ALL):
        1. Heading must be on isolated line (not inline with other text)
        2. Length must be reasonable (3-100 characters)
        3. Must look like a section title, not a question or exercise
        4. Must not be preceded/followed by text on same line

        NEGATIVE RULES (must pass NONE):
        1. Questions (ends with ?)
        2. Exercise/Practice prompts
        3. Fill-in-the-blank or multiple choice
        4. Code snippets or technical syntax
        5. List items or bullet points
        6. URLs or file paths
        7. Numbers-only or dates
        8. All lowercase (likely not a heading)

        Args:
            title: Extracted heading text (clean)
            match_text: Original matched text (with formatting)
            surrounding_context: Text around the match for isolation check

        Returns:
            True if valid section heading, False otherwise
        """
        # NEGATIVE RULE 1: Questions are NOT section headings
        if title.strip().endswith('?'):
            return False
        if title.lower().startswith(('what is ', 'what are ', 'how to ', 'how do ', 'why ', 'when ', 'where ')):
            return False

        # NEGATIVE RULE 2: Exercises/Practice prompts are NOT headings
        exercise_keywords = [
            'exercise', 'practice', 'try it', 'hands-on', 'lab', 'assignment',
            'homework', 'quiz', 'test yourself', 'challenge', 'activity'
        ]
        title_lower = title.lower()
        if any(keyword in title_lower for keyword in exercise_keywords):
            return False

        # NEGATIVE RULE 3: Fill-in-blank, multiple choice NOT headings
        if '____' in title or '___' in title:
            return False
        if re.search(r'\b[A-D]\)|^\([A-D]\)', title):  # (A) or A)
            return False

        # NEGATIVE RULE 4: Code snippets NOT headings
        code_indicators = ['()', '{}', '[]', '=>', '->', '::']
        if any(ind in title for ind in code_indicators):
            return False
        if re.search(r'[a-z]+\([^)]*\)', title):  # function() pattern
            return False

        # NEGATIVE RULE 5: List items NOT headings (unless properly numbered)
        if title.startswith(('• ', '- ', '* ', '+ ', '· ')):
            return False

        # NEGATIVE RULE 6: URLs, file paths, emails NOT headings
        if re.search(r'https?://', title) or '@' in title:
            return False
        if re.search(r'[/\\].+[/\\]', title):  # /path/to/file
            return False

        # NEGATIVE RULE 7: Numbers-only or dates NOT headings
        if re.match(r'^\d+$', title.strip()):  # Just numbers
            return False
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', title):  # Date
            return False

        # NEGATIVE RULE 8: All lowercase likely NOT a heading
        # Exception: markdown headings and numbered headings are OK
        if title.islower() and not match_text.startswith('#') and not re.match(r'^\d+\.', title):
            return False

        # NEGATIVE RULE 9: Too many special characters
        special_char_count = sum(1 for c in title if c in '!@#$%^&*()+={}[]|\\:;"<>,')
        if special_char_count > 3:
            return False

        # NEGATIVE RULE 10: Starts with coordinating conjunction (likely continuation)
        if title.lower().startswith(('and ', 'but ', 'or ', 'so ', 'yet ', 'for ', 'nor ')):
            return False

        # POSITIVE RULE 1: Length must be reasonable
        if len(title) < 3 or len(title) > 100:
            return False

        # POSITIVE RULE 2: Heading must be isolated (not inline)
        # Check if there's text before or after on the same line
        if surrounding_context:
            lines = surrounding_context.split('\n')
            for line in lines:
                if match_text in line:
                    # Remove the match itself
                    remaining = line.replace(match_text, '')
                    # If there's substantial text remaining, it's not isolated
                    if remaining.strip() and len(remaining.strip()) > 5:
                        return False
                    break

        # POSITIVE RULE 3: Must start with capital letter or number
        if not (title[0].isupper() or title[0].isdigit()):
            return False

        # POSITIVE RULE 4: Reasonable word count (not too many words = paragraph)
        word_count = len(title.split())
        if word_count > 15:  # Headings shouldn't be this long
            return False
        if word_count == 1 and len(title) < 4 and not title.isdigit():  # Single short word unlikely
            return False

        return True

    def find_section_boundaries(self, text: str) -> List[Tuple[int, str, int]]:
        """
        Find section boundaries with STRICT VALIDATION.
        Only returns TRUE section headings, not questions, exercises, or inline text.

        Returns:
            List of (position, section_title, level) tuples
            - position: character position in text
            - section_title: CLEAN heading text (validated)
            - level: heading level (1-6)
        """
        boundaries = []

        for pattern, extractor, level in self.section_patterns:
            for match in pattern.finditer(text):
                # Extract clean title
                title = extractor(match)
                match_text = match.group(0)

                # Get surrounding context for isolation check
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]

                # STRICT VALIDATION
                if not self._is_valid_section_heading(title, match_text, context):
                    continue

                # Additional validation
                if not title or len(title) < 3 or len(title) > 100:
                    continue

                # Never store placeholders or synthetic markers
                if title.startswith('<<<') or title.startswith('#'):
                    continue

                boundaries.append((match.start(), title, level))

        # Sort by position
        boundaries.sort(key=lambda x: x[0])

        # Remove duplicates (same position, different pattern matches)
        seen_positions = set()
        unique_boundaries = []
        for pos, title, level in boundaries:
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_boundaries.append((pos, title, level))

        return unique_boundaries

    def find_paragraph_boundaries(self, text: str) -> List[int]:
        """
        Find paragraph boundaries (double newlines).

        Returns:
            List of positions where paragraphs end
        """
        boundaries = []
        for match in re.finditer(r'\n\n+', text):
            boundaries.append(match.end())
        return boundaries

    def find_sentence_boundaries(self, text: str) -> List[int]:
        """
        Find sentence boundaries.

        Returns:
            List of positions where sentences end
        """
        boundaries = []
        # Simple sentence boundary detection
        sentence_end_pattern = re.compile(r'[.!?]\s+(?=[A-Z])')
        for match in sentence_end_pattern.finditer(text):
            boundaries.append(match.end())
        return boundaries

    def find_special_blocks(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        Find special blocks (tables, code, bullets).

        Returns:
            Dict with 'tables', 'code', 'bullets' keys, each containing (start, end) tuples
        """
        blocks = {
            'tables': [],
            'code': [],
            'bullets': []
        }

        # Find tables
        for pattern in self.table_patterns:
            for match in pattern.finditer(text):
                blocks['tables'].append((match.start(), match.end()))

        # Find code blocks
        for pattern in self.code_patterns:
            for match in pattern.finditer(text):
                blocks['code'].append((match.start(), match.end()))

        # Find bullet lists (consecutive bullets)
        bullet_matches = list(self.bullet_pattern.finditer(text))
        if bullet_matches:
            current_list_start = bullet_matches[0].start()
            prev_end = bullet_matches[0].end()

            for match in bullet_matches[1:]:
                # If bullets are close together (within 2 lines), they're part of same list
                if match.start() - prev_end < 100:
                    prev_end = match.end()
                else:
                    # End current list, start new one
                    blocks['bullets'].append((current_list_start, prev_end))
                    current_list_start = match.start()
                    prev_end = match.end()

            # Add final list
            blocks['bullets'].append((current_list_start, prev_end))

        return blocks

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        return self.url_pattern.findall(text)


class SemanticChunker:
    """
    Performs semantic windowing and overlap management.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.boundary_detector = BoundaryDetector()

    def apply_overlap(self, chunks: List[str]) -> List[Tuple[str, int, int]]:
        """
        Apply overlapping windows to chunks.

        Args:
            chunks: List of chunk texts

        Returns:
            List of (chunk_text, overlap_with_previous, overlap_with_next) tuples
        """
        if not self.config.enable_overlap or len(chunks) <= 1:
            return [(chunk, 0, 0) for chunk in chunks]

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            overlap_prev = 0
            overlap_next = 0

            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_text = self._get_overlap_text(
                    prev_chunk,
                    self.config.overlap_size,
                    from_end=True
                )
                chunk = overlap_text + chunk
                overlap_prev = len(overlap_text)

            # Add overlap to next chunk (tracked for metadata)
            if i < len(chunks) - 1:
                overlap_text = self._get_overlap_text(
                    chunk,
                    self.config.overlap_size,
                    from_end=True
                )
                overlap_next = len(overlap_text)

            overlapped_chunks.append((chunk, overlap_prev, overlap_next))

        return overlapped_chunks

    def _get_overlap_text(self, text: str, size: int, from_end: bool = True) -> str:
        """
        Get overlap text based on strategy.

        Args:
            text: Source text
            size: Overlap size
            from_end: If True, get from end; else from beginning

        Returns:
            Overlap text
        """
        if self.config.overlap_strategy == "character":
            if from_end:
                return text[-size:] if len(text) > size else text
            else:
                return text[:size] if len(text) > size else text

        elif self.config.overlap_strategy == "sentence":
            sentences = self.boundary_detector.find_sentence_boundaries(text)
            if not sentences:
                # Fallback to character
                return text[-size:] if from_end else text[:size]

            if from_end:
                # Get last few sentences that fit in size
                for i in range(len(sentences) - 1, -1, -1):
                    if len(text) - sentences[i] <= size:
                        return text[sentences[i]:]
                return text[-size:]
            else:
                # Get first few sentences that fit in size
                for i, boundary in enumerate(sentences):
                    if boundary >= size:
                        return text[:boundary]
                return text[:size]

        elif self.config.overlap_strategy == "token":
            # Token-based overlap (requires tokenizer)
            if self.config.tokenizer:
                tokens = self.config.tokenizer.encode(text)
                token_size = size // 4  # Rough estimate: 1 token ≈ 4 chars

                if from_end:
                    overlap_tokens = tokens[-token_size:]
                else:
                    overlap_tokens = tokens[:token_size]

                return self.config.tokenizer.decode(overlap_tokens)
            else:
                # Fallback to character
                return text[-size:] if from_end else text[:size]

        return ""

    def count_tokens(self, text: str) -> int:
        """Count tokens if tokenizer is available."""
        if self.config.tokenizer:
            return len(self.config.tokenizer.encode(text))
        else:
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4


class EnterpriseChunkingPipeline:
    """
    Main chunking pipeline that orchestrates all chunking strategies.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.boundary_detector = BoundaryDetector()
        self.semantic_chunker = SemanticChunker(self.config)
        logger.debug(
            f"ChunkingPipeline init | max={self.config.max_chunk_size} "
            f"target={self.config.target_chunk_size} overlap={self.config.overlap_size}"
        )

    def chunk_document(
        self,
        extraction_result,
        doc_id: Optional[str] = None,
        normalized_text: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[ChunkMetadata]:
        """
        Main entry point: Chunk an entire document.

        Args:
            extraction_result: ExtractionResult from document_processor
            doc_id: Optional document ID (will generate if not provided)
            normalized_text: Optional pre-normalized text (if not provided, uses extraction_result.text)
            project_id: Optional project ID for multi-tenancy

        Returns:
            List of ChunkMetadata objects
        """
        start_time = time.time()

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # Use normalized text if provided, otherwise use extracted text
        text = normalized_text if normalized_text else extraction_result.text

        # Step 1: Page-level chunking
        page_chunks = self._chunk_by_pages(text, extraction_result)

        # Step 2: Section-aware chunking
        section_chunks = self._chunk_by_sections(page_chunks, extraction_result)

        # Step 3: Apply size constraints
        sized_chunks = self._apply_size_constraints(section_chunks)

        # Step 4: Apply semantic windowing (overlap)
        overlapped_chunks = self._apply_semantic_windowing(sized_chunks)

        # Step 5: Build comprehensive metadata
        final_chunks = self._build_chunk_metadata(
            overlapped_chunks,
            extraction_result,
            doc_id,
            project_id  # Pass project_id for multi-tenancy
        )

        total_duration = time.time() - start_time

        # Single consolidated log with key metrics
        if final_chunks:
            avg_size = sum(c.chunk_char_len for c in final_chunks) / len(final_chunks)
            logger.info(
                f"Chunked {len(extraction_result.pages)} pages → {len(final_chunks)} chunks | "
                f"avg={avg_size:.0f} chars | duration={total_duration:.2f}s"
            )

        return final_chunks

    def _chunk_by_pages(self, text: str, extraction_result) -> List[Dict]:
        """
        Step 1: Page-level chunking.
        Preserves page boundaries and mapping.
        """
        chunks = []
        page_boundaries = self.boundary_detector.find_page_boundaries(text)

        if not page_boundaries:
            # No page markers found - treat as single page
            chunks.append({
                'text': text,
                'page_start': 1,
                'page_end': 1,
                'original_page': extraction_result.pages[0] if extraction_result.pages else None
            })
            return chunks

        # Split by page markers
        for i, (pos, page_num) in enumerate(page_boundaries):
            # Get text until next page marker (or end of document)
            if i < len(page_boundaries) - 1:
                next_pos = page_boundaries[i + 1][0]
                page_text = text[pos:next_pos]
            else:
                page_text = text[pos:]

            # Remove the page marker itself
            page_text = re.sub(r'<<<PAGE_\d+>>>', '', page_text).strip()

            if page_text:  # Only add non-empty pages
                # Find corresponding page metadata
                original_page = next(
                    (p for p in extraction_result.pages if p.page_number == page_num),
                    None
                )

                chunks.append({
                    'text': page_text,
                    'page_start': page_num,
                    'page_end': page_num,
                    'original_page': original_page
                })

        return chunks

    def _find_section_boundaries_enhanced(
        self,
        text: str,
        original_page
    ) -> List[Tuple[int, str, int]]:
        """
        Find section boundaries using structured headings from extraction metadata.

        This enhanced method uses the font/style analysis results from document extraction,
        providing more accurate heading detection than regex patterns alone.

        Args:
            text: Page text to search in
            original_page: PageMetadata object with headings_structured

        Returns:
            List of (position, title, level) tuples for section boundaries
        """
        # Check if structured heading usage is enabled
        if not self.config.use_structured_headings:
            return []

        if not original_page or not hasattr(original_page, 'headings_structured'):
            return []

        structured_headings = original_page.headings_structured
        if not structured_headings:
            return []

        boundaries = []

        # For each structured heading, find its position in the text
        for heading_info in structured_headings:
            heading_text = heading_info['text']
            level_str = heading_info['level']  # e.g., "H1", "H2", "H3"

            # Extract numeric level
            level = int(level_str[1]) if len(level_str) > 1 and level_str[1].isdigit() else 2

            # Search for the heading in markdown format first
            markdown_patterns = [
                rf'^{"#" * level}\s+{re.escape(heading_text)}\s*$',
                rf'^{"#" * level}\s+{re.escape(heading_text)}',
            ]

            position = None
            for pattern in markdown_patterns:
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    position = match.start()
                    break

            # If not found in markdown format, search for plain text heading
            if position is None:
                pattern = rf'(^|\n)[ \t]*{re.escape(heading_text)}[ \t]*($|\n)'
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    position = match.start()

            # Add to boundaries if found
            if position is not None:
                boundaries.append((position, heading_text, level))

        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        return boundaries

    def _chunk_by_sections(self, page_chunks: List[Dict], extraction_result) -> List[Dict]:
        """
        Step 2: Section-aware chunking (ENHANCED with structured headings).
        Splits on section boundaries while respecting page boundaries.
        Preserves exact heading text from the original document.

        ENHANCEMENT: Uses structured headings from extraction metadata when available,
        providing more accurate heading detection and level information.
        """
        section_chunks = []

        for page_chunk in page_chunks:
            text = page_chunk['text']
            page_start = page_chunk['page_start']
            page_end = page_chunk['page_end']
            original_page = page_chunk['original_page']

            # Try to use structured headings first (from font/style analysis)
            section_boundaries = self._find_section_boundaries_enhanced(
                text,
                original_page
            )

            # Fallback: Use regex-based detection if no structured headings available
            if not section_boundaries:
                section_boundaries = self.boundary_detector.find_section_boundaries(text)

            if not section_boundaries:
                # No sections found - keep as single chunk without placeholder section name
                section_chunks.append({
                    'text': text,
                    'page_start': page_start,
                    'page_end': page_end,
                    'section_title': None,  # No placeholder - leave as None
                    'section_title_raw': None,
                    'heading_path': [],
                    'original_page': original_page,
                    'boundary_type': BoundaryType.PAGE.value
                })
                continue

            # Split by sections with PROPER HIERARCHY VALIDATION
            heading_stack = []  # Stack of (title, level) tuples

            for i, (pos, title, level) in enumerate(section_boundaries):
                # VALIDATE HIERARCHY: Maintain proper parent-child relationships
                # Remove all headings at same or deeper level
                heading_stack = [h for h in heading_stack if h[1] < level]

                # PARENT SECTION INHERITANCE
                # Add current heading to stack
                heading_stack.append((title, level))

                # Build heading path from root to current
                # This preserves the full hierarchy: H1 > H2 > H3 > Current
                heading_path = [h[0] for h in heading_stack]

                # Get text until next section (or end)
                if i < len(section_boundaries) - 1:
                    next_pos = section_boundaries[i + 1][0]
                    section_text = text[pos:next_pos]
                else:
                    section_text = text[pos:]

                section_text = section_text.strip()

                if section_text:
                    # Determine boundary type based on level
                    if level == 1:
                        boundary = BoundaryType.SECTION.value
                    elif level == 2:
                        boundary = BoundaryType.SUBSECTION.value
                    else:
                        boundary = BoundaryType.SUBSECTION.value

                    # Store BOTH clean and raw section names with VALIDATED hierarchy
                    section_chunks.append({
                        'text': section_text,
                        'page_start': page_start,
                        'page_end': page_end,
                        'section_title': title,  # Current section name
                        'section_title_raw': title,  # Exact as in document
                        'heading_path': heading_path,  # Full hierarchy path with parent inheritance
                        'heading_level': level,  # Store level for validation
                        'parent_section': heading_stack[-2][0] if len(heading_stack) > 1 else None,  # Direct parent
                        'original_page': original_page,
                        'boundary_type': boundary
                    })

        return section_chunks

    def _apply_size_constraints(self, chunks: List[Dict]) -> List[Dict]:
        """
        Step 3: Apply size constraints and split large chunks.
        Respects special blocks (tables, code, bullets).
        """
        sized_chunks = []

        for chunk in chunks:
            text = chunk['text']

            # If chunk is within size limits, keep as is
            if len(text) <= self.config.max_chunk_size:
                sized_chunks.append(chunk)
                continue

            # Chunk is too large - need to split

            # Find special blocks that should be kept intact
            special_blocks = self.boundary_detector.find_special_blocks(text)
            protected_ranges = self._merge_protected_ranges(special_blocks)

            # Split while respecting protected ranges
            sub_chunks = self._split_with_protected_ranges(
                text,
                protected_ranges,
                chunk
            )

            sized_chunks.extend(sub_chunks)

        return sized_chunks

    def _merge_protected_ranges(self, special_blocks: Dict[str, List[Tuple[int, int]]]) -> List[Tuple[int, int, str]]:
        """Merge overlapping protected ranges."""
        all_ranges = []

        if self.config.keep_tables_intact:
            all_ranges.extend([(s, e, 'table') for s, e in special_blocks['tables']])

        if self.config.keep_code_blocks_intact:
            all_ranges.extend([(s, e, 'code') for s, e in special_blocks['code']])

        if self.config.keep_bullet_lists_intact:
            all_ranges.extend([(s, e, 'bullets') for s, e in special_blocks['bullets']])

        # Sort by start position
        all_ranges.sort(key=lambda x: x[0])

        # Merge overlapping ranges
        merged = []
        for start, end, type_ in all_ranges:
            if merged and start <= merged[-1][1]:
                # Overlapping - merge
                merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2])
            else:
                merged.append((start, end, type_))

        return merged

    def _split_with_protected_ranges(
        self,
        text: str,
        protected_ranges: List[Tuple[int, int, str]],
        chunk_template: Dict
    ) -> List[Dict]:
        """Split text while keeping protected ranges intact."""
        sub_chunks = []
        current_pos = 0

        # Get paragraph boundaries for splitting
        para_boundaries = self.boundary_detector.find_paragraph_boundaries(text)

        while current_pos < len(text):
            # Find next split point
            target_end = current_pos + self.config.target_chunk_size

            # Check if we're in a protected range
            in_protected = False
            for pstart, pend, ptype in protected_ranges:
                if current_pos >= pstart and current_pos < pend:
                    # We're in a protected range - take the whole range
                    chunk_text = text[pstart:pend]
                    sub_chunks.append({
                        **chunk_template,
                        'text': chunk_text,
                        'boundary_type': f"{ptype}_block",
                        f'contains_{ptype}': True
                    })
                    current_pos = pend
                    in_protected = True
                    break

            if in_protected:
                continue

            # Find best split point (prefer paragraph boundaries)
            best_split = min(target_end, len(text))

            # Look for paragraph boundary near target
            for boundary in para_boundaries:
                if target_end - self.config.min_chunk_size <= boundary <= target_end + self.config.min_chunk_size:
                    best_split = boundary
                    break

            # Make sure we don't split within a protected range
            for pstart, pend, ptype in protected_ranges:
                if current_pos < pstart < best_split < pend:
                    # Split would break a protected range - adjust
                    best_split = pstart
                    break

            chunk_text = text[current_pos:best_split].strip()

            if chunk_text:
                sub_chunks.append({
                    **chunk_template,
                    'text': chunk_text,
                    'boundary_type': BoundaryType.PARAGRAPH.value
                })

            current_pos = best_split

        return sub_chunks

    def _apply_semantic_windowing(self, chunks: List[Dict]) -> List[Dict]:
        """
        Step 4: Apply semantic windowing with overlap.
        """
        if not self.config.enable_overlap:
            return chunks

        # Extract just the text for overlap processing
        chunk_texts = [chunk['text'] for chunk in chunks]

        # Apply overlap
        overlapped = self.semantic_chunker.apply_overlap(chunk_texts)

        # Update chunks with overlapped text and overlap info
        for i, (overlapped_text, overlap_prev, overlap_next) in enumerate(overlapped):
            chunks[i]['text'] = overlapped_text
            chunks[i]['has_overlap'] = overlap_prev > 0 or overlap_next > 0
            chunks[i]['overlap_with_previous'] = overlap_prev
            chunks[i]['overlap_with_next'] = overlap_next

        return chunks

    def _build_chunk_metadata(
        self,
        chunks: List[Dict],
        extraction_result,
        doc_id: str,
        project_id: Optional[str] = None
    ) -> List[ChunkMetadata]:
        """
        Step 5: Build comprehensive metadata for each chunk.

        Args:
            chunks: List of chunk dictionaries
            extraction_result: Document extraction result
            doc_id: Document ID
            project_id: Optional project ID for multi-tenancy
        """
        chunk_metadatas = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            text = chunk['text']

            # Extract URLs if configured
            urls = []
            if self.config.extract_urls:
                urls = self.boundary_detector.extract_urls(text)

            # Count tokens if tokenizer available
            token_count = None
            if self.config.token_aware and self.config.tokenizer:
                token_count = self.semantic_chunker.count_tokens(text)

            # Get original page text if configured
            original_page_text = None
            if self.config.include_original_page_text and chunk.get('original_page'):
                original_page_text = chunk['original_page'].text

            # Detect special content
            contains_tables = '|' in text or '[Table' in text or '--- TABLES ---' in text
            contains_code = '```' in text or 'def ' in text or 'class ' in text
            contains_bullets = bool(re.search(r'^[\s]*[•·∙●○◦▪▫■□\*\-\+]\s+', text, re.MULTILINE))

            # Create metadata with VALIDATED HIERARCHY
            # Format chunk_id: {project_id}_{doc_id}_chunk_{index}
            if project_id:
                chunk_id = f"{project_id}_{doc_id}_chunk_{i:04d}"
            else:
                chunk_id = f"{doc_id}_chunk_{i:04d}"

            metadata = ChunkMetadata(
                doc_id=doc_id,
                project_id=project_id,  # Pass project_id for multi-tenancy
                file_name=extraction_result.metadata.file_name,
                chunk_id=chunk_id,
                page_number_start=chunk['page_start'],
                page_number_end=chunk['page_end'],
                section_title=chunk.get('section_title'),  # Clean section name
                section_title_raw=chunk.get('section_title_raw'),  # Raw section name from document
                heading_path=chunk.get('heading_path', []),  # Full hierarchy with parent inheritance
                heading_level=chunk.get('heading_level'),  # Validated heading level (1-6)
                parent_section=chunk.get('parent_section'),  # Direct parent section
                chunk_index=i,
                total_chunks=total_chunks,
                chunk_char_len=len(text),
                chunk_word_count=len(text.split()),
                chunk_token_count=token_count,
                boundary_type=chunk.get('boundary_type', BoundaryType.PARAGRAPH.value),
                has_overlap=chunk.get('has_overlap', False),
                overlap_with_previous=chunk.get('overlap_with_previous', 0),
                overlap_with_next=chunk.get('overlap_with_next', 0),
                normalized_text=text,
                original_page_text=original_page_text,
                contains_tables=contains_tables,
                contains_code=contains_code,
                contains_bullets=contains_bullets,
                urls_in_chunk=urls
            )

            chunk_metadatas.append(metadata)

        return chunk_metadatas

    # =========================================================================
    # HYBRID SEARCH CHUNK GENERATORS
    # These methods generate specialized chunks for OpenSearch + Vector search
    # =========================================================================

    def _generate_heading_chunks(
        self,
        content_chunks: List[ChunkMetadata],
        extraction_result,
        doc_id: str,
        project_id: Optional[str] = None
    ) -> List[ChunkMetadata]:
        """
        Generate heading-only chunks for keyword/faceted search in OpenSearch.

        Heading chunks contain:
        - Section titles and their hierarchical path
        - Page reference for navigation
        - Minimal text for keyword matching

        These chunks are optimized for:
        - Keyword search on section names
        - Faceted filtering by section
        - Table of contents generation
        - Navigation breadcrumbs

        Args:
            content_chunks: List of content chunks to extract headings from
            extraction_result: Original extraction result
            doc_id: Document ID

        Returns:
            List of heading-only ChunkMetadata objects
        """
        heading_chunks = []
        seen_headings = set()  # Avoid duplicates

        for content_chunk in content_chunks:
            section_title = content_chunk.section_title
            if not section_title or section_title in seen_headings:
                continue

            seen_headings.add(section_title)

            # Build heading text with hierarchy context
            heading_path = content_chunk.heading_path or []
            if heading_path:
                heading_text = " > ".join(heading_path)
            else:
                heading_text = section_title

            # Create heading chunk
            # Format chunk_id with project_id if available
            if project_id:
                heading_chunk_id = f"{project_id}_{doc_id}_heading_{len(heading_chunks):04d}"
            else:
                heading_chunk_id = f"{doc_id}_heading_{len(heading_chunks):04d}"

            heading_chunk = ChunkMetadata(
                doc_id=doc_id,
                project_id=project_id,
                file_name=content_chunk.file_name,
                chunk_id=heading_chunk_id,
                page_number_start=content_chunk.page_number_start,
                page_number_end=content_chunk.page_number_start,  # Headings are single-page reference
                section_title=section_title,
                section_title_raw=content_chunk.section_title_raw,
                heading_path=heading_path,
                heading_level=content_chunk.heading_level,
                parent_section=content_chunk.parent_section,
                chunk_index=len(heading_chunks),
                total_chunks=0,  # Will be updated later
                chunk_char_len=len(heading_text),
                chunk_word_count=len(heading_text.split()),
                boundary_type=BoundaryType.SECTION.value,
                chunk_type=ChunkType.HEADING.value,
                parent_chunk_id=content_chunk.chunk_id,
                normalized_text=heading_text,
                has_overlap=False
            )

            heading_chunks.append(heading_chunk)

        # Update total_chunks count
        for chunk in heading_chunks:
            chunk.total_chunks = len(heading_chunks)

        return heading_chunks

    def _generate_clause_chunks(
        self,
        content_chunks: List[ChunkMetadata],
        extraction_result,
        doc_id: str,
        project_id: Optional[str] = None
    ) -> List[ChunkMetadata]:
        """
        Generate clause/sentence-level chunks for precise retrieval.

        Clause chunks contain:
        - Individual sentences or clauses
        - Optional context from adjacent sentences
        - Reference to parent content chunk

        These chunks are optimized for:
        - Precise semantic matching
        - Question-answering systems
        - Fact extraction
        - Citation-level retrieval

        Args:
            content_chunks: List of content chunks to split into clauses
            extraction_result: Original extraction result
            doc_id: Document ID

        Returns:
            List of clause-level ChunkMetadata objects
        """
        clause_chunks = []

        # Common abbreviations that shouldn't end sentences
        abbreviations = {'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Inc', 'Ltd', 'Corp', 'vs', 'etc', 'Sr', 'Jr', 'Fig', 'No', 'Vol'}

        def split_into_sentences(text: str) -> List[str]:
            """Split text into sentences, respecting abbreviations."""
            # Simple sentence boundary pattern
            potential_splits = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

            # Rejoin splits that were false positives (abbreviations)
            sentences = []
            buffer = ""

            for segment in potential_splits:
                if buffer:
                    # Check if previous segment ended with an abbreviation
                    words = buffer.rstrip('.!?').split()
                    last_word = words[-1] if words else ""

                    if last_word.rstrip('.') in abbreviations:
                        # False positive - rejoin
                        buffer = buffer + " " + segment
                    else:
                        sentences.append(buffer)
                        buffer = segment
                else:
                    buffer = segment

            if buffer:
                sentences.append(buffer)

            return sentences

        for content_chunk in content_chunks:
            text = content_chunk.normalized_text
            if not text or len(text) < self.config.clause_min_length:
                continue

            # Split into sentences using abbreviation-aware function
            sentences = split_into_sentences(text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                continue

            for i, sentence in enumerate(sentences):
                # Skip sentences that are too short or too long
                if len(sentence) < self.config.clause_min_length:
                    continue
                if len(sentence) > self.config.clause_max_length:
                    # For very long sentences, keep them but mark them
                    pass

                # Build clause text with optional context
                clause_text = sentence
                context_sentences = []

                # Add context from adjacent sentences if configured
                if self.config.clause_overlap_sentences > 0:
                    # Previous sentence context
                    if i > 0:
                        prev_idx = max(0, i - self.config.clause_overlap_sentences)
                        context_sentences.extend(sentences[prev_idx:i])

                    # Next sentence context
                    if i < len(sentences) - 1:
                        next_idx = min(len(sentences), i + 1 + self.config.clause_overlap_sentences)
                        context_sentences.extend(sentences[i+1:next_idx])

                # Create clause chunk
                # Format chunk_id with project_id if available
                if project_id:
                    clause_chunk_id = f"{project_id}_{doc_id}_clause_{len(clause_chunks):04d}"
                else:
                    clause_chunk_id = f"{doc_id}_clause_{len(clause_chunks):04d}"

                clause_chunk = ChunkMetadata(
                    doc_id=doc_id,
                    project_id=project_id,
                    file_name=content_chunk.file_name,
                    chunk_id=clause_chunk_id,
                    page_number_start=content_chunk.page_number_start,
                    page_number_end=content_chunk.page_number_end,
                    section_title=content_chunk.section_title,
                    section_title_raw=content_chunk.section_title_raw,
                    heading_path=content_chunk.heading_path,
                    heading_level=content_chunk.heading_level,
                    parent_section=content_chunk.parent_section,
                    chunk_index=len(clause_chunks),
                    total_chunks=0,  # Will be updated later
                    chunk_char_len=len(clause_text),
                    chunk_word_count=len(clause_text.split()),
                    boundary_type=BoundaryType.SENTENCE.value,
                    chunk_type=ChunkType.CLAUSE.value,
                    parent_chunk_id=content_chunk.chunk_id,
                    normalized_text=clause_text,
                    has_overlap=len(context_sentences) > 0,
                    overlap_with_previous=len(" ".join(context_sentences[:self.config.clause_overlap_sentences])) if context_sentences else 0
                )

                clause_chunks.append(clause_chunk)

        # Update total_chunks count
        for chunk in clause_chunks:
            chunk.total_chunks = len(clause_chunks)

        return clause_chunks

    def _generate_metadata_chunks(
        self,
        extraction_result,
        doc_id: str,
        project_id: Optional[str] = None
    ) -> List[ChunkMetadata]:
        """
        Generate metadata-only chunks for filtering and faceted search.

        Metadata chunks contain:
        - Document title, author, creation date
        - File information (name, type, size)
        - Custom metadata fields
        - Keywords and tags

        These chunks are optimized for:
        - Faceted filtering in OpenSearch
        - Document-level search
        - Metadata-based retrieval
        - Document classification

        Args:
            extraction_result: Original extraction result with metadata
            doc_id: Document ID

        Returns:
            List of metadata ChunkMetadata objects
        """
        metadata_chunks = []
        doc_metadata = extraction_result.metadata

        # Build metadata text representation
        metadata_parts = []

        # Core document info
        if doc_metadata.file_name:
            metadata_parts.append(f"File: {doc_metadata.file_name}")

        if doc_metadata.file_type:
            metadata_parts.append(f"Type: {doc_metadata.file_type}")

        # PDF-specific metadata
        if hasattr(doc_metadata, 'title') and doc_metadata.title:
            metadata_parts.append(f"Title: {doc_metadata.title}")

        if hasattr(doc_metadata, 'author') and doc_metadata.author:
            metadata_parts.append(f"Author: {doc_metadata.author}")

        if hasattr(doc_metadata, 'subject') and doc_metadata.subject:
            metadata_parts.append(f"Subject: {doc_metadata.subject}")

        if hasattr(doc_metadata, 'keywords') and doc_metadata.keywords:
            metadata_parts.append(f"Keywords: {doc_metadata.keywords}")

        if hasattr(doc_metadata, 'creator') and doc_metadata.creator:
            metadata_parts.append(f"Creator: {doc_metadata.creator}")

        if hasattr(doc_metadata, 'producer') and doc_metadata.producer:
            metadata_parts.append(f"Producer: {doc_metadata.producer}")

        if hasattr(doc_metadata, 'creation_date') and doc_metadata.creation_date:
            metadata_parts.append(f"Created: {doc_metadata.creation_date}")

        if hasattr(doc_metadata, 'modification_date') and doc_metadata.modification_date:
            metadata_parts.append(f"Modified: {doc_metadata.modification_date}")

        # Page count
        if hasattr(doc_metadata, 'page_count') and doc_metadata.page_count:
            metadata_parts.append(f"Pages: {doc_metadata.page_count}")

        # Build metadata text
        metadata_text = "\n".join(metadata_parts) if metadata_parts else f"Document: {doc_metadata.file_name}"

        # Create metadata chunk
        # Format chunk_id with project_id if available
        if project_id:
            metadata_chunk_id = f"{project_id}_{doc_id}_metadata_0000"
        else:
            metadata_chunk_id = f"{doc_id}_metadata_0000"

        metadata_chunk = ChunkMetadata(
            doc_id=doc_id,
            project_id=project_id,
            file_name=doc_metadata.file_name,
            chunk_id=metadata_chunk_id,
            page_number_start=1,
            page_number_end=1,
            section_title="Document Metadata",
            heading_path=["Document Metadata"],
            heading_level=1,
            chunk_index=0,
            total_chunks=1,
            chunk_char_len=len(metadata_text),
            chunk_word_count=len(metadata_text.split()),
            boundary_type=BoundaryType.PAGE.value,
            chunk_type=ChunkType.METADATA.value,
            normalized_text=metadata_text,
            has_overlap=False
        )

        metadata_chunks.append(metadata_chunk)

        return metadata_chunks

    def _generate_summary_chunks(
        self,
        content_chunks: List[ChunkMetadata],
        extraction_result,
        doc_id: str,
        project_id: Optional[str] = None
    ) -> List[ChunkMetadata]:
        """
        Generate summary chunks for high-level document understanding.

        Summary chunks contain:
        - Document-level summary (first N sentences of document)
        - Section-level summaries (first N sentences of each section)

        These chunks are optimized for:
        - High-level semantic matching
        - Document overview retrieval
        - Topic identification
        - Executive summary search

        Note: This uses extractive summarization (first N sentences).
        For abstractive summaries, integrate with an external summarizer.

        Args:
            content_chunks: List of content chunks
            extraction_result: Original extraction result
            doc_id: Document ID

        Returns:
            List of summary ChunkMetadata objects
        """
        summary_chunks = []

        # Simple sentence splitting function
        def split_sentences(text: str) -> List[str]:
            """Split text into sentences using simple pattern."""
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            return [s.strip() for s in parts if s.strip() and len(s.strip()) > 20]

        # Generate document-level summary
        if self.config.generate_document_summary:
            # Collect text from first few content chunks
            doc_text = " ".join([c.normalized_text for c in content_chunks[:5]])
            sentences = split_sentences(doc_text)

            # Take first N sentences as summary
            summary_sentences = sentences[:self.config.summary_sentences]
            doc_summary_text = " ".join(summary_sentences)

            if doc_summary_text:
                # Format chunk_id with project_id if available
                if project_id:
                    doc_summary_chunk_id = f"{project_id}_{doc_id}_summary_doc_0000"
                else:
                    doc_summary_chunk_id = f"{doc_id}_summary_doc_0000"

                doc_summary_chunk = ChunkMetadata(
                    doc_id=doc_id,
                    project_id=project_id,
                    file_name=extraction_result.metadata.file_name,
                    chunk_id=doc_summary_chunk_id,
                    page_number_start=1,
                    page_number_end=content_chunks[-1].page_number_end if content_chunks else 1,
                    section_title="Document Summary",
                    heading_path=["Document Summary"],
                    heading_level=1,
                    chunk_index=0,
                    total_chunks=0,  # Updated later
                    chunk_char_len=len(doc_summary_text),
                    chunk_word_count=len(doc_summary_text.split()),
                    boundary_type=BoundaryType.PAGE.value,
                    chunk_type=ChunkType.SUMMARY.value,
                    normalized_text=doc_summary_text,
                    has_overlap=False
                )
                summary_chunks.append(doc_summary_chunk)

        # Generate section-level summaries
        if self.config.generate_section_summaries:
            # Group content chunks by section
            sections = {}
            for chunk in content_chunks:
                section = chunk.section_title or "Untitled Section"
                if section not in sections:
                    sections[section] = []
                sections[section].append(chunk)

            for section_title, section_chunks in sections.items():
                if section_title == "Untitled Section":
                    continue  # Skip untitled sections

                # Combine section text
                section_text = " ".join([c.normalized_text for c in section_chunks])
                sentences = split_sentences(section_text)

                # Take first N sentences as section summary
                summary_sentences = sentences[:self.config.summary_sentences]
                section_summary_text = " ".join(summary_sentences)

                if section_summary_text and len(section_summary_text) > 50:
                    first_chunk = section_chunks[0]
                    last_chunk = section_chunks[-1]

                    # Format chunk_id with project_id if available
                    if project_id:
                        section_summary_chunk_id = f"{project_id}_{doc_id}_summary_sec_{len(summary_chunks):04d}"
                    else:
                        section_summary_chunk_id = f"{doc_id}_summary_sec_{len(summary_chunks):04d}"

                    section_summary_chunk = ChunkMetadata(
                        doc_id=doc_id,
                        project_id=project_id,
                        file_name=extraction_result.metadata.file_name,
                        chunk_id=section_summary_chunk_id,
                        page_number_start=first_chunk.page_number_start,
                        page_number_end=last_chunk.page_number_end,
                        section_title=f"Summary: {section_title}",
                        section_title_raw=section_title,
                        heading_path=first_chunk.heading_path + ["Summary"] if first_chunk.heading_path else [section_title, "Summary"],
                        heading_level=first_chunk.heading_level,
                        parent_section=first_chunk.parent_section,
                        chunk_index=len(summary_chunks),
                        total_chunks=0,  # Updated later
                        chunk_char_len=len(section_summary_text),
                        chunk_word_count=len(section_summary_text.split()),
                        boundary_type=BoundaryType.SECTION.value,
                        chunk_type=ChunkType.SUMMARY.value,
                        parent_chunk_id=first_chunk.chunk_id,
                        normalized_text=section_summary_text,
                        has_overlap=False
                    )
                    summary_chunks.append(section_summary_chunk)

        # Update total_chunks count
        for chunk in summary_chunks:
            chunk.total_chunks = len(summary_chunks)

        return summary_chunks

    def generate_hybrid_chunks(
        self,
        extraction_result,
        doc_id: Optional[str] = None,
        normalized_text: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, List[ChunkMetadata]]:
        """
        Generate all chunk types for hybrid (OpenSearch + vector) search.

        This is the main entry point for hybrid chunking. It generates:
        - Content chunks (standard semantic chunks)
        - Heading chunks (for keyword search)
        - Clause chunks (for precise retrieval)
        - Metadata chunks (for filtering)
        - Summary chunks (for high-level matching)

        Args:
            extraction_result: ExtractionResult from document_processor
            doc_id: Optional document ID
            normalized_text: Optional pre-normalized text

        Returns:
            Dictionary with keys: 'content', 'heading', 'clause', 'metadata', 'summary'
            Each value is a list of ChunkMetadata objects
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # Generate standard content chunks first
        content_chunks = self.chunk_document(extraction_result, doc_id, normalized_text, project_id=project_id)

        # Initialize result dictionary
        hybrid_chunks = {
            'content': content_chunks,
            'heading': [],
            'clause': [],
            'metadata': [],
            'summary': []
        }

        # Generate heading chunks if enabled
        if self.config.generate_heading_chunks:
            hybrid_chunks['heading'] = self._generate_heading_chunks(
                content_chunks, extraction_result, doc_id, project_id=project_id
            )

        # Generate clause chunks if enabled
        if self.config.generate_clause_chunks:
            hybrid_chunks['clause'] = self._generate_clause_chunks(
                content_chunks, extraction_result, doc_id, project_id=project_id
            )

        # Generate metadata chunks if enabled
        if self.config.generate_metadata_chunks:
            hybrid_chunks['metadata'] = self._generate_metadata_chunks(
                extraction_result, doc_id, project_id=project_id
            )

        # Generate summary chunks if enabled
        if self.config.generate_summary_chunks:
            hybrid_chunks['summary'] = self._generate_summary_chunks(
                content_chunks, extraction_result, doc_id, project_id=project_id
            )

        # Single consolidated log for hybrid chunking
        logger.info(
            f"Hybrid chunks: content={len(hybrid_chunks['content'])} "
            f"clause={len(hybrid_chunks['clause'])} heading={len(hybrid_chunks['heading'])} "
            f"metadata={len(hybrid_chunks['metadata'])} summary={len(hybrid_chunks['summary'])}"
        )

        return hybrid_chunks

    def get_all_chunks_flat(
        self,
        hybrid_chunks: Dict[str, List[ChunkMetadata]]
    ) -> List[ChunkMetadata]:
        """
        Flatten hybrid chunks dictionary into a single list.

        Useful for bulk indexing where you want all chunks in one list.

        Args:
            hybrid_chunks: Dictionary from generate_hybrid_chunks()

        Returns:
            Single flat list of all ChunkMetadata objects
        """
        all_chunks = []
        for chunk_type in ['content', 'heading', 'clause', 'metadata', 'summary']:
            all_chunks.extend(hybrid_chunks.get(chunk_type, []))
        return all_chunks


# Convenience functions

def chunk_document_simple(
    extraction_result,
    max_chunk_size: int = 1000,
    enable_overlap: bool = True,
    overlap_size: int = 100
) -> List[ChunkMetadata]:
    """
    Simple convenience function for chunking.

    Args:
        extraction_result: ExtractionResult from document_processor
        max_chunk_size: Maximum chunk size in characters
        enable_overlap: Whether to enable overlap
        overlap_size: Overlap size in characters

    Returns:
        List of ChunkMetadata
    """
    config = ChunkingConfig(
        max_chunk_size=max_chunk_size,
        enable_overlap=enable_overlap,
        overlap_size=overlap_size
    )

    pipeline = EnterpriseChunkingPipeline(config)
    return pipeline.chunk_document(extraction_result)


def chunk_with_normalization(
    extraction_result,
    normalized_text: str,
    config: Optional[ChunkingConfig] = None,
    project_id: Optional[str] = None
) -> List[ChunkMetadata]:
    """
    Chunk using pre-normalized text.

    Args:
        extraction_result: ExtractionResult from document_processor
        normalized_text: Pre-normalized text (from MetadataAwareNormalizer)
        config: Optional ChunkingConfig
        project_id: Optional project ID for multi-tenancy

    Returns:
        List of ChunkMetadata
    """
    pipeline = EnterpriseChunkingPipeline(config)
    return pipeline.chunk_document(extraction_result, normalized_text=normalized_text, project_id=project_id)


def chunk_for_hybrid_search(
    extraction_result,
    max_chunk_size: int = 1000,
    enable_overlap: bool = True,
    overlap_size: int = 100,
    generate_heading_chunks: bool = True,
    generate_clause_chunks: bool = True,
    generate_metadata_chunks: bool = True,
    generate_summary_chunks: bool = True
) -> Dict[str, List[ChunkMetadata]]:
    """
    Convenience function for hybrid (OpenSearch + vector) search chunking.

    Generates multiple chunk types optimized for different search strategies:
    - Content chunks: Standard semantic chunks for vector search
    - Heading chunks: Section titles for keyword/faceted search
    - Clause chunks: Single sentences for precise retrieval
    - Metadata chunks: Document metadata for filtering
    - Summary chunks: Document/section summaries for high-level matching

    Args:
        extraction_result: ExtractionResult from document_processor
        max_chunk_size: Maximum content chunk size in characters
        enable_overlap: Whether to enable overlap for content chunks
        overlap_size: Overlap size in characters
        generate_heading_chunks: Generate heading-only chunks
        generate_clause_chunks: Generate clause/sentence chunks
        generate_metadata_chunks: Generate metadata chunks
        generate_summary_chunks: Generate summary chunks

    Returns:
        Dictionary with keys: 'content', 'heading', 'clause', 'metadata', 'summary'
    """
    config = ChunkingConfig(
        max_chunk_size=max_chunk_size,
        enable_overlap=enable_overlap,
        overlap_size=overlap_size,
        generate_heading_chunks=generate_heading_chunks,
        generate_clause_chunks=generate_clause_chunks,
        generate_metadata_chunks=generate_metadata_chunks,
        generate_summary_chunks=generate_summary_chunks
    )

    pipeline = EnterpriseChunkingPipeline(config)
    return pipeline.generate_hybrid_chunks(extraction_result)


def get_hybrid_config(
    max_chunk_size: int = 1000,
    enable_all_chunk_types: bool = True
) -> ChunkingConfig:
    """
    Get a pre-configured ChunkingConfig for hybrid search.

    Args:
        max_chunk_size: Maximum content chunk size
        enable_all_chunk_types: If True, enables all hybrid chunk types

    Returns:
        ChunkingConfig configured for hybrid search
    """
    return ChunkingConfig(
        max_chunk_size=max_chunk_size,
        target_chunk_size=max_chunk_size // 2,
        enable_overlap=True,
        overlap_size=100,
        generate_heading_chunks=enable_all_chunk_types,
        generate_clause_chunks=enable_all_chunk_types,
        generate_metadata_chunks=enable_all_chunk_types,
        generate_summary_chunks=enable_all_chunk_types
    )


if __name__ == "__main__":
    print("Enterprise Chunking Pipeline")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✅ Page-level chunking with source tracking")
    print("  ✅ Section-aware chunking using document hierarchy")
    print("  ✅ Semantic windowing with overlap")
    print("  ✅ Token-aware chunking")
    print("  ✅ Rich metadata for each chunk")
    print("\n  🆕 HYBRID SEARCH CHUNK TYPES (OpenSearch + Vector):")
    print("  ✅ Heading-only chunks - for keyword/faceted search")
    print("  ✅ Clause-only chunks - for precise sentence-level retrieval")
    print("  ✅ Metadata-only chunks - for document filtering")
    print("  ✅ Summary chunks - for high-level document matching")
    print("\nUsage example:")