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
    section_title: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)  # e.g., ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]

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

    # Content
    normalized_text: str = ""  # Actual chunk content (normalized)
    original_page_text: Optional[str] = None  # For debugging

    # Additional context
    contains_tables: bool = False
    contains_code: bool = False
    contains_bullets: bool = False
    urls_in_chunk: List[str] = field(default_factory=list)

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


class BoundaryDetector:
    """
    Detects various document boundaries for intelligent chunking.
    """

    def __init__(self):
        # Section markers
        self.section_patterns = [
            re.compile(r'^<<<HIERARCHY_L1>>>$', re.MULTILINE),
            re.compile(r'^<<<HIERARCHY_L2>>>$', re.MULTILINE),
            re.compile(r'^<<<HIERARCHY_L3>>>$', re.MULTILINE),
            re.compile(r'^[A-Z][A-Z\s]{8,}$', re.MULTILINE),  # ALL CAPS
            re.compile(r'^(?:Chapter|Section|Part)\s+\d+', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\d+\.\s+[A-Z]', re.MULTILINE),  # 1. Title
            re.compile(r'^#{1,6}\s+', re.MULTILINE),  # Markdown headings
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

    def find_section_boundaries(self, text: str) -> List[Tuple[int, str, int]]:
        """
        Find section boundaries.

        Returns:
            List of (position, section_title, level) tuples
        """
        boundaries = []

        for level, pattern in enumerate(self.section_patterns, start=1):
            for match in pattern.finditer(text):
                title = match.group(0).strip()
                boundaries.append((match.start(), title, level))

        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        return boundaries

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
        logger.info("Initializing EnterpriseChunkingPipeline")
        self.config = config or ChunkingConfig()
        logger.info(
            f"Chunking config - Max size: {self.config.max_chunk_size}, "
            f"Min size: {self.config.min_chunk_size}, Target size: {self.config.target_chunk_size}, "
            f"Overlap: {self.config.enable_overlap} ({self.config.overlap_size} chars)"
        )
        self.boundary_detector = BoundaryDetector()
        self.semantic_chunker = SemanticChunker(self.config)
        logger.info("EnterpriseChunkingPipeline initialization complete")

    def chunk_document(
        self,
        extraction_result,
        doc_id: Optional[str] = None,
        normalized_text: Optional[str] = None
    ) -> List[ChunkMetadata]:
        """
        Main entry point: Chunk an entire document.

        Args:
            extraction_result: ExtractionResult from document_processor
            doc_id: Optional document ID (will generate if not provided)
            normalized_text: Optional pre-normalized text (if not provided, uses extraction_result.text)

        Returns:
            List of ChunkMetadata objects
        """
        start_time = time.time()

        if doc_id is None:
            doc_id = str(uuid.uuid4())
            logger.debug(f"Generated doc_id: {doc_id}")

        # Use normalized text if provided, otherwise use extracted text
        text = normalized_text if normalized_text else extraction_result.text

        logger.info(
            f"Starting chunking for document: {extraction_result.metadata.file_name} "
            f"(doc_id: {doc_id[:8]}...)"
        )
        logger.info(f"Document length: {len(text)} characters, {len(extraction_result.pages)} pages")

        # Step 1: Page-level chunking with mapping
        logger.debug(f"Chunking: Starting Step 1 - Page-level chunking")
        step_start = time.time()
        page_chunks = self._chunk_by_pages(text, extraction_result)
        step_duration = time.time() - step_start
        logger.info(f"Step 1: Created {len(page_chunks)} page-level chunks in {step_duration:.2f}s")
        logger.debug(f"Chunking: Page chunks size range: {min((len(c['text']) for c in page_chunks), default=0)} - {max((len(c['text']) for c in page_chunks), default=0)} chars")

        # Step 2: Section-aware chunking
        logger.debug(f"Chunking: Starting Step 2 - Section-aware chunking")
        step_start = time.time()
        section_chunks = self._chunk_by_sections(page_chunks, extraction_result)
        step_duration = time.time() - step_start
        logger.info(f"Step 2: Created {len(section_chunks)} section-aware chunks in {step_duration:.2f}s")
        logger.debug(f"Chunking: Section chunks created from {len(page_chunks)} page chunks")

        # Step 3: Apply size constraints and split large chunks
        logger.debug(f"Chunking: Starting Step 3 - Applying size constraints (max: {self.config.max_chunk_size})")
        step_start = time.time()
        sized_chunks = self._apply_size_constraints(section_chunks)
        step_duration = time.time() - step_start
        split_count = len(sized_chunks) - len(section_chunks)
        logger.info(f"Step 3: Applied size constraints, now {len(sized_chunks)} chunks in {step_duration:.2f}s")
        if split_count > 0:
            logger.debug(f"Chunking: Split {split_count} large chunks to meet size constraints")

        # Step 4: Apply semantic windowing (overlap)
        logger.debug(f"Chunking: Starting Step 4 - Semantic windowing (overlap: {self.config.overlap_size if self.config.enable_overlap else 0})")
        step_start = time.time()
        overlapped_chunks = self._apply_semantic_windowing(sized_chunks)
        step_duration = time.time() - step_start
        logger.info(f"Step 4: Applied semantic windowing in {step_duration:.2f}s")

        # Step 5: Build comprehensive metadata
        logger.debug(f"Chunking: Starting Step 5 - Building chunk metadata")
        step_start = time.time()
        final_chunks = self._build_chunk_metadata(
            overlapped_chunks,
            extraction_result,
            doc_id
        )
        step_duration = time.time() - step_start
        logger.info(f"Step 5: Built metadata for {len(final_chunks)} final chunks in {step_duration:.2f}s")

        # Log chunk statistics
        if final_chunks:
            avg_size = sum(c.chunk_char_len for c in final_chunks) / len(final_chunks)
            logger.debug(
                f"Chunking: Final statistics - "
                f"Chunks: {len(final_chunks)}, "
                f"Avg size: {avg_size:.0f} chars, "
                f"Min: {min(c.chunk_char_len for c in final_chunks)}, "
                f"Max: {max(c.chunk_char_len for c in final_chunks)}"
            )

        total_duration = time.time() - start_time
        logger.info(
            f"Chunking complete for {extraction_result.metadata.file_name} - "
            f"Final chunks: {len(final_chunks)}, Total time: {total_duration:.2f}s"
        )

        return final_chunks

    def _chunk_by_pages(self, text: str, extraction_result) -> List[Dict]:
        """
        Step 1: Page-level chunking.
        Preserves page boundaries and mapping.
        """
        logger.debug("Chunking: Starting page-level chunking")
        chunks = []
        page_boundaries = self.boundary_detector.find_page_boundaries(text)
        logger.debug(f"Chunking: Found {len(page_boundaries)} page boundaries")

        if not page_boundaries:
            # No page markers found - treat as single page
            logger.warning("Chunking: No page markers found, treating as single page")
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

                logger.debug(f"Chunking: Page {page_num} - {len(page_text)} chars")

                chunks.append({
                    'text': page_text,
                    'page_start': page_num,
                    'page_end': page_num,
                    'original_page': original_page
                })

        logger.debug(f"Chunking: Page-level chunking complete - {len(chunks)} chunks created")
        return chunks

    def _chunk_by_sections(self, page_chunks: List[Dict], extraction_result) -> List[Dict]:
        """
        Step 2: Section-aware chunking.
        Splits on section boundaries while respecting page boundaries.
        """
        section_chunks = []

        for page_chunk in page_chunks:
            text = page_chunk['text']
            page_start = page_chunk['page_start']
            page_end = page_chunk['page_end']
            original_page = page_chunk['original_page']

            # Find section boundaries in this page
            section_boundaries = self.boundary_detector.find_section_boundaries(text)

            if not section_boundaries:
                # No sections found - keep as single chunk
                section_chunks.append({
                    'text': text,
                    'page_start': page_start,
                    'page_end': page_end,
                    'section_title': None,
                    'heading_path': [],
                    'original_page': original_page,
                    'boundary_type': BoundaryType.PAGE.value
                })
                continue

            # Split by sections
            heading_stack = []  # Track heading hierarchy

            for i, (pos, title, level) in enumerate(section_boundaries):
                # Update heading stack based on level
                heading_stack = [h for h in heading_stack if h[1] < level]
                heading_stack.append((title, level))

                # Get text until next section (or end)
                if i < len(section_boundaries) - 1:
                    next_pos = section_boundaries[i + 1][0]
                    section_text = text[pos:next_pos]
                else:
                    section_text = text[pos:]

                section_text = section_text.strip()

                if section_text:
                    section_chunks.append({
                        'text': section_text,
                        'page_start': page_start,
                        'page_end': page_end,
                        'section_title': title,
                        'heading_path': [h[0] for h in heading_stack],
                        'original_page': original_page,
                        'boundary_type': BoundaryType.SECTION.value if level == 1 else BoundaryType.SUBSECTION.value
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
            logger.debug(f"Splitting large chunk ({len(text)} chars)")

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
        doc_id: str
    ) -> List[ChunkMetadata]:
        """
        Step 5: Build comprehensive metadata for each chunk.
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

            # Create metadata
            metadata = ChunkMetadata(
                doc_id=doc_id,
                file_name=extraction_result.metadata.file_name,
                chunk_id=f"{doc_id}_chunk_{i:04d}",
                page_number_start=chunk['page_start'],
                page_number_end=chunk['page_end'],
                section_title=chunk.get('section_title'),
                heading_path=chunk.get('heading_path', []),
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
    config: Optional[ChunkingConfig] = None
) -> List[ChunkMetadata]:
    """
    Chunk using pre-normalized text.

    Args:
        extraction_result: ExtractionResult from document_processor
        normalized_text: Pre-normalized text (from MetadataAwareNormalizer)
        config: Optional ChunkingConfig

    Returns:
        List of ChunkMetadata
    """
    pipeline = EnterpriseChunkingPipeline(config)
    return pipeline.chunk_document(extraction_result, normalized_text=normalized_text)


if __name__ == "__main__":
    print("Enterprise Chunking Pipeline")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✅ Page-level chunking with source tracking")
    print("  ✅ Section-aware chunking using document hierarchy")
    print("  ✅ Semantic windowing with overlap")
    print("  ✅ Token-aware chunking")
    print("  ✅ Rich metadata for each chunk")
    print("\nUsage example:")
    print("""
from document_processor import extract_text_from_document
from enterprise_chunking_pipeline import chunk_document_simple, ChunkingConfig

# Extract document
result = extract_text_from_document('document.pdf', extract_metadata=True)

# Simple chunking
chunks = chunk_document_simple(result, max_chunk_size=1000)

# Advanced chunking
config = ChunkingConfig(
    max_chunk_size=1000,
    target_chunk_size=500,
    enable_overlap=True,
    overlap_size=100,
    respect_page_boundaries=True,
    keep_tables_intact=True,
    token_aware=True
)

from enterprise_chunking_pipeline import EnterpriseChunkingPipeline
pipeline = EnterpriseChunkingPipeline(config)
chunks = pipeline.chunk_document(result)

# Access metadata
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}")
    print(f"  Pages: {chunk.page_number_start}-{chunk.page_number_end}")
    print(f"  Section: {chunk.section_title}")
    print(f"  Path: {' > '.join(chunk.heading_path)}")
    print(f"  Size: {chunk.chunk_char_len} chars, {chunk.chunk_word_count} words")
    print(f"  Content: {chunk.normalized_text[:100]}...")
    """)
