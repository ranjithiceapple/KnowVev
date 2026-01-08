"""
IR-Based Chunking Pipeline

This module demonstrates how to chunk documents using the DocumentIR
instead of markdown parsing. It shows the complete workflow from IR to chunks.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from document_ir import (
    DocumentIR, ContentBlock, Heading, Paragraph, Table,
    Figure, CodeBlock, ListBlock, HeadingLevel, Section
)


# ============================================================================
# CHUNK DATA MODEL
# ============================================================================

@dataclass
class IRChunk:
    """
    A chunk created from DocumentIR blocks.
    Contains precise metadata and block references.
    """
    # Identification
    chunk_id: str
    doc_id: str
    chunk_index: int

    # Source blocks (THE TRUTH)
    block_ids: List[str]
    blocks: List[ContentBlock]  # Actual block objects

    # Derived metadata (computed from blocks)
    page_start: int
    page_end: int
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    section_path: List[str] = field(default_factory=list)

    # Content (generated from blocks)
    text: str = ""
    text_with_overlap: str = ""

    # Overlap tracking
    overlap_before: Optional['IRChunk'] = None
    overlap_after: Optional['IRChunk'] = None
    overlap_before_text: str = ""
    overlap_after_text: str = ""

    # Content flags (computed from blocks)
    contains_headings: bool = False
    contains_tables: bool = False
    contains_figures: bool = False
    contains_code: bool = False
    contains_lists: bool = False

    # Statistics
    char_count: int = 0
    word_count: int = 0
    block_count: int = 0

    # Bounding box aggregation
    bbox_summary: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Compute derived fields from blocks"""
        if not self.text:
            self.text = self._generate_text()

        if not self.text_with_overlap:
            self.text_with_overlap = self._generate_text_with_overlap()

        # Compute page range
        self.page_start = min(b.location.page_number for b in self.blocks)
        self.page_end = max(b.location.page_number for b in self.blocks)

        # Detect content types
        self.contains_headings = any(isinstance(b, Heading) for b in self.blocks)
        self.contains_tables = any(isinstance(b, Table) for b in self.blocks)
        self.contains_figures = any(isinstance(b, Figure) for b in self.blocks)
        self.contains_code = any(isinstance(b, CodeBlock) for b in self.blocks)
        self.contains_lists = any(isinstance(b, ListBlock) for b in self.blocks)

        # Statistics
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())
        self.block_count = len(self.blocks)

    def _generate_text(self) -> str:
        """Generate plain text from blocks"""
        parts = []
        for block in self.blocks:
            if isinstance(block, Table):
                # Include table caption and context
                if block.caption:
                    parts.append(f"[{block.caption}]")
                if block.context_before:
                    parts.append(block.context_before)
                parts.append(block.to_markdown())
                if block.context_after:
                    parts.append(block.context_after)
            elif isinstance(block, Figure):
                # Include figure caption
                caption = block.caption or block.text
                parts.append(f"[Figure: {caption}]")
            elif isinstance(block, CodeBlock):
                parts.append(f"```{block.language or ''}\n{block.text}\n```")
            else:
                parts.append(block.text)

        return '\n\n'.join(parts)

    def _generate_text_with_overlap(self) -> str:
        """Generate text including overlap regions"""
        parts = []

        # Overlap before
        if self.overlap_before_text:
            parts.append(f"[...previous context...]\n{self.overlap_before_text}")

        # Main content
        parts.append(self.text)

        # Overlap after
        if self.overlap_after_text:
            parts.append(f"{self.overlap_after_text}\n[...next context...]")

        return '\n\n'.join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'chunk_index': self.chunk_index,
            'block_ids': self.block_ids,
            'page_start': self.page_start,
            'page_end': self.page_end,
            'section_id': self.section_id,
            'section_title': self.section_title,
            'section_path': self.section_path,
            'text': self.text,
            'text_with_overlap': self.text_with_overlap,
            'contains_headings': self.contains_headings,
            'contains_tables': self.contains_tables,
            'contains_figures': self.contains_figures,
            'contains_code': self.contains_code,
            'contains_lists': self.contains_lists,
            'char_count': self.char_count,
            'word_count': self.word_count,
            'block_count': self.block_count,
            'bbox_summary': self.bbox_summary
        }


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

@dataclass
class IRChunkingConfig:
    """Configuration for IR-based chunking"""
    # Size limits (in characters)
    min_chunk_size: int = 100
    target_chunk_size: int = 500
    max_chunk_size: int = 1000
    hard_max_size: int = 2000  # For tables/code blocks that can't be split

    # Boundary respect
    respect_heading_boundaries: bool = True
    heading_levels_for_split: List[int] = field(default_factory=lambda: [1, 2])  # H1, H2
    respect_section_boundaries: bool = True
    respect_page_boundaries: bool = False  # Usually False for better context

    # Special element handling
    keep_tables_intact: bool = True
    keep_code_blocks_intact: bool = True
    keep_lists_intact: bool = True
    keep_figures_with_context: bool = True  # Include surrounding paragraphs

    # Overlap for semantic continuity
    enable_overlap: bool = True
    overlap_size: int = 100  # Characters
    overlap_blocks: int = 1  # Number of blocks to overlap

    # Section context
    include_section_title_in_chunk: bool = True  # Prepend section title to chunk


# ============================================================================
# IR-BASED CHUNKER
# ============================================================================

class IRChunker:
    """
    Chunk DocumentIR into semantically meaningful chunks.
    Operates directly on structured blocks, no markdown parsing.
    """

    def __init__(self, config: IRChunkingConfig = None):
        self.config = config or IRChunkingConfig()

    def chunk(self, ir: DocumentIR) -> List[IRChunk]:
        """
        Main chunking method.
        Returns list of IRChunk objects with precise metadata.
        """
        chunks = []
        current_chunk_blocks = []
        current_section = None
        current_size = 0

        for block in ir.blocks:
            # Track current section
            if isinstance(block, Heading) and block.section_id:
                section = ir.sections.get(block.section_id)
                if section:
                    current_section = section

                    # Check if we should split on this heading
                    if self._should_split_on_heading(block, current_chunk_blocks):
                        # Finalize current chunk
                        if current_chunk_blocks:
                            chunk = self._create_chunk(
                                ir, current_chunk_blocks, len(chunks), current_section
                            )
                            chunks.append(chunk)
                            current_chunk_blocks = []
                            current_size = 0

            # Get block size
            block_size = len(block.text)

            # Check if adding this block exceeds limits
            would_exceed = current_size + block_size > self.config.max_chunk_size

            # Special handling for large blocks that must stay intact
            if self._must_keep_intact(block):
                # If block alone exceeds hard max, we have to include it anyway
                if block_size > self.config.hard_max_size:
                    # Log warning but include it
                    pass

                # If adding this block would exceed max, finalize current chunk first
                if would_exceed and current_chunk_blocks:
                    chunk = self._create_chunk(
                        ir, current_chunk_blocks, len(chunks), current_section
                    )
                    chunks.append(chunk)
                    current_chunk_blocks = []
                    current_size = 0

                # Add the intact block to chunk
                current_chunk_blocks.append(block)
                current_size += block_size

            else:
                # Regular block - can be split
                if would_exceed:
                    # Check if we've reached minimum chunk size
                    if current_size >= self.config.min_chunk_size and current_chunk_blocks:
                        # Finalize current chunk
                        chunk = self._create_chunk(
                            ir, current_chunk_blocks, len(chunks), current_section
                        )
                        chunks.append(chunk)
                        current_chunk_blocks = []
                        current_size = 0

                # Add block
                current_chunk_blocks.append(block)
                current_size += block_size

        # Finalize last chunk
        if current_chunk_blocks:
            chunk = self._create_chunk(
                ir, current_chunk_blocks, len(chunks), current_section
            )
            chunks.append(chunk)

        # Add overlap between chunks
        if self.config.enable_overlap:
            self._add_overlap(chunks)

        return chunks

    def _should_split_on_heading(
        self, heading: Heading, current_blocks: List[ContentBlock]
    ) -> bool:
        """Determine if we should start a new chunk at this heading"""
        if not self.config.respect_heading_boundaries:
            return False

        if not current_blocks:
            return False

        # Split on specified heading levels
        return heading.level.value in self.config.heading_levels_for_split

    def _must_keep_intact(self, block: ContentBlock) -> bool:
        """Check if block must be kept intact (not split)"""
        if isinstance(block, Table) and self.config.keep_tables_intact:
            return True
        if isinstance(block, CodeBlock) and self.config.keep_code_blocks_intact:
            return True
        if isinstance(block, ListBlock) and self.config.keep_lists_intact:
            return True
        return False

    def _create_chunk(
        self,
        ir: DocumentIR,
        blocks: List[ContentBlock],
        chunk_index: int,
        section: Optional[Section]
    ) -> IRChunk:
        """Create IRChunk from blocks"""
        chunk_id = f"{ir.doc_id}::chunk::{chunk_index}"

        # Get section info
        section_id = section.section_id if section else None
        section_title = section.title if section else None
        section_path = section.get_path(ir.sections) if section else []

        # Create chunk
        chunk = IRChunk(
            chunk_id=chunk_id,
            doc_id=ir.doc_id,
            chunk_index=chunk_index,
            block_ids=[b.block_id for b in blocks],
            blocks=blocks,
            page_start=0,  # Will be computed in __post_init__
            page_end=0,
            section_id=section_id,
            section_title=section_title,
            section_path=section_path
        )

        return chunk

    def _add_overlap(self, chunks: List[IRChunk]):
        """Add overlap between consecutive chunks"""
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            # Get last N blocks from current chunk for overlap
            overlap_blocks = current.blocks[-self.config.overlap_blocks:]
            overlap_text = ' '.join(b.text for b in overlap_blocks)

            # Truncate to overlap size
            if len(overlap_text) > self.config.overlap_size:
                overlap_text = overlap_text[-self.config.overlap_size:]

            # Set overlap on next chunk
            next_chunk.overlap_before = current
            next_chunk.overlap_before_text = overlap_text

            # Update text with overlap
            next_chunk.text_with_overlap = next_chunk._generate_text_with_overlap()

            # Set backward reference
            current.overlap_after = next_chunk
            current.overlap_after_text = next_chunk.text[:self.config.overlap_size]


# ============================================================================
# CITATION GENERATOR
# ============================================================================

class IRCitationGenerator:
    """Generate precise citations from IR chunks"""

    @staticmethod
    def generate_citation(chunk: IRChunk, ir: DocumentIR) -> Dict[str, Any]:
        """
        Generate citation metadata from chunk.
        All information is precise and traceable to source blocks.
        """
        # Get section hierarchy
        section_path_str = ' > '.join(chunk.section_path) if chunk.section_path else None

        # Get bounding boxes
        bboxes = [
            b.location.bbox.to_dict()
            for b in chunk.blocks
            if b.location.bbox
        ]

        # Get block types
        block_types = [b.block_type.value for b in chunk.blocks]

        # Generate citation
        citation = {
            # Source identification
            'doc_id': ir.doc_id,
            'source_file': ir.source_file,
            'source_format': ir.source_format,

            # Location (PRECISE)
            'page_start': chunk.page_start,
            'page_end': chunk.page_end,
            'page_range': f"{chunk.page_start}-{chunk.page_end}" if chunk.page_start != chunk.page_end else str(chunk.page_start),

            # Section context
            'section_id': chunk.section_id,
            'section_title': chunk.section_title,
            'section_path': chunk.section_path,
            'section_breadcrumb': section_path_str,

            # Block-level detail
            'block_ids': chunk.block_ids,
            'block_types': block_types,
            'block_count': chunk.block_count,

            # Bounding boxes (for PDF highlighting)
            'bboxes': bboxes,

            # Content flags
            'contains_tables': chunk.contains_tables,
            'contains_figures': chunk.contains_figures,
            'contains_code': chunk.contains_code,

            # Statistics
            'char_count': chunk.char_count,
            'word_count': chunk.word_count,
        }

        return citation

    @staticmethod
    def generate_pdf_link(chunk: IRChunk, ir: DocumentIR) -> str:
        """
        Generate PDF link with page and bounding box.
        Format: file.pdf#page=5&viewrect=x0,y0,x1,y1
        """
        if not chunk.blocks or not chunk.blocks[0].location.bbox:
            return f"{ir.source_file}#page={chunk.page_start}"

        bbox = chunk.blocks[0].location.bbox
        return (
            f"{ir.source_file}#page={chunk.page_start}"
            f"&viewrect={bbox.x0},{bbox.y0},{bbox.x1},{bbox.y1}"
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_chunking_workflow():
    """
    Complete example: IR → Chunks → Citations
    """
    from document_ir import DocumentIRBuilder
    import uuid

    # 1. Build sample IR
    doc_id = str(uuid.uuid4())
    builder = DocumentIRBuilder(doc_id, "example.pdf", "pdf")

    # Add some blocks
    from document_ir import Location, BoundingBox

    builder.add_block(Heading(
        block_id=f"{doc_id}::1::0",
        text="Chapter 1: Introduction",
        level=HeadingLevel.H1,
        location=Location(page_number=1, page_offset=0,
                         bbox=BoundingBox(72, 100, 500, 120, page=1))
    ))

    builder.add_block(Paragraph(
        block_id=f"{doc_id}::1::1",
        text="This is the first paragraph. " * 20,  # ~500 chars
        location=Location(page_number=1, page_offset=50,
                         bbox=BoundingBox(72, 130, 500, 200, page=1))
    ))

    builder.add_block(Paragraph(
        block_id=f"{doc_id}::1::2",
        text="This is the second paragraph. " * 20,
        location=Location(page_number=1, page_offset=100,
                         bbox=BoundingBox(72, 210, 500, 280, page=1))
    ))

    builder.add_block(Table(
        block_id=f"{doc_id}::1::3",
        text="Performance metrics table",
        headers=["Metric", "Value"],
        rows=[["Accuracy", "95%"], ["Precision", "92%"]],
        caption="Performance Metrics",
        location=Location(page_number=1, page_offset=150,
                         bbox=BoundingBox(72, 300, 500, 400, page=1))
    ))

    builder.add_block(Paragraph(
        block_id=f"{doc_id}::1::4",
        text="Analysis of the results shows improvement. " * 15,
        location=Location(page_number=2, page_offset=0,
                         bbox=BoundingBox(72, 100, 500, 170, page=2))
    ))

    ir = builder.build()

    print("=" * 70)
    print("DocumentIR Created")
    print("=" * 70)
    print(f"Doc ID: {ir.doc_id}")
    print(f"Blocks: {ir.total_blocks}")
    print(f"Pages: {ir.total_pages}")
    print(f"Sections: {len(ir.sections)}")
    print()

    # 2. Chunk the IR
    config = IRChunkingConfig(
        target_chunk_size=400,
        max_chunk_size=800,
        enable_overlap=True,
        overlap_size=50
    )

    chunker = IRChunker(config)
    chunks = chunker.chunk(ir)

    print("=" * 70)
    print("Chunks Created")
    print("=" * 70)
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Pages: {chunk.page_start}-{chunk.page_end}")
        print(f"  Section: {chunk.section_title}")
        print(f"  Blocks: {chunk.block_count} ({', '.join(b.block_type.value for b in chunk.blocks)})")
        print(f"  Size: {chunk.char_count} chars, {chunk.word_count} words")
        print(f"  Contains: Tables={chunk.contains_tables}, Code={chunk.contains_code}")
        print(f"  Text preview: {chunk.text[:100]}...")
        print()

    # 3. Generate citations
    print("=" * 70)
    print("Citations")
    print("=" * 70)

    citation_gen = IRCitationGenerator()

    for chunk in chunks[:2]:  # First 2 chunks
        citation = citation_gen.generate_citation(chunk, ir)
        pdf_link = citation_gen.generate_pdf_link(chunk, ir)

        print(f"\nChunk {chunk.chunk_index + 1} Citation:")
        print(f"  Source: {citation['source_file']}")
        print(f"  Location: {citation['section_breadcrumb']}")
        print(f"  Pages: {citation['page_range']}")
        print(f"  Blocks: {citation['block_count']} ({', '.join(citation['block_types'])})")
        print(f"  PDF Link: {pdf_link}")

    print("\n" + "=" * 70)
    print("Key Advantages of IR-Based Chunking:")
    print("=" * 70)
    print("✓ No markdown parsing - operates on structured blocks")
    print("✓ Precise page numbers - derived directly from blocks")
    print("✓ Exact block boundaries - knows what content is included")
    print("✓ Type-safe operations - can't accidentally split tables")
    print("✓ Traceable citations - every chunk references source blocks")
    print("✓ Bounding box tracking - enables PDF highlighting")
    print("=" * 70)


if __name__ == "__main__":
    example_chunking_workflow()
