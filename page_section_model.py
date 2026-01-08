"""
Physical Page vs Logical Section Model

CRITICAL SEPARATION:
- Physical Pages: Absolute positions in source document (immutable)
- Logical Sections: Hierarchical content structure (variable depth)

Chunks MUST track BOTH independently for correct citations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


# ============================================================================
# PHYSICAL LOCATION
# ============================================================================

@dataclass
class PhysicalLocation:
    """
    Physical location in source document.
    IMMUTABLE - always refers to actual document pages.
    """
    # Source document reference
    doc_id: str
    source_file: str
    source_format: str  # pdf, docx, ppt

    # Physical page numbers (1-indexed, as shown in PDF viewer)
    page_start: int  # First page this content appears on
    page_end: int    # Last page (inclusive)

    # Optional: precise coordinates within page
    bboxes: Optional[List[dict]] = None  # [{page, x0, y0, x1, y1}, ...]

    # For non-paginated formats
    slide_numbers: Optional[List[int]] = None  # PowerPoint
    section_breaks: Optional[List[int]] = None  # DOCX

    def __post_init__(self):
        assert self.page_start > 0, "Pages are 1-indexed"
        assert self.page_end >= self.page_start

    @property
    def page_range_str(self) -> str:
        """Human-readable page range"""
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    @property
    def span_pages(self) -> int:
        """Number of pages spanned"""
        return self.page_end - self.page_start + 1

    def to_dict(self) -> dict:
        return {
            'doc_id': self.doc_id,
            'source_file': self.source_file,
            'source_format': self.source_format,
            'page_start': self.page_start,
            'page_end': self.page_end,
            'page_range': self.page_range_str,
            'span_pages': self.span_pages,
            'bboxes': self.bboxes,
            'slide_numbers': self.slide_numbers,
            'section_breaks': self.section_breaks
        }


# ============================================================================
# LOGICAL STRUCTURE
# ============================================================================

@dataclass
class LogicalSection:
    """
    Logical section in document hierarchy.
    STRUCTURAL - represents content organization, not physical layout.
    """
    # Section identification
    section_id: str  # Unique ID within document
    title: str       # Section title

    # Hierarchical path from root
    section_path: List[str]  # ["Chapter 1", "Section 1.1", "Subsection 1.1.2"]
    depth: int               # 1 = top-level, 2 = subsection, etc.

    # Parent/children relationships
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)

    # Physical mapping (where this section appears)
    # Note: One logical section can span multiple pages
    physical_page_start: int = 0
    physical_page_end: int = 0

    def __post_init__(self):
        assert self.depth == len(self.section_path), \
            "Depth must match path length"

    @property
    def breadcrumb(self) -> str:
        """Human-readable breadcrumb"""
        return " > ".join(self.section_path)

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    def to_dict(self) -> dict:
        return {
            'section_id': self.section_id,
            'title': self.title,
            'section_path': self.section_path,
            'breadcrumb': self.breadcrumb,
            'depth': self.depth,
            'parent_id': self.parent_id,
            'child_ids': self.child_ids,
            'physical_page_start': self.physical_page_start,
            'physical_page_end': self.physical_page_end
        }


# ============================================================================
# UPDATED CHUNK SCHEMA
# ============================================================================

@dataclass
class ChunkMetadata:
    """
    Chunk metadata with STRICT separation of physical and logical.

    RULES:
    1. physical_location is ALWAYS set (required)
    2. logical_section MAY be None (for unstructured content)
    3. NEVER derive physical pages from logical sections
    4. NEVER derive logical sections from physical pages
    """
    # Identity
    chunk_id: str
    chunk_index: int

    # ========== PHYSICAL LOCATION (REQUIRED) ==========
    physical_location: PhysicalLocation

    # ========== LOGICAL STRUCTURE (OPTIONAL) ==========
    logical_section: Optional[LogicalSection] = None

    # Chunk content
    text: str = ""
    text_with_context: str = ""  # With overlap

    # Block references (from DocumentIR)
    block_ids: List[str] = field(default_factory=list)

    # Content characteristics
    contains_tables: bool = False
    contains_figures: bool = False
    contains_code: bool = False
    contains_lists: bool = False

    # Statistics
    char_count: int = 0
    word_count: int = 0
    token_count: Optional[int] = None

    # Chunking metadata
    boundary_type: str = "paragraph"  # how chunk was bounded
    has_overlap: bool = False
    overlap_before_size: int = 0
    overlap_after_size: int = 0

    # Timestamps
    created_at: str = ""
    pipeline_version: str = ""

    def __post_init__(self):
        # Enforce required fields
        assert self.physical_location is not None, \
            "physical_location is REQUIRED"
        assert self.physical_location.page_start > 0, \
            "page_start must be positive"

        # Auto-compute stats
        if not self.char_count and self.text:
            self.char_count = len(self.text)
            self.word_count = len(self.text.split())

    def to_dict(self) -> dict:
        """Serialize to dict for storage"""
        return {
            'chunk_id': self.chunk_id,
            'chunk_index': self.chunk_index,

            # Physical location (always present)
            'physical_location': self.physical_location.to_dict(),

            # Logical section (may be null)
            'logical_section': self.logical_section.to_dict() if self.logical_section else None,

            # Content
            'text': self.text,
            'text_with_context': self.text_with_context,
            'block_ids': self.block_ids,

            # Flags
            'contains_tables': self.contains_tables,
            'contains_figures': self.contains_figures,
            'contains_code': self.contains_code,
            'contains_lists': self.contains_lists,

            # Stats
            'char_count': self.char_count,
            'word_count': self.word_count,
            'token_count': self.token_count,

            # Chunking metadata
            'boundary_type': self.boundary_type,
            'has_overlap': self.has_overlap,
            'overlap_before_size': self.overlap_before_size,
            'overlap_after_size': self.overlap_after_size,

            # Timestamps
            'created_at': self.created_at,
            'pipeline_version': self.pipeline_version
        }

    # Convenience accessors (safe delegation)
    @property
    def page_start(self) -> int:
        """Physical page start"""
        return self.physical_location.page_start

    @property
    def page_end(self) -> int:
        """Physical page end"""
        return self.physical_location.page_end

    @property
    def page_range_str(self) -> str:
        """Physical page range string"""
        return self.physical_location.page_range_str

    @property
    def section_breadcrumb(self) -> Optional[str]:
        """Logical section breadcrumb (may be None)"""
        return self.logical_section.breadcrumb if self.logical_section else None

    @property
    def section_title(self) -> Optional[str]:
        """Logical section title (may be None)"""
        return self.logical_section.title if self.logical_section else None


# ============================================================================
# CITATION GENERATOR
# ============================================================================

class CitationGenerator:
    """
    Generates citations from chunks using physical + logical metadata.

    SAFETY RULES:
    1. Physical pages are ALWAYS included
    2. Logical sections are included IF available
    3. Never conflate the two
    """

    @staticmethod
    def generate_citation(chunk: ChunkMetadata, style: str = "apa") -> dict:
        """
        Generate citation from chunk metadata.

        Args:
            chunk: Chunk with physical + logical metadata
            style: Citation style (apa, mla, chicago, ieee)
        """
        physical = chunk.physical_location
        logical = chunk.logical_section

        citation = {
            # Core identification
            'chunk_id': chunk.chunk_id,
            'source_file': physical.source_file,
            'doc_id': physical.doc_id,

            # PHYSICAL location (always present)
            'physical': {
                'page_start': physical.page_start,
                'page_end': physical.page_end,
                'page_range': physical.page_range_str,
                'format': physical.source_format,
                'bboxes': physical.bboxes
            },

            # LOGICAL structure (may be absent)
            'logical': logical.to_dict() if logical else None,

            # Combined citation text
            'citation_text': CitationGenerator._format_citation(
                physical, logical, style
            )
        }

        return citation

    @staticmethod
    def _format_citation(
        physical: PhysicalLocation,
        logical: Optional[LogicalSection],
        style: str
    ) -> str:
        """Format citation text in specified style"""

        if style == "apa":
            # APA: Source, page, section
            parts = [physical.source_file]

            # Physical pages (always included)
            parts.append(physical.page_range_str)

            # Logical section (if available)
            if logical:
                parts.append(f"({logical.breadcrumb})")

            return ", ".join(parts)

        elif style == "mla":
            # MLA: Source page. Section.
            citation = f"{physical.source_file} {physical.page_range_str}"
            if logical:
                citation += f". {logical.breadcrumb}"
            return citation

        elif style == "chicago":
            # Chicago: Source, page, under "Section"
            citation = f"{physical.source_file}, {physical.page_range_str}"
            if logical:
                citation += f', under "{logical.title}"'
            return citation

        elif style == "ieee":
            # IEEE: [Source], page, sec. Section
            citation = f"{physical.source_file}, {physical.page_range_str}"
            if logical:
                citation += f", sec. {logical.breadcrumb}"
            return citation

        else:
            # Default: file, pages [section]
            citation = f"{physical.source_file}, {physical.page_range_str}"
            if logical:
                citation += f" [{logical.breadcrumb}]"
            return citation

    @staticmethod
    def generate_pdf_link(chunk: ChunkMetadata) -> str:
        """
        Generate PDF link with page and bbox.

        Uses PHYSICAL location only (logical sections don't have coordinates).
        """
        physical = chunk.physical_location

        # Base: file#page
        link = f"{physical.source_file}#page={physical.page_start}"

        # Add bbox if available
        if physical.bboxes and len(physical.bboxes) > 0:
            bbox = physical.bboxes[0]
            link += f"&viewrect={bbox['x0']},{bbox['y0']},{bbox['x1']},{bbox['y1']}"

        return link

    @staticmethod
    def validate_citation(chunk: ChunkMetadata) -> Tuple[bool, List[str]]:
        """
        Validate chunk has proper citation metadata.

        Returns: (is_valid, list_of_errors)
        """
        errors = []

        # Physical location is REQUIRED
        if not chunk.physical_location:
            errors.append("Missing physical_location")
        else:
            if chunk.physical_location.page_start <= 0:
                errors.append("Invalid page_start (must be >= 1)")
            if chunk.physical_location.page_end < chunk.physical_location.page_start:
                errors.append("page_end < page_start")
            if not chunk.physical_location.source_file:
                errors.append("Missing source_file")

        # Logical section validation (if present)
        if chunk.logical_section:
            if not chunk.logical_section.section_path:
                errors.append("Logical section has empty section_path")
            if chunk.logical_section.depth != len(chunk.logical_section.section_path):
                errors.append("Section depth doesn't match path length")

        return len(errors) == 0, errors


# ============================================================================
# EXAMPLES
# ============================================================================

def example_physical_vs_logical():
    """Demonstrate physical vs logical separation"""

    print("=" * 70)
    print("EXAMPLE: Physical Pages vs Logical Sections")
    print("=" * 70)

    # Physical location: Where content PHYSICALLY appears
    physical = PhysicalLocation(
        doc_id="doc123",
        source_file="research_paper.pdf",
        source_format="pdf",
        page_start=15,
        page_end=17,
        bboxes=[
            {'page': 15, 'x0': 72, 'y0': 200, 'x1': 500, 'y1': 700},
            {'page': 16, 'x0': 72, 'y0': 100, 'x1': 500, 'y1': 750},
            {'page': 17, 'x0': 72, 'y0': 100, 'x1': 500, 'y1': 300}
        ]
    )

    # Logical section: Where content LOGICALLY belongs
    logical = LogicalSection(
        section_id="3.2.1",
        title="Experimental Results",
        section_path=["Chapter 3: Methodology", "Section 3.2: Analysis", "3.2.1: Experimental Results"],
        depth=3,
        parent_id="3.2",
        physical_page_start=15,  # Section starts on page 15
        physical_page_end=20      # Section ends on page 20
    )

    # Create chunk
    chunk = ChunkMetadata(
        chunk_id="doc123::chunk::42",
        chunk_index=42,
        physical_location=physical,
        logical_section=logical,
        text="The experimental results show that...",
        char_count=150,
        word_count=25
    )

    print("\n1. PHYSICAL LOCATION (Immutable)")
    print("-" * 70)
    print(f"   Pages: {physical.page_range_str}")
    print(f"   Spans: {physical.span_pages} pages")
    print(f"   Bboxes: {len(physical.bboxes)} boxes")
    print(f"   → This chunk PHYSICALLY appears on pages 15-17")

    print("\n2. LOGICAL SECTION (Structural)")
    print("-" * 70)
    print(f"   Breadcrumb: {logical.breadcrumb}")
    print(f"   Title: {logical.title}")
    print(f"   Depth: {logical.depth}")
    print(f"   Section spans pages: {logical.physical_page_start}-{logical.physical_page_end}")
    print(f"   → This chunk LOGICALLY belongs to Section 3.2.1")

    print("\n3. KEY DIFFERENCE")
    print("-" * 70)
    print(f"   Chunk physical pages: {chunk.page_start}-{chunk.page_end}")
    print(f"   Section total pages: {logical.physical_page_start}-{logical.physical_page_end}")
    print(f"   → Chunk is a SUBSET of the section")
    print(f"   → Physical location = where to FIND it")
    print(f"   → Logical section = what it's ABOUT")

    print("\n4. CITATIONS")
    print("-" * 70)

    cit_gen = CitationGenerator()

    # Generate citations in different styles
    for style in ['apa', 'mla', 'chicago', 'ieee']:
        citation = cit_gen.generate_citation(chunk, style)
        print(f"\n   {style.upper()}:")
        print(f"   {citation['citation_text']}")

    # PDF link uses physical location
    pdf_link = cit_gen.generate_pdf_link(chunk)
    print(f"\n   PDF Link:")
    print(f"   {pdf_link}")

    print("\n5. VALIDATION")
    print("-" * 70)
    is_valid, errors = cit_gen.validate_citation(chunk)
    print(f"   Valid: {is_valid}")
    if errors:
        for err in errors:
            print(f"   - {err}")
    else:
        print("   ✓ All citation metadata is correct")

    print("\n" + "=" * 70)


def example_edge_cases():
    """Show edge cases"""

    print("\n" + "=" * 70)
    print("EDGE CASES")
    print("=" * 70)

    # Case 1: Chunk without logical section (e.g., appendix)
    print("\n1. Chunk WITHOUT logical section")
    chunk1 = ChunkMetadata(
        chunk_id="doc::chunk::99",
        chunk_index=99,
        physical_location=PhysicalLocation(
            doc_id="doc", source_file="report.pdf", source_format="pdf",
            page_start=50, page_end=50
        ),
        logical_section=None,  # No section structure
        text="Appendix data..."
    )

    citation1 = CitationGenerator.generate_citation(chunk1)
    print(f"   Citation: {citation1['citation_text']}")
    print(f"   Logical: {citation1['logical']}")
    print("   → Physical pages always present, logical may be None")

    # Case 2: Section spanning many pages
    print("\n2. Section spanning many pages")
    large_section = LogicalSection(
        section_id="2",
        title="Literature Review",
        section_path=["Chapter 2: Literature Review"],
        depth=1,
        physical_page_start=5,
        physical_page_end=45  # 40 pages!
    )

    chunk2 = ChunkMetadata(
        chunk_id="doc::chunk::10",
        chunk_index=10,
        physical_location=PhysicalLocation(
            doc_id="doc", source_file="thesis.pdf", source_format="pdf",
            page_start=12, page_end=13  # Just 2 pages
        ),
        logical_section=large_section,
        text="Previous research shows..."
    )

    print(f"   Chunk pages: {chunk2.page_start}-{chunk2.page_end}")
    print(f"   Section pages: {large_section.physical_page_start}-{large_section.physical_page_end}")
    print("   → Chunk is small part of large section")

    # Case 3: Single page, deep nesting
    print("\n3. Single page, deeply nested section")
    deep_section = LogicalSection(
        section_id="4.2.3.1",
        title="Implementation Details",
        section_path=["Ch4", "Sec 4.2", "Subsec 4.2.3", "4.2.3.1"],
        depth=4,
        physical_page_start=82,
        physical_page_end=82
    )

    chunk3 = ChunkMetadata(
        chunk_id="doc::chunk::200",
        chunk_index=200,
        physical_location=PhysicalLocation(
            doc_id="doc", source_file="spec.pdf", source_format="pdf",
            page_start=82, page_end=82
        ),
        logical_section=deep_section,
        text="The implementation uses..."
    )

    citation3 = CitationGenerator.generate_citation(chunk3, 'apa')
    print(f"   Citation: {citation3['citation_text']}")
    print(f"   Breadcrumb: {deep_section.breadcrumb}")
    print("   → Deep nesting preserved in logical structure")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    example_physical_vs_logical()
    example_edge_cases()
