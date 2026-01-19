"""
Canonical Document Intermediate Representation (DocumentIR)

This module defines the single source of truth for document representation
across the entire ingestion pipeline. All extractors (PDF, DOCX, PPT) must
map their outputs to this IR.

Design Principles:
1. Format-agnostic: Works for PDF, DOCX, PPT, HTML, etc.
2. Structured: Uses typed dataclasses, not strings/markdown
3. Hierarchical: Preserves document structure and section nesting
4. Traceable: Every element has precise location metadata
5. Renderable: Can generate Markdown, HTML, JSON as needed

Key Advantages:
- Eliminates markdown parsing ambiguity during chunking
- Precise page/bbox tracking for every content block
- Unambiguous section hierarchy for citation
- Type-safe operations throughout pipeline
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal, Tuple
from datetime import datetime
from enum import Enum
import json


# ============================================================================
# ENUMS FOR TYPE SAFETY
# ============================================================================

class BlockType(str, Enum):
    """Content block types - exhaustive and mutually exclusive"""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    CODE_BLOCK = "code_block"
    LIST = "list"
    QUOTE = "quote"
    EQUATION = "equation"
    FOOTNOTE = "footnote"
    PAGE_BREAK = "page_break"


class ListType(str, Enum):
    """List formatting types"""
    UNORDERED = "unordered"  # bullets
    ORDERED = "ordered"  # numbered
    DEFINITION = "definition"  # term: definition


class HeadingLevel(int, Enum):
    """Heading levels 1-6"""
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6


# ============================================================================
# LOCATION METADATA
# ============================================================================

@dataclass
class BoundingBox:
    """Bounding box in page coordinates (typically PDF points: 1/72 inch)"""
    x0: float  # Left
    y0: float  # Top (origin depends on format)
    x1: float  # Right
    y1: float  # Bottom
    page_number: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x0': self.x0, 'y0': self.y0,
            'x1': self.x1, 'y1': self.y1,
            'page': self.page_number,
            'width': self.width,
            'height': self.height
        }


@dataclass
class Location:
    """Precise location of content within document"""
    page_number: int  # Physical page number (1-indexed)
    page_offset: int  # Character offset within page (0-indexed)
    bbox: Optional[BoundingBox] = None  # Spatial location if available

    # Optional: for formats with additional structure
    slide_number: Optional[int] = None  # For PPT
    section_index: Optional[int] = None  # DOCX section breaks

    def to_dict(self) -> Dict[str, Any]:
        return {
            'page': self.page_number,
            'offset': self.page_offset,
            'bbox': self.bbox.to_dict() if self.bbox else None,
            'slide': self.slide_number,
            'section': self.section_index
        }


# ============================================================================
# CONTENT BLOCKS
# ============================================================================

@dataclass
class ContentBlock:
    """
    Base class for all content blocks.
    Every block has a unique ID, type, location, and optional styling.
    """
    block_id: str  # Unique identifier: {doc_id}::{page}::{block_index}
    block_type: BlockType
    location: Location

    # Content (plain text, no markdown)
    text: str  # Plain text content (for headings, paragraphs)

    # Metadata for extraction traceability
    extraction_confidence: Optional[float] = None  # 0.0-1.0
    extraction_method: Optional[str] = None  # e.g., "font_size", "toc", "style"

    # Styling hints (optional, for rendering fidelity)
    font_family: Optional[str] = None
    font_size: Optional[float] = None
    is_bold: Optional[bool] = None
    is_italic: Optional[bool] = None
    text_color: Optional[str] = None

    # Raw data for debugging/advanced use
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict"""
        return {
            'block_id': self.block_id,
            'block_type': self.block_type.value,
            'location': self.location.to_dict(),
            'text': self.text,
            'extraction_confidence': self.extraction_confidence,
            'extraction_method': self.extraction_method,
            'font_family': self.font_family,
            'font_size': self.font_size,
            'is_bold': self.is_bold,
            'is_italic': self.is_italic,
            'text_color': self.text_color,
            'raw_metadata': self.raw_metadata
        }


@dataclass
class Heading(ContentBlock):
    """Heading block with hierarchical level"""
    level: HeadingLevel = HeadingLevel.H1

    # Section hierarchy (built by DocumentIR builder)
    section_path: List[str] = field(default_factory=list)  # ["Chapter 1", "Section 1.1", ...]
    section_id: Optional[str] = None  # e.g., "1.1.2"

    # For TOC-extracted headings
    toc_page_ref: Optional[int] = None  # Where TOC points to

    def __post_init__(self):
        self.block_type = BlockType.HEADING

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'level': self.level.value,
            'section_path': self.section_path,
            'section_id': self.section_id,
            'toc_page_ref': self.toc_page_ref
        })
        return d


@dataclass
class Paragraph(ContentBlock):
    """Regular paragraph text"""

    # Paragraph-specific metadata
    indentation_level: int = 0  # For nested content
    alignment: Optional[Literal['left', 'center', 'right', 'justify']] = None

    def __post_init__(self):
        self.block_type = BlockType.PARAGRAPH

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'indentation_level': self.indentation_level,
            'alignment': self.alignment
        })
        return d


@dataclass
class Table(ContentBlock):
    """
    Table representation - stores structured data, NOT markdown
    """
    # Structured table data
    headers: List[str] = field(default_factory=list)  # Column headers
    rows: List[List[str]] = field(default_factory=list)  # Each row is a list of cell values

    # Optional: cell-level metadata
    cell_metadata: Optional[List[List[Dict[str, Any]]]] = None

    # Table context
    caption: Optional[str] = None  # Table caption/title
    table_number: Optional[int] = None  # "Table 3"
    context_before: Optional[str] = None  # Text before table
    context_after: Optional[str] = None  # Text after table

    def __post_init__(self):
        self.block_type = BlockType.TABLE
        # Text is auto-generated summary
        if not self.text:
            self.text = f"Table with {len(self.rows)} rows and {len(self.headers)} columns"

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return len(self.headers)

    def to_markdown(self) -> str:
        """Render table as markdown (view, not storage)"""
        lines = []

        # Caption
        if self.caption:
            lines.append(f"**{self.caption}**\n")

        # Headers
        lines.append('| ' + ' | '.join(self.headers) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(self.headers)) + ' |')

        # Rows
        for row in self.rows:
            lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')

        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'headers': self.headers,
            'rows': self.rows,
            'cell_metadata': self.cell_metadata,
            'caption': self.caption,
            'table_number': self.table_number,
            'context_before': self.context_before,
            'context_after': self.context_after,
            'num_rows': self.num_rows,
            'num_cols': self.num_cols
        })
        return d


@dataclass
class Figure(ContentBlock):
    """
    Figure/image representation
    """
    # Figure identification
    figure_number: Optional[int] = None  # "Figure 5"
    caption: Optional[str] = None
    alt_text: Optional[str] = None

    # Image data
    image_path: Optional[str] = None  # Path to extracted image file
    image_format: Optional[str] = None  # png, jpg, svg, etc.
    image_data: Optional[bytes] = None  # Raw image bytes (optional)

    # Image properties
    width: Optional[float] = None
    height: Optional[float] = None

    def __post_init__(self):
        self.block_type = BlockType.FIGURE
        # Text is caption or alt text
        if not self.text:
            self.text = self.caption or self.alt_text or f"Figure {self.figure_number or ''}"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'figure_number': self.figure_number,
            'caption': self.caption,
            'alt_text': self.alt_text,
            'image_path': self.image_path,
            'image_format': self.image_format,
            'width': self.width,
            'height': self.height
        })
        # Don't include image_data in JSON (too large)
        return d


@dataclass
class CodeBlock(ContentBlock):
    """Code block"""
    language: Optional[str] = None  # python, java, etc.
    line_numbers: bool = False

    def __post_init__(self):
        self.block_type = BlockType.CODE_BLOCK

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'language': self.language,
            'line_numbers': self.line_numbers
        })
        return d


@dataclass
class ListBlock(ContentBlock):
    """
    List representation - stores structure, not markdown
    """
    list_type: ListType = ListType.UNORDERED
    items: List[str] = field(default_factory=list)  # List items as plain text

    # For nested lists
    nested_lists: Optional[List['ListBlock']] = None
    indentation_level: int = 0

    def __post_init__(self):
        self.block_type = BlockType.LIST
        # Text is concatenated items
        if not self.text:
            self.text = '\n'.join(self.items)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'list_type': self.list_type.value,
            'items': self.items,
            'indentation_level': self.indentation_level,
            'nested_lists': [nl.to_dict() for nl in self.nested_lists] if self.nested_lists else None
        })
        return d


# ============================================================================
# SECTION HIERARCHY
# ============================================================================

@dataclass
class Section:
    """
    Represents a document section with hierarchical structure.
    Built from Heading blocks during IR construction.
    """
    section_id: str  # e.g., "1", "1.1", "1.1.2"
    title: str  # Section title (from heading text)
    level: HeadingLevel  # Heading level

    # Location
    start_page: int
    end_page: int  # Inclusive

    # Hierarchy
    parent_id: Optional[str] = None
    children: List['Section'] = field(default_factory=list)

    # Content blocks in this section
    block_ids: List[str] = field(default_factory=list)

    # Heading block reference
    heading_block_id: str = None

    def get_path(self, sections_dict: Dict[str, 'Section']) -> List[str]:
        """Get hierarchical path from root to this section"""
        path = [self.title]
        current = self
        while current.parent_id:
            current = sections_dict[current.parent_id]
            path.insert(0, current.title)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return {
            'section_id': self.section_id,
            'title': self.title,
            'level': self.level.value,
            'start_page': self.start_page,
            'end_page': self.end_page,
            'parent_id': self.parent_id,
            'children': [c.section_id for c in self.children],
            'block_ids': self.block_ids,
            'heading_block_id': self.heading_block_id
        }


# ============================================================================
# DOCUMENT IR (ROOT)
# ============================================================================

@dataclass
class DocumentIR:
    """
    Canonical Document Intermediate Representation.
    This is the single source of truth for the entire document.
    """
    # Document identification
    doc_id: str
    source_file: str
    source_format: Literal['pdf', 'docx', 'ppt', 'html', 'txt']

    # Ordered content blocks (THE core representation)
    blocks: List[ContentBlock]

    # Section hierarchy (built from headings)
    sections: Dict[str, Section]  # section_id -> Section
    root_sections: List[str]  # Top-level section IDs

    # Page-to-block index (for fast page lookups)
    page_to_blocks: Dict[int, List[str]] = field(default_factory=dict)  # page_num -> block_ids

    # Document-level metadata
    total_pages: int = 0
    extraction_date: datetime = field(default_factory=datetime.now)
    extraction_pipeline: str = ""  # e.g., "marker-pdf", "pymupdf-enhanced"

    # Optional document properties
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None

    # Statistics (auto-computed)
    total_blocks: int = 0
    total_chars: int = 0
    total_words: int = 0

    def __post_init__(self):
        """Build indices after initialization"""
        self.total_blocks = len(self.blocks)
        self.total_chars = sum(len(b.text) for b in self.blocks)
        self.total_words = sum(len(b.text.split()) for b in self.blocks)

        # Build page index
        self._build_page_index()

    def _build_page_index(self):
        """Build page-to-blocks mapping"""
        self.page_to_blocks.clear()
        for block in self.blocks:
            page = block.location.page_number
            if page not in self.page_to_blocks:
                self.page_to_blocks[page] = []
            self.page_to_blocks[page].append(block.block_id)

    def get_blocks_on_page(self, page_number: int) -> List[ContentBlock]:
        """Get all blocks on a specific page"""
        block_ids = self.page_to_blocks.get(page_number, [])
        return [b for b in self.blocks if b.block_id in block_ids]

    def get_blocks_in_range(self, start_page: int, end_page: int) -> List[ContentBlock]:
        """Get all blocks in a page range (inclusive)"""
        result = []
        for page in range(start_page, end_page + 1):
            result.extend(self.get_blocks_on_page(page))
        return result

    def get_section(self, section_id: str) -> Optional[Section]:
        """Get section by ID"""
        return self.sections.get(section_id)

    def get_section_blocks(self, section_id: str) -> List[ContentBlock]:
        """Get all blocks in a section"""
        section = self.get_section(section_id)
        if not section:
            return []
        return [b for b in self.blocks if b.block_id in section.block_ids]

    def get_block_by_id(self, block_id: str) -> Optional[ContentBlock]:
        """Get block by ID"""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict"""
        return {
            'doc_id': self.doc_id,
            'source_file': self.source_file,
            'source_format': self.source_format,
            'blocks': [b.to_dict() for b in self.blocks],
            'sections': {sid: s.to_dict() for sid, s in self.sections.items()},
            'root_sections': self.root_sections,
            'page_to_blocks': self.page_to_blocks,
            'total_pages': self.total_pages,
            'extraction_date': self.extraction_date.isoformat(),
            'extraction_pipeline': self.extraction_pipeline,
            'title': self.title,
            'author': self.author,
            'creation_date': self.creation_date.isoformat() if self.creation_date else None,
            'total_blocks': self.total_blocks,
            'total_chars': self.total_chars,
            'total_words': self.total_words
        }

    def to_json(self, filepath: str, indent: int = 2):
        """Save IR to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """
        Render entire document as Markdown (VIEW, not storage).
        This demonstrates how markdown is just a rendering of the IR.
        """
        lines = []

        # Document metadata
        if self.title:
            lines.append(f"# {self.title}\n")
        if self.author:
            lines.append(f"**Author:** {self.author}\n")
        lines.append("")

        # Render blocks in order
        for block in self.blocks:
            if isinstance(block, Heading):
                prefix = '#' * block.level.value
                lines.append(f"{prefix} {block.text}\n")

            elif isinstance(block, Paragraph):
                lines.append(f"{block.text}\n")

            elif isinstance(block, Table):
                lines.append(block.to_markdown())
                lines.append("")

            elif isinstance(block, CodeBlock):
                lang = block.language or ""
                lines.append(f"```{lang}")
                lines.append(block.text)
                lines.append("```\n")

            elif isinstance(block, ListBlock):
                marker = '-' if block.list_type == ListType.UNORDERED else '1.'
                for item in block.items:
                    indent = '  ' * block.indentation_level
                    lines.append(f"{indent}{marker} {item}")
                lines.append("")

            elif isinstance(block, Figure):
                if block.caption:
                    lines.append(f"![{block.alt_text or 'Figure'}]({block.image_path})")
                    lines.append(f"*{block.caption}*\n")
                else:
                    lines.append(f"![{block.alt_text or 'Figure'}]({block.image_path})\n")

            else:
                # Default: just text
                lines.append(f"{block.text}\n")

        return '\n'.join(lines)


# ============================================================================
# IR BUILDER UTILITIES
# ============================================================================

class DocumentIRBuilder:
    """
    Helper class to build DocumentIR from extraction outputs.
    Handles section hierarchy construction from headings.
    """

    def __init__(self, doc_id: str, source_file: str, source_format: str):
        self.doc_id = doc_id
        self.source_file = source_file
        self.source_format = source_format
        self.blocks: List[ContentBlock] = []
        self.sections: Dict[str, Section] = {}
        self.section_stack: List[Section] = []  # For building hierarchy

    def add_block(self, block: ContentBlock):
        """Add a block and update section hierarchy if it's a heading"""
        self.blocks.append(block)

        if isinstance(block, Heading):
            self._process_heading(block)

    def _process_heading(self, heading: Heading):
        """Process heading to build section hierarchy"""
        # Create section
        section = Section(
            section_id=f"{len(self.sections) + 1}",
            title=heading.text,
            level=heading.level,
            start_page=heading.location.page_number,
            end_page=heading.location.page_number,  # Will update later
            heading_block_id=heading.block_id
        )

        # Find parent based on heading level
        while self.section_stack and self.section_stack[-1].level.value >= heading.level.value:
            self.section_stack.pop()

        if self.section_stack:
            parent = self.section_stack[-1]
            section.parent_id = parent.section_id
            parent.children.append(section)

        self.section_stack.append(section)
        self.sections[section.section_id] = section

        # Update heading with section info
        heading.section_id = section.section_id
        heading.section_path = section.get_path(self.sections)

    def build(self) -> DocumentIR:
        """Build final DocumentIR"""
        # Find root sections
        root_sections = [s.section_id for s in self.sections.values() if s.parent_id is None]

        # Update section end pages and assign blocks
        self._finalize_sections()

        # Determine total pages
        total_pages = max((b.location.page_number for b in self.blocks), default=0)

        return DocumentIR(
            doc_id=self.doc_id,
            source_file=self.source_file,
            source_format=self.source_format,
            blocks=self.blocks,
            sections=self.sections,
            root_sections=root_sections,
            total_pages=total_pages
        )

    def _finalize_sections(self):
        """Assign blocks to sections and update end pages"""
        current_section = None

        for block in self.blocks:
            # Update current section
            if isinstance(block, Heading) and block.section_id:
                current_section = self.sections.get(block.section_id)

            # Add block to current section
            if current_section:
                current_section.block_ids.append(block.block_id)
                current_section.end_page = max(current_section.end_page, block.location.page_number)
