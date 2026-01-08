"""
Mappers from existing extraction outputs (PDF/DOCX/PPT) to DocumentIR.

This module shows how to convert the current markdown-based extraction
outputs into the canonical DocumentIR structure.
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from document_ir import (
    DocumentIR, DocumentIRBuilder, ContentBlock, Heading, Paragraph,
    Table, Figure, CodeBlock, ListBlock, Location, BoundingBox,
    HeadingLevel, BlockType, ListType
)


# ============================================================================
# PDF EXTRACTION OUTPUT -> DocumentIR
# ============================================================================

def map_pdf_extraction_to_ir(
    pdf_file: str,
    extraction_output: List[Dict[str, Any]],
    extraction_method: str = "enhanced"
) -> DocumentIR:
    """
    Map PDF extraction output (from pdf2llm.py) to DocumentIR.

    Current PDF extraction returns list of content blocks:
    {
        'type': 'heading|table|figure|text',
        'content': str,
        'page': int,
        'y_position': float,
        'metadata': {
            'pdf_file': filename,
            'page': page_num,
            'block_type': type,
            'font_size': float,
            'heading_level': int,
            'heading_source': 'toc|font_size|pattern',
            'section': section_name,
            'section_hierarchy': list,
            'bbox': [x0, y0, x1, y1]
        }
    }
    """
    doc_id = str(uuid.uuid4())
    builder = DocumentIRBuilder(doc_id, pdf_file, 'pdf')

    for idx, block_data in enumerate(extraction_output):
        block_type = block_data.get('type', 'text')
        content = block_data.get('content', '')
        page = block_data.get('page', 1)
        metadata = block_data.get('metadata', {})

        # Create location
        bbox = None
        if 'bbox' in metadata and metadata['bbox']:
            bbox_data = metadata['bbox']
            if isinstance(bbox_data, list) and len(bbox_data) == 4:
                bbox = BoundingBox(
                    x0=bbox_data[0],
                    y0=bbox_data[1],
                    x1=bbox_data[2],
                    y1=bbox_data[3],
                    page_number=page
                )

        location = Location(
            page_number=page,
            page_offset=idx,  # Approximate
            bbox=bbox
        )

        # Generate block ID
        block_id = f"{doc_id}::{page}::{idx}"

        # Map to appropriate block type
        if block_type == 'heading':
            heading_level = metadata.get('heading_level', 1)
            heading = Heading(
                block_id=block_id,
                block_type=BlockType.HEADING,
                location=location,
                text=content,
                level=HeadingLevel(heading_level),
                extraction_method=metadata.get('heading_source', 'unknown'),
                font_size=metadata.get('font_size'),
                raw_metadata=metadata
            )
            builder.add_block(heading)

        elif block_type == 'table':
            # Parse table from markdown or use structured data
            table_block = _parse_table_from_pdf_output(
                block_id, location, content, metadata
            )
            builder.add_block(table_block)

        elif block_type == 'figure':
            figure = Figure(
                block_id=block_id,
                block_type=BlockType.FIGURE,
                location=location,
                text=content,
                caption=content,
                image_path=metadata.get('image_path'),
                figure_number=metadata.get('figure_number'),
                raw_metadata=metadata
            )
            builder.add_block(figure)

        elif block_type == 'text':
            # Check if it's a list, code block, or paragraph
            if _is_list(content):
                list_block = _parse_list_from_text(block_id, location, content, metadata)
                builder.add_block(list_block)
            elif _is_code_block(content):
                code_block = CodeBlock(
                    block_id=block_id,
                    block_type=BlockType.CODE_BLOCK,
                    location=location,
                    text=content,
                    language=_detect_code_language(content),
                    raw_metadata=metadata
                )
                builder.add_block(code_block)
            else:
                paragraph = Paragraph(
                    block_id=block_id,
                    block_type=BlockType.PARAGRAPH,
                    location=location,
                    text=content,
                    font_size=metadata.get('font_size'),
                    raw_metadata=metadata
                )
                builder.add_block(paragraph)

    return builder.build()


def _parse_table_from_pdf_output(
    block_id: str,
    location: Location,
    content: str,
    metadata: Dict[str, Any]
) -> Table:
    """Parse table from markdown content or structured data"""
    # If structured data is available, use it
    if 'table_data' in metadata:
        table_data = metadata['table_data']
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
    else:
        # Parse from markdown
        headers, rows = _parse_markdown_table(content)

    return Table(
        block_id=block_id,
        block_type=BlockType.TABLE,
        location=location,
        text=f"Table with {len(rows)} rows",
        headers=headers,
        rows=rows,
        caption=metadata.get('smart_name') or metadata.get('caption'),
        table_number=metadata.get('table_num'),
        context_before=metadata.get('context_before'),
        context_after=metadata.get('context_after'),
        raw_metadata=metadata
    )


def _parse_markdown_table(markdown: str) -> Tuple[List[str], List[List[str]]]:
    """Parse markdown table into headers and rows"""
    lines = [line.strip() for line in markdown.split('\n') if line.strip()]

    # Remove separator line
    lines = [l for l in lines if not re.match(r'^\|[\s\-:|]+\|$', l)]

    if not lines:
        return [], []

    # Parse header
    header_line = lines[0].strip('|')
    headers = [cell.strip() for cell in header_line.split('|')]

    # Parse rows
    rows = []
    for line in lines[1:]:
        row_line = line.strip('|')
        row = [cell.strip() for cell in row_line.split('|')]
        rows.append(row)

    return headers, rows


# ============================================================================
# DOCX EXTRACTION OUTPUT -> DocumentIR
# ============================================================================

def map_docx_extraction_to_ir(
    docx_file: str,
    extraction_output: List[Dict[str, Any]],
    doc_metadata: Optional[Dict[str, Any]] = None
) -> DocumentIR:
    """
    Map DOCX extraction output (from docx2llm.py) to DocumentIR.

    Current DOCX extraction returns structured content blocks with:
    {
        'type': 'heading|paragraph|table|list',
        'content': str,
        'style': str,  # DOCX style name
        'level': int,  # For headings
        'page': int,  # Estimated
        'metadata': {...}
    }
    """
    doc_id = str(uuid.uuid4())
    builder = DocumentIRBuilder(doc_id, docx_file, 'docx')

    for idx, block_data in enumerate(extraction_output):
        block_type = block_data.get('type', 'paragraph')
        content = block_data.get('content', '')
        page = block_data.get('page', 1)  # DOCX page estimation
        metadata = block_data.get('metadata', {})

        location = Location(
            page_number=page,
            page_offset=idx,
            section_index=metadata.get('section_index')
        )

        block_id = f"{doc_id}::{page}::{idx}"

        if block_type == 'heading':
            level = block_data.get('level', 1)
            heading = Heading(
                block_id=block_id,
                block_type=BlockType.HEADING,
                location=location,
                text=content,
                level=HeadingLevel(level),
                extraction_method='docx_style',
                raw_metadata=metadata
            )
            builder.add_block(heading)

        elif block_type == 'table':
            # DOCX tables are typically well-structured
            table_data = metadata.get('table_data', {})
            headers = table_data.get('headers', [])
            rows = table_data.get('rows', [])

            table = Table(
                block_id=block_id,
                block_type=BlockType.TABLE,
                location=location,
                text=content[:100],
                headers=headers,
                rows=rows,
                caption=metadata.get('caption'),
                raw_metadata=metadata
            )
            builder.add_block(table)

        elif block_type == 'list':
            list_items = block_data.get('items', content.split('\n'))
            list_type_str = metadata.get('list_type', 'unordered')
            list_type = ListType.ORDERED if 'ordered' in list_type_str else ListType.UNORDERED

            list_block = ListBlock(
                block_id=block_id,
                block_type=BlockType.LIST,
                location=location,
                text=content,
                list_type=list_type,
                items=list_items,
                indentation_level=metadata.get('indent_level', 0),
                raw_metadata=metadata
            )
            builder.add_block(list_block)

        else:  # paragraph
            paragraph = Paragraph(
                block_id=block_id,
                block_type=BlockType.PARAGRAPH,
                location=location,
                text=content,
                indentation_level=metadata.get('indent_level', 0),
                alignment=metadata.get('alignment'),
                raw_metadata=metadata
            )
            builder.add_block(paragraph)

    # Set document metadata if available
    ir = builder.build()
    if doc_metadata:
        ir.title = doc_metadata.get('title')
        ir.author = doc_metadata.get('author')
        ir.creation_date = doc_metadata.get('creation_date')

    return ir


# ============================================================================
# PPT EXTRACTION OUTPUT -> DocumentIR
# ============================================================================

def map_ppt_extraction_to_ir(
    ppt_file: str,
    extraction_output: List[Dict[str, Any]]
) -> DocumentIR:
    """
    Map PowerPoint extraction output (from ppt2llm.py) to DocumentIR.

    PPT extraction returns slide-based structure:
    {
        'slide_number': int,
        'title': str,
        'content': str,
        'notes': str,
        'tables': [...],
        'images': [...],
        'metadata': {...}
    }
    """
    doc_id = str(uuid.uuid4())
    builder = DocumentIRBuilder(doc_id, ppt_file, 'ppt')

    for slide_data in extraction_output:
        slide_num = slide_data.get('slide_number', 1)
        title = slide_data.get('title', '')
        content = slide_data.get('content', '')
        notes = slide_data.get('notes', '')
        metadata = slide_data.get('metadata', {})

        # Treat each slide as a page
        location = Location(
            page_number=slide_num,
            page_offset=0,
            slide_number=slide_num
        )

        # Add slide title as heading
        if title:
            block_id = f"{doc_id}::{slide_num}::title"
            heading = Heading(
                block_id=block_id,
                block_type=BlockType.HEADING,
                location=location,
                text=title,
                level=HeadingLevel.H1,
                extraction_method='ppt_slide_title',
                raw_metadata=metadata
            )
            builder.add_block(heading)

        # Add slide content (often bullet lists)
        if content:
            content_lines = content.split('\n')
            for idx, line in enumerate(content_lines):
                line = line.strip()
                if not line:
                    continue

                block_id = f"{doc_id}::{slide_num}::content::{idx}"
                location = Location(
                    page_number=slide_num,
                    page_offset=idx + 1,
                    slide_number=slide_num
                )

                if line.startswith(('- ', '* ', '+ ')) or re.match(r'^\d+\.', line):
                    # List item
                    list_items = [line.lstrip('- *+0123456789. ')]
                    list_type = ListType.ORDERED if re.match(r'^\d+\.', line) else ListType.UNORDERED

                    list_block = ListBlock(
                        block_id=block_id,
                        block_type=BlockType.LIST,
                        location=location,
                        text=line,
                        list_type=list_type,
                        items=list_items,
                        raw_metadata=metadata
                    )
                    builder.add_block(list_block)
                else:
                    # Regular paragraph
                    paragraph = Paragraph(
                        block_id=block_id,
                        block_type=BlockType.PARAGRAPH,
                        location=location,
                        text=line,
                        raw_metadata=metadata
                    )
                    builder.add_block(paragraph)

        # Add tables
        for table_idx, table_data in enumerate(slide_data.get('tables', [])):
            block_id = f"{doc_id}::{slide_num}::table::{table_idx}"
            location = Location(
                page_number=slide_num,
                page_offset=100 + table_idx,
                slide_number=slide_num
            )

            table = Table(
                block_id=block_id,
                block_type=BlockType.TABLE,
                location=location,
                text=f"Table on slide {slide_num}",
                headers=table_data.get('headers', []),
                rows=table_data.get('rows', []),
                table_number=table_idx + 1,
                raw_metadata=metadata
            )
            builder.add_block(table)

        # Add images/figures
        for img_idx, img_data in enumerate(slide_data.get('images', [])):
            block_id = f"{doc_id}::{slide_num}::image::{img_idx}"
            location = Location(
                page_number=slide_num,
                page_offset=200 + img_idx,
                slide_number=slide_num
            )

            figure = Figure(
                block_id=block_id,
                block_type=BlockType.FIGURE,
                location=location,
                text=img_data.get('alt_text', f"Image {img_idx + 1}"),
                caption=img_data.get('caption'),
                image_path=img_data.get('path'),
                figure_number=img_idx + 1,
                raw_metadata=metadata
            )
            builder.add_block(figure)

        # Add speaker notes as separate paragraphs
        if notes:
            block_id = f"{doc_id}::{slide_num}::notes"
            location = Location(
                page_number=slide_num,
                page_offset=300,
                slide_number=slide_num
            )

            notes_para = Paragraph(
                block_id=block_id,
                block_type=BlockType.PARAGRAPH,
                location=location,
                text=notes,
                raw_metadata={'type': 'speaker_notes', **metadata}
            )
            builder.add_block(notes_para)

    return builder.build()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _is_list(text: str) -> bool:
    """Check if text is a list"""
    lines = text.split('\n')
    list_patterns = [
        r'^\s*[-*+]\s+',  # Unordered
        r'^\s*\d+\.\s+',  # Ordered
        r'^\s*[a-z]\.\s+',  # Alphabetic
    ]
    list_lines = sum(1 for line in lines if any(re.match(p, line) for p in list_patterns))
    return list_lines / max(len(lines), 1) > 0.5


def _is_code_block(text: str) -> bool:
    """Check if text is a code block"""
    # Heuristics: high indentation, special characters, keywords
    lines = text.split('\n')
    if len(lines) < 2:
        return False

    indented = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
    if indented / len(lines) > 0.7:
        return True

    # Check for code patterns
    code_patterns = [
        r'def\s+\w+\(',
        r'class\s+\w+',
        r'import\s+\w+',
        r'function\s+\w+',
        r'var\s+\w+\s*=',
        r'const\s+\w+\s*=',
    ]
    return any(re.search(p, text) for p in code_patterns)


def _detect_code_language(text: str) -> Optional[str]:
    """Detect code language from content"""
    patterns = {
        'python': [r'def\s+\w+\(', r'import\s+\w+', r'from\s+\w+\s+import'],
        'java': [r'public\s+class', r'private\s+void', r'System\.out\.println'],
        'javascript': [r'function\s+\w+', r'const\s+\w+\s*=', r'console\.log'],
        'c': [r'#include\s*<', r'int\s+main\(', r'printf\('],
        'sql': [r'SELECT\s+.*\s+FROM', r'INSERT\s+INTO', r'CREATE\s+TABLE'],
    }

    for lang, patterns_list in patterns.items():
        if any(re.search(p, text, re.IGNORECASE) for p in patterns_list):
            return lang

    return None


def _parse_list_from_text(
    block_id: str,
    location: Location,
    text: str,
    metadata: Dict[str, Any]
) -> ListBlock:
    """Parse list from text"""
    lines = text.split('\n')
    items = []
    list_type = ListType.UNORDERED

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect list type
        if re.match(r'^\d+\.', line):
            list_type = ListType.ORDERED
            item = re.sub(r'^\d+\.\s*', '', line)
        elif re.match(r'^[-*+]\s+', line):
            item = re.sub(r'^[-*+]\s*', '', line)
        else:
            item = line

        items.append(item)

    return ListBlock(
        block_id=block_id,
        block_type=BlockType.LIST,
        location=location,
        text=text,
        list_type=list_type,
        items=items,
        raw_metadata=metadata
    )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def convert_to_ir(
    source_file: str,
    extraction_output: Any,
    source_format: str,
    **kwargs
) -> DocumentIR:
    """
    Convenience function to convert any extraction output to DocumentIR.

    Args:
        source_file: Path to source document
        extraction_output: Output from pdf2llm/docx2llm/ppt2llm
        source_format: 'pdf', 'docx', or 'ppt'
        **kwargs: Additional format-specific arguments

    Returns:
        DocumentIR instance
    """
    if source_format == 'pdf':
        return map_pdf_extraction_to_ir(
            source_file,
            extraction_output,
            kwargs.get('extraction_method', 'enhanced')
        )
    elif source_format == 'docx':
        return map_docx_extraction_to_ir(
            source_file,
            extraction_output,
            kwargs.get('doc_metadata')
        )
    elif source_format == 'ppt':
        return map_ppt_extraction_to_ir(source_file, extraction_output)
    else:
        raise ValueError(f"Unsupported format: {source_format}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Convert PDF extraction to IR
    pdf_extraction_output = [
        {
            'type': 'heading',
            'content': 'Introduction',
            'page': 1,
            'metadata': {
                'heading_level': 1,
                'heading_source': 'font_size',
                'font_size': 18.0,
                'bbox': [72, 100, 500, 120]
            }
        },
        {
            'type': 'text',
            'content': 'This is the first paragraph of the document.',
            'page': 1,
            'metadata': {
                'font_size': 12.0,
                'bbox': [72, 130, 500, 150]
            }
        },
        {
            'type': 'table',
            'content': '| Header 1 | Header 2 |\n|----------|----------|\n| Cell 1 | Cell 2 |',
            'page': 2,
            'metadata': {
                'table_num': 1,
                'smart_name': 'Performance Metrics',
                'bbox': [72, 200, 500, 350]
            }
        }
    ]

    # Convert to IR
    ir = map_pdf_extraction_to_ir(
        'example.pdf',
        pdf_extraction_output,
        extraction_method='enhanced'
    )

    # Access structured data
    print(f"Total blocks: {ir.total_blocks}")
    print(f"Total pages: {ir.total_pages}")
    print(f"Sections: {len(ir.sections)}")

    # Get blocks on page 1
    page1_blocks = ir.get_blocks_on_page(1)
    print(f"Blocks on page 1: {len(page1_blocks)}")

    # Render as markdown (view)
    markdown = ir.to_markdown()
    print("\nMarkdown view:")
    print(markdown)

    # Save as JSON (structured storage)
    ir.to_json('example_ir.json')
