"""
Markdown as Presentation Layer

CORE PRINCIPLE: Markdown is WRITE-ONLY at pipeline edges.
Never parse markdown during processing.

Pipeline Flow:
  Extract → DocumentIR → Chunk → Embed → Store
                ↓                           ↓
            to_markdown()              to_markdown()
                ↓                           ↓
         (for export)              (for UI display)
"""

from dataclasses import dataclass
from typing import List, Optional
from document_ir import DocumentIR, ContentBlock, Heading, Paragraph, Table, CodeBlock


# ============================================================================
# MARKDOWN RESPONSIBILITIES (PROHIBITED)
# ============================================================================

"""
Markdown should NOT:

1. ✗ Store Structure
   - Don't store section hierarchy as "# Heading"
   - Use: DocumentIR.sections with explicit parent/child

2. ✗ Define Boundaries
   - Don't use "# Heading" to detect chunk boundaries
   - Use: isinstance(block, Heading) and block.level

3. ✗ Represent Tables
   - Don't store tables as "| col | col |" strings
   - Use: Table(headers=[...], rows=[[...]])

4. ✗ Track Pages
   - Don't use "<<<PAGE_5>>>" markers
   - Use: block.location.page_number

5. ✗ Preserve Metadata
   - Don't store "<!-- bbox: ... -->" comments
   - Use: block.location.bbox

6. ✗ Be Parsed
   - Don't regex markdown during chunking/normalization
   - Use: Direct object operations on IR

7. ✗ Carry Citations
   - Don't embed "[Source: page 5]" in text
   - Use: chunk.physical_location.page_start
"""


# ============================================================================
# WRONG: Markdown-Based Processing
# ============================================================================

def chunking_wrong_markdown_based(markdown_text: str) -> List[dict]:
    """
    ✗ WRONG: Operates on markdown string
    Problems:
    - Fragile regex parsing
    - Heading detection fails with formatting variations
    - Table boundaries unclear
    - Page numbers reconstructed from markers
    """
    import re

    chunks = []
    current_chunk = []

    lines = markdown_text.split('\n')

    for line in lines:
        # ✗ FRAGILE: Regex to detect headings
        if re.match(r'^#{1,2}\s+', line):
            # ✗ FRAGILE: Assumes heading format
            if current_chunk:
                chunks.append({
                    'text': '\n'.join(current_chunk),
                    'page': extract_page_from_marker(current_chunk)  # ✗ FRAGILE
                })
                current_chunk = []

        # ✗ FRAGILE: Detect tables by markdown syntax
        if line.startswith('|'):
            # ✗ PROBLEM: Can't tell if this is in a code block
            pass

        current_chunk.append(line)

    return chunks


def extract_page_from_marker(lines: List[str]) -> int:
    """✗ FRAGILE: Extract page from <<<PAGE_N>>> marker"""
    import re
    for line in lines:
        match = re.search(r'<<<PAGE_(\d+)>>>', line)
        if match:
            return int(match.group(1))
    return 1  # ✗ WRONG: Default fallback


# ============================================================================
# CORRECT: IR-Based Processing
# ============================================================================

def chunking_correct_ir_based(ir: DocumentIR, max_size: int = 1000) -> List[dict]:
    """
    ✓ CORRECT: Operates on structured IR
    Benefits:
    - Type-safe operations
    - Exact block boundaries
    - Precise page numbers
    - No parsing ambiguity
    """
    chunks = []
    current_blocks = []
    current_size = 0

    for block in ir.blocks:
        block_size = len(block.text)

        # ✓ EXACT: Check block type without regex
        if isinstance(block, Heading) and block.level.value <= 2:
            if current_blocks:
                chunks.append(create_chunk(current_blocks, ir))
                current_blocks = []
                current_size = 0

        # ✓ EXACT: Table handling
        if isinstance(block, Table):
            # Keep tables intact, no parsing needed
            if current_size + block_size > max_size and current_blocks:
                chunks.append(create_chunk(current_blocks, ir))
                current_blocks = []
                current_size = 0

        # Size check
        if current_size + block_size > max_size and current_blocks:
            chunks.append(create_chunk(current_blocks, ir))
            current_blocks = []
            current_size = 0

        current_blocks.append(block)
        current_size += block_size

    if current_blocks:
        chunks.append(create_chunk(current_blocks, ir))

    return chunks


def create_chunk(blocks: List[ContentBlock], ir: DocumentIR) -> dict:
    """Create chunk from IR blocks"""
    # ✓ EXACT: Page numbers from blocks
    page_start = min(b.location.page_number for b in blocks)
    page_end = max(b.location.page_number for b in blocks)

    # ✓ EXACT: Section from first block's heading
    section = None
    for block in blocks:
        if isinstance(block, Heading) and block.section_id:
            section = ir.sections[block.section_id]
            break

    # ✓ Generate markdown ONLY for output
    markdown_text = generate_chunk_markdown(blocks)

    return {
        'blocks': blocks,
        'block_ids': [b.block_id for b in blocks],
        'page_start': page_start,  # From blocks, not markers
        'page_end': page_end,
        'section_path': section.section_path if section else [],
        'text': markdown_text,  # View only
        'contains_tables': any(isinstance(b, Table) for b in blocks)
    }


def generate_chunk_markdown(blocks: List[ContentBlock]) -> str:
    """
    ✓ Generate markdown for OUTPUT only.
    This is WRITE-ONLY - never parsed back.
    """
    parts = []

    for block in blocks:
        if isinstance(block, Heading):
            prefix = '#' * block.level.value
            parts.append(f"{prefix} {block.text}")

        elif isinstance(block, Paragraph):
            parts.append(block.text)

        elif isinstance(block, Table):
            parts.append(block.to_markdown())  # Generate from structure

        elif isinstance(block, CodeBlock):
            parts.append(f"```{block.language or ''}\n{block.text}\n```")

        else:
            parts.append(block.text)

    return '\n\n'.join(parts)


# ============================================================================
# NORMALIZATION: IR-Based vs Markdown-Based
# ============================================================================

def normalize_wrong_markdown(markdown: str) -> str:
    """
    ✗ WRONG: Normalize markdown text
    Problems:
    - May break markdown syntax
    - Can't normalize table cells independently
    - Loses structure during normalization
    """
    import re

    # ✗ PROBLEM: Can accidentally break markdown
    markdown = re.sub(r'\s+', ' ', markdown)  # Breaks code blocks!
    markdown = re.sub(r'#+\s*', '# ', markdown)  # Breaks heading levels!

    return markdown


def normalize_correct_ir(ir: DocumentIR) -> DocumentIR:
    """
    ✓ CORRECT: Normalize structured blocks
    Benefits:
    - Type-aware normalization
    - Preserve structure
    - Precise operations
    """
    for block in ir.blocks:
        # Normalize text content
        block.text = normalize_text(block.text)

        # ✓ Type-specific normalization
        if isinstance(block, Table):
            # Normalize table cells independently
            for i, row in enumerate(block.rows):
                block.rows[i] = [normalize_text(cell) for cell in row]

        elif isinstance(block, CodeBlock):
            # DON'T normalize code blocks
            pass

    return ir


def normalize_text(text: str) -> str:
    """Normalize plain text (not markdown)"""
    import re
    text = re.sub(r'\s+', ' ', text)  # Safe on plain text
    text = text.strip()
    return text


# ============================================================================
# PRESENTATION LAYER: Where Markdown IS Used
# ============================================================================

def export_to_markdown_file(ir: DocumentIR, output_path: str):
    """
    ✓ CORRECT: Generate markdown for export
    This is PRESENTATION, not storage.
    """
    markdown = ir.to_markdown()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)


def render_chunk_for_ui(chunk: dict) -> str:
    """
    ✓ CORRECT: Generate markdown for UI display
    This is VIEW, not storage.
    """
    return chunk['text']  # Already generated from IR


def api_response_with_markdown(chunk: dict) -> dict:
    """
    ✓ CORRECT: Include markdown in API response
    Clients receive markdown for display, but citation uses structured data.
    """
    return {
        'chunk_id': chunk['block_ids'][0],

        # Structured citation data
        'citation': {
            'page_start': chunk['page_start'],
            'page_end': chunk['page_end'],
            'section_path': chunk['section_path']
        },

        # Markdown for display
        'content': {
            'markdown': chunk['text'],  # View only
            'format': 'markdown'
        }
    }


# ============================================================================
# COMPARISON: Text-Based vs IR-Based Chunking
# ============================================================================

"""
TEXT-BASED CHUNKING (Markdown):
================================

Input: String
    "# Chapter 1\n\nParagraph text...\n\n| Table |"

Process:
    1. Split by newlines
    2. Regex match "^#" for headings
    3. Regex match "^|" for tables
    4. Count characters for size
    5. Extract <<<PAGE_N>>> markers for pages

Problems:
    - Heading detection fails: "# Chapter" vs "#Chapter"
    - Table detection fails in code blocks
    - Page markers can be lost/misplaced
    - Can't distinguish heading levels without parsing
    - Metadata detached from content

Chunking Logic:
    if re.match(r'^#{1,2}\s', line):  # Fragile!
        start_new_chunk()


IR-BASED CHUNKING:
==================

Input: DocumentIR
    blocks=[
        Heading(level=1, text="Chapter 1", location=Location(page=5)),
        Paragraph(text="Paragraph text...", location=Location(page=5)),
        Table(headers=[...], rows=[[...]], location=Location(page=6))
    ]

Process:
    1. Iterate blocks (typed objects)
    2. Check isinstance(block, Heading)
    3. Access block.level directly
    4. Access block.location.page_number
    5. Size = len(block.text)

Benefits:
    - Type-safe: isinstance(block, Heading)
    - Exact levels: block.level.value
    - Precise pages: block.location.page_number
    - Metadata attached: block.location.bbox
    - No parsing ambiguity

Chunking Logic:
    if isinstance(block, Heading) and block.level <= HeadingLevel.H2:
        start_new_chunk()  # Always correct!
"""


# ============================================================================
# EXAMPLE: Complete Pipeline Without Markdown Parsing
# ============================================================================

def complete_pipeline_ir_based(pdf_file: str):
    """
    Complete pipeline operating on IR, markdown only at edges.
    """
    from document_ir_mappers import map_pdf_extraction_to_ir

    # 1. Extract (produces structured output)
    from pdf2llm import extract_pdf_enhanced
    raw_output = extract_pdf_enhanced(pdf_file)

    # 2. Convert to IR (NO MARKDOWN)
    ir = map_pdf_extraction_to_ir(pdf_file, raw_output)
    # ir.blocks = [ContentBlock objects]

    # 3. Normalize (operates on IR blocks)
    normalized_ir = normalize_correct_ir(ir)

    # 4. Chunk (operates on IR blocks)
    chunks = chunking_correct_ir_based(normalized_ir, max_size=1000)

    # 5. Embed (text already in chunks, generated from IR)
    embeddings = []
    for chunk in chunks:
        embedding = embed_text(chunk['text'])  # Markdown for embedding input
        embeddings.append({
            'vector': embedding,
            'metadata': {
                'chunk_id': chunk['block_ids'][0],
                'page_start': chunk['page_start'],  # From IR, not parsed
                'page_end': chunk['page_end'],
                'section_path': chunk['section_path']
            }
        })

    # 6. Store (structured metadata)
    store_in_qdrant(embeddings)

    # 7. Export (ONLY NOW generate markdown for files)
    export_to_markdown_file(ir, 'output.md')

    return {
        'doc_id': ir.doc_id,
        'chunks': len(chunks),
        'embeddings': len(embeddings)
    }


def embed_text(text: str) -> List[float]:
    """Dummy embedding function"""
    return [0.1] * 1536


def store_in_qdrant(embeddings: list):
    """Dummy storage function"""
    pass


# ============================================================================
# KEY TRANSFORMATIONS
# ============================================================================

"""
1. HEADING DETECTION
====================
Before (Markdown):
    if re.match(r'^#{1,2}\s+', line):
        level = len(line) - len(line.lstrip('#'))

After (IR):
    if isinstance(block, Heading) and block.level <= HeadingLevel.H2:
        level = block.level.value


2. TABLE HANDLING
=================
Before (Markdown):
    if line.startswith('|'):
        table_lines.append(line)
        headers = parse_markdown_table_header(table_lines)

After (IR):
    if isinstance(block, Table):
        headers = block.headers  # Already structured
        rows = block.rows


3. PAGE TRACKING
================
Before (Markdown):
    page_match = re.search(r'<<<PAGE_(\d+)>>>', text)
    page = int(page_match.group(1)) if page_match else 1

After (IR):
    page = block.location.page_number  # Always accurate


4. CHUNK SIZE
=============
Before (Markdown):
    size = len(markdown_text)  # Includes formatting

After (IR):
    size = sum(len(b.text) for b in blocks)  # Content only


5. SECTION HIERARCHY
====================
Before (Markdown):
    # Reconstruct from heading levels
    if line.startswith('# '):
        section_stack = [line]
    elif line.startswith('## '):
        section_stack.append(line)

After (IR):
    section_path = block.section_path  # Pre-built during IR construction
"""

if __name__ == "__main__":
    print("Markdown as Presentation Layer")
    print("=" * 70)
    print("\n✗ Markdown should NOT:")
    print("  - Store structure")
    print("  - Define boundaries")
    print("  - Be parsed during processing")
    print("  - Carry metadata")
    print("\n✓ Markdown should:")
    print("  - Be generated at edges (export/UI)")
    print("  - Serve as view/presentation only")
    print("  - Come from IR, never create IR")
