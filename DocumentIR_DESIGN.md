# DocumentIR: Canonical Document Intermediate Representation

## Table of Contents
1. [Overview](#overview)
2. [The Problem with Markdown](#the-problem-with-markdown)
3. [DocumentIR Design](#documentir-design)
4. [How IR Removes Ambiguity](#how-ir-removes-ambiguity)
5. [Integration with Existing Pipeline](#integration-with-existing-pipeline)
6. [Migration Strategy](#migration-strategy)

---

## Overview

**DocumentIR** is a canonical, structured intermediate representation (IR) for documents in the KnowVev ingestion pipeline. It serves as the **single source of truth** between extraction and downstream processing (chunking, embedding, retrieval).

### Key Principles

1. **Format-Agnostic**: Works uniformly for PDF, DOCX, PPT, HTML
2. **Structured First**: Uses typed dataclasses, not text-based formats
3. **Location-Aware**: Preserves precise page numbers and bounding boxes
4. **Hierarchical**: Maintains section structure for context
5. **Renderable**: Can generate Markdown/HTML as views, not storage

---

## The Problem with Markdown

### Current Pipeline Issue

```
PDF/DOCX/PPT → Markdown String → Parse Markdown → Chunk → Embed
```

**Problems:**

1. **Lossy Conversion**: Bounding boxes, page numbers, and metadata are stored separately from content
2. **Ambiguous Parsing**: Markdown parsing during chunking is fragile
   - `# Heading` vs `#Heading` (missing space)
   - Table formats vary between extractors
   - List detection relies on regex patterns
3. **Metadata Detachment**: Page numbers and sections are "bolted on" after chunking
4. **Citation Reconstruction**: Must reverse-engineer page/section from chunk text

### Example Ambiguity

**Markdown representation:**
```markdown
# Chapter 1: Introduction

This is a paragraph.

| Name | Age |
|------|-----|
| Alice | 30 |

Another paragraph.
```

**Questions that can't be answered without external metadata:**
- Which page is "Chapter 1" on?
- What's the bounding box of the table?
- What's the heading hierarchy? (Is this H1 or H2?)
- If we chunk at the table, what's the section context?

**Current Solution:** Store metadata separately in dictionaries and hope the mapping doesn't break.

---

## DocumentIR Design

### Core Structure

```python
DocumentIR
├── doc_id: str
├── source_file: str
├── source_format: pdf|docx|ppt
├── blocks: List[ContentBlock]  # Ordered content
│   ├── Heading(level, section_path, bbox)
│   ├── Paragraph(text, bbox)
│   ├── Table(headers, rows, bbox)
│   ├── Figure(caption, image_path, bbox)
│   ├── CodeBlock(language, text, bbox)
│   └── ListBlock(items, list_type, bbox)
├── sections: Dict[section_id, Section]  # Hierarchy
│   └── Section(title, level, parent, children, block_ids)
└── page_to_blocks: Dict[page_num, block_ids]  # Fast lookup
```

### Example DocumentIR Instance

```python
ir = DocumentIR(
    doc_id="abc-123",
    source_file="report.pdf",
    source_format="pdf",
    blocks=[
        Heading(
            block_id="abc-123::1::0",
            text="Chapter 1: Introduction",
            level=HeadingLevel.H1,
            location=Location(page_number=1, page_offset=0,
                            bbox=BoundingBox(72, 100, 500, 120, page=1)),
            section_id="1",
            section_path=["Chapter 1: Introduction"]
        ),
        Paragraph(
            block_id="abc-123::1::1",
            text="This is a paragraph.",
            location=Location(page_number=1, page_offset=50,
                            bbox=BoundingBox(72, 130, 500, 150, page=1))
        ),
        Table(
            block_id="abc-123::1::2",
            headers=["Name", "Age"],
            rows=[["Alice", "30"]],
            location=Location(page_number=1, page_offset=100,
                            bbox=BoundingBox(72, 200, 500, 300, page=1)),
            caption="Employee Data",
            table_number=1
        ),
        Paragraph(
            block_id="abc-123::1::3",
            text="Another paragraph.",
            location=Location(page_number=1, page_offset=150,
                            bbox=BoundingBox(72, 320, 500, 340, page=1))
        )
    ],
    sections={
        "1": Section(
            section_id="1",
            title="Chapter 1: Introduction",
            level=HeadingLevel.H1,
            start_page=1,
            end_page=1,
            block_ids=["abc-123::1::0", "abc-123::1::1", "abc-123::1::2", "abc-123::1::3"]
        )
    },
    page_to_blocks={
        1: ["abc-123::1::0", "abc-123::1::1", "abc-123::1::2", "abc-123::1::3"]
    }
)
```

### Markdown as a View (Not Storage)

```python
# Generate markdown ON DEMAND from IR
markdown = ir.to_markdown()

# Output:
# # Chapter 1: Introduction
#
# This is a paragraph.
#
# | Name | Age |
# |------|-----|
# | Alice | 30 |
# **Employee Data**
#
# Another paragraph.
```

---

## How IR Removes Ambiguity

### 1. **Chunking Without Parsing**

#### Before (Markdown-based):
```python
# Chunking must parse markdown
text = "# Heading\n\nParagraph\n\n| Table |"
# Parse headings with regex: ^#+ (.*)
# Detect tables with regex: ^\|.*\|$
# Hope formatting is consistent!

chunks = chunk_markdown(text)  # Fragile
```

#### After (IR-based):
```python
# Chunking operates on structured blocks
for block in ir.blocks:
    if isinstance(block, Heading):
        # Exact heading level, no parsing
        start_new_chunk(block.level)
    elif isinstance(block, Table):
        # Keep table intact, no regex
        add_to_chunk(block, keep_intact=True)
    elif isinstance(block, Paragraph):
        add_to_chunk(block)
```

**Benefits:**
- No markdown parsing errors
- Exact block boundaries
- Type-safe operations
- Reliable heading hierarchy

---

### 2. **Precise Citation**

#### Before (Markdown-based):
```python
# After retrieval, reconstruct citation from metadata dict
chunk_text = "...paragraph text..."
metadata = {
    'page_start': 5,
    'page_end': 6,
    'section_title': 'Introduction',
    'heading_path': ['Chapter 1', 'Introduction']
}
# Hope the mapping is correct!
citation = f"{metadata['file_name']}, pages {metadata['page_start']}-{metadata['page_end']}"
```

**Problem:** If chunking crosses boundaries incorrectly, page numbers are wrong.

#### After (IR-based):
```python
# Each block has precise location
chunk_blocks = [block1, block2, block3]
page_start = min(b.location.page_number for b in chunk_blocks)
page_end = max(b.location.page_number for b in chunk_blocks)
section = get_section_for_block(chunk_blocks[0])

citation = Citation(
    file_name=ir.source_file,
    doc_id=ir.doc_id,
    page_start=page_start,
    page_end=page_end,
    section_path=section.get_path(ir.sections),
    bbox=chunk_blocks[0].location.bbox
)
```

**Benefits:**
- Page numbers are **always** accurate (derived from blocks, not guessed)
- Bounding boxes enable PDF highlighting
- Section hierarchy is explicit, not inferred
- No metadata detachment

---

### 3. **Unambiguous Section Hierarchy**

#### Before (Markdown-based):
```markdown
# Chapter 1
## Section 1.1
### Subsection 1.1.1
Paragraph in 1.1.1
## Section 1.2
Paragraph in 1.2
```

**Problems:**
- Must parse heading levels correctly
- Missing `#` breaks entire hierarchy
- Section boundaries are implicit
- Heading path must be reconstructed during chunking

#### After (IR-based):
```python
sections = {
    "1": Section(
        section_id="1",
        title="Chapter 1",
        level=HeadingLevel.H1,
        parent_id=None,
        children=["1.1", "1.2"],
        block_ids=["abc::1::0"]
    ),
    "1.1": Section(
        section_id="1.1",
        title="Section 1.1",
        level=HeadingLevel.H2,
        parent_id="1",
        children=["1.1.1"],
        block_ids=["abc::2::0", "abc::2::1", ...]
    ),
    "1.1.1": Section(
        section_id="1.1.1",
        title="Subsection 1.1.1",
        level=HeadingLevel.H3,
        parent_id="1.1",
        children=[],
        block_ids=[...]
    )
}

# Get section path
section = ir.sections["1.1.1"]
path = section.get_path(ir.sections)
# ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]
```

**Benefits:**
- Hierarchy is explicit, not inferred
- Section boundaries are precise (start_page, end_page)
- Fast section lookups
- Citation breadcrumbs are trivial

---

### 4. **Table Handling**

#### Before (Markdown-based):
```markdown
| Name | Age |
|------|-----|
| Alice | 30 |
```

**Problems:**
- Is this `|---|` or `|:---|` or `| --- |`?
- Cell content with `|` breaks parsing
- Must re-parse markdown table during chunking
- Context (before/after) is lost

#### After (IR-based):
```python
table = Table(
    block_id="abc::2::5",
    headers=["Name", "Age"],
    rows=[["Alice", "30"]],
    caption="Employee Data",
    location=Location(page_number=2, bbox=BoundingBox(...)),
    context_before="The following table shows employee data:",
    context_after="As shown above, Alice is 30."
)

# Chunking: keep table intact
if isinstance(block, Table):
    chunk.add_block(block)
    # Include context in chunk text
    chunk.text = f"{block.context_before}\n{block.to_markdown()}\n{block.context_after}"
```

**Benefits:**
- No markdown parsing
- Context is preserved
- Can render to markdown, HTML, JSON on demand
- Structured queries: "Find tables with > 10 rows"

---

### 5. **Bounding Box Tracking**

#### Before (Markdown-based):
```python
# Bounding box stored in separate metadata dict
metadata = {
    'bbox': [72, 100, 500, 120],
    'page': 3
}
# Hope the chunk text aligns with the bbox!
```

**Problem:** If chunking splits text, which part gets the bbox?

#### After (IR-based):
```python
# Every block has its own bbox
heading = Heading(
    text="Chapter 1",
    location=Location(
        page_number=3,
        bbox=BoundingBox(x0=72, y0=100, x1=500, y1=120, page=3)
    )
)

# When chunking, aggregate bboxes
chunk_bbox = aggregate_bboxes([b.location.bbox for b in chunk_blocks])
```

**Benefits:**
- Precise PDF highlighting
- Visual citation (show user exactly where content is)
- Multi-column handling (bbox X coordinates)
- Figure/table placement accuracy

---

## Integration with Existing Pipeline

### Current Pipeline

```
PDF/DOCX/PPT
    ↓
[Extraction: pdf2llm, docx2llm, ppt2llm]
    ↓
Markdown string + metadata dict
    ↓
[Document Processor]
    ↓
DocumentMetadata + PageMetadata
    ↓
[Chunking: enterprise_chunking_pipeline]
    ↓
ChunkMetadata with page_start, page_end, heading_path
    ↓
[Embedding Preparation]
    ↓
EmbeddingRecord
    ↓
Qdrant
```

### New Pipeline with DocumentIR

```
PDF/DOCX/PPT
    ↓
[Extraction: pdf2llm, docx2llm, ppt2llm]
    ↓
Extraction output (structured blocks)
    ↓
[MAP TO DocumentIR]  ← NEW STEP
    ↓
DocumentIR (single source of truth)
    ↓
[IR-based Chunking]
    ↓
Chunks with precise block references
    ↓
[Embedding Preparation]
    ↓
EmbeddingRecord with citation metadata
    ↓
Qdrant
```

### Migration Steps

1. **Phase 1: Add IR Layer**
   - Keep existing extraction outputs
   - Add `document_ir_mappers.py` to convert outputs to IR
   - Store IR as JSON alongside current outputs

2. **Phase 2: Update Chunking**
   - Create `ir_chunking_pipeline.py` that operates on DocumentIR
   - Run in parallel with existing chunking
   - Validate outputs match

3. **Phase 3: Update Embedding**
   - Update `embedding_preparation.py` to use IR metadata
   - Ensure Qdrant payloads include block IDs and bboxes

4. **Phase 4: Deprecate Markdown**
   - Switch all pipelines to IR
   - Remove markdown-based chunking
   - Keep markdown rendering for UI/export

---

## Example: IR-Based Chunking

### Algorithm

```python
def chunk_document_ir(
    ir: DocumentIR,
    max_chunk_size: int = 1000,
    respect_boundaries: bool = True
) -> List[Chunk]:
    """
    Chunk DocumentIR with precise boundary control.
    """
    chunks = []
    current_chunk = Chunk()
    current_section = None

    for block in ir.blocks:
        # Track section
        if isinstance(block, Heading):
            current_section = ir.sections[block.section_id]

            # Start new chunk on major headings
            if block.level <= HeadingLevel.H2 and current_chunk.blocks:
                chunks.append(finalize_chunk(current_chunk, ir))
                current_chunk = Chunk()

        # Check size limits
        if current_chunk.size + len(block.text) > max_chunk_size:
            # Respect boundaries
            if respect_boundaries and isinstance(block, (Table, CodeBlock)):
                # Finish current chunk, start new one with this block
                if current_chunk.blocks:
                    chunks.append(finalize_chunk(current_chunk, ir))
                current_chunk = Chunk()
            elif current_chunk.blocks:
                chunks.append(finalize_chunk(current_chunk, ir))
                current_chunk = Chunk()

        # Add block to chunk
        current_chunk.add_block(block)
        current_chunk.section = current_section

    # Final chunk
    if current_chunk.blocks:
        chunks.append(finalize_chunk(current_chunk, ir))

    return chunks


def finalize_chunk(chunk: Chunk, ir: DocumentIR) -> ChunkMetadata:
    """
    Create ChunkMetadata with precise citation info.
    """
    blocks = chunk.blocks

    # Precise page range from blocks
    page_start = min(b.location.page_number for b in blocks)
    page_end = max(b.location.page_number for b in blocks)

    # Section hierarchy
    section = chunk.section
    section_path = section.get_path(ir.sections) if section else []

    # Aggregate bounding boxes
    bboxes = [b.location.bbox for b in blocks if b.location.bbox]
    combined_bbox = aggregate_bboxes(bboxes) if bboxes else None

    # Generate text (with overlap handling)
    text = generate_chunk_text(blocks, chunk.overlap_before, chunk.overlap_after)

    return ChunkMetadata(
        chunk_id=f"{ir.doc_id}::chunk::{len(chunks)}",
        doc_id=ir.doc_id,
        file_name=ir.source_file,
        page_number_start=page_start,
        page_number_end=page_end,
        section_title=section.title if section else None,
        heading_path=section_path,
        block_ids=[b.block_id for b in blocks],
        bbox=combined_bbox,
        normalized_text=text,
        chunk_char_len=len(text),
        chunk_word_count=len(text.split()),
        boundary_type=detect_boundary_type(blocks),
        contains_tables=any(isinstance(b, Table) for b in blocks),
        contains_code=any(isinstance(b, CodeBlock) for b in blocks)
    )
```

### Benefits of IR-Based Chunking

1. **No Parsing Errors**: Operates on structured blocks, not text
2. **Exact Boundaries**: Knows precisely where each block starts/ends
3. **Precise Citations**: Page/bbox directly from blocks
4. **Type-Safe**: Can't accidentally split a table or heading
5. **Fast Lookups**: `page_to_blocks` index for quick page access
6. **Provenance**: Every chunk knows its source blocks

---

## Example: Citation Rendering

### From Chunk to Citation

```python
# After retrieval from Qdrant
retrieved_chunk = {
    'chunk_id': 'abc-123::chunk::5',
    'score': 0.92,
    'payload': {
        'doc_id': 'abc-123',
        'file_name': 'report.pdf',
        'page_start': 5,
        'page_end': 6,
        'section_path': ['Chapter 2', 'Results', 'Performance Analysis'],
        'bbox': {'x0': 72, 'y0': 150, 'x1': 500, 'y1': 400, 'page': 5},
        'block_ids': ['abc-123::5::3', 'abc-123::5::4', 'abc-123::6::0']
    }
}

# Reconstruct from IR
ir = load_document_ir('abc-123')
chunk_blocks = [ir.get_block_by_id(bid) for bid in retrieved_chunk['payload']['block_ids']]

# Render citation
citation = render_citation(retrieved_chunk, chunk_blocks)
```

### Citation Output

```json
{
  "source": "report.pdf",
  "location": "Chapter 2 > Results > Performance Analysis",
  "pages": "5-6",
  "bbox": {
    "page": 5,
    "x0": 72,
    "y0": 150,
    "x1": 500,
    "y1": 400
  },
  "pdf_link": "report.pdf#page=5&viewrect=72,150,500,400",
  "blocks": [
    {
      "type": "paragraph",
      "text": "The performance analysis shows...",
      "page": 5
    },
    {
      "type": "table",
      "caption": "Performance Metrics",
      "page": 5
    },
    {
      "type": "paragraph",
      "text": "As shown in the table...",
      "page": 6
    }
  ]
}
```

### UI Rendering

```html
<div class="citation">
  <div class="source">
    <strong>Source:</strong>
    <a href="report.pdf#page=5&viewrect=72,150,500,400">
      report.pdf (Pages 5-6)
    </a>
  </div>
  <div class="breadcrumb">
    Chapter 2 > Results > Performance Analysis
  </div>
  <div class="preview">
    <p>The performance analysis shows...</p>
    <table>
      <caption>Performance Metrics</caption>
      ...
    </table>
    <p>As shown in the table...</p>
  </div>
</div>
```

---

## Comparison: Before vs After

| Aspect | Before (Markdown-based) | After (IR-based) |
|--------|-------------------------|------------------|
| **Representation** | String + metadata dict | Typed dataclasses |
| **Heading Detection** | Regex parsing (`^#+ `) | `isinstance(block, Heading)` |
| **Section Hierarchy** | Reconstructed during chunking | Built once in IR |
| **Page Numbers** | Guessed from markdown markers | Exact from `block.location.page_number` |
| **Bounding Boxes** | Separate metadata, hope it aligns | Attached to each block |
| **Tables** | Markdown string, re-parse | Structured `Table(headers, rows)` |
| **Chunking** | Parse markdown, regex boundaries | Iterate structured blocks |
| **Citation** | Reconstruct from metadata | Direct from block references |
| **Ambiguity** | High (parsing errors, format variations) | None (structured, typed) |
| **Type Safety** | No types (strings and dicts) | Full type checking |
| **Markdown** | Storage format | View/rendering only |

---

## JSON Schema

For reference, here's the JSON schema representation:

```json
{
  "DocumentIR": {
    "doc_id": "string (UUID)",
    "source_file": "string",
    "source_format": "pdf|docx|ppt|html|txt",
    "blocks": [
      {
        "block_id": "string",
        "block_type": "heading|paragraph|table|figure|code_block|list",
        "location": {
          "page_number": "integer",
          "page_offset": "integer",
          "bbox": {
            "x0": "float",
            "y0": "float",
            "x1": "float",
            "y1": "float",
            "page": "integer"
          }
        },
        "text": "string",

        "// Heading-specific": {
          "level": "1-6",
          "section_path": ["string"],
          "section_id": "string"
        },

        "// Table-specific": {
          "headers": ["string"],
          "rows": [["string"]],
          "caption": "string",
          "table_number": "integer"
        },

        "// Figure-specific": {
          "figure_number": "integer",
          "caption": "string",
          "image_path": "string"
        }
      }
    ],
    "sections": {
      "section_id": {
        "section_id": "string",
        "title": "string",
        "level": "1-6",
        "start_page": "integer",
        "end_page": "integer",
        "parent_id": "string|null",
        "children": ["string"],
        "block_ids": ["string"]
      }
    },
    "page_to_blocks": {
      "page_number": ["block_id"]
    },
    "total_pages": "integer",
    "total_blocks": "integer",
    "total_chars": "integer",
    "total_words": "integer"
  }
}
```

---

## Summary

**DocumentIR solves three core problems:**

1. **Chunking Ambiguity**
   - No markdown parsing
   - Exact block boundaries
   - Type-safe operations

2. **Citation Accuracy**
   - Precise page numbers from blocks
   - Bounding boxes for PDF highlighting
   - Explicit section hierarchy

3. **Data Integrity**
   - Single source of truth
   - Metadata never detaches from content
   - Format-agnostic representation

**Markdown becomes a view, not storage:**
- Generate markdown for UI display
- Generate HTML for web rendering
- Generate JSON for API responses
- But **never parse markdown for processing**

**The pipeline becomes:**
```
Extract → DocumentIR → Process → Render
         ↑
    Single Source of Truth
```
