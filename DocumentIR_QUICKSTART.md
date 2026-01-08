# DocumentIR Quick Start Guide

## Overview

You now have a complete **Canonical Document Intermediate Representation (DocumentIR)** system that eliminates markdown parsing ambiguity in your ingestion pipeline.

## Files Created

| File | Purpose |
|------|---------|
| `document_ir.py` | Core IR dataclasses and types (500+ lines) |
| `document_ir_mappers.py` | Converters from PDF/DOCX/PPT extraction to IR |
| `document_ir_chunking.py` | IR-based chunking implementation with examples |
| `DocumentIR_DESIGN.md` | Complete design documentation and rationale |
| `DocumentIR_QUICKSTART.md` | This file - quick reference |

## Quick Example

### 1. Convert Extraction Output to IR

```python
from document_ir_mappers import map_pdf_extraction_to_ir

# Your existing PDF extraction output
pdf_output = extract_pdf_enhanced('document.pdf')  # From pdf2llm.py

# Convert to IR
ir = map_pdf_extraction_to_ir('document.pdf', pdf_output, 'enhanced')

# IR is now the single source of truth
print(f"Blocks: {ir.total_blocks}")
print(f"Sections: {len(ir.sections)}")
print(f"Pages: {ir.total_pages}")

# Save IR for later use
ir.to_json('document_ir.json')
```

### 2. Chunk the IR (No Markdown Parsing!)

```python
from document_ir_chunking import IRChunker, IRChunkingConfig

# Configure chunking
config = IRChunkingConfig(
    target_chunk_size=500,
    max_chunk_size=1000,
    enable_overlap=True,
    keep_tables_intact=True
)

# Chunk directly from structured blocks
chunker = IRChunker(config)
chunks = chunker.chunk(ir)

# Every chunk has precise metadata
for chunk in chunks:
    print(f"Pages: {chunk.page_start}-{chunk.page_end}")
    print(f"Section: {' > '.join(chunk.section_path)}")
    print(f"Blocks: {chunk.block_ids}")
    print(f"Contains tables: {chunk.contains_tables}")
```

### 3. Generate Citations

```python
from document_ir_chunking import IRCitationGenerator

citation_gen = IRCitationGenerator()

# Get precise citation from chunk
citation = citation_gen.generate_citation(chunks[0], ir)

# Result has exact page/section/bbox info
{
    'source_file': 'document.pdf',
    'page_range': '5-6',
    'section_breadcrumb': 'Chapter 1 > Introduction',
    'block_ids': ['abc::5::3', 'abc::5::4'],
    'bboxes': [{'x0': 72, 'y0': 150, ...}],
    'contains_tables': True
}

# Generate PDF link with highlighting
pdf_link = citation_gen.generate_pdf_link(chunks[0], ir)
# → "document.pdf#page=5&viewrect=72,150,500,400"
```

### 4. Render to Markdown (View Only)

```python
# Markdown is generated ON DEMAND, never parsed
markdown = ir.to_markdown()

# Or render individual blocks
for block in ir.blocks:
    if isinstance(block, Table):
        table_md = block.to_markdown()
        print(table_md)
```

## Key Data Structures

### DocumentIR (Root)
```python
ir = DocumentIR(
    doc_id="uuid",
    source_file="document.pdf",
    blocks=[...],           # Ordered content blocks
    sections={...},         # Section hierarchy
    page_to_blocks={...}    # Fast page lookup
)
```

### Content Blocks
```python
# Heading with precise level and section info
Heading(text="Chapter 1", level=HeadingLevel.H1,
        section_path=["Chapter 1"], location=...)

# Table with structured data (NOT markdown)
Table(headers=["Col1", "Col2"], rows=[["A", "B"]],
      caption="Data", location=...)

# Every block has location
Location(page_number=5, page_offset=100,
         bbox=BoundingBox(x0=72, y0=150, x1=500, y1=400))
```

### IR Chunks
```python
chunk = IRChunk(
    chunk_id="doc::chunk::5",
    blocks=[block1, block2],      # Actual block objects
    block_ids=["doc::5::3", ...], # References
    page_start=5,                  # Computed from blocks
    page_end=6,
    section_path=["Ch1", "Intro"], # From section hierarchy
    text="...",                    # Generated from blocks
    contains_tables=True           # Flags computed
)
```

## Benefits Over Markdown

| Aspect | Markdown-Based | IR-Based |
|--------|----------------|----------|
| Heading detection | Regex: `^#+ ` | `isinstance(block, Heading)` |
| Page numbers | Guessed from markers | `block.location.page_number` |
| Tables | Parse markdown string | `Table(headers, rows)` |
| Chunking | Parse then split | Iterate structured blocks |
| Citations | Reconstruct metadata | Direct from block references |
| Type safety | Strings + dicts | Typed dataclasses |

## Integration Path

### Current Pipeline
```
PDF → Markdown + Metadata Dict → Chunk → Embed
```

### New Pipeline
```
PDF → Extraction Output → DocumentIR → IR-Chunk → Embed
                          ↑
                   Single Source of Truth
```

### Migration Steps

1. **Add IR Layer (No Breaking Changes)**
   ```python
   # In your document_processor.py
   from document_ir_mappers import map_pdf_extraction_to_ir

   # After extraction
   extraction_output = extract_pdf_enhanced(pdf_file)

   # Convert to IR
   ir = map_pdf_extraction_to_ir(pdf_file, extraction_output)
   ir.to_json(f"{output_dir}/document_ir.json")

   # Continue with existing pipeline (parallel run)
   ```

2. **Update Chunking (Gradual)**
   ```python
   # New IR-based chunking
   from document_ir_chunking import IRChunker

   ir = DocumentIR.from_json("document_ir.json")
   chunker = IRChunker()
   ir_chunks = chunker.chunk(ir)

   # Compare with existing chunks for validation
   ```

3. **Update Embedding Metadata**
   ```python
   # Use IR chunks for embedding
   for chunk in ir_chunks:
       citation = IRCitationGenerator.generate_citation(chunk, ir)

       embedding_record = EmbeddingRecord(
           chunk_id=chunk.chunk_id,
           page_number_start=chunk.page_start,  # From blocks
           page_number_end=chunk.page_end,
           section_path=chunk.section_path,     # From hierarchy
           block_ids=chunk.block_ids,           # NEW: block references
           bboxes=[b.location.bbox for b in chunk.blocks],  # NEW
           ...
       )
   ```

4. **Deprecate Markdown Processing**
   - Remove markdown parsing from chunking pipeline
   - Keep markdown rendering for UI/export only

## Example Workflow

```python
from document_ir_mappers import convert_to_ir
from document_ir_chunking import IRChunker, IRCitationGenerator

# 1. Extract document (existing code)
extraction_output = extract_pdf_enhanced('research.pdf')

# 2. Convert to IR (new step)
ir = convert_to_ir(
    source_file='research.pdf',
    extraction_output=extraction_output,
    source_format='pdf'
)

# 3. Chunk using IR (replaces markdown chunking)
chunker = IRChunker()
chunks = chunker.chunk(ir)

# 4. Generate embeddings with precise citations
citation_gen = IRCitationGenerator()

for chunk in chunks:
    # Get citation metadata
    citation = citation_gen.generate_citation(chunk, ir)

    # Create embedding record
    embedding_text = chunk.text_with_overlap  # Includes semantic overlap

    # Store in Qdrant with rich metadata
    qdrant_payload = {
        'chunk_id': chunk.chunk_id,
        'text': embedding_text,
        'doc_id': ir.doc_id,
        'source_file': ir.source_file,

        # Precise location (no guessing!)
        'page_start': chunk.page_start,
        'page_end': chunk.page_end,
        'page_range': citation['page_range'],

        # Section context
        'section_path': chunk.section_path,
        'section_breadcrumb': citation['section_breadcrumb'],

        # Block-level traceability
        'block_ids': chunk.block_ids,
        'block_types': citation['block_types'],

        # Content flags
        'contains_tables': chunk.contains_tables,
        'contains_code': chunk.contains_code,
        'contains_figures': chunk.contains_figures,

        # Bounding boxes for PDF highlighting
        'bboxes': citation['bboxes'],

        # Full citation
        'citation': citation
    }

    # Store vector + payload
    # qdrant_client.upsert(collection, ...)
```

## Testing the Implementation

Run the example to see it in action:

```bash
python document_ir_chunking.py
```

This will:
1. Create a sample DocumentIR
2. Chunk it using IR-based algorithm
3. Generate precise citations
4. Show all metadata

Expected output:
```
======================================================================
DocumentIR Created
======================================================================
Doc ID: abc-123-...
Blocks: 5
Pages: 2
Sections: 1

======================================================================
Chunks Created
======================================================================
Total chunks: 2

Chunk 1:
  ID: abc-123::chunk::0
  Pages: 1-1
  Section: Chapter 1: Introduction
  Blocks: 3 (heading, paragraph, paragraph)
  Size: 1120 chars, 180 words
  Contains: Tables=False, Code=False

Chunk 2:
  ID: abc-123::chunk::1
  Pages: 1-2
  Section: Chapter 1: Introduction
  Blocks: 2 (table, paragraph)
  Size: 520 chars, 85 words
  Contains: Tables=True, Code=False

======================================================================
Citations
======================================================================

Chunk 1 Citation:
  Source: example.pdf
  Location: Chapter 1: Introduction
  Pages: 1
  Blocks: 3 (heading, paragraph, paragraph)
  PDF Link: example.pdf#page=1&viewrect=72,100,500,120

...
```

## FAQs

### Q: Do I need to rewrite all my extraction code?

**A:** No! Your existing `pdf2llm.py`, `docx2llm.py`, `ppt2llm.py` work as-is. Just add the IR mapping step after extraction.

### Q: What about existing documents already processed?

**A:** You can:
1. Reprocess with IR (recommended for best accuracy)
2. Create IR from existing metadata (if you have structured metadata)
3. Run both pipelines in parallel during migration

### Q: How do I handle custom document types?

**A:** Implement a new mapper in `document_ir_mappers.py`:

```python
def map_custom_format_to_ir(file, extraction_output):
    builder = DocumentIRBuilder(doc_id, file, 'custom')
    # Map your extraction output to blocks
    builder.add_block(...)
    return builder.build()
```

### Q: Can I still generate markdown?

**A:** Yes! Markdown is now a **view**:
```python
markdown = ir.to_markdown()  # Generate on demand
```

### Q: Does this work with your current chunking config?

**A:** Yes, `IRChunkingConfig` mirrors your existing `ChunkingConfig`:
- `max_chunk_size` → Same
- `respect_page_boundaries` → Same
- `keep_tables_intact` → Same
- All existing settings supported

## Next Steps

1. **Test on Sample Document**
   ```bash
   python document_ir_chunking.py
   ```

2. **Integrate into Pipeline**
   - Add IR conversion after extraction
   - Run in parallel with existing chunking
   - Compare outputs

3. **Validate Accuracy**
   - Check page numbers match
   - Verify section hierarchy is correct
   - Ensure citations are precise

4. **Gradual Rollout**
   - Start with new documents
   - Reprocess critical documents
   - Deprecate markdown-based chunking

## Support

For detailed explanations:
- **Design rationale**: See `DocumentIR_DESIGN.md`
- **Implementation**: See code in `document_ir.py`
- **Mapping examples**: See `document_ir_mappers.py`
- **Chunking examples**: See `document_ir_chunking.py`

---

**Remember:** DocumentIR is the **single source of truth**. Markdown is just a view.
