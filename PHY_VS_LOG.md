# Physical Pages vs Logical Sections

## Core Separation

```python
ChunkMetadata:
    physical_location: PhysicalLocation  # REQUIRED - where content IS
    logical_section: LogicalSection      # OPTIONAL - what content is ABOUT
```

## Physical Location

**Definition**: Absolute position in source document

**Properties**:
- **Immutable**: Always refers to actual document pages
- **Absolute**: Page 15 is always page 15
- **Required**: Every chunk MUST have physical location
- **Observable**: You can open PDF to page N and see it

**Example**:
```python
PhysicalLocation(
    source_file="report.pdf",
    page_start=15,
    page_end=17,
    bboxes=[{page: 15, x0: 72, y0: 200, ...}]
)
```

## Logical Section

**Definition**: Hierarchical content structure

**Properties**:
- **Variable**: Depth and nesting vary per document
- **Structural**: Represents organization, not layout
- **Optional**: Unstructured content has no section
- **Relative**: Depends on heading hierarchy

**Example**:
```python
LogicalSection(
    title="Experimental Results",
    section_path=["Ch3: Methodology", "3.2: Analysis", "3.2.1: Results"],
    depth=3,
    physical_page_start=15,  # Section spans pages 15-20
    physical_page_end=20
)
```

## Key Differences

| Aspect | Physical | Logical |
|--------|----------|---------|
| **What** | Where content appears | What content is about |
| **Units** | Pages, coordinates | Chapters, sections |
| **Required** | Always | Optional |
| **Mutability** | Immutable | Can change with re-structuring |
| **Span** | Exact pages | Can span many pages |
| **Used for** | Finding content | Understanding context |

## Example Scenario

**Document**: "Machine Learning Research Paper"

**Chunk**: Text on pages 15-17

```python
# Physical: WHERE to find it
physical_location:
  page_start: 15
  page_end: 17
  source_file: "ml_paper.pdf"

# Logical: WHAT it's about
logical_section:
  section_path: ["Chapter 3: Methodology",
                 "Section 3.2: Experiments",
                 "3.2.1: Results"]
  depth: 3
  # Note: Section 3.2.1 spans pages 15-20
  physical_page_start: 15
  physical_page_end: 20
```

**Interpretation**:
- **Find it**: Go to pages 15-17 in ml_paper.pdf
- **Context**: It's part of Section 3.2.1 (which is longer)
- **Chunk**: Small piece of a larger section

## Safe Citation Generation

### Rule 1: Always Use Physical Pages

```python
# CORRECT
citation = f"{chunk.physical_location.source_file}, pp. {chunk.page_start}-{chunk.page_end}"
# → "report.pdf, pp. 15-17"

# WRONG - Don't use section's page range for chunk
citation = f"pp. {chunk.logical_section.physical_page_start}-{chunk.logical_section.physical_page_end}"
# → "pp. 15-20" (WRONG! Chunk is only 15-17)
```

### Rule 2: Logical Section Provides Context

```python
# CORRECT - Use both
citation = f"{chunk.physical_location.page_range_str} ({chunk.logical_section.breadcrumb})"
# → "pp. 15-17 (Chapter 3 > Section 3.2 > Results)"

# WRONG - Logical section alone
citation = chunk.logical_section.breadcrumb
# → Missing physical pages (can't find it)
```

### Rule 3: Handle Missing Logical Section

```python
# CORRECT - Check if logical exists
if chunk.logical_section:
    citation = f"{physical.page_range_str} [{logical.breadcrumb}]"
else:
    citation = physical.page_range_str  # Physical only

# WRONG - Assume logical exists
citation = chunk.logical_section.breadcrumb  # May be None!
```

## Citation Templates

### APA Style
```python
# With section
"report.pdf, pp. 15-17 (Chapter 3 > Results)"

# Without section
"report.pdf, pp. 15-17"
```

### MLA Style
```python
"report.pdf pp. 15-17. Chapter 3, Results."
```

### Chicago Style
```python
'report.pdf, pp. 15-17, under "Experimental Results"'
```

## PDF Link Generation

```python
# Use PHYSICAL location only
link = f"{physical.source_file}#page={physical.page_start}"

# With bbox (physical coordinates)
if physical.bboxes:
    bbox = physical.bboxes[0]
    link += f"&viewrect={bbox['x0']},{bbox['y0']},{bbox['x1']},{bbox['y1']}"

# NEVER use logical section for coordinates
# (sections don't have bboxes)
```

## Validation Checklist

```python
def validate_chunk_citation(chunk):
    errors = []

    # Physical location (REQUIRED)
    if not chunk.physical_location:
        errors.append("Missing physical_location")

    if chunk.physical_location.page_start <= 0:
        errors.append("Invalid page number")

    if chunk.physical_location.page_end < chunk.physical_location.page_start:
        errors.append("page_end < page_start")

    # Logical section (OPTIONAL, but validate if present)
    if chunk.logical_section:
        if not chunk.logical_section.section_path:
            errors.append("Empty section_path")

        if chunk.logical_section.depth != len(chunk.logical_section.section_path):
            errors.append("Depth mismatch")

    return len(errors) == 0, errors
```

## Edge Cases

### Case 1: Chunk Without Logical Section
```python
# Appendix, bibliography, unstructured content
chunk = ChunkMetadata(
    physical_location=PhysicalLocation(page_start=50, page_end=50, ...),
    logical_section=None  # OK
)

citation = chunk.physical_location.page_range_str  # "p. 50"
```

### Case 2: Section Spanning Many Pages
```python
# Large chapter with many chunks
section = LogicalSection(
    title="Literature Review",
    physical_page_start=5,
    physical_page_end=45  # 40 pages
)

chunk = ChunkMetadata(
    physical_location=PhysicalLocation(page_start=12, page_end=13),  # 2 pages
    logical_section=section
)

# Chunk citation uses chunk's physical pages, NOT section's
citation = "pp. 12-13 (Literature Review)"  # CORRECT
```

### Case 3: Deeply Nested Section
```python
section = LogicalSection(
    section_path=["Ch4", "Sec 4.2", "4.2.3", "4.2.3.1"],
    depth=4
)

# Full breadcrumb preserved
breadcrumb = " > ".join(section.section_path)
# "Ch4 > Sec 4.2 > 4.2.3 > 4.2.3.1"
```

## Integration with Qdrant

### Payload Structure
```python
qdrant_payload = {
    # Physical location (for finding)
    'page_start': chunk.physical_location.page_start,
    'page_end': chunk.physical_location.page_end,
    'page_range': chunk.physical_location.page_range_str,
    'source_file': chunk.physical_location.source_file,
    'bboxes': chunk.physical_location.bboxes,

    # Logical section (for context)
    'section_title': chunk.section_title,  # May be None
    'section_breadcrumb': chunk.section_breadcrumb,  # May be None
    'section_path': chunk.logical_section.section_path if chunk.logical_section else None,

    # Full objects for reconstruction
    'physical_location': chunk.physical_location.to_dict(),
    'logical_section': chunk.logical_section.to_dict() if chunk.logical_section else None
}
```

### Filtering Examples
```python
# Filter by physical pages
results = qdrant.search(
    filter={
        'must': [
            {'key': 'page_start', 'range': {'gte': 10, 'lte': 20}}
        ]
    }
)

# Filter by logical section
results = qdrant.search(
    filter={
        'must': [
            {'key': 'section_title', 'match': {'value': 'Results'}}
        ]
    }
)
```

## Summary

### DO ✓
- Store physical location for ALL chunks
- Use physical pages for citations
- Add logical section for context
- Handle None logical sections
- Validate both independently

### DON'T ✗
- Derive physical pages from sections
- Derive sections from physical pages
- Use section page range for chunk citation
- Assume logical section exists
- Mix physical and logical concepts

### Citation Formula
```
Citation = Physical Location (REQUIRED) + Logical Context (OPTIONAL)

Example: "report.pdf, pp. 15-17 (Chapter 3 > Results)"
         └─── Physical ───┘  └─── Logical ───┘
```
