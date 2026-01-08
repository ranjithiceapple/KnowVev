# Markdown as Presentation Layer

## What Markdown Should NOT Do

### 1. Store Structure ✗
```python
# WRONG: Structure in markdown
text = "# Chapter 1\n## Section 1.1\n### Subsection"

# CORRECT: Structure in IR
ir.sections = {
    "1": Section(title="Chapter 1", level=1, children=["1.1"]),
    "1.1": Section(title="Section 1.1", level=2, parent="1")
}
```

### 2. Define Boundaries ✗
```python
# WRONG: Parse markdown for chunk boundaries
if re.match(r'^#{1,2}\s', line):
    start_new_chunk()

# CORRECT: Use block types
if isinstance(block, Heading) and block.level <= HeadingLevel.H2:
    start_new_chunk()
```

### 3. Track Pages ✗
```python
# WRONG: Page markers in text
text = "Content... <<<PAGE_5>>> More content..."
page = extract_page_from_marker(text)

# CORRECT: Page in block metadata
page = block.location.page_number
```

### 4. Represent Tables ✗
```python
# WRONG: Parse markdown tables
text = "| Name | Age |\n|------|-----|\n| Alice | 30 |"
headers, rows = parse_markdown_table(text)

# CORRECT: Structured table data
table = Table(headers=["Name", "Age"], rows=[["Alice", "30"]])
```

### 5. Carry Metadata ✗
```python
# WRONG: Metadata in comments/markers
text = "<!-- bbox: 72,100,500,200 -->\nParagraph text"

# CORRECT: Metadata in block
block = Paragraph(
    text="Paragraph text",
    location=Location(bbox=BoundingBox(72, 100, 500, 200))
)
```

### 6. Be Parsed During Processing ✗
```python
# WRONG: Process markdown
normalized = normalize_markdown(markdown_text)
chunks = chunk_markdown(markdown_text)

# CORRECT: Process IR
normalized_ir = normalize_ir(document_ir)
chunks = chunk_ir(document_ir)
```

### 7. Act as Source of Truth ✗
```python
# WRONG: Markdown is storage
with open('doc.md') as f:
    markdown = f.read()
    chunks = parse_and_chunk(markdown)

# CORRECT: IR is storage
ir = DocumentIR.from_json('doc_ir.json')
chunks = chunk_ir(ir)
markdown = ir.to_markdown()  # Generate for display
```

## Chunking Logic: Text vs IR

### Text-Based Chunking (Wrong)
```python
def chunk_markdown(markdown: str, max_size: int) -> List[dict]:
    chunks = []
    current = []
    current_size = 0

    for line in markdown.split('\n'):
        # ✗ Fragile regex parsing
        if re.match(r'^#{1,2}\s', line):
            if current_size > 0:
                chunks.append({
                    'text': '\n'.join(current),
                    'page': extract_page(current),  # ✗ Reconstructed
                })
                current = []
                current_size = 0

        current.append(line)
        current_size += len(line)

    return chunks
```

**Problems:**
- Regex fails on formatting variations: `"# Title"` vs `"#Title"`
- Page numbers reconstructed from markers (fragile)
- Can't detect tables inside code blocks
- Heading level requires string parsing
- Metadata detached from content

### IR-Based Chunking (Correct)
```python
def chunk_ir(ir: DocumentIR, max_size: int) -> List[dict]:
    chunks = []
    current_blocks = []
    current_size = 0

    for block in ir.blocks:
        # ✓ Type-safe, no regex
        if isinstance(block, Heading) and block.level <= HeadingLevel.H2:
            if current_size > 0:
                chunks.append(create_chunk(current_blocks, ir))
                current_blocks = []
                current_size = 0

        current_blocks.append(block)
        current_size += len(block.text)

    return chunks


def create_chunk(blocks: List[ContentBlock], ir: DocumentIR) -> dict:
    # ✓ Exact page numbers from blocks
    page_start = min(b.location.page_number for b in blocks)
    page_end = max(b.location.page_number for b in blocks)

    # ✓ Section from block metadata
    section = get_section_for_block(blocks[0], ir)

    # ✓ Generate markdown ONLY for output
    text = '\n\n'.join(block_to_markdown(b) for b in blocks)

    return {
        'blocks': blocks,
        'page_start': page_start,
        'page_end': page_end,
        'section_path': section.section_path if section else [],
        'text': text  # View only
    }
```

**Benefits:**
- No regex parsing
- Exact page numbers (from `block.location.page_number`)
- Type-safe operations (`isinstance(block, Heading)`)
- Precise heading levels (`block.level.value`)
- Metadata attached to blocks

## Key Transformations

| Operation | Text-Based (Wrong) | IR-Based (Correct) |
|-----------|-------------------|-------------------|
| **Detect heading** | `re.match(r'^#', line)` | `isinstance(block, Heading)` |
| **Get heading level** | `len(line) - len(line.lstrip('#'))` | `block.level.value` |
| **Track pages** | `re.search(r'<<<PAGE_(\d+)>>>', text)` | `block.location.page_number` |
| **Handle tables** | `parse_markdown_table(text)` | `block.headers, block.rows` |
| **Get section** | Reconstruct from `#` levels | `block.section_path` |
| **Chunk boundaries** | Regex + string parsing | Type checks on blocks |

## Pipeline Flow

```
WRONG (Markdown-centric):
  Extract → Markdown → Parse → Chunk → Parse → Embed

CORRECT (IR-centric):
  Extract → DocumentIR → Chunk → Embed
                ↓                  ↓
           to_markdown()      to_markdown()
                ↓                  ↓
            (export)          (UI display)
```

## Where Markdown IS Used

### 1. Export
```python
# Generate markdown file for users
markdown = ir.to_markdown()
with open('output.md', 'w') as f:
    f.write(markdown)
```

### 2. UI Display
```python
# Render chunk in UI
@app.get('/chunks/{chunk_id}')
def get_chunk(chunk_id: str):
    chunk = load_chunk(chunk_id)
    return {
        'chunk_id': chunk_id,
        'content': chunk.to_markdown(),  # Generate for display
        'citation': {
            'page_start': chunk.page_start,  # From IR
            'page_end': chunk.page_end
        }
    }
```

### 3. Embedding Input
```python
# Generate text for embedding model
text = chunk.to_markdown()  # View generation
vector = embed_model.encode(text)
```

**Key**: Markdown is WRITE-ONLY at pipeline edges. Never read/parsed.

## Summary

**Markdown should:**
- Be generated from IR
- Serve display/export only
- Never be parsed during processing

**Markdown should NOT:**
- Store structure
- Define boundaries
- Track metadata
- Be source of truth

**Chunking changes:**
- From regex parsing → type-safe object operations
- From string processing → block iteration
- From reconstructed pages → direct block.location.page_number
- From markdown parsing → IR traversal
