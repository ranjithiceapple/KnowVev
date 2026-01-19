# Block-Based Chunking

## Chunk Schema

```python
@dataclass
class BlockChunk:
    chunk_id: str

    # EXACT source blocks (reversible)
    block_ids: List[str]              # ["doc::1::0", "doc::1::1", ...]
    block_types: List[str]            # ["heading", "paragraph", ...]

    # EXPLAINABLE boundaries
    start_boundary: ChunkBoundary     # Why chunk starts here
    end_boundary: ChunkBoundary       # Why chunk ends here

    # COMPUTABLE confidence
    confidence: float                 # Overall quality (0-1)
    semantic_coherence: float         # Block relatedness (0-1)
    size_score: float                 # Size appropriateness (0-1)
```

### Boundary Tracking
```python
@dataclass
class ChunkBoundary:
    block_id: str                     # Exact block at boundary
    reason: BoundaryReason            # Why boundary here
    confidence: float                 # How confident (0-1)

BoundaryReason:
    HEADING_MAJOR                     # H1, H2 split (conf: 0.95)
    SIZE_LIMIT                        # Max size reached (conf: 0.6)
    TABLE_BOUNDARY                    # Before/after table (conf: 0.85)
    SECTION_END                       # Logical section ended (conf: 1.0)
```

## Chunking Algorithm

### Input: DocumentIR (structured blocks)
```python
ir.blocks = [
    Heading(block_id="doc::1::0", level=1, text="Chapter 1"),
    Paragraph(block_id="doc::1::1", text="..."),
    Table(block_id="doc::1::2", headers=[...], rows=[[...]]),
    Paragraph(block_id="doc::1::3", text="...")
]
```

### Process
```python
chunker = BlockChunker(config)
chunks = chunker.chunk(ir)

for block in ir.blocks:
    # Type-safe boundary detection
    should_split, reason, confidence = _should_split(current_blocks, block)

    if should_split:
        # Create chunk with exact block references
        chunk = BlockChunk(
            block_ids=[b.block_id for b in current_blocks],
            start_boundary=ChunkBoundary(
                block_id=current_blocks[0].block_id,
                reason=reason,
                confidence=confidence
            )
        )
```

### Output: Chunks with explainable boundaries
```python
chunks[0] = BlockChunk(
    chunk_id="doc::chunk::0",
    block_ids=["doc::1::0", "doc::1::1"],
    start_boundary=ChunkBoundary(
        block_id="doc::1::0",
        reason=BoundaryReason.SECTION_END,
        confidence=1.0
    ),
    end_boundary=ChunkBoundary(
        block_id="doc::1::1",
        reason=BoundaryReason.HEADING_MAJOR,
        confidence=0.95
    ),
    confidence=0.92,
    semantic_coherence=0.95,
    size_score=0.88
)
```

## Benefits

### 1. Reversibility
```python
# Reconstruct chunk from block IDs
chunk = load_chunk("doc::chunk::0")
blocks = [ir.get_block_by_id(bid) for bid in chunk.block_ids]

# EXACT reconstruction
original_text = chunk.text
reconstructed_text = '\n\n'.join(b.text for b in blocks)
assert original_text == reconstructed_text  # ✓ Always true
```

**Debugging**: Trace any chunk back to source blocks.

### 2. Explainability
```python
# Why did this chunk split here?
print(explain_chunk_boundary(chunk, ir))

# Output:
# Chunk 5 Boundary Explanation:
# START:
#   Block: doc::3::12
#   Reason: heading_major
#   Confidence: 0.95
#   Block Type: heading
#   Page: 15
# END:
#   Block: doc::3::18
#   Reason: size_limit
#   Confidence: 0.6
#   Block Type: paragraph
#   Page: 16
```

**Debugging**: Understand why chunking made specific decisions.

### 3. Confidence Scoring
```python
# Compute chunk quality
chunk.confidence = compute_confidence(
    boundary_confidence=0.95,  # Heading boundary is strong
    size_score=0.88,           # Near target size
    coherence=0.95             # All blocks in same section
)
# → 0.92 overall

# Filter low-quality chunks
bad_chunks = [c for c in chunks if c.confidence < 0.7]
for chunk in bad_chunks:
    print(f"Chunk {chunk.chunk_id}: {chunk.confidence}")
    print(f"  Issue: {chunk.end_boundary.reason}")
```

**Debugging**: Find problematic chunks automatically.

### 4. Retrieval Accuracy Improvements

#### Precise Block References
```python
# Qdrant payload
{
    'chunk_id': 'doc::chunk::5',
    'block_ids': ['doc::3::12', 'doc::3::13', 'doc::3::14'],

    # Can reconstruct exact source
    'pages': [15, 16],           # From block.location.page_number
    'section': 'Ch3 > Results',  # From block.section_path

    # Quality filtering
    'confidence': 0.92,
    'coherence': 0.95
}

# Filter by quality during search
results = qdrant.search(
    vector=query_vector,
    filter={'confidence': {'gte': 0.8}}  # Only high-quality chunks
)
```

#### Boundary Context
```python
# Show why chunk boundaries exist
retrieved = search_qdrant(query)
for result in retrieved:
    chunk = load_chunk(result['chunk_id'])

    print(f"Score: {result['score']}")
    print(f"Quality: {chunk.confidence}")
    print(f"Boundary: {chunk.start_boundary.reason.value}")

    # User sees: "This chunk starts at a major heading (H1)"
    # vs: "This chunk starts due to size limit" (lower confidence)
```

#### Block-Level Retrieval
```python
# Can expand to neighboring blocks
chunk = search_result['chunk']
ir = load_ir(chunk.doc_id)

# Get block before chunk
prev_block_id = get_prev_block_id(chunk.block_ids[0], ir)
prev_block = ir.get_block_by_id(prev_block_id)

# Get block after chunk
next_block_id = get_next_block_id(chunk.block_ids[-1], ir)
next_block = ir.get_block_by_id(next_block_id)

# Show extended context to user
extended_context = [prev_block] + chunk.blocks + [next_block]
```

## Comparison: Text-Based vs Block-Based

| Aspect | Text-Based | Block-Based |
|--------|-----------|-------------|
| **Boundaries** | Text positions | Block IDs |
| **Reversibility** | Approximate | Exact |
| **Explainability** | None | Reason + confidence |
| **Confidence** | Not computable | Computed per chunk |
| **Debugging** | Guess from text | Trace to blocks |
| **Quality filtering** | Not possible | Filter by confidence |

### Debugging Example

**Text-Based** (opaque):
```python
chunk = {
    'text': "# Chapter 1\n\nParagraph...",
    'page_start': 5,  # How was this determined?
    'page_end': 6     # Why does it end here?
}

# No way to know why chunk was created this way
```

**Block-Based** (transparent):
```python
chunk = {
    'block_ids': ['doc::5::0', 'doc::5::1', 'doc::5::2'],
    'start_boundary': {
        'reason': 'heading_major',
        'confidence': 0.95,
        'block_id': 'doc::5::0'
    },
    'end_boundary': {
        'reason': 'size_limit',
        'confidence': 0.6,
        'block_id': 'doc::5::2'
    }
}

# Exact explanation:
# "Chunk starts at H1 heading (strong boundary)
#  Chunk ends due to size limit (weaker boundary - consider adjusting?)"
```

## Usage

```python
from block_based_chunking import BlockChunker, ChunkingConfig

# Configure
config = ChunkingConfig(
    target_size=500,
    max_size=1000,
    split_on_h1_h2=True,
    heading_boundary_confidence=0.95
)

# Chunk
chunker = BlockChunker(config)
chunks = chunker.chunk(ir)

# Analyze quality
from block_based_chunking import analyze_chunking_quality
analysis = analyze_chunking_quality(chunks)
print(f"Avg confidence: {analysis['avg_confidence']}")
print(f"Low quality chunks: {len(analysis['low_confidence_chunks'])}")

# Debug specific chunk
from block_based_chunking import explain_chunk_boundary
explanation = explain_chunk_boundary(chunks[5], ir)
print(explanation)

# Reconstruct
from block_based_chunking import reconstruct_chunk
reconstructed = reconstruct_chunk(chunks[0], ir)
assert reconstructed['text_matches']  # ✓ Exact match
```

## Summary

Block-based chunking provides:
1. **Exact traceability**: Every chunk → source blocks
2. **Explainable boundaries**: Why chunks split where they do
3. **Quality metrics**: Confidence scores per chunk
4. **Better debugging**: Understand chunking decisions
5. **Improved retrieval**: Filter by quality, expand context
