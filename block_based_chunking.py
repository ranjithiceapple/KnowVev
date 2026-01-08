"""
Block-Based Chunking Pipeline

Operates on structured DocumentIR blocks instead of text/markers.
Every chunk is traceable to exact source blocks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
from document_ir import DocumentIR, ContentBlock, Heading, Paragraph, Table, HeadingLevel


# ============================================================================
# CHUNK SCHEMA
# ============================================================================

class BoundaryReason(str, Enum):
    """Why chunk boundary was created"""
    HEADING_MAJOR = "heading_major"      # H1, H2 heading
    HEADING_MINOR = "heading_minor"      # H3-H6 heading
    SIZE_LIMIT = "size_limit"            # Max size reached
    TABLE_BOUNDARY = "table_boundary"    # Before/after table
    PAGE_BOUNDARY = "page_boundary"      # Page break
    SECTION_END = "section_end"          # Logical section ended
    EXPLICIT_BREAK = "explicit_break"    # Manual break marker


@dataclass
class ChunkBoundary:
    """Chunk boundary with reasoning"""
    block_id: str                        # Block where boundary occurs
    reason: BoundaryReason
    confidence: float                    # 0.0-1.0, how good is this boundary
    alternative_blocks: List[str] = field(default_factory=list)  # Other possible boundaries


@dataclass
class BlockChunk:
    """
    Chunk composed of DocumentIR blocks.

    Key Properties:
    - block_ids: Exact source blocks (reversible)
    - boundaries: Why chunk starts/ends here
    - confidence: Overall chunk quality score
    """
    chunk_id: str
    chunk_index: int

    # Source blocks (EXACT references)
    block_ids: List[str]                 # ["doc::1::0", "doc::1::1", ...]
    block_types: List[str]               # ["heading", "paragraph", ...]

    # Boundaries (explainable)
    start_boundary: ChunkBoundary
    end_boundary: ChunkBoundary

    # Physical location (from blocks)
    page_start: int
    page_end: int

    # Logical structure (from blocks)
    section_id: Optional[str] = None
    section_path: List[str] = field(default_factory=list)

    # Content (generated from blocks)
    text: str = ""
    text_tokens: Optional[int] = None

    # Overlap tracking
    overlap_before_block_ids: List[str] = field(default_factory=list)
    overlap_after_block_ids: List[str] = field(default_factory=list)

    # Quality metrics
    confidence: float = 1.0              # Overall chunk quality
    semantic_coherence: float = 1.0      # How related are blocks
    size_score: float = 1.0              # How close to target size

    # Metadata
    contains_tables: bool = False
    contains_code: bool = False
    contains_figures: bool = False

    def to_dict(self) -> dict:
        return {
            'chunk_id': self.chunk_id,
            'chunk_index': self.chunk_index,
            'block_ids': self.block_ids,
            'block_types': self.block_types,
            'start_boundary': {
                'block_id': self.start_boundary.block_id,
                'reason': self.start_boundary.reason.value,
                'confidence': self.start_boundary.confidence
            },
            'end_boundary': {
                'block_id': self.end_boundary.block_id,
                'reason': self.end_boundary.reason.value,
                'confidence': self.end_boundary.confidence
            },
            'page_start': self.page_start,
            'page_end': self.page_end,
            'section_id': self.section_id,
            'section_path': self.section_path,
            'text': self.text,
            'text_tokens': self.text_tokens,
            'confidence': self.confidence,
            'semantic_coherence': self.semantic_coherence,
            'size_score': self.size_score,
            'contains_tables': self.contains_tables,
            'contains_code': self.contains_code,
            'contains_figures': self.contains_figures
        }


# ============================================================================
# CHUNKING ALGORITHM
# ============================================================================

@dataclass
class ChunkingConfig:
    """Chunking configuration"""
    min_size: int = 200
    target_size: int = 500
    max_size: int = 1000
    hard_max: int = 2000                 # For tables/code

    # Boundary preferences
    split_on_h1_h2: bool = True
    split_on_h3_h6: bool = False
    keep_tables_intact: bool = True
    keep_code_intact: bool = True

    # Overlap
    overlap_blocks: int = 1

    # Confidence weights
    heading_boundary_confidence: float = 0.95
    size_boundary_confidence: float = 0.6
    table_boundary_confidence: float = 0.85


class BlockChunker:
    """Chunks DocumentIR by operating on structured blocks"""

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()

    def chunk(self, ir: DocumentIR) -> List[BlockChunk]:
        """
        Chunk DocumentIR into BlockChunks.

        Returns chunks with exact block references and boundary explanations.
        """
        chunks = []
        current_blocks = []
        current_size = 0
        current_section = None

        for i, block in enumerate(ir.blocks):
            block_size = len(block.text)

            # Update current section
            if isinstance(block, Heading) and block.section_id:
                section = ir.sections.get(block.section_id)
                if section:
                    current_section = section

            # Evaluate boundary
            should_split, boundary_reason, confidence = self._should_split(
                current_blocks, block, current_size, block_size, ir
            )

            if should_split and current_blocks:
                # Create chunk
                chunk = self._create_chunk(
                    ir, current_blocks, len(chunks),
                    current_section, boundary_reason, confidence,
                    end_block_id=current_blocks[-1].block_id
                )
                chunks.append(chunk)

                # Reset
                current_blocks = []
                current_size = 0

            # Add block
            current_blocks.append(block)
            current_size += block_size

        # Final chunk
        if current_blocks:
            chunk = self._create_chunk(
                ir, current_blocks, len(chunks),
                current_section, BoundaryReason.SECTION_END, 1.0,
                end_block_id=current_blocks[-1].block_id
            )
            chunks.append(chunk)

        # Add overlap
        self._add_overlap(chunks, ir)

        return chunks

    def _should_split(
        self,
        current_blocks: List[ContentBlock],
        next_block: ContentBlock,
        current_size: int,
        next_size: int,
        ir: DocumentIR
    ) -> Tuple[bool, BoundaryReason, float]:
        """
        Determine if chunk should split before next_block.

        Returns: (should_split, reason, confidence)
        """
        if not current_blocks:
            return False, None, 0.0

        # 1. Heading boundaries (high confidence)
        if isinstance(next_block, Heading):
            if next_block.level <= HeadingLevel.H2 and self.config.split_on_h1_h2:
                return True, BoundaryReason.HEADING_MAJOR, self.config.heading_boundary_confidence
            elif next_block.level >= HeadingLevel.H3 and self.config.split_on_h3_h6:
                return True, BoundaryReason.HEADING_MINOR, self.config.heading_boundary_confidence * 0.8

        # 2. Table boundaries (medium-high confidence)
        if isinstance(next_block, Table) and self.config.keep_tables_intact:
            if current_size + next_size > self.config.max_size:
                return True, BoundaryReason.TABLE_BOUNDARY, self.config.table_boundary_confidence

        # 3. Size limit (medium confidence)
        if current_size + next_size > self.config.max_size:
            return True, BoundaryReason.SIZE_LIMIT, self.config.size_boundary_confidence

        # 4. Page boundary (optional, low confidence)
        current_page = current_blocks[-1].location.page_number
        next_page = next_block.location.page_number
        if next_page > current_page:
            if current_size > self.config.min_size:
                return True, BoundaryReason.PAGE_BOUNDARY, 0.5

        return False, None, 0.0

    def _create_chunk(
        self,
        ir: DocumentIR,
        blocks: List[ContentBlock],
        chunk_index: int,
        section,
        boundary_reason: BoundaryReason,
        boundary_confidence: float,
        end_block_id: str
    ) -> BlockChunk:
        """Create chunk from blocks"""

        # Extract metadata from blocks
        block_ids = [b.block_id for b in blocks]
        block_types = [b.block_type.value for b in blocks]

        page_start = min(b.location.page_number for b in blocks)
        page_end = max(b.location.page_number for b in blocks)

        # Generate text
        text = self._generate_text(blocks)

        # Compute confidence scores
        confidence = self._compute_confidence(blocks, boundary_confidence)
        semantic_coherence = self._compute_semantic_coherence(blocks, section)
        size_score = self._compute_size_score(len(text))

        # Boundaries
        start_boundary = ChunkBoundary(
            block_id=blocks[0].block_id,
            reason=boundary_reason if chunk_index > 0 else BoundaryReason.SECTION_END,
            confidence=boundary_confidence
        )
        end_boundary = ChunkBoundary(
            block_id=end_block_id,
            reason=boundary_reason,
            confidence=boundary_confidence
        )

        chunk = BlockChunk(
            chunk_id=f"{ir.doc_id}::chunk::{chunk_index}",
            chunk_index=chunk_index,
            block_ids=block_ids,
            block_types=block_types,
            start_boundary=start_boundary,
            end_boundary=end_boundary,
            page_start=page_start,
            page_end=page_end,
            section_id=section.section_id if section else None,
            section_path=section.section_path if section else [],
            text=text,
            confidence=confidence,
            semantic_coherence=semantic_coherence,
            size_score=size_score,
            contains_tables=any(b.block_type.value == 'table' for b in blocks),
            contains_code=any(b.block_type.value == 'code_block' for b in blocks),
            contains_figures=any(b.block_type.value == 'figure' for b in blocks)
        )

        return chunk

    def _generate_text(self, blocks: List[ContentBlock]) -> str:
        """Generate text from blocks"""
        parts = []
        for block in blocks:
            if hasattr(block, 'to_markdown'):
                parts.append(block.to_markdown())
            else:
                parts.append(block.text)
        return '\n\n'.join(parts)

    def _compute_confidence(
        self, blocks: List[ContentBlock], boundary_confidence: float
    ) -> float:
        """
        Compute overall chunk confidence.

        Factors:
        - Boundary quality
        - Block coherence
        - Size appropriateness
        """
        # Start with boundary confidence
        conf = boundary_confidence

        # Penalize very small chunks
        total_size = sum(len(b.text) for b in blocks)
        if total_size < self.config.min_size:
            conf *= 0.7

        # Reward chunks with single section
        sections = set(
            getattr(b, 'section_id', None) for b in blocks
            if isinstance(b, Heading) and hasattr(b, 'section_id')
        )
        if len(sections) <= 1:
            conf *= 1.1

        return min(1.0, conf)

    def _compute_semantic_coherence(
        self, blocks: List[ContentBlock], section
    ) -> float:
        """
        Compute semantic coherence score.

        Higher if:
        - All blocks in same section
        - Similar block types
        - No abrupt heading level changes
        """
        score = 1.0

        # Same section bonus
        if section:
            blocks_in_section = sum(
                1 for b in blocks
                if isinstance(b, Heading) and getattr(b, 'section_id', None) == section.section_id
            )
            if blocks_in_section > 0:
                score *= 1.1

        # Heading level consistency
        heading_levels = [
            b.level.value for b in blocks if isinstance(b, Heading)
        ]
        if heading_levels:
            level_range = max(heading_levels) - min(heading_levels)
            if level_range > 2:
                score *= 0.8

        return min(1.0, score)

    def _compute_size_score(self, size: int) -> float:
        """
        Score based on proximity to target size.

        1.0 at target_size, lower when far from target.
        """
        target = self.config.target_size

        if size < self.config.min_size:
            return size / self.config.min_size * 0.5

        if size > self.config.max_size:
            penalty = (size - self.config.max_size) / self.config.max_size
            return max(0.3, 1.0 - penalty)

        # Near target
        distance = abs(size - target) / target
        return max(0.7, 1.0 - distance)

    def _add_overlap(self, chunks: List[BlockChunk], ir: DocumentIR):
        """Add overlap between chunks"""
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            # Overlap: last N blocks of current
            overlap_count = min(self.config.overlap_blocks, len(current.block_ids))
            overlap_block_ids = current.block_ids[-overlap_count:]

            next_chunk.overlap_before_block_ids = overlap_block_ids
            current.overlap_after_block_ids = next_chunk.block_ids[:overlap_count]


# ============================================================================
# CHUNK RECONSTRUCTION
# ============================================================================

def reconstruct_chunk(chunk: BlockChunk, ir: DocumentIR) -> dict:
    """
    Reconstruct chunk from block IDs.

    Demonstrates reversibility: given block_ids, recreate exact chunk.
    """
    blocks = [ir.get_block_by_id(bid) for bid in chunk.block_ids]
    blocks = [b for b in blocks if b is not None]

    # Verify
    reconstructed_text = '\n\n'.join(b.text for b in blocks)

    return {
        'chunk_id': chunk.chunk_id,
        'blocks': blocks,
        'block_count': len(blocks),
        'text_matches': chunk.text == reconstructed_text,
        'pages': [b.location.page_number for b in blocks],
        'boundary_explanation': {
            'start': f"{chunk.start_boundary.reason.value} (confidence: {chunk.start_boundary.confidence})",
            'end': f"{chunk.end_boundary.reason.value} (confidence: {chunk.end_boundary.confidence})"
        }
    }


# ============================================================================
# DEBUGGING & ANALYSIS
# ============================================================================

def analyze_chunking_quality(chunks: List[BlockChunk]) -> dict:
    """Analyze overall chunking quality"""

    return {
        'total_chunks': len(chunks),
        'avg_confidence': sum(c.confidence for c in chunks) / len(chunks),
        'avg_coherence': sum(c.semantic_coherence for c in chunks) / len(chunks),
        'avg_size_score': sum(c.size_score for c in chunks) / len(chunks),

        'boundary_reasons': {
            reason.value: sum(1 for c in chunks if c.start_boundary.reason == reason)
            for reason in BoundaryReason
        },

        'low_confidence_chunks': [
            {'chunk_id': c.chunk_id, 'confidence': c.confidence}
            for c in chunks if c.confidence < 0.7
        ],

        'size_distribution': {
            'min': min(len(c.text) for c in chunks),
            'max': max(len(c.text) for c in chunks),
            'avg': sum(len(c.text) for c in chunks) / len(chunks)
        }
    }


def explain_chunk_boundary(chunk: BlockChunk, ir: DocumentIR) -> str:
    """Generate human-readable boundary explanation"""

    start_block = ir.get_block_by_id(chunk.start_boundary.block_id)
    end_block = ir.get_block_by_id(chunk.end_boundary.block_id)

    explanation = f"""
Chunk {chunk.chunk_index} Boundary Explanation:
================================================

START:
  Block: {chunk.start_boundary.block_id}
  Reason: {chunk.start_boundary.reason.value}
  Confidence: {chunk.start_boundary.confidence:.2f}
  Block Type: {start_block.block_type.value if start_block else 'unknown'}
  Page: {start_block.location.page_number if start_block else '?'}

END:
  Block: {chunk.end_boundary.block_id}
  Reason: {chunk.end_boundary.reason.value}
  Confidence: {chunk.end_boundary.confidence:.2f}
  Block Type: {end_block.block_type.value if end_block else 'unknown'}
  Page: {end_block.location.page_number if end_block else '?'}

QUALITY SCORES:
  Overall Confidence: {chunk.confidence:.2f}
  Semantic Coherence: {chunk.semantic_coherence:.2f}
  Size Score: {chunk.size_score:.2f}

COMPOSITION:
  Blocks: {len(chunk.block_ids)}
  Block Types: {', '.join(set(chunk.block_types))}
  Pages: {chunk.page_start}-{chunk.page_end}
  Section: {' > '.join(chunk.section_path) if chunk.section_path else 'None'}
"""

    return explanation


# ============================================================================
# EXAMPLE
# ============================================================================

def example_block_chunking():
    """Example: Block-based chunking with explanations"""
    from document_ir import DocumentIRBuilder, Location, BoundingBox
    import uuid

    # Build IR
    doc_id = str(uuid.uuid4())
    builder = DocumentIRBuilder(doc_id, "example.pdf", "pdf")

    builder.add_block(Heading(
        block_id=f"{doc_id}::1::0",
        text="Chapter 1",
        level=HeadingLevel.H1,
        location=Location(page_number=1, page_offset=0)
    ))

    for i in range(5):
        builder.add_block(Paragraph(
            block_id=f"{doc_id}::1::{i+1}",
            text=f"Paragraph {i+1}. " * 50,
            location=Location(page_number=1, page_offset=(i+1)*100)
        ))

    ir = builder.build()

    # Chunk
    config = ChunkingConfig(target_size=400, max_size=800)
    chunker = BlockChunker(config)
    chunks = chunker.chunk(ir)

    print("=" * 70)
    print("BLOCK-BASED CHUNKING RESULTS")
    print("=" * 70)

    for chunk in chunks:
        print(f"\nChunk {chunk.chunk_index}:")
        print(f"  Blocks: {len(chunk.block_ids)}")
        print(f"  Block IDs: {chunk.block_ids}")
        print(f"  Start Boundary: {chunk.start_boundary.reason.value} (conf: {chunk.start_boundary.confidence})")
        print(f"  End Boundary: {chunk.end_boundary.reason.value} (conf: {chunk.end_boundary.confidence})")
        print(f"  Confidence: {chunk.confidence:.2f}")
        print(f"  Coherence: {chunk.semantic_coherence:.2f}")
        print(f"  Size Score: {chunk.size_score:.2f}")
        print(f"  Text Length: {len(chunk.text)}")

    # Analysis
    print("\n" + "=" * 70)
    print("QUALITY ANALYSIS")
    print("=" * 70)
    analysis = analyze_chunking_quality(chunks)
    print(f"Total Chunks: {analysis['total_chunks']}")
    print(f"Avg Confidence: {analysis['avg_confidence']:.2f}")
    print(f"Avg Coherence: {analysis['avg_coherence']:.2f}")
    print(f"Boundary Reasons: {analysis['boundary_reasons']}")

    # Reconstruction
    print("\n" + "=" * 70)
    print("RECONSTRUCTION TEST")
    print("=" * 70)
    reconstructed = reconstruct_chunk(chunks[0], ir)
    print(f"Chunk {chunks[0].chunk_id}")
    print(f"  Blocks Retrieved: {reconstructed['block_count']}")
    print(f"  Text Matches: {reconstructed['text_matches']}")
    print(f"  Boundary: {reconstructed['boundary_explanation']}")


if __name__ == "__main__":
    example_block_chunking()
