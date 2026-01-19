"""
Artifact Persistence Strategy for Document Ingestion Pipeline

Stores intermediate outputs at each pipeline stage to enable:
- Reprocessing without re-extraction
- Debugging and validation
- Incremental updates
- Audit trails
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import json
import hashlib
from datetime import datetime


# ============================================================================
# ARTIFACT STORAGE LAYOUT
# ============================================================================

@dataclass
class ArtifactPaths:
    """
    Directory structure for a single document's artifacts.

    Layout:
    artifacts/
    └── {doc_id}/
        ├── metadata.json              # Doc-level metadata
        ├── 00_raw/
        │   ├── source.pdf             # Original file (optional)
        │   ├── extraction_raw.json    # Raw extractor output
        │   └── extraction_metadata.json
        ├── 01_normalized/
        │   ├── document_ir.json       # Canonical IR
        │   ├── normalized_text.txt    # Clean text
        │   └── normalization_log.json
        ├── 02_chunks/
        │   ├── chunks.json            # All chunks
        │   ├── chunk_metadata.json    # Chunk stats
        │   └── chunks/                # Individual chunks
        │       ├── chunk_000.json
        │       ├── chunk_001.json
        │       └── ...
        ├── 03_embeddings/
        │   ├── embeddings.npy         # Vectors
        │   ├── embedding_records.json # Records with metadata
        │   └── embedding_config.json
        └── 04_enrichment/
            ├── topics.json            # Topic modeling output
            ├── summaries.json         # Section/doc summaries
            ├── entities.json          # NER outputs
            └── keywords.json
    """
    base_dir: Path
    doc_id: str

    @property
    def root(self) -> Path:
        return self.base_dir / self.doc_id

    # Stage directories
    @property
    def raw_dir(self) -> Path:
        return self.root / "00_raw"

    @property
    def normalized_dir(self) -> Path:
        return self.root / "01_normalized"

    @property
    def chunks_dir(self) -> Path:
        return self.root / "02_chunks"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "03_embeddings"

    @property
    def enrichment_dir(self) -> Path:
        return self.root / "04_enrichment"

    # Specific files
    @property
    def metadata_file(self) -> Path:
        return self.root / "metadata.json"

    @property
    def source_file(self) -> Path:
        return self.raw_dir / "source_file"

    @property
    def extraction_raw(self) -> Path:
        return self.raw_dir / "extraction_raw.json"

    @property
    def document_ir(self) -> Path:
        return self.normalized_dir / "document_ir.json"

    @property
    def chunks_file(self) -> Path:
        return self.chunks_dir / "chunks.json"

    @property
    def embeddings_file(self) -> Path:
        return self.embeddings_dir / "embeddings.npy"

    @property
    def embedding_records(self) -> Path:
        return self.embeddings_dir / "embedding_records.json"

    def create_dirs(self):
        """Create all directories"""
        for d in [self.raw_dir, self.normalized_dir, self.chunks_dir,
                  self.embeddings_dir, self.enrichment_dir]:
            d.mkdir(parents=True, exist_ok=True)
        (self.chunks_dir / "chunks").mkdir(exist_ok=True)


# ============================================================================
# STAGE-SPECIFIC ARTIFACTS
# ============================================================================

@dataclass
class Stage00_RawArtifacts:
    """Stage 0: Raw extraction outputs"""

    # Raw extraction output (from pdf2llm, docx2llm, etc.)
    extraction_output: list  # Raw blocks from extractor

    # Extraction metadata
    extractor_name: str  # "marker-pdf", "pymupdf-enhanced", etc.
    extractor_version: str
    extraction_timestamp: str
    extraction_config: Dict[str, Any]

    # Source file info
    source_filename: str
    source_hash: str  # SHA256 of source file
    source_size_bytes: int
    source_format: str  # pdf, docx, ppt


@dataclass
class Stage01_NormalizedArtifacts:
    """Stage 1: Normalized/canonical representation"""

    # DocumentIR (JSON serialized)
    document_ir: Dict[str, Any]

    # Normalization metadata
    normalization_rules: list  # Rules applied
    normalization_stats: Dict[str, Any]  # Chars removed, etc.

    # Plain text (for quick access)
    full_text: str


@dataclass
class Stage02_ChunkArtifacts:
    """Stage 2: Chunked content"""

    # All chunks
    chunks: list  # List of IRChunk.to_dict()

    # Chunking config
    chunking_config: Dict[str, Any]

    # Stats
    total_chunks: int
    avg_chunk_size: float
    chunk_size_distribution: Dict[str, int]


@dataclass
class Stage03_EmbeddingArtifacts:
    """Stage 3: Embeddings"""

    # Embedding records (metadata + text)
    embedding_records: list  # EmbeddingRecord.to_dict()

    # Embedding config
    model_name: str
    embedding_dim: int
    embedding_timestamp: str

    # Note: Actual vectors stored in .npy file separately


@dataclass
class Stage04_EnrichmentArtifacts:
    """Stage 4: Enrichments (topics, summaries, etc.)"""

    # Topics
    topics: Optional[Dict[str, Any]] = None  # Topic modeling results

    # Summaries
    summaries: Optional[Dict[str, Any]] = None  # Section/doc summaries

    # Entities
    entities: Optional[list] = None  # NER results

    # Keywords
    keywords: Optional[list] = None


# ============================================================================
# PERSISTENCE MANAGER
# ============================================================================

class ArtifactManager:
    """Manages artifact persistence for a document"""

    def __init__(self, base_dir: str, doc_id: str):
        self.paths = ArtifactPaths(Path(base_dir), doc_id)
        self.paths.create_dirs()

    # ========== WRITE METHODS ==========

    def save_stage00_raw(self, artifacts: Stage00_RawArtifacts):
        """Save raw extraction artifacts"""
        self.paths.extraction_raw.write_text(
            json.dumps({
                'extraction_output': artifacts.extraction_output,
                'metadata': {
                    'extractor_name': artifacts.extractor_name,
                    'extractor_version': artifacts.extractor_version,
                    'extraction_timestamp': artifacts.extraction_timestamp,
                    'extraction_config': artifacts.extraction_config,
                    'source_filename': artifacts.source_filename,
                    'source_hash': artifacts.source_hash,
                    'source_size_bytes': artifacts.source_size_bytes,
                    'source_format': artifacts.source_format
                }
            }, indent=2, ensure_ascii=False)
        )

    def save_stage01_normalized(self, artifacts: Stage01_NormalizedArtifacts):
        """Save normalized artifacts"""
        self.paths.document_ir.write_text(
            json.dumps(artifacts.document_ir, indent=2, ensure_ascii=False)
        )

        (self.paths.normalized_dir / "full_text.txt").write_text(
            artifacts.full_text, encoding='utf-8'
        )

        (self.paths.normalized_dir / "normalization_log.json").write_text(
            json.dumps({
                'rules': artifacts.normalization_rules,
                'stats': artifacts.normalization_stats
            }, indent=2)
        )

    def save_stage02_chunks(self, artifacts: Stage02_ChunkArtifacts):
        """Save chunk artifacts"""
        # All chunks in one file
        self.paths.chunks_file.write_text(
            json.dumps({
                'chunks': artifacts.chunks,
                'metadata': {
                    'chunking_config': artifacts.chunking_config,
                    'total_chunks': artifacts.total_chunks,
                    'avg_chunk_size': artifacts.avg_chunk_size,
                    'chunk_size_distribution': artifacts.chunk_size_distribution
                }
            }, indent=2, ensure_ascii=False)
        )

        # Individual chunks (for easy access)
        chunks_subdir = self.paths.chunks_dir / "chunks"
        for i, chunk in enumerate(artifacts.chunks):
            chunk_file = chunks_subdir / f"chunk_{i:03d}.json"
            chunk_file.write_text(json.dumps(chunk, indent=2, ensure_ascii=False))

    def save_stage03_embeddings(self, artifacts: Stage03_EmbeddingArtifacts, vectors=None):
        """Save embedding artifacts"""
        # Metadata
        self.paths.embedding_records.write_text(
            json.dumps({
                'records': artifacts.embedding_records,
                'metadata': {
                    'model_name': artifacts.model_name,
                    'embedding_dim': artifacts.embedding_dim,
                    'embedding_timestamp': artifacts.embedding_timestamp
                }
            }, indent=2, ensure_ascii=False)
        )

        # Vectors (if provided)
        if vectors is not None:
            import numpy as np
            np.save(self.paths.embeddings_file, vectors)

    def save_stage04_enrichment(self, artifacts: Stage04_EnrichmentArtifacts):
        """Save enrichment artifacts"""
        if artifacts.topics:
            (self.paths.enrichment_dir / "topics.json").write_text(
                json.dumps(artifacts.topics, indent=2)
            )

        if artifacts.summaries:
            (self.paths.enrichment_dir / "summaries.json").write_text(
                json.dumps(artifacts.summaries, indent=2, ensure_ascii=False)
            )

        if artifacts.entities:
            (self.paths.enrichment_dir / "entities.json").write_text(
                json.dumps(artifacts.entities, indent=2)
            )

        if artifacts.keywords:
            (self.paths.enrichment_dir / "keywords.json").write_text(
                json.dumps(artifacts.keywords, indent=2)
            )

    # ========== READ METHODS ==========

    def load_stage00_raw(self) -> Stage00_RawArtifacts:
        """Load raw artifacts"""
        data = json.loads(self.paths.extraction_raw.read_text())
        return Stage00_RawArtifacts(
            extraction_output=data['extraction_output'],
            extractor_name=data['metadata']['extractor_name'],
            extractor_version=data['metadata']['extractor_version'],
            extraction_timestamp=data['metadata']['extraction_timestamp'],
            extraction_config=data['metadata']['extraction_config'],
            source_filename=data['metadata']['source_filename'],
            source_hash=data['metadata']['source_hash'],
            source_size_bytes=data['metadata']['source_size_bytes'],
            source_format=data['metadata']['source_format']
        )

    def load_document_ir(self) -> Dict[str, Any]:
        """Load DocumentIR"""
        return json.loads(self.paths.document_ir.read_text())

    def load_chunks(self) -> list:
        """Load all chunks"""
        data = json.loads(self.paths.chunks_file.read_text())
        return data['chunks']

    def load_embedding_records(self) -> list:
        """Load embedding records"""
        data = json.loads(self.paths.embedding_records.read_text())
        return data['records']

    # ========== REPROCESSING ==========

    def can_skip_extraction(self, source_file_hash: str) -> bool:
        """Check if extraction can be skipped"""
        if not self.paths.extraction_raw.exists():
            return False

        raw = self.load_stage00_raw()
        return raw.source_hash == source_file_hash

    def reprocess_from_stage(self, stage: int) -> Dict[str, Any]:
        """
        Reprocess pipeline from a specific stage.
        Returns artifacts needed for that stage.
        """
        if stage == 1:  # Start from normalization
            return {'raw': self.load_stage00_raw()}
        elif stage == 2:  # Start from chunking
            return {'document_ir': self.load_document_ir()}
        elif stage == 3:  # Start from embedding
            return {'chunks': self.load_chunks()}
        elif stage == 4:  # Start from enrichment
            return {
                'chunks': self.load_chunks(),
                'document_ir': self.load_document_ir()
            }
        else:
            raise ValueError(f"Invalid stage: {stage}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_pipeline_with_persistence():
    """Example: Full pipeline with artifact persistence"""

    doc_id = "doc_12345"
    artifacts = ArtifactManager("./artifacts", doc_id)

    # Stage 0: Extract (expensive)
    print("Stage 0: Extraction")
    raw_artifacts = Stage00_RawArtifacts(
        extraction_output=[{'type': 'heading', 'content': 'Title'}],
        extractor_name='pymupdf-enhanced',
        extractor_version='1.0',
        extraction_timestamp=datetime.now().isoformat(),
        extraction_config={'method': 'enhanced'},
        source_filename='document.pdf',
        source_hash=hashlib.sha256(b'file_content').hexdigest(),
        source_size_bytes=1024000,
        source_format='pdf'
    )
    artifacts.save_stage00_raw(raw_artifacts)

    # Stage 1: Normalize
    print("Stage 1: Normalization → DocumentIR")
    normalized_artifacts = Stage01_NormalizedArtifacts(
        document_ir={'doc_id': doc_id, 'blocks': []},
        normalization_rules=['remove_extra_whitespace', 'fix_encoding'],
        normalization_stats={'chars_removed': 150},
        full_text='Full document text...'
    )
    artifacts.save_stage01_normalized(normalized_artifacts)

    # Stage 2: Chunk
    print("Stage 2: Chunking")
    chunk_artifacts = Stage02_ChunkArtifacts(
        chunks=[{'chunk_id': 'c1', 'text': 'chunk 1'}],
        chunking_config={'max_size': 1000},
        total_chunks=1,
        avg_chunk_size=500.0,
        chunk_size_distribution={'0-500': 1}
    )
    artifacts.save_stage02_chunks(chunk_artifacts)

    # Stage 3: Embed
    print("Stage 3: Embedding")
    embedding_artifacts = Stage03_EmbeddingArtifacts(
        embedding_records=[{'chunk_id': 'c1', 'text': 'chunk 1'}],
        model_name='text-embedding-3-small',
        embedding_dim=1536,
        embedding_timestamp=datetime.now().isoformat()
    )
    artifacts.save_stage03_embeddings(embedding_artifacts)

    # Stage 4: Enrich
    print("Stage 4: Enrichment")
    enrichment_artifacts = Stage04_EnrichmentArtifacts(
        topics={'topic_1': 0.8},
        summaries={'doc_summary': 'Document about...'},
        entities=[{'text': 'John', 'type': 'PERSON'}],
        keywords=['machine learning', 'AI']
    )
    artifacts.save_stage04_enrichment(enrichment_artifacts)

    print("\n=== Reprocessing Scenarios ===\n")

    # Scenario 1: Re-chunk with different config (skip extraction)
    print("Scenario: Re-chunk with new config")
    doc_ir = artifacts.load_document_ir()
    print(f"  Loaded DocumentIR: {len(doc_ir.get('blocks', []))} blocks")
    print("  → Skip expensive extraction, start from IR")

    # Scenario 2: Re-embed with new model (skip extraction + chunking)
    print("\nScenario: Re-embed with new model")
    chunks = artifacts.load_chunks()
    print(f"  Loaded chunks: {len(chunks)}")
    print("  → Reuse chunks, just re-embed")

    # Scenario 3: Check if extraction needed
    print("\nScenario: Check if extraction needed")
    current_hash = hashlib.sha256(b'file_content').hexdigest()
    can_skip = artifacts.can_skip_extraction(current_hash)
    print(f"  Can skip extraction: {can_skip}")


if __name__ == "__main__":
    example_pipeline_with_persistence()
