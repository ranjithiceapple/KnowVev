"""
Embedding Preparation Module

Prepares chunks for embedding and vector storage (e.g., Qdrant, Pinecone).
Handles deduplication, metadata formatting, and versioning.

Features:
- Clean text preparation for embeddings
- Hash-based deduplication
- Qdrant-compatible metadata
- Re-indexing without re-embedding
- Version tracking
"""

import hashlib
import re
import logging
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import time
from logger_config import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingRecord:
    """
    Complete embedding record ready for vector storage.
    Compatible with Qdrant, Pinecone, Weaviate, etc.
    """
    # Unique identifiers
    chunk_id: str                    # Original chunk ID
    embedding_id: str                # UUID for vector DB
    embedding_hash: str              # SHA256 hash for deduplication

    # Embedding input
    embedding_input_text: str        # Clean text for embedding model

    # Source tracking
    doc_id: str
    file_name: str
    page_number_start: int
    page_number_end: int

    # Section hierarchy (for filtering)
    section_title: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    heading_path_str: str = ""       # Flattened for easier querying

    # Chunk position
    chunk_index: int = 0
    total_chunks: int = 0

    # Content metrics
    chunk_char_len: int = 0
    chunk_word_count: int = 0
    chunk_token_count: Optional[int] = None

    # Boundary information
    boundary_type: str = "paragraph"
    has_overlap: bool = False

    # Content flags (for filtering)
    contains_tables: bool = False
    contains_code: bool = False
    contains_bullets: bool = False
    has_urls: bool = False

    # Full metadata (for re-indexing)
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)

    # Version tracking
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_pipeline: str = "default"

    # Optional: store original normalized text
    normalized_text: Optional[str] = None  # For debugging/comparison

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """
        Convert to Qdrant-compatible payload.
        Optimized for filtering and field-based queries.
        """
        # Extract project_id from embedding_metadata if present (for multi-tenancy)
        project_id = self.embedding_metadata.get('project_id', None)

        payload = {
            # IDs
            'chunk_id': self.chunk_id,
            'embedding_id': self.embedding_id,
            'embedding_hash': self.embedding_hash,

            # Source
            'doc_id': self.doc_id,
            'file_name': self.file_name,
            'page_start': self.page_number_start,
            'page_end': self.page_number_end,
            'page_range': f"{self.page_number_start}-{self.page_number_end}",

            # Hierarchy (for filtering)
            'section_title': self.section_title or "",
            'heading_path': self.heading_path,
            'heading_path_str': self.heading_path_str,
            'hierarchy_level': len(self.heading_path),

            # Position
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,

            # Metrics
            'char_len': self.chunk_char_len,
            'word_count': self.chunk_word_count,
            'token_count': self.chunk_token_count or 0,

            # Boundary
            'boundary_type': self.boundary_type,
            'has_overlap': self.has_overlap,

            # Content flags (for filtering)
            'contains_tables': self.contains_tables,
            'contains_code': self.contains_code,
            'contains_bullets': self.contains_bullets,
            'has_urls': self.has_urls,

            # Text content
            'text': self.embedding_input_text,

            # Full metadata (for re-indexing)
            'metadata': self.embedding_metadata,

            # Version
            'version': self.version,
            'created_at': self.created_at,
            'pipeline': self.processing_pipeline,
        }

        # Add project_id as top-level field for efficient filtering (multi-tenancy)
        if project_id:
            payload['project_id'] = project_id

        return payload

    def to_pinecone_metadata(self) -> Dict[str, Any]:
        """
        Convert to Pinecone-compatible metadata.
        Note: Pinecone has limitations on metadata types.
        """
        return {
            'chunk_id': self.chunk_id,
            'embedding_hash': self.embedding_hash,
            'doc_id': self.doc_id,
            'file_name': self.file_name,
            'page_start': self.page_number_start,
            'page_end': self.page_number_end,
            'section_title': self.section_title or "",
            'heading_path_str': self.heading_path_str,
            'chunk_index': self.chunk_index,
            'word_count': self.chunk_word_count,
            'boundary_type': self.boundary_type,
            'contains_tables': self.contains_tables,
            'contains_code': self.contains_code,
            'text': self.embedding_input_text,
            'version': self.version,
        }


@dataclass
class DeduplicationStats:
    """Statistics from deduplication process."""
    total_chunks: int = 0
    unique_chunks: int = 0
    duplicate_chunks: int = 0
    deduplication_rate: float = 0.0
    duplicate_groups: Dict[str, List[str]] = field(default_factory=dict)


class TextCleaner:
    """
    Cleans text for optimal embedding quality.
    """

    def __init__(self):
        # Patterns to clean
        self.page_marker_pattern = re.compile(r'<<<PAGE_\d+>>>')
        self.hierarchy_marker_pattern = re.compile(r'<<<HIERARCHY_L\d+>>>')
        self.protected_marker_pattern = re.compile(r'<<<PROTECTED_\w+_\d+>>>')
        self.excessive_whitespace = re.compile(r'\s{3,}')
        self.multiple_newlines = re.compile(r'\n{3,}')

        # Control characters to remove
        self.control_chars = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]')

    def clean_for_embedding(self, text: str, aggressive: bool = False) -> str:
        """
        Clean text for embedding models.

        Args:
            text: Input text
            aggressive: If True, apply more aggressive cleaning

        Returns:
            Cleaned text optimized for embeddings
        """
        # Step 1: Remove markers
        text = self.page_marker_pattern.sub('', text)
        text = self.hierarchy_marker_pattern.sub('', text)
        text = self.protected_marker_pattern.sub('', text)

        # Step 2: Remove control characters
        text = self.control_chars.sub('', text)

        # Step 3: Normalize whitespace
        text = self.excessive_whitespace.sub(' ', text)
        text = self.multiple_newlines.sub('\n\n', text)

        if aggressive:
            # Step 4: Remove special formatting (aggressive mode)
            # Remove markdown formatting
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
            text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
            text = re.sub(r'`(.+?)`', r'\1', text)        # Code

            # Remove excessive punctuation
            text = re.sub(r'\.{3,}', '...', text)
            text = re.sub(r'-{3,}', '---', text)

        # Step 5: Final cleanup
        text = text.strip()

        # Step 6: Normalize line breaks
        text = '\n'.join(line.strip() for line in text.split('\n'))

        return text


class HashGenerator:
    """
    Generates content hashes for deduplication.
    """

    @staticmethod
    def generate_hash(text: str, algorithm: str = 'sha256') -> str:
        """
        Generate hash of text content.

        Args:
            text: Input text
            algorithm: Hash algorithm ('md5', 'sha256', 'sha1')

        Returns:
            Hex digest of hash
        """
        # Normalize text before hashing
        normalized = text.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)

        # Generate hash
        if algorithm == 'md5':
            return hashlib.md5(normalized.encode('utf-8')).hexdigest()
        elif algorithm == 'sha1':
            return hashlib.sha1(normalized.encode('utf-8')).hexdigest()
        else:  # sha256 (default)
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    @staticmethod
    def generate_semantic_hash(text: str, ngram_size: int = 3) -> str:
        """
        Generate semantic hash using n-grams.
        More robust to minor text variations.

        Args:
            text: Input text
            ngram_size: Size of n-grams

        Returns:
            Semantic hash
        """
        # Normalize
        normalized = text.lower().strip()
        words = normalized.split()

        # Generate n-grams
        ngrams = []
        for i in range(len(words) - ngram_size + 1):
            ngram = ' '.join(words[i:i + ngram_size])
            ngrams.append(ngram)

        # Hash n-grams
        ngram_hashes = sorted([hashlib.md5(ng.encode()).hexdigest()[:8] for ng in ngrams])

        # Combine into final hash
        combined = ''.join(ngram_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()


class MetadataBuilder:
    """
    Builds comprehensive metadata for embeddings.
    """

    @staticmethod
    def build_embedding_metadata(chunk_metadata) -> Dict[str, Any]:
        """
        Build metadata dictionary from ChunkMetadata.

        Args:
            chunk_metadata: ChunkMetadata object

        Returns:
            Metadata dictionary
        """
        return {
            # Source identification
            'source': {
                'doc_id': chunk_metadata.doc_id,
                'file_name': chunk_metadata.file_name,
                'file_type': Path(chunk_metadata.file_name).suffix.lstrip('.'),
            },

            # Location
            'location': {
                'page_start': chunk_metadata.page_number_start,
                'page_end': chunk_metadata.page_number_end,
                'chunk_index': chunk_metadata.chunk_index,
                'total_chunks': chunk_metadata.total_chunks,
            },

            # Hierarchy
            'hierarchy': {
                'section_title': chunk_metadata.section_title,
                'heading_path': chunk_metadata.heading_path,
                'level': len(chunk_metadata.heading_path),
            },

            # Content properties
            'content': {
                'char_length': chunk_metadata.chunk_char_len,
                'word_count': chunk_metadata.chunk_word_count,
                'token_count': chunk_metadata.chunk_token_count,
                'boundary_type': chunk_metadata.boundary_type,
                'has_overlap': chunk_metadata.has_overlap,
            },

            # Content flags
            'flags': {
                'contains_tables': chunk_metadata.contains_tables,
                'contains_code': chunk_metadata.contains_code,
                'contains_bullets': chunk_metadata.contains_bullets,
                'has_urls': len(chunk_metadata.urls_in_chunk) > 0,
            },

            # Timestamps
            'timestamps': {
                'created_at': chunk_metadata.created_at,
            },
        }


class EmbeddingPreparationPipeline:
    """
    Main pipeline for preparing chunks for embedding.
    """

    def __init__(
        self,
        aggressive_cleaning: bool = False,
        hash_algorithm: str = 'sha256',
        use_semantic_hash: bool = False,
        version: str = "1.0",
        pipeline_name: str = "default"
    ):
        logger.info("Initializing EmbeddingPreparationPipeline")
        self.text_cleaner = TextCleaner()
        self.hash_generator = HashGenerator()
        self.metadata_builder = MetadataBuilder()

        self.aggressive_cleaning = aggressive_cleaning
        self.hash_algorithm = hash_algorithm
        self.use_semantic_hash = use_semantic_hash
        self.version = version
        self.pipeline_name = pipeline_name
        logger.info(
            f"EmbeddingPreparationPipeline initialized - Pipeline: {pipeline_name}, "
            f"Version: {version}, Aggressive cleaning: {aggressive_cleaning}, "
            f"Hash algorithm: {hash_algorithm}, Semantic hash: {use_semantic_hash}"
        )

    def prepare_chunks(
        self,
        chunks: List,
        deduplicate: bool = True,
        keep_normalized_text: bool = False
    ) -> Tuple[List[EmbeddingRecord], DeduplicationStats]:
        """
        Prepare chunks for embedding.

        Args:
            chunks: List of ChunkMetadata objects
            deduplicate: Whether to remove duplicates
            keep_normalized_text: Whether to keep original normalized text

        Returns:
            Tuple of (embedding_records, deduplication_stats)
        """
        start_time = time.time()
        logger.info(
            f"Starting chunk preparation - Chunks: {len(chunks)}, "
            f"Deduplicate: {deduplicate}, Keep text: {keep_normalized_text}"
        )

        embedding_records = []
        seen_hashes: Dict[str, str] = {}  # hash -> chunk_id
        duplicate_groups: Dict[str, List[str]] = {}

        for chunk in chunks:
            # Step 1: Clean text for embedding
            embedding_input_text = self.text_cleaner.clean_for_embedding(
                chunk.normalized_text,
                aggressive=self.aggressive_cleaning
            )

            # Step 2: Generate hash
            if self.use_semantic_hash:
                embedding_hash = self.hash_generator.generate_semantic_hash(embedding_input_text)
            else:
                embedding_hash = self.hash_generator.generate_hash(
                    embedding_input_text,
                    algorithm=self.hash_algorithm
                )

            # Step 3: Check for duplicates
            if deduplicate and embedding_hash in seen_hashes:
                original_chunk_id = seen_hashes[embedding_hash]
                logger.debug(f"Duplicate found: {chunk.chunk_id} matches {original_chunk_id}")

                # Track duplicate group
                if embedding_hash not in duplicate_groups:
                    duplicate_groups[embedding_hash] = [original_chunk_id]
                duplicate_groups[embedding_hash].append(chunk.chunk_id)

                continue  # Skip duplicate

            # Step 4: Build metadata
            embedding_metadata = self.metadata_builder.build_embedding_metadata(chunk)

            # Step 5: Create embedding record
            record = EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                embedding_id=f"{chunk.doc_id}_emb_{chunk.chunk_index:04d}",
                embedding_hash=embedding_hash,
                embedding_input_text=embedding_input_text,
                doc_id=chunk.doc_id,
                file_name=chunk.file_name,
                page_number_start=chunk.page_number_start,
                page_number_end=chunk.page_number_end,
                section_title=chunk.section_title,
                heading_path=chunk.heading_path,
                heading_path_str=' > '.join(chunk.heading_path),
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                chunk_char_len=chunk.chunk_char_len,
                chunk_word_count=chunk.chunk_word_count,
                chunk_token_count=chunk.chunk_token_count,
                boundary_type=chunk.boundary_type,
                has_overlap=chunk.has_overlap,
                contains_tables=chunk.contains_tables,
                contains_code=chunk.contains_code,
                contains_bullets=chunk.contains_bullets,
                has_urls=len(chunk.urls_in_chunk) > 0,
                embedding_metadata=embedding_metadata,
                version=self.version,
                processing_pipeline=self.pipeline_name,
                normalized_text=chunk.normalized_text if keep_normalized_text else None
            )

            embedding_records.append(record)
            seen_hashes[embedding_hash] = chunk.chunk_id

        # Step 6: Calculate deduplication stats
        stats = DeduplicationStats(
            total_chunks=len(chunks),
            unique_chunks=len(embedding_records),
            duplicate_chunks=len(chunks) - len(embedding_records),
            deduplication_rate=(len(chunks) - len(embedding_records)) / len(chunks) * 100 if chunks else 0,
            duplicate_groups=duplicate_groups
        )

        duration = time.time() - start_time
        logger.info(
            f"Chunk preparation complete - Total: {stats.total_chunks}, "
            f"Unique: {stats.unique_chunks}, Duplicates: {stats.duplicate_chunks} "
            f"({stats.deduplication_rate:.1f}% deduplication rate), "
            f"Duration: {duration:.2f}s"
        )

        return embedding_records, stats

    def prepare_single_chunk(self, chunk) -> EmbeddingRecord:
        """
        Prepare a single chunk for embedding.

        Args:
            chunk: ChunkMetadata object

        Returns:
            EmbeddingRecord
        """
        records, _ = self.prepare_chunks([chunk], deduplicate=False)
        return records[0] if records else None


class EmbeddingExporter:
    """
    Export embedding records to various formats.
    """

    @staticmethod
    def export_to_json(
        records: List[EmbeddingRecord],
        output_file: str,
        include_text: bool = True
    ):
        """
        Export to JSON file.

        Args:
            records: List of EmbeddingRecord objects
            output_file: Output file path
            include_text: Whether to include full text
        """
        data = []
        for record in records:
            record_dict = record.to_dict()
            if not include_text:
                record_dict.pop('embedding_input_text', None)
                record_dict.pop('normalized_text', None)
            data.append(record_dict)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_records': len(records),
                'version': records[0].version if records else "1.0",
                'records': data
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(records)} records to {output_file}")

    @staticmethod
    def export_to_qdrant_format(
        records: List[EmbeddingRecord],
        output_file: str
    ):
        """
        Export in Qdrant-compatible format.

        Args:
            records: List of EmbeddingRecord objects
            output_file: Output file path
        """
        points = []
        for i, record in enumerate(records):
            point = {
                'id': i,  # Qdrant will assign IDs, or use embedding_id
                'payload': record.to_qdrant_payload()
            }
            points.append(point)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'points': points
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(points)} points in Qdrant format to {output_file}")

    @staticmethod
    def export_to_csv(
        records: List[EmbeddingRecord],
        output_file: str
    ):
        """
        Export to CSV file.

        Args:
            records: List of EmbeddingRecord objects
            output_file: Output file path
        """
        import csv

        fieldnames = [
            'chunk_id', 'embedding_id', 'embedding_hash',
            'doc_id', 'file_name', 'page_start', 'page_end',
            'section_title', 'heading_path_str',
            'chunk_index', 'word_count',
            'contains_tables', 'contains_code', 'contains_bullets',
            'embedding_input_text'
        ]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                row = {
                    'chunk_id': record.chunk_id,
                    'embedding_id': record.embedding_id,
                    'embedding_hash': record.embedding_hash,
                    'doc_id': record.doc_id,
                    'file_name': record.file_name,
                    'page_start': record.page_number_start,
                    'page_end': record.page_number_end,
                    'section_title': record.section_title or '',
                    'heading_path_str': record.heading_path_str,
                    'chunk_index': record.chunk_index,
                    'word_count': record.chunk_word_count,
                    'contains_tables': record.contains_tables,
                    'contains_code': record.contains_code,
                    'contains_bullets': record.contains_bullets,
                    'embedding_input_text': record.embedding_input_text[:500]  # Truncate for CSV
                }
                writer.writerow(row)

        logger.info(f"Exported {len(records)} records to CSV: {output_file}")


# Convenience functions

def prepare_for_embedding(
    chunks: List,
    deduplicate: bool = True,
    aggressive_cleaning: bool = False,
    version: str = "1.0"
) -> Tuple[List[EmbeddingRecord], DeduplicationStats]:
    """
    Convenience function to prepare chunks for embedding.

    Args:
        chunks: List of ChunkMetadata objects
        deduplicate: Whether to remove duplicates
        aggressive_cleaning: Whether to apply aggressive text cleaning
        version: Version string for tracking

    Returns:
        Tuple of (embedding_records, stats)
    """
    pipeline = EmbeddingPreparationPipeline(
        aggressive_cleaning=aggressive_cleaning,
        version=version
    )
    return pipeline.prepare_chunks(chunks, deduplicate=deduplicate)


# Import Path for metadata builder
from pathlib import Path


if __name__ == "__main__":
    print("Embedding Preparation Module")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✅ Clean text preparation for embeddings")
    print("  ✅ Hash-based deduplication")
    print("  ✅ Qdrant/Pinecone compatible metadata")
    print("  ✅ Re-indexing without re-embedding")
    print("  ✅ Version tracking")
    print("\nUsage example:")
    print("""
from enterprise_chunking_pipeline import chunk_document_simple
from embedding_preparation import prepare_for_embedding, EmbeddingExporter

# Get chunks
chunks = chunk_document_simple(extraction_result)

# Prepare for embedding
embedding_records, stats = prepare_for_embedding(
    chunks,
    deduplicate=True,
    version="1.0"
)

print(f"Unique chunks: {stats.unique_chunks}")
print(f"Duplicates removed: {stats.duplicate_chunks}")

# Use records
for record in embedding_records:
    # Generate embedding
    embedding = model.encode(record.embedding_input_text)

    # Store in vector DB
    qdrant_client.upsert(
        collection_name="documents",
        points=[{
            'id': record.embedding_id,
            'vector': embedding,
            'payload': record.to_qdrant_payload()
        }]
    )

# Export
EmbeddingExporter.export_to_json(embedding_records, 'embeddings.json')
EmbeddingExporter.export_to_qdrant_format(embedding_records, 'qdrant_points.json')
    """)
