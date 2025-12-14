"""
Document to Vector Service

A unified service that handles the complete pipeline:
Document → Text → Normalized Text → Chunks → Embeddings → Qdrant DB

Simply upload a document and it's automatically processed and stored in Qdrant.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import uuid
from datetime import datetime

# Import pipeline components
from document_processor import extract_text_from_document
from metadata_aware_normalizer import normalize_with_metadata, NormalizationConfig
from enterprise_chunking_pipeline import chunk_with_normalization, ChunkingConfig
from embedding_preparation import prepare_for_embedding
from qdrant_storage import QdrantStorage, QdrantConfig, setup_qdrant_collection

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for the Document to Vector Service."""

    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    qdrant_api_key: Optional[str] = None

    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"
    vector_size: int = 384

    # Normalization settings
    remove_toc_pages: bool = True
    protect_headings: bool = True
    protect_tables: bool = True
    protect_code_blocks: bool = True
    detect_multi_column: bool = True

    # Chunking settings
    max_chunk_size: int = 1000
    target_chunk_size: int = 500
    enable_overlap: bool = True
    overlap_size: int = 100
    respect_page_boundaries: bool = True
    keep_tables_intact: bool = True

    # Processing settings
    deduplicate_chunks: bool = True
    aggressive_text_cleaning: bool = False

    # Pipeline version
    version: str = "1.0"


@dataclass
class ProcessingResult:
    """Result of document processing."""
    success: bool
    doc_id: str
    file_name: str

    # Statistics
    pages_extracted: int = 0
    chunks_created: int = 0
    unique_chunks: int = 0
    duplicates_removed: int = 0
    vectors_stored: int = 0

    # Processing time
    extraction_time: float = 0.0
    normalization_time: float = 0.0
    chunking_time: float = 0.0
    embedding_time: float = 0.0
    storage_time: float = 0.0
    total_time: float = 0.0

    # Error information
    error_message: Optional[str] = None
    error_stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'doc_id': self.doc_id,
            'file_name': self.file_name,
            'statistics': {
                'pages_extracted': self.pages_extracted,
                'chunks_created': self.chunks_created,
                'unique_chunks': self.unique_chunks,
                'duplicates_removed': self.duplicates_removed,
                'vectors_stored': self.vectors_stored,
            },
            'timing': {
                'extraction_time': f"{self.extraction_time:.2f}s",
                'normalization_time': f"{self.normalization_time:.2f}s",
                'chunking_time': f"{self.chunking_time:.2f}s",
                'embedding_time': f"{self.embedding_time:.2f}s",
                'storage_time': f"{self.storage_time:.2f}s",
                'total_time': f"{self.total_time:.2f}s",
            },
            'error': {
                'message': self.error_message,
                'stage': self.error_stage
            } if not self.success else None
        }


class DocumentToVectorService:
    """
    Unified service for processing documents to vector storage.
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()

        # 🔥 Load environment variables (Docker override)
        import os
        env_qdrant_url = os.getenv("QDRANT_URL")
        env_model_name = os.getenv("EMBEDDING_MODEL")

        if env_qdrant_url:
            self.config.qdrant_url = env_qdrant_url

        if env_model_name:
            self.config.embedding_model_name = env_model_name

        logger.info(f"Using Qdrant URL: {self.config.qdrant_url}")
        logger.info(f"Using Embedding Model: {self.config.embedding_model_name}")

        # Initialize embedding model
        if EMBEDDING_MODEL_AVAILABLE:
            logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
            logger.info("✅ Embedding model loaded")
        else:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

        # Initialize Qdrant storage
        self.qdrant_config = QdrantConfig(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            collection_name=self.config.qdrant_collection,
            vector_size=self.config.vector_size,
            batch_size=100
        )

        self.storage = QdrantStorage(self.qdrant_config)

        # Setup collection
        self._ensure_collection_exists()

        logger.info("✅ Document to Vector Service initialized")


    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with proper indexes."""
        try:
            self.storage.collection_manager.create_collection(recreate=False)
            logger.info("✅ Qdrant collection ready")
        except Exception as e:
            logger.error(f"Failed to setup Qdrant collection: {e}")
            raise

    def process_document(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        custom_metadata: Optional[Dict] = None
    ) -> ProcessingResult:
        """
        Process a document through the complete pipeline.

        Args:
            file_path: Path to document file
            doc_id: Optional document ID (generated if not provided)
            custom_metadata: Optional custom metadata to attach

        Returns:
            ProcessingResult with statistics and timing
        """
        import time

        start_time = time.time()

        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        file_name = Path(file_path).name

        result = ProcessingResult(
            success=False,
            doc_id=doc_id,
            file_name=file_name
        )

        try:
            # ================================================================
            # STAGE 1: EXTRACT DOCUMENT
            # ================================================================
            logger.info(f"[{doc_id}] Stage 1/5: Extracting document...")
            stage_start = time.time()

            extraction_result = extract_text_from_document(
                file_path,
                extract_metadata=True
            )

            result.pages_extracted = extraction_result.metadata.total_pages
            result.extraction_time = time.time() - stage_start

            logger.info(f"[{doc_id}] ✅ Extracted {result.pages_extracted} pages in {result.extraction_time:.2f}s")

            # ================================================================
            # STAGE 2: NORMALIZE TEXT
            # ================================================================
            logger.info(f"[{doc_id}] Stage 2/5: Normalizing text...")
            stage_start = time.time()

            norm_config = NormalizationConfig(
                # Structural
                normalize_line_breaks=True,
                remove_hyphen_line_breaks=True,
                collapse_whitespace=True,
                unicode_normalize=True,

                # Noise removal
                remove_urls=True,
                remove_page_numbers=True,
                remove_headers_footers=True,
                remove_toc_pages=self.config.remove_toc_pages,

                # Protection
                protect_headings=self.config.protect_headings,
                protect_tables=self.config.protect_tables,
                protect_code_blocks=self.config.protect_code_blocks,

                # Advanced
                detect_multi_column=self.config.detect_multi_column,
                preserve_hierarchy=True,
                add_page_markers=True,
            )

            normalized_text, page_results, norm_stats = normalize_with_metadata(
                extraction_result,
                norm_config
            )

            result.normalization_time = time.time() - stage_start

            logger.info(f"[{doc_id}] ✅ Normalized in {result.normalization_time:.2f}s ({norm_stats['char_reduction_percent']:.1f}% reduction)")

            # ================================================================
            # STAGE 3: CHUNK DOCUMENT
            # ================================================================
            logger.info(f"[{doc_id}] Stage 3/5: Chunking document...")
            stage_start = time.time()

            chunk_config = ChunkingConfig(
                max_chunk_size=self.config.max_chunk_size,
                target_chunk_size=self.config.target_chunk_size,
                enable_overlap=self.config.enable_overlap,
                overlap_size=self.config.overlap_size,
                overlap_strategy="sentence",
                respect_page_boundaries=self.config.respect_page_boundaries,
                keep_tables_intact=self.config.keep_tables_intact,
                keep_code_blocks_intact=self.config.protect_code_blocks,
            )

            chunks = chunk_with_normalization(
                extraction_result,
                normalized_text,
                chunk_config
            )

            result.chunks_created = len(chunks)
            result.chunking_time = time.time() - stage_start

            logger.info(f"[{doc_id}] ✅ Created {result.chunks_created} chunks in {result.chunking_time:.2f}s")

            # ================================================================
            # STAGE 4: PREPARE FOR EMBEDDING
            # ================================================================
            logger.info(f"[{doc_id}] Stage 4/5: Preparing embeddings...")
            stage_start = time.time()

            embedding_records, dedup_stats = prepare_for_embedding(
                chunks,
                deduplicate=self.config.deduplicate_chunks,
                aggressive_cleaning=self.config.aggressive_text_cleaning,
                version=self.config.version
            )

            result.unique_chunks = dedup_stats.unique_chunks
            result.duplicates_removed = dedup_stats.duplicate_chunks

            # Generate embeddings
            texts = [record.embedding_input_text for record in embedding_records]
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Convert to list of lists
            embeddings = [emb.tolist() for emb in embeddings]

            result.embedding_time = time.time() - stage_start

            logger.info(f"[{doc_id}] ✅ Generated {len(embeddings)} embeddings in {result.embedding_time:.2f}s")
            logger.info(f"[{doc_id}]    Deduplication: {dedup_stats.deduplication_rate:.1f}%")

            # ================================================================
            # STAGE 5: STORE IN QDRANT
            # ================================================================
            logger.info(f"[{doc_id}] Stage 5/5: Storing in Qdrant...")
            stage_start = time.time()

            # Add custom metadata if provided
            if custom_metadata:
                for record in embedding_records:
                    record.embedding_metadata.update(custom_metadata)

            upload_stats = self.storage.store_embeddings(
                embedding_records,
                embeddings,
                show_progress=False
            )

            result.vectors_stored = upload_stats['uploaded']
            result.storage_time = time.time() - stage_start

            logger.info(f"[{doc_id}] ✅ Stored {result.vectors_stored} vectors in {result.storage_time:.2f}s")

            # ================================================================
            # SUCCESS
            # ================================================================
            result.success = True
            result.total_time = time.time() - start_time

            logger.info(f"[{doc_id}] ✅ COMPLETE: {file_name} processed in {result.total_time:.2f}s")
            logger.info(f"[{doc_id}]    Pages: {result.pages_extracted}")
            logger.info(f"[{doc_id}]    Chunks: {result.chunks_created} → {result.unique_chunks} unique")
            logger.info(f"[{doc_id}]    Vectors: {result.vectors_stored} stored in Qdrant")

        except Exception as e:
            # Handle errors
            result.success = False
            result.error_message = str(e)
            result.total_time = time.time() - start_time

            # Determine stage where error occurred
            if result.pages_extracted == 0:
                result.error_stage = "extraction"
            elif result.chunks_created == 0:
                result.error_stage = "normalization_or_chunking"
            elif result.unique_chunks == 0:
                result.error_stage = "embedding_preparation"
            else:
                result.error_stage = "storage"

            logger.error(f"[{doc_id}] ❌ FAILED at {result.error_stage}: {e}")

        return result

    def process_multiple_documents(
        self,
        file_paths: List[str],
        show_progress: bool = True
    ) -> List[ProcessingResult]:
        """
        Process multiple documents.

        Args:
            file_paths: List of document paths
            show_progress: Whether to show progress

        Returns:
            List of ProcessingResult objects
        """
        results = []

        logger.info(f"Processing {len(file_paths)} documents...")

        for i, file_path in enumerate(file_paths, 1):
            if show_progress:
                logger.info(f"\n{'='*80}")
                logger.info(f"Document {i}/{len(file_paths)}: {Path(file_path).name}")
                logger.info(f"{'='*80}")

            result = self.process_document(file_path)
            results.append(result)

            if show_progress:
                if result.success:
                    logger.info(f"✅ Success: {result.vectors_stored} vectors stored")
                else:
                    logger.error(f"❌ Failed: {result.error_message}")

        # Summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_vectors = sum(r.vectors_stored for r in results)

        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total documents: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total vectors stored: {total_vectors}")

        return results

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for documents.

        Args:
            query: Search query text
            limit: Number of results
            filters: Optional metadata filters
            score_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Search in Qdrant
        results = self.storage.search(
            query_vector=query_embedding,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold
        )

        return results

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        return self.storage.collection_manager.collection_info()


# Convenience functions

def process_document_simple(
    file_path: str,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "documents"
) -> ProcessingResult:
    """
    Simple function to process a document.

    Args:
        file_path: Path to document
        qdrant_url: Qdrant server URL
        collection_name: Collection name

    Returns:
        ProcessingResult
    """
    config = ServiceConfig(
        qdrant_url=qdrant_url,
        qdrant_collection=collection_name
    )

    service = DocumentToVectorService(config)
    return service.process_document(file_path)


# Main execution
if __name__ == "__main__":
    import sys
    import json

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 80)
    print("DOCUMENT TO VECTOR SERVICE")
    print("=" * 80)

    if len(sys.argv) < 2:
        print("\nUsage: python document_to_vector_service.py <document_path> [collection_name]")
        print("\nExample:")
        print("  python document_to_vector_service.py document.pdf")
        print("  python document_to_vector_service.py document.pdf my_collection")
        print("\nSupported formats: PDF, DOCX, TXT")
        print("\nThis will:")
        print("  1. Extract text from document")
        print("  2. Normalize text (metadata-aware)")
        print("  3. Chunk with semantic windowing")
        print("  4. Generate embeddings")
        print("  5. Store in Qdrant vector database")
        sys.exit(1)

    file_path = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "documents"

    # Check file exists
    if not Path(file_path).exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)

    # Process document
    config = ServiceConfig(
        qdrant_collection=collection_name,
        max_chunk_size=1000,
        enable_overlap=True,
        deduplicate_chunks=True
    )

    service = DocumentToVectorService(config)
    result = service.process_document(file_path)

    # Display result
    print("\n" + "=" * 80)
    print("PROCESSING RESULT")
    print("=" * 80)
    print(json.dumps(result.to_dict(), indent=2))

    if result.success:
        print("\n✅ SUCCESS!")
        print(f"\nYour document is now searchable in Qdrant collection '{collection_name}'")
        print("\nTest search:")
        print(f'  results = service.search("your query here", limit=5)')
    else:
        print("\n❌ FAILED!")
        print(f"Error: {result.error_message}")
        sys.exit(1)
