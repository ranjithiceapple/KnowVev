"""
Document to Vector Service

A unified service that handles the complete pipeline:
Document → Text → Normalized Text → Chunks → Embeddings → Qdrant DB

Simply upload a document and it's automatically processed and stored in Qdrant.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import uuid
from datetime import datetime
import time
from logger_config import get_logger

# Import pipeline components
from document_processor_llm import extract_text_from_document
from metadata_aware_normalizer import normalize_with_metadata, NormalizationConfig
from enterprise_chunking_pipeline import (
    chunk_with_normalization,
    ChunkingConfig,
    EnterpriseChunkingPipeline
)
from embedding_preparation import prepare_for_embedding
from qdrant_storage import QdrantStorage, QdrantConfig
from document_summarizer import DocumentSummarizer


# Import OpenSearch for hybrid search
try:
    from opensearch_keyword_store import OpenSearchKeywordStore
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logger = None  # Will be set after logger initialization

import os
import sys
import logging
from pathlib import Path

# Disable tqdm progress bars globally
os.environ["TQDM_DISABLE"] = "1"

# Disable huggingface_hub progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Disable transformers progress bars and reduce verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress specific loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
    logger.info("SentenceTransformer module loaded successfully")
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")


@dataclass
class ServiceConfig:
    """Configuration for the Document to Vector Service."""

    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    qdrant_api_key: Optional[str] = None

    # OpenSearch settings (for hybrid keyword search)
    opensearch_enabled: bool = True
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_username: Optional[str] = "admin"           # Default OpenSearch username
    opensearch_password: Optional[str] = "ArivurAI@123"    # Your OpenSearch Docker password
    opensearch_use_ssl: bool = True                        # OpenSearch Docker uses SSL by default
    opensearch_verify_certs: bool = False                  # Disable cert verification for self-signed certs
    opensearch_index: str = "document_keywords"

    # Hybrid chunking settings (for OpenSearch)
    generate_heading_chunks: bool = True
    generate_clause_chunks: bool = True
    generate_metadata_chunks: bool = True
    generate_summary_chunks: bool = True

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

    # Document summary settings
    generate_document_summary: bool = True
    summary_max_length: int = 2000
    summary_method: str = "hybrid"  # 'extractive', 'abstractive', 'hybrid'

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
    has_summary: bool = False
    summary_length: int = 0

    # OpenSearch hybrid search stats
    opensearch_indexed: bool = False
    opensearch_content_chunks: int = 0
    opensearch_heading_chunks: int = 0
    opensearch_clause_chunks: int = 0
    opensearch_metadata_chunks: int = 0
    opensearch_summary_chunks: int = 0
    opensearch_total_chunks: int = 0

    # Vector IDs (for Base Model tracking and deletion)
    vector_ids: List[str] = field(default_factory=list)
    embedding_ids: List[str] = field(default_factory=list)  # Alias for vector_ids

    # Processing time
    extraction_time: float = 0.0
    normalization_time: float = 0.0
    chunking_time: float = 0.0
    summary_time: float = 0.0
    embedding_time: float = 0.0
    storage_time: float = 0.0
    opensearch_time: float = 0.0
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
                'has_summary': self.has_summary,
                'summary_length': self.summary_length,
            },
            'opensearch': {
                'indexed': self.opensearch_indexed,
                'content_chunks': self.opensearch_content_chunks,
                'heading_chunks': self.opensearch_heading_chunks,
                'clause_chunks': self.opensearch_clause_chunks,
                'metadata_chunks': self.opensearch_metadata_chunks,
                'summary_chunks': self.opensearch_summary_chunks,
                'total_chunks': self.opensearch_total_chunks,
            },
            'vector_ids': self.vector_ids,  # For Base Model tracking
            'embedding_ids': self.embedding_ids,  # Alias
            'timing': {
                'extraction_time': f"{self.extraction_time:.2f}s",
                'normalization_time': f"{self.normalization_time:.2f}s",
                'chunking_time': f"{self.chunking_time:.2f}s",
                'summary_time': f"{self.summary_time:.2f}s",
                'embedding_time': f"{self.embedding_time:.2f}s",
                'topic_modeling_time': f"{self.topic_modeling_time:.2f}s",
                'storage_time': f"{self.storage_time:.2f}s",
                'opensearch_time': f"{self.opensearch_time:.2f}s",
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
        init_start = time.time()

        self.config = config or ServiceConfig()

        # Load environment variables (Docker override)
        import os
        env_qdrant_url = os.getenv("QDRANT_URL")
        env_model_name = os.getenv("EMBEDDING_MODEL")

        if env_qdrant_url:
            self.config.qdrant_url = env_qdrant_url
        if env_model_name:
            self.config.embedding_model_name = env_model_name

        # Initialize embedding model
        if EMBEDDING_MODEL_AVAILABLE:
            self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
        else:
            logger.error("sentence-transformers not available")
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
        self._ensure_collection_exists()

        # Initialize OpenSearch for hybrid keyword search
        self.opensearch_store = None
        if self.config.opensearch_enabled and OPENSEARCH_AVAILABLE:
            try:
                # Load OpenSearch settings from environment
                import os
                env_opensearch_host = os.getenv("OPENSEARCH_HOST")
                env_opensearch_port = os.getenv("OPENSEARCH_PORT")
                env_opensearch_user = os.getenv("OPENSEARCH_USERNAME")
                env_opensearch_pass = os.getenv("OPENSEARCH_PASSWORD")

                if env_opensearch_host:
                    self.config.opensearch_host = env_opensearch_host
                if env_opensearch_port:
                    self.config.opensearch_port = int(env_opensearch_port)
                if env_opensearch_user:
                    self.config.opensearch_username = env_opensearch_user
                if env_opensearch_pass:
                    self.config.opensearch_password = env_opensearch_pass

                self.opensearch_store = OpenSearchKeywordStore(
                    host=self.config.opensearch_host,
                    port=self.config.opensearch_port,
                    username=self.config.opensearch_username,
                    password=self.config.opensearch_password,
                    use_ssl=self.config.opensearch_use_ssl,
                    verify_certs=self.config.opensearch_verify_certs
                )

                # Ensure index exists
                self.opensearch_store.create_index(
                    self.config.opensearch_index,
                    delete_if_exists=False
                )
            except Exception as e:
                logger.warning(f"OpenSearch unavailable: {e}")
                self.opensearch_store = None
                self.config.opensearch_enabled = False
        elif self.config.opensearch_enabled and not OPENSEARCH_AVAILABLE:
            self.config.opensearch_enabled = False

        init_duration = time.time() - init_start

        # Single consolidated initialization log
        logger.info(
            f"Service initialized | qdrant={self.config.qdrant_url} "
            f"model={self.config.embedding_model_name} "
            f"opensearch={'enabled' if self.config.opensearch_enabled else 'disabled'} "
            f"duration={init_duration:.2f}s"
        )


    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with proper indexes."""
        try:
            self.storage.collection_manager.create_collection(recreate=False)
        except Exception as e:
            logger.error(f"Failed to setup Qdrant collection: {e}")
            raise

    def process_document(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        project_id: Optional[str] = None,
        custom_metadata: Optional[Dict] = None,
        enforce_contract: bool = True
    ) -> ProcessingResult:
        """
        Process a document through the complete pipeline.

        IMPORTANT: In production, use `ingest_document()` instead which enforces
        mandatory metadata validation.

        Args:
            file_path: Path to document file
            doc_id: Document ID (REQUIRED when enforce_contract=True)
            project_id: Project ID for multi-tenancy/data isolation (REQUIRED when enforce_contract=True)
            custom_metadata: Optional custom metadata to attach
            enforce_contract: If True (default), requires doc_id and project_id

        Returns:
            ProcessingResult with statistics and timing

        """
        start_time = time.time()

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
            stage_start = time.time()

            extraction_result = extract_text_from_document(
                file_path,
                extract_metadata=True
            )

            result.pages_extracted = extraction_result.metadata.total_pages
            result.extraction_time = time.time() - stage_start

            # ================================================================
            # STAGE 2: NORMALIZE TEXT
            # ================================================================
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

            # ================================================================
            # STAGE 3: CHUNK DOCUMENT
            # ================================================================
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
                chunk_config,
                project_id=project_id  # Pass project_id for multi-tenancy
            )

            for chunk in chunks:
                if not chunk.project_id:
                    chunk.project_id = project_id


            result.chunks_created = len(chunks)
            result.chunking_time = time.time() - stage_start

            # ================================================================
            # STAGE 3.5: GENERATE DOCUMENT SUMMARY (Virtual Chunk)
            # ================================================================
            # Extract document_title for citations
            document_title = None

            if self.config.generate_document_summary:
                stage_start = time.time()

                summarizer = DocumentSummarizer(
                    max_summary_length=self.config.summary_max_length,
                    method=self.config.summary_method
                )

                document_summary = summarizer.generate_summary(
                    chunks=chunks,
                    extraction_result=extraction_result,
                    doc_id=doc_id,
                    file_name=file_name
                )

                # CRITICAL: Capture document_title for citations
                document_title = document_summary.document_title

                summary_chunk = summarizer.create_summary_chunk(document_summary)
                chunks = [summary_chunk] + chunks

                result.summary_time = time.time() - stage_start
                result.has_summary = True
                result.summary_length = document_summary.summary_length

            # If no summary, derive title from file_name
            if not document_title:
                # Clean file_name: remove UUID prefix and extension
                document_title = file_name
                if '_' in document_title and len(document_title.split('_')[0]) == 36:
                    document_title = '_'.join(document_title.split('_')[1:])
                # Remove extension
                if '.' in document_title:
                    document_title = document_title.rsplit('.', 1)[0]
                # Replace underscores with spaces
                document_title = document_title.replace('_', ' ')

            # ================================================================
            # STAGE 4: PREPARE FOR EMBEDDING
            # ================================================================
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

            
            # ================================================================
            # STAGE 4: STORE IN QDRANT
            # ================================================================
            stage_start = time.time()

            # Prepare metadata to inject into all chunks
            metadata_to_inject = {
                'doc_id': doc_id,
                'document_title': document_title,
                'source': file_name,
            }

            # Add project_id if provided (for multi-tenancy)
            if project_id:
                metadata_to_inject['project_id'] = project_id

            # Add custom metadata if provided
            if custom_metadata:
                metadata_to_inject.update(custom_metadata)

            # Inject metadata into ALL embedding records
            for record in embedding_records:
                record.embedding_metadata.update(metadata_to_inject)
                if record.embedding_metadata.get('chunk_index', 0) < 0:
                    record.embedding_metadata['chunk_index'] = 0

            # Collect all embedding_ids for tracking
            vector_ids = [record.embedding_id for record in embedding_records]

            upload_stats = self.storage.store_embeddings(
                embedding_records,
                embeddings,
                show_progress=False
            )

            result.vectors_stored = upload_stats['uploaded']
            result.vector_ids = vector_ids
            result.embedding_ids = vector_ids
            result.storage_time = time.time() - stage_start

            # ================================================================
            # STAGE 5: INDEX IN OPENSEARCH (Hybrid Keyword Search)
            # ================================================================
            if self.opensearch_store and self.config.opensearch_enabled:
                stage_start = time.time()

                try:
                    # Create hybrid chunking config
                    hybrid_chunk_config = ChunkingConfig(
                        max_chunk_size=self.config.max_chunk_size,
                        target_chunk_size=self.config.target_chunk_size,
                        enable_overlap=self.config.enable_overlap,
                        overlap_size=self.config.overlap_size,
                        generate_heading_chunks=self.config.generate_heading_chunks,
                        generate_clause_chunks=self.config.generate_clause_chunks,
                        generate_metadata_chunks=self.config.generate_metadata_chunks,
                        generate_summary_chunks=self.config.generate_summary_chunks
                    )

                    # Create pipeline and generate hybrid chunks
                    hybrid_pipeline = EnterpriseChunkingPipeline(hybrid_chunk_config)
                    hybrid_chunks = hybrid_pipeline.generate_hybrid_chunks(
                        extraction_result,
                        doc_id=doc_id,
                        normalized_text=normalized_text,
                        project_id=project_id
                    )

                    # Inject metadata into ALL OpenSearch documents
                    for chunk_type, chunk_list in hybrid_chunks.items():
                        for chunk in chunk_list:
                            if project_id:
                                setattr(chunk, 'project_id', project_id)
                            setattr(chunk, 'document_title', document_title)
                            if hasattr(chunk, 'chunk_index') and chunk.chunk_index < 0:
                                chunk.chunk_index = 0

                    # Index hybrid chunks in OpenSearch
                    opensearch_stats = self.opensearch_store.index_hybrid_chunks(
                        self.config.opensearch_index,
                        hybrid_chunks,
                        doc_id=doc_id
                    )

                    # Update result statistics
                    result.opensearch_indexed = True
                    result.opensearch_content_chunks = len(hybrid_chunks.get('content', []))
                    result.opensearch_heading_chunks = len(hybrid_chunks.get('heading', []))
                    result.opensearch_clause_chunks = len(hybrid_chunks.get('clause', []))
                    result.opensearch_metadata_chunks = len(hybrid_chunks.get('metadata', []))
                    result.opensearch_summary_chunks = len(hybrid_chunks.get('summary', []))
                    result.opensearch_total_chunks = opensearch_stats.get('success', 0)
                    result.opensearch_time = time.time() - stage_start

                except Exception as e:
                    logger.error(f"OpenSearch indexing failed: {e}")
                    result.opensearch_time = time.time() - stage_start

            # ================================================================
            # SUCCESS - Single consolidated summary log
            # ================================================================
            result.success = True
            result.total_time = time.time() - start_time

            # Single consolidated completion log
            os_info = f" opensearch={result.opensearch_total_chunks}" if result.opensearch_indexed else ""
            logger.info(
                f"✅ {file_name} | pages={result.pages_extracted} chunks={result.unique_chunks} "
                f"vectors={result.vectors_stored}{os_info} | duration={result.total_time:.2f}s"
            )

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

            logger.error(f"❌ {file_name} | stage={result.error_stage} | error={e}")

        return result


    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
        expand_to_section: bool = False,
        project_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for documents.

        Args:
            query: Search query text
            limit: Number of results
            filters: Optional metadata filters
            score_threshold: Minimum similarity score
            expand_to_section: If True, expands matched chunks to include all chunks from the same section using heading_path
            project_id: Optional project ID filter for data isolation

        Returns:
            List of search results
        """
        # Build filters with project_id if provided
        search_filters = filters.copy() if filters else {}
        if project_id:
            search_filters['project_id'] = project_id

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Search in Qdrant
        results = self.storage.search(
            query_vector=query_embedding,
            limit=limit,
            filters=search_filters if search_filters else None,
            score_threshold=score_threshold
        )

        # Expand to full sections if requested
        if expand_to_section and results:
            results = self._expand_to_full_sections(results)

        return results

    def _expand_to_full_sections(self, initial_results: List[Dict]) -> List[Dict]:
        """
        Expand matched chunks to include all chunks from the same section.

        Uses heading_path to identify and fetch all sibling chunks in the same section,
        providing complete context instead of fragmented chunks.

        Args:
            initial_results: Initial search results with matched chunks

        Returns:
            Expanded results with full sections
        """
        logger.info(f"Expanding {len(initial_results)} matched chunks to full sections")

        # Track which sections we've already fetched to avoid duplicates
        seen_sections = set()
        expanded_results = []

        for result in initial_results:
            payload = result.get('payload', {})
            heading_path = payload.get('heading_path', [])
            doc_id = payload.get('doc_id')

            # Create a unique identifier for this section
            section_key = (doc_id, tuple(heading_path)) if heading_path else None

            # If we've already processed this section, skip
            if section_key and section_key in seen_sections:
                logger.debug(f"Section already expanded: {' > '.join(heading_path)}")
                continue

            if section_key:
                seen_sections.add(section_key)

            # Fetch all chunks with the same heading_path
            if heading_path:
                heading_path_str = ' > '.join(heading_path)
                logger.debug(f"Fetching full section for: {heading_path_str}")

                # Build filter to get all chunks in this section
                section_filter = {
                    'doc_id': doc_id,
                    'heading_path_str': heading_path_str
                }

                # Fetch all chunks in this section
                section_chunks = self.storage.filter_by_metadata(
                    filters=section_filter,
                    limit=1000  # High limit to get all chunks in section
                )

                logger.info(f"Expanded 1 match to {len(section_chunks)} chunks in section: {heading_path_str}")

                # Add all section chunks with the original match score for context
                for chunk in section_chunks:
                    expanded_result = {
                        'id': chunk.get('id'),
                        'score': result.get('score'),  # Preserve original match score
                        'payload': chunk.get('payload'),
                        'matched_chunk': chunk.get('payload', {}).get('chunk_id') == payload.get('chunk_id'),  # Mark which was the original match
                        'section_expansion': True  # Flag to indicate this is part of section expansion
                    }
                    expanded_results.append(expanded_result)
            else:
                # No heading_path, just add the original result
                logger.debug("No heading_path found, adding original result")
                result['section_expansion'] = False
                result['matched_chunk'] = True
                expanded_results.append(result)

        logger.info(f"Expansion complete: {len(initial_results)} matches → {len(expanded_results)} chunks")
        return expanded_results

    def filter_by_metadata(
        self,
        filters: Dict,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Filter documents by metadata only (no vector search).

        Args:
            filters: Metadata filters to apply
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of filtered results
        """
        return self.storage.filter_by_metadata(
            filters=filters,
            limit=limit,
            offset=offset
        )

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        return self.storage.collection_manager.collection_info()

    # =========================================================================
    # DOCUMENT DELETION METHODS
    # =========================================================================

    def delete_document(
        self,
        doc_id: str,
        delete_from_qdrant: bool = True,
        delete_from_opensearch: bool = True
    ) -> Dict[str, Any]:
        """
        Delete a document from both Qdrant and OpenSearch.

        This method removes all vectors and keywords associated with a document.
        Use this when a document is deleted from your system.

        Args:
            doc_id: Document ID to delete
            delete_from_qdrant: Whether to delete from Qdrant vector store
            delete_from_opensearch: Whether to delete from OpenSearch keyword store

        Returns:
            Dict with deletion statistics:
            - qdrant_deleted: Number of vectors deleted from Qdrant
            - opensearch_deleted: Number of chunks deleted from OpenSearch
            - success: Overall success status
        """
        logger.info(f"Deleting document: {doc_id}")

        result = {
            'doc_id': doc_id,
            'qdrant_deleted': 0,
            'opensearch_deleted': 0,
            'qdrant_success': False,
            'opensearch_success': False,
            'success': False
        }

        # Delete from Qdrant
        if delete_from_qdrant:
            try:
                logger.info(f"[{doc_id}] Deleting from Qdrant...")

                # Use scroll to find all vectors for this doc_id
                qdrant_filter = {"doc_id": doc_id}
                vectors_to_delete = self.storage.filter_by_metadata(
                    filters=qdrant_filter,
                    limit=10000  # High limit to get all
                )

                if vectors_to_delete:
                    # Extract vector IDs
                    vector_ids = [v.get('id') for v in vectors_to_delete if v.get('id')]

                    # Delete vectors
                    if vector_ids:
                        delete_result = self.storage.delete_by_ids(vector_ids)
                        result['qdrant_deleted'] = len(vector_ids)
                        result['qdrant_success'] = True
                        logger.info(f"[{doc_id}] ✅ Deleted {len(vector_ids)} vectors from Qdrant")
                else:
                    logger.info(f"[{doc_id}] No vectors found in Qdrant")
                    result['qdrant_success'] = True

            except Exception as e:
                logger.error(f"[{doc_id}] Failed to delete from Qdrant: {e}")
                result['qdrant_error'] = str(e)

        # Delete from OpenSearch
        if delete_from_opensearch and self.opensearch_store:
            try:
                logger.info(f"[{doc_id}] Deleting from OpenSearch...")

                opensearch_result = self.opensearch_store.delete_document(
                    self.config.opensearch_index,
                    doc_id
                )

                result['opensearch_deleted'] = opensearch_result.get('deleted', 0)
                result['opensearch_success'] = True
                logger.info(f"[{doc_id}] ✅ Deleted {result['opensearch_deleted']} chunks from OpenSearch")

            except Exception as e:
                logger.error(f"[{doc_id}] Failed to delete from OpenSearch: {e}")
                result['opensearch_error'] = str(e)
        elif delete_from_opensearch and not self.opensearch_store:
            logger.debug(f"[{doc_id}] OpenSearch not available, skipping")
            result['opensearch_success'] = True  # Not a failure if not configured

        # Overall success
        result['success'] = result['qdrant_success'] and result['opensearch_success']

        total_deleted = result['qdrant_deleted'] + result['opensearch_deleted']
        logger.info(f"[{doc_id}] Delete complete: {total_deleted} total items removed")

        return result

    def delete_document_by_file_name(
        self,
        file_name: str,
        delete_from_qdrant: bool = True,
        delete_from_opensearch: bool = True
    ) -> Dict[str, Any]:
        """
        Delete a document by file name from both Qdrant and OpenSearch.

        Args:
            file_name: File name to delete
            delete_from_qdrant: Whether to delete from Qdrant
            delete_from_opensearch: Whether to delete from OpenSearch

        Returns:
            Dict with deletion statistics
        """
        logger.info(f"Deleting document by file name: {file_name}")

        result = {
            'file_name': file_name,
            'qdrant_deleted': 0,
            'opensearch_deleted': 0,
            'success': False
        }

        # Delete from Qdrant
        if delete_from_qdrant:
            try:
                qdrant_filter = {"file_name": file_name}
                vectors_to_delete = self.storage.filter_by_metadata(
                    filters=qdrant_filter,
                    limit=10000
                )

                if vectors_to_delete:
                    vector_ids = [v.get('id') for v in vectors_to_delete if v.get('id')]
                    if vector_ids:
                        self.storage.delete_by_ids(vector_ids)
                        result['qdrant_deleted'] = len(vector_ids)

                logger.info(f"Deleted {result['qdrant_deleted']} vectors from Qdrant")

            except Exception as e:
                logger.error(f"Failed to delete from Qdrant: {e}")
                result['qdrant_error'] = str(e)

        # Delete from OpenSearch
        if delete_from_opensearch and self.opensearch_store:
            try:
                opensearch_result = self.opensearch_store.delete_by_file_name(
                    self.config.opensearch_index,
                    file_name
                )
                result['opensearch_deleted'] = opensearch_result.get('deleted', 0)
                logger.info(f"Deleted {result['opensearch_deleted']} chunks from OpenSearch")

            except Exception as e:
                logger.error(f"Failed to delete from OpenSearch: {e}")
                result['opensearch_error'] = str(e)

        result['success'] = 'qdrant_error' not in result and 'opensearch_error' not in result
        return result

    def check_document_exists(self, doc_id: str) -> Dict[str, Any]:
        """
        Check if a document exists in both Qdrant and OpenSearch.

        Args:
            doc_id: Document ID to check

        Returns:
            Dict with existence information
        """
        result = {
            'doc_id': doc_id,
            'exists_in_qdrant': False,
            'exists_in_opensearch': False,
            'qdrant_count': 0,
            'opensearch_count': 0
        }

        # Check Qdrant
        try:
            qdrant_filter = {"doc_id": doc_id}
            vectors = self.storage.filter_by_metadata(
                filters=qdrant_filter,
                limit=1
            )
            result['exists_in_qdrant'] = len(vectors) > 0

            # Get count
            all_vectors = self.storage.filter_by_metadata(
                filters=qdrant_filter,
                limit=10000
            )
            result['qdrant_count'] = len(all_vectors)

        except Exception as e:
            logger.error(f"Error checking Qdrant: {e}")

        # Check OpenSearch
        if self.opensearch_store:
            try:
                os_info = self.opensearch_store.check_document_exists(
                    self.config.opensearch_index,
                    doc_id
                )
                result['exists_in_opensearch'] = os_info.get('exists', False)
                result['opensearch_count'] = os_info.get('total_chunks', 0)
                result['opensearch_chunks_by_type'] = os_info.get('chunks_by_type', {})

            except Exception as e:
                logger.error(f"Error checking OpenSearch: {e}")

        return result


    def search_headings(
        self,
        query: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search only heading chunks - useful for TOC-style queries.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of heading chunks matching the query
        """
        if not self.opensearch_store:
            logger.warning("OpenSearch not available for heading search")
            return []

        return self.opensearch_store.search_headings_only(
            self.config.opensearch_index,
            query,
            size=limit
        )

    def get_document_toc(self, doc_id: str) -> List[Dict]:
        """
        Get table of contents (all headings) for a document.

        Args:
            doc_id: Document ID

        Returns:
            List of heading chunks sorted by page number
        """
        if not self.opensearch_store:
            logger.warning("OpenSearch not available for TOC retrieval")
            return []

        return self.opensearch_store.get_document_sections(
            self.config.opensearch_index,
            doc_id
        )

    def get_opensearch_stats(self) -> Dict:
        """Get OpenSearch index statistics."""
        if not self.opensearch_store:
            return {'available': False}

        try:
            stats = self.opensearch_store.get_stats(self.config.opensearch_index)
            stats['available'] = True
            return stats
        except Exception as e:
            logger.error(f"Failed to get OpenSearch stats: {e}")
            return {'available': False, 'error': str(e)}


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
