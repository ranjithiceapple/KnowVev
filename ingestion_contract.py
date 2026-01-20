"""
Ingestion Contract Module

This module defines the production-grade contract for document ingestion in an
Enterprise RAG system. It enforces mandatory metadata validation and ensures
consistent data identity across all storage backends (Qdrant, OpenSearch).

Architecture Context:
- Embeddings stored in Qdrant
- Keywords stored in OpenSearch
- Metadata stored in PostgreSQL (NOT accessed by ingestion pipeline)
- Ingestion pipeline is STATELESS and receives validated metadata from API service

Critical Guarantees:
1. Mandatory metadata (project_id, doc_id) MUST be provided - no auto-generation
2. project_id is stored in EVERY Qdrant vector and OpenSearch document
3. A single immutable doc_id is used across ALL chunks and vectors
4. Ingestion FAILS FAST if required metadata is missing
5. Schema versioning is enforced for payload contracts

Usage:
    from ingestion_contract import (
        IngestionRequest,
        IngestionResponse,
        validate_ingestion_request,
        IngestionError
    )

    # Create and validate request
    request = IngestionRequest(
        file_path="/path/to/document.pdf",
        doc_id="doc_abc123",
        project_id="proj_xyz789",
        file_name="document.pdf"
    )

    # Validation happens automatically in __post_init__
    # If validation fails, IngestionValidationError is raised
"""

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid

from logger_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# SCHEMA VERSION - INCREMENT ON BREAKING CHANGES
# =============================================================================
INGESTION_SCHEMA_VERSION = "2.0.0"
PAYLOAD_SCHEMA_VERSION = "2.0.0"


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class IngestionError(Exception):
    """Base exception for ingestion errors."""
    pass


class IngestionValidationError(IngestionError):
    """Raised when ingestion request validation fails."""

    def __init__(self, message: str, field: str = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(f"Validation failed: {message}")


class IngestionMissingMetadataError(IngestionValidationError):
    """Raised when mandatory metadata is missing."""

    def __init__(self, field: str):
        super().__init__(
            f"Mandatory field '{field}' is missing or None. "
            f"Ingestion cannot proceed without {field}.",
            field=field
        )


class IngestionInvalidMetadataError(IngestionValidationError):
    """Raised when metadata format is invalid."""

    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Field '{field}' has invalid value '{value}': {reason}",
            field=field,
            value=value
        )


# =============================================================================
# ENUMS
# =============================================================================

class ChunkType(str, Enum):
    """Canonical chunk types for hybrid search."""
    CONTENT = "content"       # Standard content chunks for semantic search
    HEADING = "heading"       # Heading-only chunks for keyword/TOC search
    CLAUSE = "clause"         # Single-sentence chunks for precise retrieval
    METADATA = "metadata"     # Document metadata chunks for filtering
    SUMMARY = "summary"       # Document/section summary chunks


class IngestionStatus(str, Enum):
    """Ingestion operation status."""
    SUCCESS = "success"
    PARTIAL = "partial"       # Some operations succeeded, some failed
    FAILED = "failed"
    VALIDATION_ERROR = "validation_error"


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_uuid_format(value: str, field_name: str) -> bool:
    """
    Validate that a string is a valid UUID format.

    Note: We don't require strict UUID4 format - any valid UUID or
    prefixed ID (e.g., 'doc_abc123', 'proj_xyz') is acceptable.
    """
    if not value:
        raise IngestionMissingMetadataError(field_name)

    if not isinstance(value, str):
        raise IngestionInvalidMetadataError(
            field_name, value, "Must be a string"
        )

    if len(value) < 3:
        raise IngestionInvalidMetadataError(
            field_name, value, "ID must be at least 3 characters"
        )

    if len(value) > 255:
        raise IngestionInvalidMetadataError(
            field_name, value, "ID must be less than 255 characters"
        )

    # Allow alphanumeric, hyphens, underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', value):
        raise IngestionInvalidMetadataError(
            field_name, value,
            "ID must contain only alphanumeric characters, hyphens, and underscores"
        )

    return True


def validate_file_path(file_path: str) -> bool:
    """Validate file path is provided and non-empty."""
    if not file_path:
        raise IngestionMissingMetadataError("file_path")

    if not isinstance(file_path, str):
        raise IngestionInvalidMetadataError(
            "file_path", file_path, "Must be a string"
        )

    return True


# =============================================================================
# REQUEST/RESPONSE CONTRACTS
# =============================================================================

@dataclass
class IngestionRequest:
    """
    Ingestion Request Contract.

    This is the ONLY valid way to request document ingestion. All fields
    marked as mandatory MUST be provided by the API service. The ingestion
    pipeline will FAIL FAST if mandatory fields are missing.

    Mandatory Fields:
    - file_path: Path to the document file
    - doc_id: Immutable document ID (provided by API service from Postgres)
    - project_id: Project/tenant ID for data isolation

    Optional Fields:
    - file_name: Human-readable file name (extracted from path if not provided)
    - custom_metadata: Additional metadata to attach to all vectors/documents
    - document_title: Clean document title (extracted from content if not provided)
    """

    # MANDATORY - Must be provided by API service
    file_path: str
    doc_id: str
    project_id: str

    # OPTIONAL - Can be derived or omitted
    file_name: Optional[str] = None
    document_title: Optional[str] = None
    custom_metadata: Optional[Dict[str, Any]] = None

    # Internal tracking
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    schema_version: str = INGESTION_SCHEMA_VERSION

    def __post_init__(self):
        """Validate all mandatory fields on creation."""
        self._validate()

    def _validate(self):
        """
        Validate the ingestion request.

        Raises:
            IngestionMissingMetadataError: If mandatory fields are missing
            IngestionInvalidMetadataError: If field formats are invalid
        """
        # Validate mandatory fields - FAIL FAST
        validate_file_path(self.file_path)
        validate_uuid_format(self.doc_id, "doc_id")
        validate_uuid_format(self.project_id, "project_id")

        # Derive file_name if not provided
        if not self.file_name:
            from pathlib import Path
            self.file_name = Path(self.file_path).name

        logger.info(
            f"[{self.request_id[:8]}] Ingestion request validated - "
            f"doc_id: {self.doc_id}, project_id: {self.project_id}, "
            f"file: {self.file_name}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'request_id': self.request_id,
            'doc_id': self.doc_id,
            'project_id': self.project_id,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'document_title': self.document_title,
            'custom_metadata': self.custom_metadata,
            'request_timestamp': self.request_timestamp,
            'schema_version': self.schema_version
        }


@dataclass
class ChunkIdentity:
    """
    Immutable chunk identity.

    Every chunk in the system must have a consistent identity that includes
    both doc_id and project_id. This ensures proper data isolation and
    enables efficient filtering/deletion.
    """
    doc_id: str
    project_id: str
    chunk_id: str
    chunk_type: ChunkType
    chunk_index: int  # Always non-negative (0, 1, 2, ...)

    @property
    def embedding_id(self) -> str:
        """
        Generate embedding ID.

        Format: {doc_id}_{chunk_type}_{chunk_index:04d}
        Example: doc_abc123_content_0001
        """
        return f"{self.doc_id}_{self.chunk_type.value}_{self.chunk_index:04d}"

    @property
    def opensearch_id(self) -> str:
        """
        Generate OpenSearch document ID.

        Same format as embedding_id for consistency.
        """
        return self.embedding_id


@dataclass
class VectorPayloadContract:
    """
    Qdrant Vector Payload Contract.

    This defines the EXACT structure of payloads stored in Qdrant.
    All fields are mandatory for proper retrieval and filtering.
    """
    # Identity - MANDATORY for filtering and deletion
    doc_id: str
    project_id: str
    chunk_id: str
    chunk_type: str  # ChunkType.value

    # Content
    text: str
    normalized_text: str

    # Position/Navigation
    page_number_start: int
    page_number_end: int
    chunk_index: int
    total_chunks: int

    # Hierarchy
    section_title: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    heading_level: Optional[int] = None

    # Metadata
    file_name: str = ""
    document_title: Optional[str] = None
    char_count: int = 0
    word_count: int = 0

    # Content flags
    contains_tables: bool = False
    contains_code: bool = False
    contains_bullets: bool = False
    is_summary: bool = False

    # Schema versioning
    schema_version: str = PAYLOAD_SCHEMA_VERSION
    indexed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Qdrant payload dictionary."""
        payload = asdict(self)
        # Ensure heading_path is a list (for Qdrant filtering)
        if not isinstance(payload.get('heading_path'), list):
            payload['heading_path'] = []
        # Add heading_path_str for exact matching
        payload['heading_path_str'] = ' > '.join(payload['heading_path'])
        return payload

    @classmethod
    def from_chunk_metadata(
        cls,
        chunk,
        project_id: str,
        document_title: Optional[str] = None
    ) -> 'VectorPayloadContract':
        """
        Create payload from ChunkMetadata.

        Args:
            chunk: ChunkMetadata object
            project_id: Project ID (MANDATORY)
            document_title: Clean document title

        Returns:
            VectorPayloadContract instance
        """
        return cls(
            doc_id=chunk.doc_id,
            project_id=project_id,
            chunk_id=chunk.chunk_id,
            chunk_type=chunk.chunk_type if hasattr(chunk, 'chunk_type') else ChunkType.CONTENT.value,
            text=chunk.normalized_text,
            normalized_text=chunk.normalized_text.lower() if chunk.normalized_text else "",
            page_number_start=chunk.page_number_start,
            page_number_end=chunk.page_number_end,
            chunk_index=max(0, chunk.chunk_index),  # Ensure non-negative
            total_chunks=chunk.total_chunks,
            section_title=chunk.section_title,
            heading_path=chunk.heading_path or [],
            heading_level=chunk.heading_level,
            file_name=chunk.file_name,
            document_title=document_title,
            char_count=chunk.chunk_char_len,
            word_count=chunk.chunk_word_count,
            contains_tables=getattr(chunk, 'contains_tables', False),
            contains_code=getattr(chunk, 'contains_code', False),
            contains_bullets=getattr(chunk, 'contains_bullets', False),
            is_summary=chunk.chunk_type == ChunkType.SUMMARY.value if hasattr(chunk, 'chunk_type') else False
        )


@dataclass
class OpenSearchDocumentContract:
    """
    OpenSearch Document Contract.

    This defines the EXACT structure of documents stored in OpenSearch
    for keyword/hybrid search.
    """
    # Identity - MANDATORY
    doc_id: str
    project_id: str
    chunk_id: str
    chunk_type: str

    # Searchable text
    text: str
    text_normalized: str
    keywords: List[str] = field(default_factory=list)

    # Section info
    section_title: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    heading_level: Optional[int] = None

    # Source tracking
    page_number_start: int = 1
    page_number_end: int = 1
    file_name: str = ""
    document_title: Optional[str] = None

    # Metadata
    chunk_index: int = 0
    total_chunks: int = 0
    char_count: int = 0
    word_count: int = 0

    # Schema versioning
    schema_version: str = PAYLOAD_SCHEMA_VERSION
    indexed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenSearch document dictionary."""
        return asdict(self)


@dataclass
class IngestionResult:
    """
    Result of a single storage operation (Qdrant or OpenSearch).
    """
    backend: str  # 'qdrant' or 'opensearch'
    success: bool
    items_stored: int = 0
    items_failed: int = 0
    error_message: Optional[str] = None


@dataclass
class IngestionResponse:
    """
    Ingestion Response Contract.

    This is returned to the API service after ingestion completes.
    It provides complete transparency about what was stored and where.
    """
    # Request tracking
    request_id: str
    doc_id: str
    project_id: str
    file_name: str

    # Status
    status: IngestionStatus
    success: bool

    # Statistics
    pages_extracted: int = 0
    chunks_created: int = 0
    unique_chunks: int = 0

    # Storage results
    qdrant_result: Optional[IngestionResult] = None
    opensearch_result: Optional[IngestionResult] = None

    # Chunk breakdown
    content_chunks: int = 0
    heading_chunks: int = 0
    clause_chunks: int = 0
    metadata_chunks: int = 0
    summary_chunks: int = 0

    # Vector IDs for tracking/deletion
    vector_ids: List[str] = field(default_factory=list)

    # Timing
    extraction_time: float = 0.0
    chunking_time: float = 0.0
    embedding_time: float = 0.0
    qdrant_time: float = 0.0
    opensearch_time: float = 0.0
    total_time: float = 0.0

    # Schema version
    schema_version: str = INGESTION_SCHEMA_VERSION
    completed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Error details (if any)
    error_message: Optional[str] = None
    error_stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            'request_id': self.request_id,
            'doc_id': self.doc_id,
            'project_id': self.project_id,
            'file_name': self.file_name,
            'status': self.status.value,
            'success': self.success,
            'statistics': {
                'pages_extracted': self.pages_extracted,
                'chunks_created': self.chunks_created,
                'unique_chunks': self.unique_chunks,
                'content_chunks': self.content_chunks,
                'heading_chunks': self.heading_chunks,
                'clause_chunks': self.clause_chunks,
                'metadata_chunks': self.metadata_chunks,
                'summary_chunks': self.summary_chunks,
            },
            'storage': {
                'qdrant': {
                    'success': self.qdrant_result.success if self.qdrant_result else False,
                    'items_stored': self.qdrant_result.items_stored if self.qdrant_result else 0,
                    'error': self.qdrant_result.error_message if self.qdrant_result else None
                },
                'opensearch': {
                    'success': self.opensearch_result.success if self.opensearch_result else False,
                    'items_stored': self.opensearch_result.items_stored if self.opensearch_result else 0,
                    'error': self.opensearch_result.error_message if self.opensearch_result else None
                }
            },
            'vector_ids': self.vector_ids,
            'timing': {
                'extraction': f"{self.extraction_time:.2f}s",
                'chunking': f"{self.chunking_time:.2f}s",
                'embedding': f"{self.embedding_time:.2f}s",
                'qdrant': f"{self.qdrant_time:.2f}s",
                'opensearch': f"{self.opensearch_time:.2f}s",
                'total': f"{self.total_time:.2f}s"
            },
            'schema_version': self.schema_version,
            'completed_at': self.completed_at
        }

        if self.error_message:
            result['error'] = {
                'message': self.error_message,
                'stage': self.error_stage
            }

        return result


# =============================================================================
# CHUNK INDEX GENERATOR
# =============================================================================

class ChunkIndexGenerator:
    """
    Generates consistent, non-negative chunk indices.

    This ensures:
    - Summary chunks get index 0 (first chunk)
    - Content chunks start at index 1
    - All indices are non-negative
    """

    def __init__(self, doc_id: str, project_id: str):
        self.doc_id = doc_id
        self.project_id = project_id
        self._counters = {
            ChunkType.SUMMARY: 0,
            ChunkType.CONTENT: 0,
            ChunkType.HEADING: 0,
            ChunkType.CLAUSE: 0,
            ChunkType.METADATA: 0
        }

    def next_index(self, chunk_type: ChunkType) -> int:
        """Get next index for a chunk type."""
        index = self._counters[chunk_type]
        self._counters[chunk_type] += 1
        return index

    def generate_chunk_id(self, chunk_type: ChunkType, index: int) -> str:
        """
        Generate a consistent chunk ID.

        Format: {doc_id}_{chunk_type}_{index:04d}
        Example: doc_abc123_content_0001
        """
        return f"{self.doc_id}_{chunk_type.value}_{index:04d}"

    def generate_embedding_id(self, chunk_type: ChunkType, index: int) -> str:
        """
        Generate embedding ID (same as chunk_id for consistency).
        """
        return self.generate_chunk_id(chunk_type, index)


# =============================================================================
# VALIDATION HELPER FUNCTIONS
# =============================================================================

def validate_ingestion_request(
    file_path: str,
    doc_id: Optional[str],
    project_id: Optional[str],
    **kwargs
) -> IngestionRequest:
    """
    Validate and create an IngestionRequest.

    This is the preferred way to create ingestion requests as it provides
    clear error messages when validation fails.

    Args:
        file_path: Path to document file
        doc_id: Document ID from API service
        project_id: Project ID from API service
        **kwargs: Additional optional parameters

    Returns:
        IngestionRequest if validation passes

    Raises:
        IngestionMissingMetadataError: If mandatory fields are missing
        IngestionInvalidMetadataError: If field formats are invalid
    """
    return IngestionRequest(
        file_path=file_path,
        doc_id=doc_id,
        project_id=project_id,
        **kwargs
    )


def create_failed_response(
    request: Optional[IngestionRequest],
    error_message: str,
    error_stage: str = "validation"
) -> IngestionResponse:
    """
    Create a failed response for error cases.

    Args:
        request: Original request (may be None if validation failed)
        error_message: Error description
        error_stage: Stage where error occurred

    Returns:
        IngestionResponse with failure status
    """
    return IngestionResponse(
        request_id=request.request_id if request else str(uuid.uuid4()),
        doc_id=request.doc_id if request else "unknown",
        project_id=request.project_id if request else "unknown",
        file_name=request.file_name if request else "unknown",
        status=IngestionStatus.FAILED if error_stage != "validation" else IngestionStatus.VALIDATION_ERROR,
        success=False,
        error_message=error_message,
        error_stage=error_stage
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Ingestion Contract Module")
    print("=" * 60)
    print(f"\nSchema Version: {INGESTION_SCHEMA_VERSION}")
    print(f"Payload Version: {PAYLOAD_SCHEMA_VERSION}")

    print("\nContract Components:")
    print("  - IngestionRequest: Validated request from API service")
    print("  - IngestionResponse: Response returned to API service")
    print("  - VectorPayloadContract: Qdrant payload structure")
    print("  - OpenSearchDocumentContract: OpenSearch document structure")
    print("  - ChunkIdentity: Immutable chunk identity")
    print("  - ChunkIndexGenerator: Non-negative index generator")

    print("\nMandatory Fields:")
    print("  - file_path: Path to document")
    print("  - doc_id: Document ID (from API service)")
    print("  - project_id: Project ID (from API service)")

    print("\nUsage Example:")
    print("""
# Valid request
request = IngestionRequest(
    file_path="/path/to/doc.pdf",
    doc_id="doc_abc123",
    project_id="proj_xyz789"
)

# This will FAIL - missing project_id
try:
    bad_request = IngestionRequest(
        file_path="/path/to/doc.pdf",
        doc_id="doc_abc123",
        project_id=None  # FAILS!
    )
except IngestionMissingMetadataError as e:
    print(f"Validation failed: {e}")
""")
