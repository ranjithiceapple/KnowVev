import os
import logging

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress verbose loggers early
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("surya").setLevel(logging.WARNING)
logging.getLogger("marker").setLevel(logging.WARNING)

# =============================================================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import shutil
import uuid
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE initializing logger
load_dotenv()

from document_to_vector_service import (
    DocumentToVectorService,
    ServiceConfig
)
from logger_config import get_logger
from list_documents_and_embeddings import DocumentLister
from config import get_config

# Initialize logger
logger = get_logger(__name__)

# ---------------------------------------------------------
# Load config from environment
# ---------------------------------------------------------
logger.info("Loading configuration from environment variables")
config = get_config()
logger.info(f"Configuration loaded - QDRANT_URL: {config.qdrant_url}, EMBEDDING_MODEL: {config.embedding_model}")

# Create service config from loaded configuration
service_config = ServiceConfig(
    qdrant_url=config.qdrant_url,
    qdrant_collection=config.qdrant_collection,
    qdrant_api_key=config.qdrant_api_key,
    embedding_model_name=config.embedding_model,
    vector_size=config.embedding_dimension,
    # Processing settings
    max_chunk_size=config.max_chunk_size,
    target_chunk_size=config.target_chunk_size,
    enable_overlap=config.enable_overlap,
    overlap_size=config.overlap_size,
    respect_page_boundaries=config.respect_page_boundaries,
    keep_tables_intact=config.keep_tables_intact,
    # Normalization settings
    remove_toc_pages=config.remove_toc_pages,
    protect_headings=config.protect_headings,
    protect_tables=config.protect_tables,
    protect_code_blocks=config.protect_code_blocks,
    detect_multi_column=config.detect_multi_column,
    # Processing
    deduplicate_chunks=config.deduplicate_chunks,
    aggressive_text_cleaning=config.aggressive_text_cleaning,
    # Summary settings
    generate_document_summary=config.generate_document_summary,
    summary_max_length=config.summary_max_length,
    summary_method=config.summary_method,
    # Topic modeling settings
    enable_topic_modeling=config.enable_topic_modeling,
    topic_n_topics=config.topic_n_topics,
    topic_model_dir=config.topic_model_dir,
    topic_retrain_threshold=config.topic_retrain_threshold_docs,
)
logger.info(f"Service config created - Collection: {service_config.qdrant_collection}, Vector size: {service_config.vector_size}")

logger.info("Initializing DocumentToVectorService")
service = DocumentToVectorService(service_config)
logger.info("DocumentToVectorService initialized successfully")

# Initialize DocumentLister for listing/managing documents
logger.info("Initializing DocumentLister")
document_lister = DocumentLister(
    qdrant_url=service_config.qdrant_url,
    collection_name=service_config.qdrant_collection
)
logger.info("DocumentLister initialized successfully")

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
logger.info("Creating FastAPI application")
app = FastAPI(title="KnowVec RAG Pipeline API")
logger.info("FastAPI application created successfully")

# CORS
logger.info("Configuring CORS middleware")
cors_origins = config.cors_origins.split(",") if config.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS middleware configured - Origins: {cors_origins}")


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests and responses."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Request started - ID: {request_id}, Method: {request.method}, "
        f"Path: {request.url.path}, Client: {request.client.host if request.client else 'unknown'}"
    )

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        logger.info(
            f"Request completed - ID: {request_id}, Method: {request.method}, "
            f"Path: {request.url.path}, Status: {response.status_code}, "
            f"Duration: {duration:.3f}s"
        )

        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Request failed - ID: {request_id}, Method: {request.method}, "
            f"Path: {request.url.path}, Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
class SearchResponse(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict
    matched_chunk: bool = True  # Indicates if this was the originally matched chunk
    section_expansion: bool = False  # Indicates if this chunk was added via section expansion


class FilterSearchRequest(BaseModel):
    """Request model for filtered search"""
    query: str = Field(..., description="Search query text")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters to apply",
        examples=[{
            "doc_id": "document-123",
            "page_start": {"gte": 5, "lte": 10},
            "contains_code": True
        }]
    )
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    score_threshold: Optional[float] = Field(default=None, description="Minimum score threshold")
    expand_to_section: bool = Field(default=False, description="Expand matched chunks to include full section context using heading_path")


class MetadataFilterRequest(BaseModel):
    """Request model for pure metadata filtering (no vector search)"""
    filters: Dict[str, Any] = Field(
        ...,
        description="Metadata filters to apply",
        examples=[{
            "doc_id": "document-123",
            "contains_tables": True
        }]
    )
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class DocumentResponse(BaseModel):
    """Response model for document information"""
    doc_id: str
    file_name: str
    total_chunks: int
    total_pages: int
    has_summary: bool
    created_at: str
    version: str


class DocumentDetailResponse(BaseModel):
    """Response model for detailed document information"""
    doc_id: str
    file_name: str
    total_chunks: int
    total_pages: int
    has_summary: bool
    chunks: List[Dict[str, Any]]


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion"""
    success: bool
    doc_id: str
    file_name: Optional[str] = None
    chunks_deleted: int
    message: str


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/")
def root():
    logger.debug("Root endpoint accessed")
    return {"message": "KnowVec RAG Pipeline API is running 🚀 booooooooo"}


# ---------------------------------------------------------
# 1️⃣ DOCUMENT INGESTION ENDPOINT
# ---------------------------------------------------------
@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    doc_id: str = Form(..., description="Mandatory document ID from API service"),
    project_id: str = Form(..., description="Mandatory project ID for data isolation"),
    debug: bool = False
):
    """
    Upload a document → Extract → Normalize → Chunk → Embed → Store in Qdrant

    IMPORTANT: doc_id and project_id are MANDATORY for production use.
    The ingestion pipeline will reject requests without these parameters.

    Args:
        file: Document file to process
        doc_id: MANDATORY document ID (provided by your API service from Postgres)
        project_id: MANDATORY project ID for multi-tenancy and data isolation
        debug: If True, saves pipeline stage outputs to pipeline_debug/ folder

    Multi-Tenancy:
        All chunks from this document are tagged with doc_id and project_id.
        This enables:
        - Filtered searches where users only see documents from their own projects
        - Clean document deletion by doc_id
        - Consistent identity across Qdrant and OpenSearch
    """
    start_time = time.time()
    logger.info(
        f"Document processing started - Filename: {file.filename}, "
        f"Content-Type: {file.content_type}, Doc ID: {doc_id}, Project ID: {project_id}, Debug mode: {debug}"
    )

    try:
        # Temporary file path
        temp_file_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        logger.debug(f"Created temporary file path: {temp_file_path}")

        # Save uploaded file to temp folder
        logger.info(f"Saving uploaded file to temporary location: {temp_file_path}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"File saved successfully - Size: {file_size} bytes")

        # DEBUG MODE: Run pipeline inspector if debug=True
        debug_files = None
        if debug:
            logger.info(f"Debug mode enabled - Running pipeline stage inspector")
            from pipeline_stage_inspector import PipelineStageInspector
            inspector = PipelineStageInspector(output_dir="pipeline_debug")
            debug_files = inspector.inspect_pipeline(temp_file_path)
            logger.info(f"Debug files created: {len(debug_files)} files in pipeline_debug/")

        # Process document using the full pipeline with mandatory metadata
        logger.info(f"Starting document processing pipeline for: {file.filename} (doc_id: {doc_id}, project_id: {project_id})")
        result = service.process_document(
            file_path=temp_file_path,
            doc_id=doc_id,
            project_id=project_id
        )
        logger.info(f"Document processing pipeline completed for: {file.filename}")

        # Cleanup
        logger.debug(f"Removing temporary file: {temp_file_path}")
        os.remove(temp_file_path)
        logger.debug("Temporary file removed successfully")

        if not result.success:
            logger.error(
                f"Document processing failed - Filename: {file.filename}, "
                f"Error: {result.error_message}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result.error_message}"
            )

        duration = time.time() - start_time
        logger.info(
            f"Document processed successfully - Filename: {file.filename}, "
            f"Doc ID: {result.doc_id}, Pages: {result.pages_extracted}, "
            f"Chunks: {result.chunks_created}, Unique: {result.unique_chunks}, "
            f"Vectors: {result.vectors_stored}, Pipeline time: {result.total_time:.2f}s, "
            f"Total time: {duration:.2f}s"
        )

        response = {
            "message": "Document processed successfully",
            "doc_id": result.doc_id,
            "project_id": project_id,
            "pages_extracted": result.pages_extracted,
            "chunks_created": result.chunks_created,
            "unique_chunks": result.unique_chunks,
            "vectors_stored": result.vectors_stored,
            "vector_ids": result.vector_ids,  # For Base Model tracking and deletion
            "embedding_ids": result.embedding_ids,  # Alias
            "total_time": result.total_time
        }

        # Add debug files to response if debug mode was enabled
        if debug and debug_files:
            response["debug"] = {
                "enabled": True,
                "output_dir": "pipeline_debug",
                "files_created": debug_files
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Unexpected error during document processing - Filename: {file.filename}, "
            f"Duration: {duration:.2f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 1️⃣A DOCUMENT INGESTION VIA SHARED VOLUME PATH
# ---------------------------------------------------------
class ProcessPathRequest(BaseModel):
    """Request model for processing documents via shared volume path"""
    file_path: str = Field(..., description="Path to document file in shared volume")
    project_id: Optional[str] = Field(None, description="Project ID for multi-tenancy")
    callback_url: Optional[str] = Field(None, description="URL to callback when processing completes")
    debug: bool = Field(False, description="Enable debug mode")


@app.post("/process-path")
async def process_document_by_path(request: ProcessPathRequest):
    """
    Process a document from a shared volume path.

    This endpoint is optimized for Docker deployments where the Base Model
    and RAG Microservice share a volume (e.g., /data/uploads).

    Instead of uploading the file via HTTP, the Base Model:
    1. Saves the file to the shared volume
    2. Sends the file path to this endpoint
    3. (Optional) Provides a callback_url for async notification

    Benefits:
    - Reduces network latency
    - Avoids duplicating large files in memory
    - Enables async processing with callbacks

    Args:
        file_path: Absolute path to document in shared volume
        project_id: Optional project ID for multi-tenancy
        callback_url: Optional webhook URL for completion notification
        debug: Enable debug mode for pipeline inspection

    Returns:
        Processing result with doc_id, vector_ids, and statistics
    """
    start_time = time.time()
    logger.info(
        f"Path-based processing started - Path: {request.file_path}, "
        f"Project ID: {request.project_id}, Callback: {request.callback_url}, Debug: {request.debug}"
    )

    try:
        # Validate file exists
        if not os.path.exists(request.file_path):
            logger.error(f"File not found: {request.file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

        # Validate file is readable
        if not os.access(request.file_path, os.R_OK):
            logger.error(f"File not readable: {request.file_path}")
            raise HTTPException(status_code=403, detail=f"File not readable: {request.file_path}")

        file_size = os.path.getsize(request.file_path)
        file_name = os.path.basename(request.file_path)
        logger.info(f"Processing file - Name: {file_name}, Size: {file_size} bytes")

        # DEBUG MODE: Run pipeline inspector if debug=True
        debug_files = None
        if request.debug:
            logger.info(f"Debug mode enabled - Running pipeline stage inspector")
            from pipeline_stage_inspector import PipelineStageInspector
            inspector = PipelineStageInspector(output_dir="pipeline_debug")
            debug_files = inspector.inspect_pipeline(request.file_path)
            logger.info(f"Debug files created: {len(debug_files)} files in pipeline_debug/")

        # Process document using the full pipeline
        logger.info(f"Starting document processing pipeline for: {file_name}")
        result = service.process_document(
            file_path=request.file_path,
            project_id=request.project_id
        )
        logger.info(f"Document processing pipeline completed for: {file_name}")

        if not result.success:
            logger.error(
                f"Document processing failed - Filename: {file_name}, "
                f"Error: {result.error_message}"
            )

            # Send failure callback if provided
            if request.callback_url:
                await _send_callback(request.callback_url, {
                    "success": False,
                    "doc_id": result.doc_id,
                    "project_id": request.project_id,
                    "error": result.error_message
                })

            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result.error_message}"
            )

        duration = time.time() - start_time
        logger.info(
            f"Document processed successfully - Filename: {file_name}, "
            f"Doc ID: {result.doc_id}, Pages: {result.pages_extracted}, "
            f"Chunks: {result.chunks_created}, Unique: {result.unique_chunks}, "
            f"Vectors: {result.vectors_stored}, Vector IDs: {len(result.vector_ids)}, "
            f"Pipeline time: {result.total_time:.2f}s, Total time: {duration:.2f}s"
        )

        response = {
            "message": "Document processed successfully",
            "doc_id": result.doc_id,
            "project_id": request.project_id,
            "file_name": file_name,
            "file_path": request.file_path,
            "pages_extracted": result.pages_extracted,
            "chunks_created": result.chunks_created,
            "unique_chunks": result.unique_chunks,
            "vectors_stored": result.vectors_stored,
            "vector_ids": result.vector_ids,  # For Base Model tracking
            "embedding_ids": result.embedding_ids,  # Alias
            "total_time": result.total_time
        }

        # Add debug files to response if debug mode was enabled
        if request.debug and debug_files:
            response["debug"] = {
                "enabled": True,
                "output_dir": "pipeline_debug",
                "files_created": debug_files
            }

        # Send success callback if provided
        if request.callback_url:
            await _send_callback(request.callback_url, response)
            response["callback_sent"] = True

        return response

    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Unexpected error during path-based processing - Path: {request.file_path}, "
            f"Duration: {duration:.2f}s, Error: {str(e)}",
            exc_info=True
        )

        # Send error callback if provided
        if request.callback_url:
            await _send_callback(request.callback_url, {
                "success": False,
                "file_path": request.file_path,
                "project_id": request.project_id,
                "error": str(e)
            })

        raise HTTPException(status_code=500, detail=str(e))


# Helper function for sending callbacks
async def _send_callback(callback_url: str, payload: dict):
    """Send async callback to Base Model."""
    import httpx
    try:
        logger.info(f"Sending callback to: {callback_url}")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(callback_url, json=payload)
            logger.info(f"Callback sent - Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send callback to {callback_url}: {str(e)}", exc_info=True)


# ---------------------------------------------------------
# 2️⃣ VECTOR SEARCH ENDPOINT
# ---------------------------------------------------------
@app.get("/search", response_model=List[SearchResponse])
def search_documents(
    q: str = Query(..., description="Search query text"),
    limit: int = 10,
    score_threshold: Optional[float] = None,
    expand_to_section: bool = Query(False, description="Expand matched chunks to include full section context using heading_path"),
    project_id: Optional[str] = Query(None, description="Filter results by project ID (multi-tenancy)")
):
    """
    Semantic search through Qdrant embeddings

    Args:
        q: Search query text
        limit: Maximum number of results
        score_threshold: Minimum similarity score threshold
        expand_to_section: If True, retrieves complete sections instead of individual chunks.
                          Uses heading_path metadata to fetch all chunks in the same section,
                          preventing context fragmentation (Parent-Child Retrieval).
        project_id: Optional project ID to filter results (only returns documents from this project)

    Multi-Tenancy:
        When project_id is provided, only documents tagged with that project_id will be returned.
        This ensures data isolation between different projects/users.
    """
    start_time = time.time()
    logger.info(
        f"Search request received - Query: '{q[:100]}{'...' if len(q) > 100 else ''}', "
        f"Limit: {limit}, Score threshold: {score_threshold}, Expand to section: {expand_to_section}, "
        f"Project ID: {project_id}"
    )

    try:
        # Build filters with project_id if provided
        filters = {}
        if project_id:
            filters['project_id'] = project_id
            logger.debug(f"Filtering results by project_id: {project_id}")

        logger.debug(f"Executing search with query: {q}")
        results = service.search(
            query=q,
            limit=limit,
            score_threshold=score_threshold,
            expand_to_section=expand_to_section,
            filters=filters if filters else None
        )
        logger.debug(f"Search returned {len(results)} results")

        formatted = [
            {
                "id": str(r.get("payload", {}).get("chunk_id")),
                "score": r.get("score"),
                "text": r.get("payload", {}).get("text"),
                "metadata": r.get("payload", {}),
                "matched_chunk": r.get("matched_chunk", True),
                "section_expansion": r.get("section_expansion", False)
            }
            for r in results
        ]


        duration = time.time() - start_time
        logger.info(
            f"Search completed successfully - Query: '{q[:50]}{'...' if len(q) > 50 else ''}', "
            f"Results: {len(formatted)}, Duration: {duration:.3f}s"
        )

        return formatted

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Search failed - Query: '{q[:50]}{'...' if len(q) > 50 else ''}', "
            f"Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 3️⃣ FILTERED SEMANTIC SEARCH
# ---------------------------------------------------------
@app.post("/search/filtered", response_model=List[SearchResponse])
def filtered_search(
    request: FilterSearchRequest,
    project_id: Optional[str] = Query(None, description="Filter results by project ID (multi-tenancy)")
):
    """
    Semantic search with metadata filtering.

    Combines vector similarity search with metadata filters for precise results.

    Example filters:
    - By document: {"doc_id": "document-123"}
    - By page range: {"page_start": {"gte": 5, "lte": 10}}
    - By content type: {"contains_code": True, "contains_tables": False}
    - By section: {"section_title": "Introduction"}
    - Combined: {"doc_id": "doc-123", "page_start": {"gte": 1}, "contains_code": True}

    Parent-Child Retrieval:
    - Set expand_to_section=True to retrieve complete sections instead of fragments
    - Uses heading_path to fetch all chunks in the same section

    Multi-Tenancy:
    - Provide project_id to filter results by project (data isolation)
    - project_id can also be included in the filters dictionary
    """
    start_time = time.time()
    logger.info(
        f"Filtered search request - Query: '{request.query[:50]}', "
        f"Filters: {request.filters}, Limit: {request.limit}, "
        f"Expand to section: {request.expand_to_section}, Project ID: {project_id}"
    )

    try:
        # Merge project_id into filters if provided
        merged_filters = request.filters.copy() if request.filters else {}
        if project_id:
            merged_filters['project_id'] = project_id
            logger.debug(f"Added project_id to filters: {project_id}")

        results = service.search(
            query=request.query,
            limit=request.limit,
            filters=merged_filters if merged_filters else None,
            score_threshold=request.score_threshold,
            expand_to_section=request.expand_to_section
        )

        formatted = [
            {
                "id": str(r.get("payload", {}).get("chunk_id")),
                "score": r.get("score"),
                "text": r.get("payload", {}).get("text"),
                "metadata": r.get("payload", {}),
                "matched_chunk": r.get("matched_chunk", True),
                "section_expansion": r.get("section_expansion", False)
            }
            for r in results
        ]

        duration = time.time() - start_time
        logger.info(
            f"Filtered search completed - Query: '{request.query[:50]}', "
            f"Results: {len(formatted)}, Duration: {duration:.3f}s"
        )

        return formatted

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Filtered search failed - Query: '{request.query[:50]}', "
            f"Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 4️⃣ METADATA-ONLY FILTERING (No Vector Search)
# ---------------------------------------------------------
@app.post("/filter")
def filter_by_metadata(request: MetadataFilterRequest):
    """
    Filter documents by metadata only (no semantic search).

    Useful for:
    - Finding all chunks from a specific document
    - Finding all chunks with tables or code
    - Filtering by page ranges
    - Browsing document structure

    Example filters:
    - All tables: {"contains_tables": True}
    - Document pages: {"doc_id": "doc-123", "page_start": {"gte": 10, "lte": 20}}
    - Code blocks: {"contains_code": True}
    - Specific section: {"section_title": "Chapter 1"}
    """
    start_time = time.time()
    logger.info(
        f"Metadata filter request - Filters: {request.filters}, "
        f"Limit: {request.limit}, Offset: {request.offset}"
    )

    try:
        results = service.filter_by_metadata(
            filters=request.filters,
            limit=request.limit,
            offset=request.offset
        )

        formatted = [
            {
                "id": str(r.get("payload", {}).get("chunk_id")),
                "text": r.get("payload", {}).get("text"),
                "metadata": r.get("payload", {}),
            }
            for r in results
        ]

        duration = time.time() - start_time
        logger.info(
            f"Metadata filter completed - Results: {len(formatted)}, Duration: {duration:.3f}s"
        )

        return {
            "results": formatted,
            "count": len(formatted),
            "limit": request.limit,
            "offset": request.offset
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Metadata filter failed - Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 5️⃣ COLLECTION STATS
# ---------------------------------------------------------
@app.get("/stats")
def get_stats():
    """
    Returns Qdrant collection stats
    """
    logger.info("Stats request received")
    try:
        logger.debug("Fetching collection stats from service")
        stats = service.get_collection_stats()
        logger.info(f"Stats retrieved successfully: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Failed to retrieve stats - Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 6️⃣ LIST ALL DOCUMENTS
# ---------------------------------------------------------
@app.get("/documents", response_model=List[DocumentResponse])
def list_documents(
    limit: int = Query(1000, ge=1, le=10000, description="Maximum documents to return"),
    project_id: Optional[str] = Query(None, description="Filter documents by project ID")
):
    """
    List all documents stored in the vector database.

    Returns basic metadata for each document including:
    - Document ID
    - File name
    - Total chunks
    - Total pages
    - Whether document has a summary
    - Creation timestamp

    Multi-Tenancy:
    - Provide project_id to only list documents from a specific project

    Example response:
    ```json
    [
        {
            "doc_id": "abc-123-xyz",
            "file_name": "report.pdf",
            "total_chunks": 45,
            "total_pages": 25,
            "has_summary": true,
            "created_at": "2024-01-15T10:30:00",
            "version": "1.0"
        }
    ]
    ```
    """
    start_time = time.time()
    logger.info(f"List documents request received - Limit: {limit}, Project ID: {project_id}")

    try:
        # If project_id is provided, filter documents by project
        if project_id:
            logger.debug(f"Filtering documents by project_id: {project_id}")
            # Get all chunks from this project
            chunks = service.filter_by_metadata(
                filters={'project_id': project_id},
                limit=limit * 100  # Higher limit to get all chunks
            )

            # Group by doc_id to get unique documents
            doc_map = {}
            for chunk in chunks:
                payload = chunk.get('payload', {})
                doc_id = payload.get('doc_id')
                if doc_id and doc_id not in doc_map:
                    doc_map[doc_id] = payload

            # Format as DocumentResponse
            documents = []
            for doc_id, first_chunk in doc_map.items():
                # Get chunk count for this document
                doc_chunks = [c for c in chunks if c.get('payload', {}).get('doc_id') == doc_id]
                documents.append({
                    'doc_id': doc_id,
                    'file_name': first_chunk.get('file_name', 'Unknown'),
                    'total_chunks': len(doc_chunks),
                    'total_pages': max((c.get('payload', {}).get('page_end', 0) for c in doc_chunks), default=0),
                    'has_summary': any(c.get('payload', {}).get('section_title') == '[DOCUMENT SUMMARY]' for c in doc_chunks),
                    'created_at': first_chunk.get('created_at', ''),
                    'version': first_chunk.get('version', '1.0')
                })

            documents = documents[:limit]  # Apply limit
        else:
            # No project filter - list all documents
            documents = document_lister.list_documents(limit=limit)

        duration = time.time() - start_time
        logger.info(
            f"List documents completed - Found: {len(documents)} documents, "
            f"Duration: {duration:.3f}s"
        )

        return documents

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"List documents failed - Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 7️⃣ GET DOCUMENT DETAILS
# ---------------------------------------------------------
@app.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
def get_document_details(
    doc_id: str,
    include_embeddings: bool = Query(False, description="Include vector embeddings in response")
):
    """
    Get detailed information about a specific document.

    Returns all chunks with metadata for the document.
    Optionally includes vector embeddings if requested.

    Args:
        doc_id: Document ID (full UUID)
        include_embeddings: Whether to include 384-dim vectors (default: false)

    Example response:
    ```json
    {
        "doc_id": "abc-123-xyz",
        "file_name": "report.pdf",
        "total_chunks": 45,
        "total_pages": 25,
        "has_summary": true,
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 2,
                "section_title": "Introduction",
                "text_preview": "This document describes..."
            }
        ]
    }
    ```
    """
    start_time = time.time()
    logger.info(f"Get document details request - Doc ID: {doc_id[:8]}..., Include embeddings: {include_embeddings}")

    try:
        # Get chunks for document
        chunks = document_lister.get_document_chunks(doc_id, include_vectors=include_embeddings)

        if not chunks:
            logger.warning(f"Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        # Build response
        response = {
            "doc_id": doc_id,
            "file_name": chunks[0].get('file_name', 'Unknown') if chunks else "Unknown",
            "total_chunks": len(chunks),
            "total_pages": chunks[-1]['page_end'] if chunks else 0,
            "has_summary": any(c.get('section_title') == '[DOCUMENT SUMMARY]' for c in chunks),
            "chunks": chunks
        }

        duration = time.time() - start_time
        logger.info(
            f"Get document details completed - Chunks: {len(chunks)}, Duration: {duration:.3f}s"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Get document details failed - Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 8️⃣ GET DOCUMENT EMBEDDINGS
# ---------------------------------------------------------
@app.get("/documents/{doc_id}/embeddings")
def get_document_embeddings(doc_id: str):
    """
    Get all embeddings (vectors) for a specific document.

    Returns the 384-dimensional embedding vectors for each chunk.
    Useful for:
    - Analyzing document representation
    - Exporting for external processing
    - Debugging embedding quality

    Warning: Response can be large for documents with many chunks.
    Each vector is 384 dimensions (floats).

    Example response:
    ```json
    {
        "doc_id": "abc-123-xyz",
        "total_chunks": 45,
        "vector_dimension": 384,
        "embeddings": [
            {
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "vector": [0.123, -0.456, 0.789, ...]
            }
        ]
    }
    ```
    """
    start_time = time.time()
    logger.info(f"Get document embeddings request - Doc ID: {doc_id[:8]}...")

    try:
        chunks = document_lister.get_document_chunks(doc_id, include_vectors=True)

        if not chunks:
            logger.warning(f"Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        # Extract embeddings
        embeddings = []
        for chunk in chunks:
            if 'vector' in chunk:
                embeddings.append({
                    "chunk_id": chunk['chunk_id'],
                    "chunk_index": chunk['chunk_index'],
                    "vector": chunk['vector']
                })

        response = {
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "vector_dimension": embeddings[0].get('vector') and len(embeddings[0]['vector']) if embeddings else 0,
            "embeddings": embeddings
        }

        duration = time.time() - start_time
        logger.info(
            f"Get document embeddings completed - Embeddings: {len(embeddings)}, Duration: {duration:.3f}s"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Get document embeddings failed - Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 9️⃣ DELETE DOCUMENT
# ---------------------------------------------------------
@app.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(doc_id: str):
    """
    Delete a document and all its embeddings from the vector database.

    This will:
    - Delete all chunks associated with the document
    - Delete all embeddings/vectors for the document
    - Remove document from the collection

    ⚠️ WARNING: This action is irreversible!

    Args:
        doc_id: Document ID to delete

    Example response:
    ```json
    {
        "success": true,
        "doc_id": "abc-123-xyz",
        "file_name": "report.pdf",
        "chunks_deleted": 45,
        "message": "Document and 45 embeddings deleted successfully"
    }
    ```
    """
    start_time = time.time()
    logger.warning(f"Delete document request - Doc ID: {doc_id[:8]}...")

    try:
        # First, get document info before deletion
        chunks = document_lister.get_document_chunks(doc_id, include_vectors=False)

        if not chunks:
            logger.warning(f"Document not found for deletion: {doc_id}")
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        file_name = chunks[0].get('file_name', 'Unknown') if chunks else None
        chunks_count = len(chunks)

        # Delete from Qdrant using filter
        logger.warning(f"Deleting document {doc_id} with {chunks_count} chunks...")

        delete_result = service.storage.delete_by_filter({
            "doc_id": doc_id
        })

        duration = time.time() - start_time
        logger.warning(
            f"Document deleted - Doc ID: {doc_id[:8]}..., File: {file_name}, "
            f"Chunks: {chunks_count}, Duration: {duration:.3f}s"
        )

        return {
            "success": True,
            "doc_id": doc_id,
            "file_name": file_name,
            "chunks_deleted": chunks_count,
            "message": f"Document '{file_name}' and {chunks_count} embeddings deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Delete document failed - Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 🔟 BULK DELETE DOCUMENTS
# ---------------------------------------------------------
@app.post("/documents/bulk-delete")
def bulk_delete_documents(doc_ids: List[str] = Body(..., description="List of document IDs to delete")):
    """
    Delete multiple documents at once.

    ⚠️ WARNING: This action is irreversible!

    Request body:
    ```json
    ["doc-id-1", "doc-id-2", "doc-id-3"]
    ```

    Response:
    ```json
    {
        "success": true,
        "total_requested": 3,
        "deleted": 3,
        "failed": 0,
        "results": [
            {
                "doc_id": "doc-id-1",
                "success": true,
                "chunks_deleted": 45
            }
        ]
    }
    ```
    """
    start_time = time.time()
    logger.warning(f"Bulk delete request - {len(doc_ids)} documents")

    try:
        results = []
        deleted_count = 0
        failed_count = 0

        for doc_id in doc_ids:
            try:
                # Get chunk count
                chunks = document_lister.get_document_chunks(doc_id, include_vectors=False)
                chunks_count = len(chunks)

                if chunks_count == 0:
                    results.append({
                        "doc_id": doc_id,
                        "success": False,
                        "error": "Document not found"
                    })
                    failed_count += 1
                    continue

                # Delete
                service.storage.delete_by_filter({"doc_id": doc_id})

                results.append({
                    "doc_id": doc_id,
                    "success": True,
                    "chunks_deleted": chunks_count
                })
                deleted_count += 1

            except Exception as e:
                results.append({
                    "doc_id": doc_id,
                    "success": False,
                    "error": str(e)
                })
                failed_count += 1

        duration = time.time() - start_time
        logger.warning(
            f"Bulk delete completed - Requested: {len(doc_ids)}, "
            f"Deleted: {deleted_count}, Failed: {failed_count}, Duration: {duration:.3f}s"
        )

        return {
            "success": True,
            "total_requested": len(doc_ids),
            "deleted": deleted_count,
            "failed": failed_count,
            "results": results
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Bulk delete failed - Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))

