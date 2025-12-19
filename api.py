from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
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
# COMMENTED OUT: Advanced search features (optional modules)
# from query_intent_classifier import QueryIntentClassifier, QueryIntent
# from query_analyzer import EnhancedQueryAnalyzer
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

# COMMENTED OUT: Advanced search features (optional modules)
# # Initialize Query Intent Classifier
# logger.info("Initializing QueryIntentClassifier")
# intent_classifier = QueryIntentClassifier()
# logger.info("QueryIntentClassifier initialized successfully")
#
# # Initialize Enhanced Query Analyzer
# logger.info("Initializing EnhancedQueryAnalyzer")
# query_analyzer = EnhancedQueryAnalyzer()
# logger.info("EnhancedQueryAnalyzer initialized successfully")

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


# COMMENTED OUT: Response models for advanced search features
"""
class IntentClassificationResponse(BaseModel):
    '''Response model for intent classification'''
    query: str
    primary_intent: str
    confidence: float
    secondary_intents: Optional[List[Dict[str, Any]]] = None
    recommended_filters: Optional[Dict[str, Any]] = None
    recommended_limit: int
    recommended_score_threshold: float


class SearchWithIntentResponse(BaseModel):
    '''Response model for search with intent classification'''
    query: str
    intent: IntentClassificationResponse
    results: List[SearchResponse]
    total_results: int
    search_time: float
"""


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get("/")
def root():
    logger.debug("Root endpoint accessed")
    return {"message": "KnowVec RAG Pipeline API is running 🚀"}


# ---------------------------------------------------------
# 1️⃣ DOCUMENT INGESTION ENDPOINT
# ---------------------------------------------------------
@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    debug: bool = False
):
    """
    Upload a document → Extract → Normalize → Chunk → Embed → Store in Qdrant

    Args:
        file: Document file to process
        debug: If True, saves pipeline stage outputs to pipeline_debug/ folder
    """
    start_time = time.time()
    logger.info(f"Document processing started - Filename: {file.filename}, Content-Type: {file.content_type}, Debug mode: {debug}")

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

        # Process document using the full pipeline
        logger.info(f"Starting document processing pipeline for: {file.filename}")
        result = service.process_document(temp_file_path)
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
            "pages_extracted": result.pages_extracted,
            "chunks_created": result.chunks_created,
            "unique_chunks": result.unique_chunks,
            "vectors_stored": result.vectors_stored,
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
# 2️⃣ VECTOR SEARCH ENDPOINT
# ---------------------------------------------------------
@app.get("/search", response_model=List[SearchResponse])
def search_documents(
    q: str = Query(..., description="Search query text"),
    limit: int = 10,
    score_threshold: Optional[float] = None
):
    """
    Semantic search through Qdrant embeddings
    """
    start_time = time.time()
    logger.info(
        f"Search request received - Query: '{q[:100]}{'...' if len(q) > 100 else ''}', "
        f"Limit: {limit}, Score threshold: {score_threshold}"
    )

    try:
        logger.debug(f"Executing search with query: {q}")
        results = service.search(
            query=q,
            limit=limit,
            score_threshold=score_threshold
        )
        logger.debug(f"Search returned {len(results)} results")

        formatted = [
            {
                "id": str(r.get("payload", {}).get("chunk_id")),
                "score": r.get("score"),
                "text": r.get("payload", {}).get("text"),
                "metadata": r.get("payload", {}),
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
def filtered_search(request: FilterSearchRequest):
    """
    Semantic search with metadata filtering.

    Combines vector similarity search with metadata filters for precise results.

    Example filters:
    - By document: {"doc_id": "document-123"}
    - By page range: {"page_start": {"gte": 5, "lte": 10}}
    - By content type: {"contains_code": True, "contains_tables": False}
    - By section: {"section_title": "Introduction"}
    - Combined: {"doc_id": "doc-123", "page_start": {"gte": 1}, "contains_code": True}
    """
    start_time = time.time()
    logger.info(
        f"Filtered search request - Query: '{request.query[:50]}', "
        f"Filters: {request.filters}, Limit: {request.limit}"
    )

    try:
        results = service.search(
            query=request.query,
            limit=request.limit,
            filters=request.filters,
            score_threshold=request.score_threshold
        )

        formatted = [
            {
                "id": str(r.get("payload", {}).get("chunk_id")),
                "score": r.get("score"),
                "text": r.get("payload", {}).get("text"),
                "metadata": r.get("payload", {}),
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
def list_documents(limit: int = Query(1000, ge=1, le=10000, description="Maximum documents to return")):
    """
    List all documents stored in the vector database.

    Returns basic metadata for each document including:
    - Document ID
    - File name
    - Total chunks
    - Total pages
    - Whether document has a summary
    - Creation timestamp

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
    logger.info(f"List documents request received - Limit: {limit}")

    try:
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


# ---------------------------------------------------------
# ADVANCED SEARCH ENDPOINTS (COMMENTED OUT)
# ---------------------------------------------------------
# The following endpoints require optional modules:
# - query_intent_classifier (for /intent/classify and /search/smart)
# - query_analyzer (for /search/advanced)
#
# To enable these endpoints:
# 1. Implement the required modules
# 2. Uncomment the imports at the top of this file
# 3. Uncomment the endpoint code below
#
# Available endpoints when enabled:
# - POST /intent/classify - Classify query intent
# - POST /search/smart - Intent-aware search
# - POST /search/advanced - Advanced search with fallback
# ---------------------------------------------------------


# ---------------------------------------------------------
# MAIN
    - **How-to**: Instructions, tutorials, guides
    - **Definition**: Explanations, definitions
    - **Comparison**: Comparing options
    - **Code/Technical**: Code examples, API docs
    - **Summary**: Overview, highlights
    - **Troubleshooting**: Error fixing, debugging
    - **Recommendation**: Best practices, suggestions
    - **Procedural**: Steps, workflows
    - **Conceptual**: Why, theory, concepts

    Returns intent classification with confidence scores and search recommendations.

    Example request:
    ```json
    {
        "query": "How to implement authentication in Python"
    }
    ```

    Example response:
    ```json
    {
        "query": "How to implement authentication in Python",
        "primary_intent": "how_to",
        "confidence": 0.85,
        "secondary_intents": [
            {"intent": "code_technical", "confidence": 0.45}
        ],
        "recommended_filters": {"contains_code": true},
        "recommended_limit": 10,
        "recommended_score_threshold": 0.5
    }
    ```
    """
    start_time = time.time()
    logger.info(f"Intent classification request - Query: '{query[:50]}...'")

    try:
        classification = intent_classifier.classify(query)

        duration = time.time() - start_time

        response = IntentClassificationResponse(
            query=query,
            primary_intent=classification.primary_intent.value,
            confidence=classification.confidence,
            secondary_intents=[
                {"intent": intent.value, "confidence": conf}
                for intent, conf in classification.secondary_intents
            ],
            recommended_filters=classification.recommended_filters,
            recommended_limit=classification.recommended_limit,
            recommended_score_threshold=classification.recommended_score_threshold
        )

        logger.info(
            f"Intent classified - Intent: {classification.primary_intent.value}, "
            f"Confidence: {classification.confidence:.2f}, Duration: {duration:.3f}s"
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Intent classification failed - Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 1️⃣2️⃣ SMART SEARCH (with Intent Classification)
# ---------------------------------------------------------
@app.post("/search/smart", response_model=SearchWithIntentResponse)
def smart_search(
    query: str = Body(..., embed=True, description="Search query"),
    use_intent_filters: bool = Body(True, description="Apply intent-based filters"),
    use_intent_limits: bool = Body(True, description="Apply intent-based limits"),
    override_filters: Optional[Dict[str, Any]] = Body(None, description="Manual filter overrides")
):
    """
    Smart search with automatic intent classification and optimization.

    This endpoint:
    1. Classifies the query intent
    2. Applies intent-specific search strategies
    3. Returns results optimized for the query type

    **Benefits:**
    - Better relevance for different query types
    - Automatic filter selection
    - Optimized result limits
    - Intent-aware scoring thresholds

    **Intent-Specific Strategies:**
    - **Summary queries** → Search document summaries first
    - **Code queries** → Filter for code-heavy chunks
    - **Troubleshooting** → Cast wider net, lower threshold
    - **Definition queries** → Prefer early chunks, higher threshold
    - **Comparison** → Return more results for multiple perspectives

    Example request:
    ```json
    {
        "query": "Show me code examples for authentication",
        "use_intent_filters": true,
        "use_intent_limits": true
    }
    ```

    Example response:
    ```json
    {
        "query": "Show me code examples for authentication",
        "intent": {
            "primary_intent": "code_technical",
            "confidence": 0.82,
            "recommended_filters": {"contains_code": true}
        },
        "results": [
            {"id": "...", "score": 0.89, "text": "...", "metadata": {...}}
        ],
        "total_results": 8,
        "search_time": 0.234
    }
    ```
    """
    start_time = time.time()
    logger.info(f"Smart search request - Query: '{query[:50]}...', Use filters: {use_intent_filters}")

    try:
        # Step 1: Classify intent
        classification = intent_classifier.classify(query)

        logger.info(
            f"Intent detected: {classification.primary_intent.value} "
            f"(confidence: {classification.confidence:.2f})"
        )

        # Step 2: Build search parameters based on intent
        filters = {}
        limit = 10
        score_threshold = 0.5

        if use_intent_filters and classification.recommended_filters:
            filters.update(classification.recommended_filters)
            logger.debug(f"Applied intent filters: {filters}")

        if use_intent_limits:
            limit = classification.recommended_limit
            score_threshold = classification.recommended_score_threshold
            logger.debug(f"Applied intent limits: limit={limit}, threshold={score_threshold}")

        # Apply manual overrides
        if override_filters:
            filters.update(override_filters)
            logger.debug(f"Applied override filters: {override_filters}")

        # Step 3: Execute search
        if filters:
            results = service.search(
                query=query,
                limit=limit,
                filters=filters,
                score_threshold=score_threshold
            )
        else:
            results = service.search(
                query=query,
                limit=limit,
                score_threshold=score_threshold
            )

        # Step 4: Format response
        formatted_results = [
            {
                "id": str(r.get("payload", {}).get("chunk_id")),
                "score": r.get("score"),
                "text": r.get("payload", {}).get("text"),
                "metadata": r.get("payload", {}),
            }
            for r in results
        ]

        duration = time.time() - start_time

        response = SearchWithIntentResponse(
            query=query,
            intent=IntentClassificationResponse(
                query=query,
                primary_intent=classification.primary_intent.value,
                confidence=classification.confidence,
                secondary_intents=[
                    {"intent": intent.value, "confidence": conf}
                    for intent, conf in classification.secondary_intents
                ],
                recommended_filters=classification.recommended_filters,
                recommended_limit=classification.recommended_limit,
                recommended_score_threshold=classification.recommended_score_threshold
            ),
            results=formatted_results,
            total_results=len(formatted_results),
            search_time=duration
        )

        logger.info(
            f"Smart search completed - Intent: {classification.primary_intent.value}, "
            f"Results: {len(formatted_results)}, Duration: {duration:.3f}s"
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Smart search failed - Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 1️⃣3️⃣ ADVANCED SEARCH (with Enhanced Analysis & Fallback)
# ---------------------------------------------------------
@app.post("/search/advanced")
def advanced_search(
    query: str = Body(..., embed=True, description="Search query"),
    min_results: int = Body(1, description="Minimum results before fallback"),
    max_results: int = Body(20, description="Maximum results to return")
):
    """
    Advanced search with enhanced query analysis and automatic fallback strategies.

    **Key Features:**
    - Enhanced scope detection (document vs section level)
    - Specificity analysis (broad vs specific)
    - Automatic summary routing
    - Fallback strategies when no results found
    - Post-filtering to exclude summaries for specific queries

    **Handles Edge Cases:**
    - "what is covered in document" → Summary (with fallback to sections)
    - "git staging area explanation" → Sections only (excludes summary)
    - "git working directory and staging area" → Sections (multiple topics)
    - "overview of git" → Summary first (with fallback)

    **Search Strategies:**
    1. **summary_first**: Search summaries first, fallback to sections if needed
    2. **section_only**: Search sections only, exclude summaries
    3. **hybrid**: Search all chunks, rank by relevance
    4. **summary_only**: Only return document summaries

    Example request:
    ```json
    {
        "query": "what is covered in document",
        "min_results": 1,
        "max_results": 10
    }
    ```

    Example response:
    ```json
    {
        "query": "what is covered in document",
        "analysis": {
            "scope": "document_level",
            "specificity": "very_broad",
            "strategy": "summary_first",
            "confidence": 0.85
        },
        "results": [...],
        "total_results": 3,
        "search_time": 0.123,
        "fallback_used": false,
        "summary_excluded": false
    }
    ```
    """
    start_time = time.time()
    logger.info(f"Advanced search request - Query: '{query[:50]}...'")

    try:
        # Step 1: Analyze query
        analysis = query_analyzer.analyze(query)

        logger.info(
            f"Query analyzed - Scope: {analysis.scope.value}, "
            f"Specificity: {analysis.specificity.value}, "
            f"Strategy: {analysis.search_strategy}"
        )

        # Step 2: Execute search based on strategy
        results = []
        fallback_used = False
        summary_excluded_count = 0

        if analysis.search_strategy == 'summary_first':
            # Try summary first
            logger.debug("Executing summary-first search...")
            if analysis.recommended_filters:
                results = service.search(
                    query=query,
                    limit=max_results,
                    filters=analysis.recommended_filters,
                    score_threshold=0.5
                )
            else:
                results = service.search(query=query, limit=max_results, score_threshold=0.5)

            # Fallback to sections if not enough results
            if len(results) < min_results and analysis.fallback_strategy:
                logger.info(f"Insufficient results ({len(results)}), using fallback strategy")
                fallback_results = service.search(query=query, limit=max_results, score_threshold=0.4)

                # Filter out summaries from fallback
                fallback_results = [
                    r for r in fallback_results
                    if r.get('payload', {}).get('section_title') != '[DOCUMENT SUMMARY]'
                ]

                results.extend(fallback_results)
                fallback_used = True

        elif analysis.search_strategy == 'section_only':
            # Search all, then exclude summaries
            logger.debug("Executing section-only search...")
            all_results = service.search(query=query, limit=max_results * 2, score_threshold=0.4)

            # Post-filter: exclude summary chunks
            results = [
                r for r in all_results
                if r.get('payload', {}).get('section_title') != '[DOCUMENT SUMMARY]'
            ]

            summary_excluded_count = len(all_results) - len(results)
            results = results[:max_results]

            logger.debug(f"Excluded {summary_excluded_count} summary chunks")

        elif analysis.search_strategy == 'hybrid':
            # Search all chunks, let relevance decide
            logger.debug("Executing hybrid search...")
            results = service.search(query=query, limit=max_results, score_threshold=0.45)

            # If specificity is high and summaries appear, consider excluding
            if analysis.specificity_score > 0.7:
                non_summary_results = [
                    r for r in results
                    if r.get('payload', {}).get('section_title') != '[DOCUMENT SUMMARY]'
                ]

                # Only use non-summary if we have enough
                if len(non_summary_results) >= min_results:
                    summary_excluded_count = len(results) - len(non_summary_results)
                    results = non_summary_results
                    logger.debug(f"Hybrid: Excluded {summary_excluded_count} summaries due to high specificity")

        else:  # 'summary_only'
            logger.debug("Executing summary-only search...")
            results = service.search(
                query=query,
                limit=max_results,
                filters={"section_title": "[DOCUMENT SUMMARY]"},
                score_threshold=0.5
            )

            # Fallback if no summaries found
            if len(results) == 0:
                logger.info("No summary results, falling back to sections")
                results = service.search(query=query, limit=max_results, score_threshold=0.4)
                fallback_used = True

        # Step 3: Format results
        formatted_results = [
            {
                "id": str(r.get("payload", {}).get("chunk_id")),
                "score": r.get("score"),
                "text": r.get("payload", {}).get("text"),
                "metadata": r.get("payload", {}),
                "is_summary": r.get("payload", {}).get("section_title") == "[DOCUMENT SUMMARY]"
            }
            for r in results[:max_results]
        ]

        duration = time.time() - start_time

        response = {
            "query": query,
            "analysis": {
                "scope": analysis.scope.value,
                "specificity": analysis.specificity.value,
                "strategy": analysis.search_strategy,
                "confidence": analysis.confidence,
                "should_exclude_summary": analysis.should_exclude_summary,
                "summary_only": analysis.summary_only
            },
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_time": duration,
            "fallback_used": fallback_used,
            "summary_excluded": summary_excluded_count > 0,
            "summary_excluded_count": summary_excluded_count
        }

        logger.info(
            f"Advanced search completed - Strategy: {analysis.search_strategy}, "
            f"Results: {len(formatted_results)}, Fallback: {fallback_used}, "
            f"Summaries excluded: {summary_excluded_count}, Duration: {duration:.3f}s"
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Advanced search failed - Duration: {duration:.3f}s, Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))
"""
