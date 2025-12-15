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

# Initialize logger
logger = get_logger(__name__)

# ---------------------------------------------------------
# Load config from environment (Docker passes variables)
# ---------------------------------------------------------
logger.info("Loading configuration from environment variables")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
logger.info(f"Configuration loaded - QDRANT_URL: {QDRANT_URL}, EMBEDDING_MODEL: {EMBEDDING_MODEL}")

service_config = ServiceConfig(
    qdrant_url="http://localhost:6333",  # Your existing Qdrant
    qdrant_collection="documents_services",        # Use existing or create new
    embedding_model_name="all-MiniLM-L6-v2",
    vector_size=384
)
logger.info(f"Service config created - Collection: {service_config.qdrant_collection}, Vector size: {service_config.vector_size}")

logger.info("Initializing DocumentToVectorService")
service = DocumentToVectorService(service_config)
logger.info("DocumentToVectorService initialized successfully")

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
logger.info("Creating FastAPI application")
app = FastAPI(title="KnowVec RAG Pipeline API")
logger.info("FastAPI application created successfully")

# CORS
logger.info("Configuring CORS middleware")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")


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
async def process_document(file: UploadFile = File(...)):
    """
    Upload a document → Extract → Normalize → Chunk → Embed → Store in Qdrant
    """
    start_time = time.time()
    logger.info(f"Document processing started - Filename: {file.filename}, Content-Type: {file.content_type}")

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

        return {
            "message": "Document processed successfully",
            "doc_id": result.doc_id,
            "pages_extracted": result.pages_extracted,
            "chunks_created": result.chunks_created,
            "unique_chunks": result.unique_chunks,
            "vectors_stored": result.vectors_stored,
            "total_time": result.total_time
        }

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
