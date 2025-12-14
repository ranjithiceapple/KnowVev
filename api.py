from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import shutil
import uuid
import os

from document_to_vector_service import (
    DocumentToVectorService,
    ServiceConfig
)

# ---------------------------------------------------------
# Load config from environment (Docker passes variables)
# ---------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

service_config = ServiceConfig(
    qdrant_url="http://localhost:6333",  # Your existing Qdrant
    qdrant_collection="documents_services",        # Use existing or create new
    embedding_model_name="all-MiniLM-L6-v2",
    vector_size=384
)

service = DocumentToVectorService(service_config)

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(title="KnowVec RAG Pipeline API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
class SearchResponse(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get("/")
def root():
    return {"message": "KnowVec RAG Pipeline API is running 🚀"}


# ---------------------------------------------------------
# 1️⃣ DOCUMENT INGESTION ENDPOINT
# ---------------------------------------------------------
@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """
    Upload a document → Extract → Normalize → Chunk → Embed → Store in Qdrant
    """
    try:
        # Temporary file path
        temp_file_path = f"/tmp/{uuid.uuid4()}_{file.filename}"

        # Save uploaded file to temp folder
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process document using the full pipeline
        result = service.process_document(temp_file_path)

        # Cleanup
        os.remove(temp_file_path)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result.error_message}"
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

    except Exception as e:
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
    try:
        results = service.search(
            query=q,
            limit=limit,
            score_threshold=score_threshold
        )

        formatted = [
            SearchResponse(
                id=str(r.payload.get("chunk_id")),
                score=r.score,
                text=r.payload.get("text", ""),
                metadata=r.payload
            )
            for r in results
        ]

        return formatted

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 3️⃣ COLLECTION STATS
# ---------------------------------------------------------
@app.get("/stats")
def get_stats():
    """
    Returns Qdrant collection stats
    """
    try:
        return service.get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
