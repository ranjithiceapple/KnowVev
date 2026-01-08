"""
FastAPI endpoints for V2 pipeline
"""

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
import shutil
from pathlib import Path

from ingestion_pipeline_v2 import IngestionPipelineV2, PipelineConfig

app = FastAPI(title="KnowVec API V2")
pipeline = IngestionPipelineV2()


class JobSubmitResponse(BaseModel):
    job_id: str
    doc_id: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    state: str
    current_stage: Optional[str]
    progress: Dict


@app.post("/v2/ingest", response_model=JobSubmitResponse)
async def ingest_document(file: UploadFile, background_tasks: BackgroundTasks):
    """Upload and ingest document"""

    # Save uploaded file
    temp_path = f"./temp/{file.filename}"
    Path(temp_path).parent.mkdir(exist_ok=True)

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Submit job
    job_id = pipeline.ingest_document(temp_path)
    job = pipeline.job_store.load_job(job_id)

    return JobSubmitResponse(
        job_id=job_id,
        doc_id=job.doc_id,
        message="Document ingestion started"
    )


@app.get("/v2/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status"""

    status = pipeline.get_job_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")

    # Calculate progress
    completed = sum(1 for s in status['stages'].values() if s['state'] == 'completed')
    total = len(status['stages'])
    progress_pct = int(completed / total * 100)

    return JobStatusResponse(
        job_id=job_id,
        state=status['state'],
        current_stage=status['current_stage'],
        progress={'percent': progress_pct, 'stages': status['stages']}
    )


@app.get("/v2/search")
async def search(query: str, limit: int = 10):
    """Search with new metadata"""

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_vector = model.encode(query).tolist()

    results = pipeline.qdrant.search(
        vector=query_vector,
        limit=limit
    )

    return {
        'query': query,
        'results': [
            {
                'chunk_id': r['id'],
                'score': r['score'],
                'text': r['payload'].get('text', ''),
                'pages': f"{r['payload']['physical_location']['page_start']}-{r['payload']['physical_location']['page_end']}",
                'section': r['payload'].get('logical_section', {}).get('breadcrumb') if r['payload'].get('logical_section') else None,
                'confidence': r['payload'].get('confidence', 1.0)
            }
            for r in results
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
