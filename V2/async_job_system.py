"""
Asynchronous Job-Based Ingestion Pipeline

Features:
- Immediate job_id return
- Stage-by-stage resumability
- Failure isolation (completed stages preserved)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
from pathlib import Path


# ============================================================================
# JOB STATE MACHINE
# ============================================================================

class JobState(str, Enum):
    """Job states"""
    PENDING = "pending"              # Created, not started
    EXTRACTING = "extracting"        # Stage 0: Running extraction
    NORMALIZING = "normalizing"      # Stage 1: Creating DocumentIR
    CHUNKING = "chunking"            # Stage 2: Creating chunks
    EMBEDDING = "embedding"          # Stage 3: Generating embeddings
    ENRICHING = "enriching"          # Stage 4: Topic/summary generation
    COMPLETED = "completed"          # All stages done
    FAILED = "failed"                # Unrecoverable error
    PAUSED = "paused"                # Manual pause
    CANCELLED = "cancelled"          # User cancelled


class StageState(str, Enum):
    """Individual stage states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# JOB MODEL
# ============================================================================

@dataclass
class StageStatus:
    """Status of a single pipeline stage"""
    stage_name: str
    stage_index: int
    state: StageState = StageState.PENDING

    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Artifacts produced
    artifacts: List[str] = field(default_factory=list)

    # Progress (0-100)
    progress_percent: int = 0
    progress_message: str = ""


@dataclass
class IngestionJob:
    """Job tracking ingestion pipeline execution"""
    job_id: str
    doc_id: str
    source_file: str

    # Overall job state
    state: JobState = JobState.PENDING

    # Stage statuses
    stages: Dict[str, StageStatus] = field(default_factory=dict)

    # Config
    pipeline_config: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Current stage
    current_stage: Optional[str] = None

    # Retry info
    retry_count: int = 0
    max_retries: int = 3

    # Results
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

        # Initialize stages
        if not self.stages:
            stage_names = [
                "extraction", "normalization", "chunking",
                "embedding", "enrichment"
            ]
            for i, name in enumerate(stage_names):
                self.stages[name] = StageStatus(
                    stage_name=name,
                    stage_index=i
                )

    def to_dict(self) -> dict:
        return {
            'job_id': self.job_id,
            'doc_id': self.doc_id,
            'source_file': self.source_file,
            'state': self.state.value,
            'current_stage': self.current_stage,
            'stages': {k: {
                'state': v.state.value,
                'started_at': v.started_at,
                'completed_at': v.completed_at,
                'duration_seconds': v.duration_seconds,
                'progress_percent': v.progress_percent,
                'progress_message': v.progress_message,
                'error_message': v.error_message,
                'artifacts': v.artifacts
            } for k, v in self.stages.items()},
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'retry_count': self.retry_count,
            'result': self.result
        }


# ============================================================================
# JOB STORAGE
# ============================================================================

class JobStore:
    """Persistent job storage"""

    def __init__(self, base_dir: str = "./jobs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def save_job(self, job: IngestionJob):
        """Save job to disk"""
        job_file = self.base_dir / f"{job.job_id}.json"
        job_file.write_text(json.dumps(job.to_dict(), indent=2))

    def load_job(self, job_id: str) -> Optional[IngestionJob]:
        """Load job from disk"""
        job_file = self.base_dir / f"{job_id}.json"
        if not job_file.exists():
            return None

        data = json.loads(job_file.read_text())

        # Reconstruct job
        job = IngestionJob(
            job_id=data['job_id'],
            doc_id=data['doc_id'],
            source_file=data['source_file'],
            state=JobState(data['state']),
            created_at=data['created_at'],
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            current_stage=data.get('current_stage'),
            retry_count=data['retry_count'],
            result=data.get('result'),
            pipeline_config=data.get('pipeline_config', {})
        )

        # Reconstruct stages
        for stage_name, stage_data in data['stages'].items():
            job.stages[stage_name] = StageStatus(
                stage_name=stage_name,
                stage_index=job.stages[stage_name].stage_index,
                state=StageState(stage_data['state']),
                started_at=stage_data.get('started_at'),
                completed_at=stage_data.get('completed_at'),
                duration_seconds=stage_data.get('duration_seconds'),
                progress_percent=stage_data.get('progress_percent', 0),
                progress_message=stage_data.get('progress_message', ''),
                error_message=stage_data.get('error_message'),
                artifacts=stage_data.get('artifacts', [])
            )

        return job

    def list_jobs(self, state: Optional[JobState] = None) -> List[str]:
        """List all jobs, optionally filtered by state"""
        jobs = []
        for job_file in self.base_dir.glob("*.json"):
            if state:
                data = json.loads(job_file.read_text())
                if data['state'] == state.value:
                    jobs.append(job_file.stem)
            else:
                jobs.append(job_file.stem)
        return jobs


# ============================================================================
# PIPELINE EXECUTOR
# ============================================================================

class PipelineExecutor:
    """Executes pipeline stages with state tracking"""

    def __init__(self, job_store: JobStore, artifact_base_dir: str = "./artifacts"):
        self.job_store = job_store
        self.artifact_base_dir = Path(artifact_base_dir)

    def start_stage(self, job: IngestionJob, stage_name: str):
        """Mark stage as started"""
        stage = job.stages[stage_name]
        stage.state = StageState.IN_PROGRESS
        stage.started_at = datetime.now().isoformat()

        job.current_stage = stage_name
        job.state = self._get_job_state_for_stage(stage_name)

        self.job_store.save_job(job)

    def complete_stage(
        self, job: IngestionJob, stage_name: str,
        artifacts: List[str], result: Optional[dict] = None
    ):
        """Mark stage as completed"""
        stage = job.stages[stage_name]
        stage.state = StageState.COMPLETED
        stage.completed_at = datetime.now().isoformat()
        stage.artifacts = artifacts
        stage.progress_percent = 100

        # Calculate duration
        if stage.started_at:
            start = datetime.fromisoformat(stage.started_at)
            end = datetime.fromisoformat(stage.completed_at)
            stage.duration_seconds = (end - start).total_seconds()

        # Check if all stages complete
        if self._all_stages_complete(job):
            job.state = JobState.COMPLETED
            job.completed_at = datetime.now().isoformat()
            job.result = result

        self.job_store.save_job(job)

    def fail_stage(
        self, job: IngestionJob, stage_name: str,
        error_message: str, traceback: Optional[str] = None
    ):
        """Mark stage as failed"""
        stage = job.stages[stage_name]
        stage.state = StageState.FAILED
        stage.error_message = error_message
        stage.error_traceback = traceback
        stage.completed_at = datetime.now().isoformat()

        # Decide if job should fail or retry
        if job.retry_count < job.max_retries:
            job.retry_count += 1
            job.state = JobState.PAUSED  # Manual resume required
        else:
            job.state = JobState.FAILED

        self.job_store.save_job(job)

    def update_progress(
        self, job: IngestionJob, stage_name: str,
        progress_percent: int, message: str = ""
    ):
        """Update stage progress"""
        stage = job.stages[stage_name]
        stage.progress_percent = min(100, max(0, progress_percent))
        stage.progress_message = message

        self.job_store.save_job(job)

    def resume_from_stage(self, job: IngestionJob, stage_name: str):
        """Resume job from a specific stage"""
        # Reset stage and subsequent stages
        stage_idx = job.stages[stage_name].stage_index

        for name, stage in job.stages.items():
            if stage.stage_index >= stage_idx:
                stage.state = StageState.PENDING
                stage.started_at = None
                stage.completed_at = None
                stage.error_message = None

        job.state = JobState.PENDING
        job.current_stage = None

        self.job_store.save_job(job)

    def _get_job_state_for_stage(self, stage_name: str) -> JobState:
        """Map stage name to job state"""
        mapping = {
            'extraction': JobState.EXTRACTING,
            'normalization': JobState.NORMALIZING,
            'chunking': JobState.CHUNKING,
            'embedding': JobState.EMBEDDING,
            'enrichment': JobState.ENRICHING
        }
        return mapping.get(stage_name, JobState.PENDING)

    def _all_stages_complete(self, job: IngestionJob) -> bool:
        """Check if all stages are complete"""
        return all(
            stage.state == StageState.COMPLETED
            for stage in job.stages.values()
        )


# ============================================================================
# API ENDPOINTS (FastAPI)
# ============================================================================

"""
API Implementation (FastAPI):

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

app = FastAPI()
job_store = JobStore()
executor = PipelineExecutor(job_store)


# ========== MODELS ==========

class JobSubmitRequest(BaseModel):
    source_file: str
    pipeline_config: Optional[Dict[str, Any]] = {}


class JobStatusResponse(BaseModel):
    job_id: str
    state: str
    current_stage: Optional[str]
    stages: Dict[str, Any]
    progress_percent: int
    result: Optional[Dict[str, Any]]


# ========== ENDPOINTS ==========

@app.post("/api/v1/jobs/submit")
async def submit_job(
    request: JobSubmitRequest,
    background_tasks: BackgroundTasks
):
    '''Submit document for ingestion'''
    import uuid

    job_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    job = IngestionJob(
        job_id=job_id,
        doc_id=doc_id,
        source_file=request.source_file,
        pipeline_config=request.pipeline_config
    )

    job_store.save_job(job)

    # Run pipeline in background
    background_tasks.add_task(run_pipeline, job_id)

    return {
        'job_id': job_id,
        'doc_id': doc_id,
        'state': job.state.value,
        'message': 'Job submitted successfully'
    }


@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    '''Get job status'''
    job = job_store.load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Calculate overall progress
    total_progress = sum(s.progress_percent for s in job.stages.values())
    avg_progress = total_progress // len(job.stages)

    return {
        'job_id': job.job_id,
        'doc_id': job.doc_id,
        'state': job.state.value,
        'current_stage': job.current_stage,
        'progress_percent': avg_progress,
        'stages': job.to_dict()['stages'],
        'created_at': job.created_at,
        'started_at': job.started_at,
        'completed_at': job.completed_at,
        'result': job.result
    }


@app.post("/api/v1/jobs/{job_id}/resume")
async def resume_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    from_stage: Optional[str] = None
):
    '''Resume failed/paused job'''
    job = job_store.load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.state not in [JobState.FAILED, JobState.PAUSED]:
        raise HTTPException(400, "Job is not resumable")

    # Find failed stage or use specified
    resume_stage = from_stage
    if not resume_stage:
        for name, stage in job.stages.items():
            if stage.state == StageState.FAILED:
                resume_stage = name
                break

    if not resume_stage:
        raise HTTPException(400, "No failed stage to resume from")

    executor.resume_from_stage(job, resume_stage)
    background_tasks.add_task(run_pipeline, job_id)

    return {'message': f'Job resumed from {resume_stage}'}


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    '''Cancel running job'''
    job = job_store.load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.state in [JobState.COMPLETED, JobState.FAILED]:
        raise HTTPException(400, "Job already finished")

    job.state = JobState.CANCELLED
    job_store.save_job(job)

    return {'message': 'Job cancelled'}


@app.get("/api/v1/jobs")
async def list_jobs(state: Optional[str] = None):
    '''List all jobs'''
    filter_state = JobState(state) if state else None
    job_ids = job_store.list_jobs(filter_state)

    jobs = []
    for job_id in job_ids:
        job = job_store.load_job(job_id)
        if job:
            jobs.append({
                'job_id': job.job_id,
                'doc_id': job.doc_id,
                'state': job.state.value,
                'current_stage': job.current_stage,
                'created_at': job.created_at
            })

    return {'jobs': jobs, 'total': len(jobs)}


# ========== PIPELINE EXECUTION ==========

async def run_pipeline(job_id: str):
    '''Execute pipeline stages'''
    job = job_store.load_job(job_id)
    if not job:
        return

    job.started_at = datetime.now().isoformat()
    job_store.save_job(job)

    from artifact_persistence import ArtifactManager
    artifacts = ArtifactManager('./artifacts', job.doc_id)

    try:
        # Stage 0: Extraction
        if job.stages['extraction'].state != StageState.COMPLETED:
            await run_extraction_stage(job, artifacts)

        # Stage 1: Normalization
        if job.stages['normalization'].state != StageState.COMPLETED:
            await run_normalization_stage(job, artifacts)

        # Stage 2: Chunking
        if job.stages['chunking'].state != StageState.COMPLETED:
            await run_chunking_stage(job, artifacts)

        # Stage 3: Embedding
        if job.stages['embedding'].state != StageState.COMPLETED:
            await run_embedding_stage(job, artifacts)

        # Stage 4: Enrichment
        if job.stages['enrichment'].state != StageState.COMPLETED:
            await run_enrichment_stage(job, artifacts)

    except Exception as e:
        # Current stage already marked as failed by stage function
        pass


async def run_extraction_stage(job: IngestionJob, artifacts: ArtifactManager):
    '''Stage 0: Extract document'''
    executor.start_stage(job, 'extraction')

    try:
        # Run extraction
        from pdf2llm import extract_pdf_enhanced

        executor.update_progress(job, 'extraction', 10, "Starting extraction")

        raw_output = extract_pdf_enhanced(job.source_file)

        executor.update_progress(job, 'extraction', 80, "Saving artifacts")

        # Save artifacts
        from artifact_persistence import Stage00_RawArtifacts
        import hashlib

        raw_artifacts = Stage00_RawArtifacts(
            extraction_output=raw_output,
            extractor_name='pymupdf-enhanced',
            extractor_version='1.0',
            extraction_timestamp=datetime.now().isoformat(),
            extraction_config={},
            source_filename=job.source_file,
            source_hash=hashlib.sha256(
                Path(job.source_file).read_bytes()
            ).hexdigest(),
            source_size_bytes=Path(job.source_file).stat().st_size,
            source_format='pdf'
        )

        artifacts.save_stage00_raw(raw_artifacts)

        executor.complete_stage(
            job, 'extraction',
            artifacts=[str(artifacts.paths.extraction_raw)]
        )

    except Exception as e:
        import traceback
        executor.fail_stage(
            job, 'extraction',
            str(e), traceback.format_exc()
        )
        raise


async def run_normalization_stage(job: IngestionJob, artifacts: ArtifactManager):
    '''Stage 1: Create DocumentIR'''
    executor.start_stage(job, 'normalization')

    try:
        # Load raw artifacts
        raw = artifacts.load_stage00_raw()

        executor.update_progress(job, 'normalization', 20, "Converting to IR")

        # Convert to IR
        from document_ir_mappers import map_pdf_extraction_to_ir

        ir = map_pdf_extraction_to_ir(
            job.source_file,
            raw.extraction_output,
            raw.extractor_name
        )

        executor.update_progress(job, 'normalization', 80, "Saving IR")

        # Save
        from artifact_persistence import Stage01_NormalizedArtifacts

        normalized = Stage01_NormalizedArtifacts(
            document_ir=ir.to_dict(),
            normalization_rules=['remove_whitespace'],
            normalization_stats={},
            full_text=ir.to_markdown()
        )

        artifacts.save_stage01_normalized(normalized)

        executor.complete_stage(
            job, 'normalization',
            artifacts=[str(artifacts.paths.document_ir)]
        )

    except Exception as e:
        import traceback
        executor.fail_stage(
            job, 'normalization',
            str(e), traceback.format_exc()
        )
        raise


# Similar for chunking, embedding, enrichment stages...
"""


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_async_pipeline():
    """Example: Async job submission and tracking"""

    import uuid

    job_store = JobStore("./example_jobs")
    executor = PipelineExecutor(job_store)

    # Submit job
    job_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    job = IngestionJob(
        job_id=job_id,
        doc_id=doc_id,
        source_file="document.pdf",
        pipeline_config={'chunking': {'max_size': 1000}}
    )

    job_store.save_job(job)
    print(f"Job submitted: {job_id}")

    # Simulate stage execution
    print("\nStage 1: Extraction")
    executor.start_stage(job, 'extraction')
    executor.update_progress(job, 'extraction', 50, "Extracting pages")
    executor.complete_stage(job, 'extraction', ['raw.json'])

    print("\nStage 2: Normalization")
    executor.start_stage(job, 'normalization')
    executor.update_progress(job, 'normalization', 30, "Building IR")
    # Simulate failure
    executor.fail_stage(job, 'normalization', "Out of memory")

    print(f"\nJob failed at normalization stage")
    print(f"Retry count: {job.retry_count}/{job.max_retries}")

    # Resume from failed stage
    print("\nResuming from normalization...")
    executor.resume_from_stage(job, 'normalization')

    # Re-run normalization
    executor.start_stage(job, 'normalization')
    executor.complete_stage(job, 'normalization', ['ir.json'])

    # Continue with other stages
    for stage in ['chunking', 'embedding', 'enrichment']:
        executor.start_stage(job, stage)
        executor.complete_stage(job, stage, [f'{stage}.json'])

    print(f"\nJob completed: {job.state}")
    print(f"Total duration: {(datetime.fromisoformat(job.completed_at) - datetime.fromisoformat(job.created_at)).total_seconds()}s")


if __name__ == "__main__":
    example_async_pipeline()
