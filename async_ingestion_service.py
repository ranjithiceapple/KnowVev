"""
Async Document Ingestion Service

Separates document ingestion from HTTP requests using background workers.
Manages cache and temp file cleanup to prevent disk exhaustion.

Architecture:
    API Request → Job Queue → Background Worker → Status Updates

    Client flow:
    1. POST /ingest → Returns job_id immediately
    2. GET /ingest/{job_id}/status → Poll for completion
    3. GET /ingest/{job_id}/result → Get final result

Features:
- Async processing with job queue
- Automatic temp file cleanup
- Model cache management
- Progress tracking
- Graceful shutdown
"""

import os
import uuid
import time
import shutil
import threading
import tempfile
from pathlib import Path
from queue import Queue, Empty
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import gc

from logger_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# JOB STATUS AND MODELS
# =============================================================================

class JobStatus(str, Enum):
    """Ingestion job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IngestionJob:
    """Represents an ingestion job in the queue."""
    job_id: str
    file_path: str  # Temp file path
    original_filename: str
    doc_id: str
    project_id: str

    # Status tracking
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0-100
    current_stage: str = "queued"

    # Timing
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Result (populated on completion)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # Custom metadata
    custom_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'job_id': self.job_id,
            'doc_id': self.doc_id,
            'project_id': self.project_id,
            'original_filename': self.original_filename,
            'status': self.status.value,
            'progress': self.progress,
            'current_stage': self.current_stage,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'result': self.result,
            'error_message': self.error_message
        }


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """
    Manages cache directories for marker-pdf and embedding models.

    Cleans up caches periodically to prevent disk exhaustion.
    """

    # Default cache locations
    MARKER_CACHE_DIRS = [
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch",
        Path("/tmp") / "marker_cache",
        Path("/tmp") / "surya_cache",
    ]

    TEMP_PATTERNS = [
        "/tmp/*_*.pdf",  # Temp uploaded files
        "/tmp/marker_*",
        "/tmp/torch_*",
    ]

    def __init__(
        self,
        max_cache_size_gb: float = 10.0,
        max_temp_age_hours: int = 1,
        cleanup_interval_minutes: int = 30
    ):
        self.max_cache_size_gb = max_cache_size_gb
        self.max_temp_age_hours = max_temp_age_hours
        self.cleanup_interval_minutes = cleanup_interval_minutes

        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

        logger.info(
            f"CacheManager initialized - max_cache: {max_cache_size_gb}GB, "
            f"temp_age: {max_temp_age_hours}h, interval: {cleanup_interval_minutes}m"
        )

    def start_periodic_cleanup(self):
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.warning("Cleanup thread already running")
            return

        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CacheCleanupThread"
        )
        self._cleanup_thread.start()
        logger.info("Started periodic cache cleanup thread")

    def stop_periodic_cleanup(self):
        """Stop background cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
            logger.info("Stopped cache cleanup thread")

    def _cleanup_loop(self):
        """Background cleanup loop."""
        while not self._stop_cleanup.is_set():
            try:
                self.cleanup_all()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

            # Wait for next cleanup interval
            self._stop_cleanup.wait(self.cleanup_interval_minutes * 60)

    def cleanup_all(self) -> Dict[str, Any]:
        """Run all cleanup tasks."""
        stats = {
            'temp_files_removed': 0,
            'temp_bytes_freed': 0,
            'cache_bytes_freed': 0,
            'gc_collected': 0
        }

        # 1. Clean old temp files
        temp_stats = self.cleanup_temp_files()
        stats['temp_files_removed'] = temp_stats['files_removed']
        stats['temp_bytes_freed'] = temp_stats['bytes_freed']

        # 2. Clean model caches if over limit
        cache_stats = self.cleanup_model_caches()
        stats['cache_bytes_freed'] = cache_stats['bytes_freed']

        # 3. Force garbage collection
        stats['gc_collected'] = gc.collect()

        logger.info(
            f"Cleanup complete - temp: {stats['temp_files_removed']} files "
            f"({stats['temp_bytes_freed'] / 1024 / 1024:.1f}MB), "
            f"cache: {stats['cache_bytes_freed'] / 1024 / 1024:.1f}MB freed, "
            f"gc: {stats['gc_collected']} objects"
        )

        return stats

    def cleanup_temp_files(self) -> Dict[str, Any]:
        """Remove old temporary files."""
        stats = {'files_removed': 0, 'bytes_freed': 0}
        cutoff = datetime.now() - timedelta(hours=self.max_temp_age_hours)

        # Clean /tmp directory
        tmp_dir = Path("/tmp")
        if tmp_dir.exists():
            for item in tmp_dir.iterdir():
                try:
                    # Check if it's our temp file (UUID pattern)
                    if item.is_file() and "_" in item.name:
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        if mtime < cutoff:
                            size = item.stat().st_size
                            item.unlink()
                            stats['files_removed'] += 1
                            stats['bytes_freed'] += size
                            logger.debug(f"Removed old temp file: {item.name}")
                except Exception as e:
                    logger.debug(f"Could not remove {item}: {e}")

        return stats

    def cleanup_model_caches(self) -> Dict[str, Any]:
        """Clean model caches if over size limit."""
        stats = {'bytes_freed': 0}

        total_cache_size = self._get_total_cache_size()
        max_bytes = self.max_cache_size_gb * 1024 * 1024 * 1024

        if total_cache_size <= max_bytes:
            logger.debug(f"Cache size OK: {total_cache_size / 1024 / 1024 / 1024:.2f}GB")
            return stats

        logger.warning(
            f"Cache size {total_cache_size / 1024 / 1024 / 1024:.2f}GB "
            f"exceeds limit {self.max_cache_size_gb}GB"
        )

        # Clean oldest files from cache directories
        for cache_dir in self.MARKER_CACHE_DIRS:
            if not cache_dir.exists():
                continue

            try:
                # Get all files sorted by modification time (oldest first)
                files = []
                for f in cache_dir.rglob("*"):
                    if f.is_file():
                        files.append((f, f.stat().st_mtime, f.stat().st_size))

                files.sort(key=lambda x: x[1])  # Sort by mtime

                # Remove oldest files until under limit
                for file_path, mtime, size in files:
                    if self._get_total_cache_size() <= max_bytes * 0.8:  # Target 80%
                        break
                    try:
                        file_path.unlink()
                        stats['bytes_freed'] += size
                        logger.debug(f"Removed cache file: {file_path}")
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Error cleaning cache dir {cache_dir}: {e}")

        return stats

    def _get_total_cache_size(self) -> int:
        """Get total size of all cache directories."""
        total = 0
        for cache_dir in self.MARKER_CACHE_DIRS:
            if cache_dir.exists():
                try:
                    for f in cache_dir.rglob("*"):
                        if f.is_file():
                            total += f.stat().st_size
                except Exception:
                    pass
        return total

    def cleanup_job_temp_file(self, file_path: str):
        """Clean up a specific job's temp file."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.debug(f"Removed job temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not remove temp file {file_path}: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        stats = {
            'total_cache_size_mb': 0,
            'cache_directories': {}
        }

        for cache_dir in self.MARKER_CACHE_DIRS:
            if cache_dir.exists():
                try:
                    size = sum(
                        f.stat().st_size
                        for f in cache_dir.rglob("*")
                        if f.is_file()
                    )
                    stats['cache_directories'][str(cache_dir)] = size / 1024 / 1024
                    stats['total_cache_size_mb'] += size / 1024 / 1024
                except Exception:
                    pass

        return stats


# =============================================================================
# ASYNC INGESTION SERVICE
# =============================================================================

class AsyncIngestionService:
    """
    Async document ingestion service with job queue.

    Separates HTTP requests from long-running ingestion tasks.
    """

    def __init__(
        self,
        document_service,  # DocumentToVectorService instance
        max_workers: int = 2,
        max_queue_size: int = 100,
        job_retention_hours: int = 24
    ):
        self.document_service = document_service
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.job_retention_hours = job_retention_hours

        # Job storage
        self._jobs: Dict[str, IngestionJob] = {}
        self._job_lock = threading.Lock()

        # Job queue
        self._queue: Queue = Queue(maxsize=max_queue_size)

        # Worker pool
        self._executor: Optional[ThreadPoolExecutor] = None
        self._workers_running = False

        # Cache manager
        self.cache_manager = CacheManager()

        logger.info(
            f"AsyncIngestionService initialized - workers: {max_workers}, "
            f"queue_size: {max_queue_size}"
        )

    def start(self):
        """Start the async ingestion service."""
        if self._workers_running:
            logger.warning("Workers already running")
            return

        # Start worker threads
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="IngestionWorker"
        )

        # Submit worker tasks
        for i in range(self.max_workers):
            self._executor.submit(self._worker_loop, i)

        self._workers_running = True

        # Start cache cleanup
        self.cache_manager.start_periodic_cleanup()

        logger.info(f"Started {self.max_workers} ingestion workers")

    def stop(self):
        """Stop the async ingestion service gracefully."""
        logger.info("Stopping async ingestion service...")

        self._workers_running = False

        # Signal workers to stop
        for _ in range(self.max_workers):
            self._queue.put(None)  # Poison pill

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)

        # Stop cache cleanup
        self.cache_manager.stop_periodic_cleanup()

        logger.info("Async ingestion service stopped")

    def submit_job(
        self,
        file_path: str,
        original_filename: str,
        doc_id: str,
        project_id: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionJob:
        """
        Submit a document for async ingestion.

        Args:
            file_path: Path to the uploaded file (temp location)
            original_filename: Original file name
            doc_id: Document ID (from API service)
            project_id: Project ID (from API service)
            custom_metadata: Optional additional metadata

        Returns:
            IngestionJob with job_id for tracking

        Raises:
            RuntimeError: If queue is full
        """
        if self._queue.full():
            raise RuntimeError("Ingestion queue is full. Please try again later.")

        # Create job
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            file_path=file_path,
            original_filename=original_filename,
            doc_id=doc_id,
            project_id=project_id,
            custom_metadata=custom_metadata
        )

        # Store job
        with self._job_lock:
            self._jobs[job.job_id] = job

        # Add to queue
        self._queue.put(job.job_id)

        logger.info(
            f"[{job.job_id[:8]}] Job submitted - doc_id: {doc_id}, "
            f"project_id: {project_id}, file: {original_filename}"
        )

        return job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """Get job by ID."""
        with self._job_lock:
            return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status as dictionary."""
        job = self.get_job(job_id)
        if job:
            return job.to_dict()
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow().isoformat()
                # Clean up temp file
                self.cache_manager.cleanup_job_temp_file(job.file_path)
                logger.info(f"[{job_id[:8]}] Job cancelled")
                return True
        return False

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._job_lock:
            status_counts = {}
            for job in self._jobs.values():
                status = job.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'queue_size': self._queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'total_jobs': len(self._jobs),
            'jobs_by_status': status_counts,
            'workers': self.max_workers,
            'workers_running': self._workers_running
        }

    def _worker_loop(self, worker_id: int):
        """Worker thread loop."""
        logger.info(f"Worker {worker_id} started")

        while self._workers_running:
            try:
                # Get job from queue (with timeout for graceful shutdown)
                job_id = self._queue.get(timeout=1.0)

                # Check for poison pill
                if job_id is None:
                    break

                # Process job
                self._process_job(job_id, worker_id)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.info(f"Worker {worker_id} stopped")

    def _process_job(self, job_id: str, worker_id: int):
        """Process a single ingestion job."""
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Check if cancelled
        if job.status == JobStatus.CANCELLED:
            logger.info(f"[{job_id[:8]}] Job was cancelled, skipping")
            return

        logger.info(f"[{job_id[:8]}] Worker {worker_id} processing job")

        # Update status
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow().isoformat()
        job.current_stage = "initializing"
        job.progress = 5

        try:
            # Run ingestion pipeline
            job.current_stage = "extracting"
            job.progress = 10

            result = self.document_service.process_document(
                file_path=job.file_path,
                doc_id=job.doc_id,
                project_id=job.project_id,
                custom_metadata=job.custom_metadata
            )

            # Update progress based on pipeline stages
            job.progress = 100
            job.current_stage = "completed"

            if result.success:
                job.status = JobStatus.COMPLETED
                job.result = result.to_dict()
                logger.info(
                    f"[{job_id[:8]}] Job completed successfully - "
                    f"vectors: {result.vectors_stored}"
                )
            else:
                job.status = JobStatus.FAILED
                job.error_message = result.error_message
                logger.error(f"[{job_id[:8]}] Job failed: {result.error_message}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            logger.error(f"[{job_id[:8]}] Job exception: {e}", exc_info=True)

        finally:
            job.completed_at = datetime.utcnow().isoformat()

            # CRITICAL: Clean up temp file
            self.cache_manager.cleanup_job_temp_file(job.file_path)

            # Force garbage collection after each job
            gc.collect()

    def cleanup_old_jobs(self):
        """Remove completed jobs older than retention period."""
        cutoff = datetime.utcnow() - timedelta(hours=self.job_retention_hours)
        cutoff_str = cutoff.isoformat()

        with self._job_lock:
            jobs_to_remove = []
            for job_id, job in self._jobs.items():
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    if job.completed_at and job.completed_at < cutoff_str:
                        jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self._jobs[job_id]

            if jobs_to_remove:
                logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_async_service: Optional[AsyncIngestionService] = None


def get_async_ingestion_service() -> Optional[AsyncIngestionService]:
    """Get the singleton async ingestion service."""
    return _async_service


def init_async_ingestion_service(document_service, **kwargs) -> AsyncIngestionService:
    """Initialize and start the async ingestion service."""
    global _async_service

    if _async_service is not None:
        logger.warning("Async ingestion service already initialized")
        return _async_service

    _async_service = AsyncIngestionService(document_service, **kwargs)
    _async_service.start()

    return _async_service


def shutdown_async_ingestion_service():
    """Shutdown the async ingestion service."""
    global _async_service

    if _async_service:
        _async_service.stop()
        _async_service = None
