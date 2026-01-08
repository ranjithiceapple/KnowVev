"""
Ingestion Pipeline V2 - IR-Based

Integrates new designs:
- DocumentIR (structured representation)
- Block-based chunking
- Artifact persistence
- Async job system
- Topic modeling as enrichment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

from document_ir import DocumentIR
from document_ir_mappers import map_pdf_extraction_to_ir, map_docx_extraction_to_ir
from block_based_chunking import BlockChunker, ChunkingConfig as BlockChunkingConfig
from artifact_persistence import ArtifactManager, Stage00_RawArtifacts, Stage01_NormalizedArtifacts, Stage02_ChunkArtifacts
from async_job_system import IngestionJob, JobStore, PipelineExecutor, JobState
from page_section_model import PhysicalLocation, LogicalSection, ChunkMetadata as V2ChunkMetadata

# Existing components
from document_processor_llm import extract_text_from_document
from qdrant_storage import QdrantStorage, QdrantConfig


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Chunking
    target_chunk_size: int = 500
    max_chunk_size: int = 1000

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents_v2"

    # Artifacts
    artifact_dir: str = "./artifacts"
    jobs_dir: str = "./jobs"


class IngestionPipelineV2:
    """New ingestion pipeline using IR and block-based chunking"""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.job_store = JobStore(self.config.jobs_dir)
        self.executor = PipelineExecutor(self.job_store)
        self.qdrant = QdrantStorage(QdrantConfig(
            url=self.config.qdrant_url,
            collection_name=self.config.qdrant_collection
        ))
        # Ensure collection exists
        self.qdrant.collection_manager.create_collection(recreate=False)

    def ingest_document(self, file_path: str) -> str:
        """
        Ingest document, return job_id immediately.
        Processing continues async.
        """
        doc_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())

        job = IngestionJob(
            job_id=job_id,
            doc_id=doc_id,
            source_file=file_path
        )
        self.job_store.save_job(job)

        # Run pipeline (could be background task)
        try:
            self._run_pipeline(job)
        except Exception as e:
            job.state = JobState.FAILED
            self.job_store.save_job(job)
            raise

        return job_id

    def _run_pipeline(self, job: IngestionJob):
        """Execute pipeline stages"""
        artifacts = ArtifactManager(self.config.artifact_dir, job.doc_id)

        # Stage 0: Extract
        self.executor.start_stage(job, 'extraction')
        raw_output = self._extract(job.source_file)
        artifacts.save_stage00_raw(Stage00_RawArtifacts(
            extraction_output=raw_output,
            extractor_name='pdf2llm',
            extractor_version='1.0',
            extraction_timestamp=datetime.now().isoformat(),
            extraction_config={},
            source_filename=job.source_file,
            source_hash='',
            source_size_bytes=0,
            source_format=self._get_format(job.source_file)
        ))
        self.executor.complete_stage(job, 'extraction', [str(artifacts.paths.extraction_raw)])

        # Stage 1: Normalize (create IR)
        self.executor.start_stage(job, 'normalization')
        ir = self._create_ir(job.source_file, raw_output)
        artifacts.save_stage01_normalized(Stage01_NormalizedArtifacts(
            document_ir=ir.to_dict(),
            normalization_rules=[],
            normalization_stats={},
            full_text=ir.to_markdown()
        ))
        self.executor.complete_stage(job, 'normalization', [str(artifacts.paths.document_ir)])

        # Stage 2: Chunk (block-based)
        self.executor.start_stage(job, 'chunking')
        chunks = self._chunk_ir(ir)
        artifacts.save_stage02_chunks(Stage02_ChunkArtifacts(
            chunks=[c.to_dict() for c in chunks],
            chunking_config={'target_size': self.config.target_chunk_size},
            total_chunks=len(chunks),
            avg_chunk_size=sum(len(c.text) for c in chunks) / len(chunks),
            chunk_size_distribution={}
        ))
        self.executor.complete_stage(job, 'chunking', [str(artifacts.paths.chunks_file)])

        # Stage 3: Embed + Store
        self.executor.start_stage(job, 'embedding')
        self._embed_and_store(chunks, ir, job.doc_id)
        self.executor.complete_stage(job, 'embedding', [])

        # Stage 4: Enrichment (skip for now, runs async)
        self.executor.start_stage(job, 'enrichment')
        self.executor.complete_stage(job, 'enrichment', [])

        job.state = JobState.COMPLETED
        job.completed_at = datetime.now().isoformat()
        self.job_store.save_job(job)

    def _extract(self, file_path: str) -> List[Dict]:
        """Extract using existing extractors"""
        result = extract_text_from_document(file_path)

        # Convert to expected format
        if hasattr(result, 'pages'):
            return [{'type': 'text', 'content': p.text, 'page': i+1, 'metadata': {}}
                   for i, p in enumerate(result.pages)]
        return []

    def _create_ir(self, file_path: str, raw_output: List[Dict]) -> DocumentIR:
        """Create DocumentIR from extraction"""
        format_type = self._get_format(file_path)

        if format_type == 'pdf':
            return map_pdf_extraction_to_ir(file_path, raw_output)
        elif format_type == 'docx':
            return map_docx_extraction_to_ir(file_path, raw_output)
        else:
            return map_pdf_extraction_to_ir(file_path, raw_output)

    def _chunk_ir(self, ir: DocumentIR):
        """Chunk using block-based chunker"""
        config = BlockChunkingConfig(
            target_size=self.config.target_chunk_size,
            max_size=self.config.max_chunk_size
        )
        chunker = BlockChunker(config)
        return chunker.chunk(ir)

    def _embed_and_store(self, chunks, ir: DocumentIR, doc_id: str):
        """Embed and store in Qdrant"""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')

        points = []
        for chunk in chunks:
            vector = model.encode(chunk.text).tolist()

            # Create v2 metadata with physical/logical separation
            physical = PhysicalLocation(
                doc_id=doc_id,
                source_file=ir.source_file,
                source_format=ir.source_format,
                page_start=chunk.page_start,
                page_end=chunk.page_end
            )

            logical = LogicalSection(
                section_id=chunk.section_id or '',
                title=chunk.section_path[0] if chunk.section_path else '',
                section_path=chunk.section_path,
                depth=len(chunk.section_path)
            ) if chunk.section_path else None

            metadata = V2ChunkMetadata(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                physical_location=physical,
                logical_section=logical,
                text=chunk.text,
                block_ids=chunk.block_ids
            )

            points.append({
                'id': chunk.chunk_id,
                'vector': vector,
                'payload': {
                    **metadata.to_dict(),
                    'confidence': chunk.confidence,
                    'boundary_reason': chunk.start_boundary.reason.value
                }
            })

        self.qdrant.upsert_points(points)

    def _get_format(self, file_path: str) -> str:
        """Get file format"""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.docx', '.doc']:
            return 'docx'
        elif ext in ['.pptx', '.ppt']:
            return 'ppt'
        return 'pdf'

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status"""
        job = self.job_store.load_job(job_id)
        return job.to_dict() if job else None


# Usage
if __name__ == "__main__":
    pipeline = IngestionPipelineV2()

    # Ingest document
    job_id = pipeline.ingest_document("example.pdf")
    print(f"Job submitted: {job_id}")

    # Check status
    status = pipeline.get_job_status(job_id)
    print(f"Status: {status['state']}")
