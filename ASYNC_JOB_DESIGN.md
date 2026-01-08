# Async Job-Based Ingestion Pipeline

## Job State Machine

```
                    submit
                      ↓
    ┌─────────── PENDING ──────────┐
    │                ↓             │
    │          EXTRACTING          │
    │                ↓             │
    │         NORMALIZING          │
    │                ↓             │
    │           CHUNKING           │
    │                ↓             │
    │          EMBEDDING           │
    │                ↓             │
    │          ENRICHING           │
    │                ↓             │
    │          COMPLETED           │
    │                               │
    │     (any stage failure)      │
    │                ↓             │
    └──────────→ PAUSED ───resume──┘
                     ↓
              (max retries)
                     ↓
                  FAILED

    CANCELLED (manual at any stage)
```

## Stage States

Each stage tracks:
- `PENDING` → `IN_PROGRESS` → `COMPLETED`
- `IN_PROGRESS` → `FAILED` (with error)

**Resumability**: Failed stages can be retried; completed stages are preserved.

## API Endpoints

### 1. Submit Job
```http
POST /api/v1/jobs/submit
Content-Type: application/json

{
  "source_file": "document.pdf",
  "pipeline_config": {
    "chunking": {"max_size": 1000},
    "embedding": {"model": "text-embedding-3-small"}
  }
}

Response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "doc_id": "650e8400-e29b-41d4-a716-446655440001",
  "state": "pending",
  "message": "Job submitted successfully"
}
```

### 2. Get Job Status
```http
GET /api/v1/jobs/{job_id}/status

Response:
{
  "job_id": "550e8400...",
  "doc_id": "650e8400...",
  "state": "chunking",
  "current_stage": "chunking",
  "progress_percent": 60,
  "stages": {
    "extraction": {
      "state": "completed",
      "duration_seconds": 45.2,
      "artifacts": ["artifacts/doc_id/00_raw/extraction_raw.json"]
    },
    "normalization": {
      "state": "completed",
      "duration_seconds": 12.5,
      "artifacts": ["artifacts/doc_id/01_normalized/document_ir.json"]
    },
    "chunking": {
      "state": "in_progress",
      "progress_percent": 60,
      "progress_message": "Chunking page 30/50"
    },
    "embedding": {"state": "pending"},
    "enrichment": {"state": "pending"}
  },
  "created_at": "2024-01-15T10:30:00",
  "started_at": "2024-01-15T10:30:05"
}
```

### 3. Resume Job
```http
POST /api/v1/jobs/{job_id}/resume
Content-Type: application/json

{
  "from_stage": "normalization"  # Optional, auto-detects failed stage
}

Response:
{
  "message": "Job resumed from normalization"
}
```

### 4. Cancel Job
```http
DELETE /api/v1/jobs/{job_id}

Response:
{
  "message": "Job cancelled"
}
```

### 5. List Jobs
```http
GET /api/v1/jobs?state=failed

Response:
{
  "jobs": [
    {
      "job_id": "550e8400...",
      "doc_id": "650e8400...",
      "state": "failed",
      "current_stage": "embedding",
      "created_at": "2024-01-15T10:30:00"
    }
  ],
  "total": 1
}
```

## Artifact Updates Per Stage

### Stage 0: Extraction
```
artifacts/{doc_id}/
└── 00_raw/
    └── extraction_raw.json  ← Created

Job stages['extraction']:
  state: COMPLETED
  artifacts: ["artifacts/{doc_id}/00_raw/extraction_raw.json"]
```

### Stage 1: Normalization
```
artifacts/{doc_id}/
├── 00_raw/              (preserved)
└── 01_normalized/
    ├── document_ir.json  ← Created
    └── full_text.txt     ← Created

Job stages['normalization']:
  state: COMPLETED
  artifacts: [
    "artifacts/{doc_id}/01_normalized/document_ir.json",
    "artifacts/{doc_id}/01_normalized/full_text.txt"
  ]
```

### Stage 2: Chunking
```
artifacts/{doc_id}/
├── 00_raw/              (preserved)
├── 01_normalized/       (preserved)
└── 02_chunks/
    ├── chunks.json      ← Created
    └── chunks/          ← Created
        ├── chunk_000.json
        └── ...
```

### Stage 3: Embedding
```
artifacts/{doc_id}/
├── 00_raw/              (preserved)
├── 01_normalized/       (preserved)
├── 02_chunks/           (preserved)
└── 03_embeddings/
    ├── embeddings.npy   ← Created
    └── embedding_records.json  ← Created
```

### Stage 4: Enrichment
```
artifacts/{doc_id}/
├── 00_raw/              (preserved)
├── 01_normalized/       (preserved)
├── 02_chunks/           (preserved)
├── 03_embeddings/       (preserved)
└── 04_enrichment/
    ├── topics.json      ← Created
    ├── summaries.json   ← Created
    └── entities.json    ← Created
```

## Failure Handling

### Example: Stage 2 Fails

**Before failure**:
```
✓ Stage 0: extraction (COMPLETED)
✓ Stage 1: normalization (COMPLETED)
✗ Stage 2: chunking (FAILED - "Out of memory")
- Stage 3: embedding (PENDING)
- Stage 4: enrichment (PENDING)
```

**Artifacts preserved**:
```
artifacts/{doc_id}/
├── 00_raw/              ✓ Preserved
├── 01_normalized/       ✓ Preserved
└── 02_chunks/           ✗ Incomplete/missing
```

**Resume from Stage 2**:
```http
POST /api/v1/jobs/{job_id}/resume
{"from_stage": "chunking"}
```

**Resumption logic**:
1. Load `artifacts/{doc_id}/01_normalized/document_ir.json` (from completed stage)
2. Re-run chunking with potentially different config
3. Continue to stages 3, 4

**Result**: No re-extraction needed, saves minutes per document.

## Progress Tracking

Each stage reports progress:
```python
# During extraction
executor.update_progress(job, 'extraction', 25, "Extracting page 10/40")

# During embedding
executor.update_progress(job, 'embedding', 60, "Embedding chunk 300/500")
```

Client polls:
```javascript
async function pollJobStatus(jobId) {
  const response = await fetch(`/api/v1/jobs/${jobId}/status`);
  const data = await response.json();

  console.log(`Overall: ${data.progress_percent}%`);
  console.log(`Stage: ${data.current_stage} - ${data.stages[data.current_stage].progress_message}`);

  if (data.state === 'completed') {
    console.log('Done!', data.result);
  } else if (data.state === 'failed') {
    console.error('Failed:', data.stages[data.current_stage].error_message);
  } else {
    setTimeout(() => pollJobStatus(jobId), 2000);  // Poll every 2s
  }
}
```

## Benefits

1. **Immediate response**: API returns job_id instantly
2. **Resumability**: Failed stages restart from last completed stage
3. **Artifact preservation**: Completed stages never re-run
4. **Progress visibility**: Real-time updates via polling
5. **Error isolation**: One stage failure doesn't corrupt others
6. **Retry logic**: Auto-retry with backoff (up to max_retries)

## Implementation

See `async_job_system.py` for:
- `IngestionJob` model
- `JobStore` for persistence
- `PipelineExecutor` for stage management
- FastAPI endpoints (commented)
