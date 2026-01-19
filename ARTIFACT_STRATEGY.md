# Artifact Persistence Strategy

## Directory Layout

```
artifacts/{doc_id}/
├── metadata.json
├── 00_raw/
│   └── extraction_raw.json         # Raw extractor output + metadata
├── 01_normalized/
│   ├── document_ir.json            # Canonical DocumentIR
│   ├── full_text.txt               # Plain text
│   └── normalization_log.json      # What was cleaned
├── 02_chunks/
│   ├── chunks.json                 # All chunks + config
│   └── chunks/
│       ├── chunk_000.json          # Individual chunks
│       └── ...
├── 03_embeddings/
│   ├── embeddings.npy              # Vectors (NumPy)
│   └── embedding_records.json      # Metadata + config
└── 04_enrichment/
    ├── topics.json
    ├── summaries.json
    ├── entities.json
    └── keywords.json
```

## Data Stored Per Stage

### Stage 0: Raw Extraction
```json
{
  "extraction_output": [...],       // Raw blocks from pdf2llm/docx2llm
  "metadata": {
    "extractor_name": "marker-pdf",
    "extractor_version": "1.0",
    "extraction_timestamp": "...",
    "source_hash": "sha256...",     // For change detection
    "source_format": "pdf"
  }
}
```

### Stage 1: Normalized
```json
{
  "document_ir": {...},             // Full DocumentIR (blocks, sections)
  "normalization_rules": [...],     // Rules applied
  "full_text": "..."                // Quick text access
}
```

### Stage 2: Chunks
```json
{
  "chunks": [                       // All IRChunks
    {"chunk_id": "...", "blocks": [...], "page_start": 1, ...}
  ],
  "chunking_config": {              // Config used
    "max_chunk_size": 1000,
    "keep_tables_intact": true
  }
}
```

### Stage 3: Embeddings
```json
{
  "records": [                      // Embedding records
    {"chunk_id": "...", "text": "...", "metadata": {...}}
  ],
  "metadata": {
    "model_name": "text-embedding-3-small",
    "embedding_dim": 1536
  }
}
// + embeddings.npy (vectors)
```

### Stage 4: Enrichment
```json
{
  "topics": {...},
  "summaries": {...},
  "entities": [...],
  "keywords": [...]
}
```

## Reprocessing Without Re-Extraction

| Change | Start From | Skip |
|--------|-----------|------|
| New chunking config | Stage 1 (IR) | Extraction |
| New embedding model | Stage 2 (Chunks) | Extraction + Chunking |
| Re-run topic modeling | Stage 2 (Chunks) | Extraction + Chunking |
| Update normalization | Stage 0 (Raw) | Extraction only |

### Example: Re-chunk

```python
artifacts = ArtifactManager("./artifacts", doc_id)

# Load IR (skip extraction)
document_ir = artifacts.load_document_ir()

# Re-chunk with new config
new_config = IRChunkingConfig(max_chunk_size=500)  # Changed
chunker = IRChunker(new_config)
new_chunks = chunker.chunk(document_ir)

# Save new chunks
artifacts.save_stage02_chunks(new_chunks)
```

### Example: Re-embed

```python
# Load existing chunks (skip extraction + chunking)
chunks = artifacts.load_chunks()

# Embed with new model
new_embeddings = embed_with_model(chunks, "new-model")

# Save
artifacts.save_stage03_embeddings(new_embeddings)
```

## Change Detection

```python
# Hash source file
current_hash = sha256(file_content)

# Check if extraction needed
if artifacts.can_skip_extraction(current_hash):
    print("Using cached extraction")
    raw = artifacts.load_stage00_raw()
else:
    print("Re-extracting (file changed)")
    raw = extract_pdf(file)
    artifacts.save_stage00_raw(raw)
```

## Benefits

1. **Skip expensive steps**: Re-chunk without re-extracting (saves minutes per doc)
2. **Experiment fast**: Try different configs without re-running entire pipeline
3. **Audit trail**: See exactly what happened at each stage
4. **Debug**: Inspect intermediate outputs
5. **Incremental updates**: Update only changed stages
6. **Rollback**: Revert to previous artifacts if needed

## Storage Estimates

Per 100-page PDF:
- Raw extraction: ~5-10 MB
- DocumentIR: ~2-5 MB
- Chunks: ~3-7 MB
- Embeddings: ~1-2 MB (records) + vectors size
- Enrichment: ~500 KB - 2 MB

Total: ~15-30 MB per document
