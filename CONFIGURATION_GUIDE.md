# Configuration Guide - KnowVec RAG Pipeline

## ✅ What Was Fixed

### Before (Problems):
1. **api.py** loaded env variables but ignored them - used hardcoded values
2. **ServiceConfig** had all hardcoded defaults
3. Most .env variables were completely unused

### After (Fixed):
1. ✅ Created **config.py** - centralized configuration loader
2. ✅ Updated **api.py** - now uses config from .env
3. ✅ Updated **metadata_aware_normalizer.py** - fixed division by zero bug
4. ✅ All .env variables are now properly loaded and used

---

## 📋 Testing the Configuration

### Step 1: Test Configuration Loading

```bash
# Test that config.py loads all environment variables correctly
python config.py
```

**Expected Output:**
```
================================================================================
KnowVec RAG Pipeline - Configuration
================================================================================

📊 QDRANT DATABASE
  URL: http://localhost:6333
  Collection: knowvec_documents
  API Key: Not set

🔢 EMBEDDING MODEL
  Model: all-MiniLM-L6-v2
  Dimension: 384
  Batch Size: 32

⚙️  PROCESSING
  Max Chunk Size: 1000
  Target Chunk Size: 800
  Overlap: True (200 chars)
  Deduplicate: True
  Remove TOC Pages: True
  Protect Headings: True
  Protect Tables: True
  Protect Code Blocks: True
  Detect Multi-Column: True

📝 DOCUMENT SUMMARY
  Enabled: True
  Method: hybrid
  Max Length: 500 chars

🔍 SEARCH
  Min Similarity Score: 0.3
  Default Limit: 10
  Max Limit: 100

🌐 API SERVER
  Host: 0.0.0.0
  Port: 8000
  Debug Mode: False
  Max Upload: 50 MB
  CORS Origins: *

⚡ PERFORMANCE
  Workers: 4
  Cache Size: 100

📝 LOGGING
  Level: DEBUG
  Format: standard
  File Logging: True
  Log Directory: logs
  Show Progress: True
  Timing Logs: True

🏷️  VERSION
  Pipeline Version: 1.0
================================================================================
```

### Step 2: Restart Your API Server

```bash
# Stop current server (Ctrl+C if running)

# Start server
uvicorn api:app --reload
```

**Watch for these log messages:**
```
INFO: Loading configuration from environment variables
INFO: Configuration loaded - QDRANT_URL: http://localhost:6333, EMBEDDING_MODEL: all-MiniLM-L6-v2
INFO: Service config created - Collection: knowvec_documents, Vector size: 384
INFO: Initializing DocumentToVectorService
INFO: Using Qdrant URL: http://localhost:6333
INFO: Using Embedding Model: all-MiniLM-L6-v2
INFO: Pipeline settings - Chunking: max=1000, target=800, overlap=200, Deduplication: True
```

### Step 3: Retry Document Upload

Now retry uploading the document that failed before:

```bash
# The document should now process successfully
# The bug fixes will:
# 1. Prevent filtering out all pages
# 2. Use correct configuration from .env
# 3. Apply proper processing settings
```

---

## 🔧 How to Modify Configuration

### Option 1: Edit .env File (Recommended)

Simply edit the `.env` file and change values:

```bash
# Example: Change chunk size
MAX_CHUNK_SIZE=1500
TARGET_CHUNK_SIZE=1200

# Example: Disable TOC removal
REMOVE_TOC_PAGES=false

# Example: Change summary settings
GENERATE_DOCUMENT_SUMMARY=true
SUMMARY_METHOD=extractive
SUMMARY_MAX_LENGTH=1000
```

Then restart the server - changes will be applied automatically.

### Option 2: Override with Environment Variables

```bash
# Override specific settings when starting server
QDRANT_URL=http://remote-qdrant:6333 MAX_CHUNK_SIZE=2000 uvicorn api:app --reload
```

---

## 📊 Configuration Variables Reference

### Qdrant Database
| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | (empty) | API key for Qdrant Cloud |
| `QDRANT_COLLECTION` | `knowvec_documents` | Collection name |

### Embedding Model
| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `EMBEDDING_DIMENSION` | `384` | Vector dimension |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size for embeddings |

### Document Processing
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `TARGET_CHUNK_SIZE` | `800` | Target chunk size |
| `ENABLE_OVERLAP` | `true` | Enable chunk overlap |
| `OVERLAP_SIZE` | `200` | Overlap size in characters |
| `DEDUPLICATE_CHUNKS` | `true` | Remove duplicate chunks |
| `REMOVE_TOC_PAGES` | `true` | Filter out table of contents |
| `PROTECT_HEADINGS` | `true` | Preserve heading formatting |
| `PROTECT_TABLES` | `true` | Keep tables intact |
| `PROTECT_CODE_BLOCKS` | `true` | Preserve code blocks |
| `DETECT_MULTI_COLUMN` | `true` | Detect multi-column layouts |

### Document Summarization
| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATE_DOCUMENT_SUMMARY` | `true` | Generate doc summaries |
| `SUMMARY_METHOD` | `hybrid` | Method: extractive/abstractive/hybrid |
| `SUMMARY_MAX_LENGTH` | `500` | Max summary length (chars) |
| `SUMMARY_MIN_DOC_LENGTH` | `1000` | Min doc length for summary |

### Search
| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_SIMILARITY_SCORE` | `0.3` | Minimum similarity threshold |
| `DEFAULT_SEARCH_LIMIT` | `10` | Default results to return |
| `MAX_SEARCH_LIMIT` | `100` | Maximum results allowed |

### API Server
| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server host |
| `API_PORT` | `8000` | Server port |
| `API_DEBUG` | `false` | Debug mode |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max file upload size |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

### Performance
| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_WORKERS` | `4` | Worker processes |
| `CACHE_SIZE` | `100` | Cache size |

### Logging
| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `LOG_FORMAT` | `standard` | standard/json |
| `ENABLE_FILE_LOGGING` | `true` | Save logs to file |
| `LOG_DIR` | `logs` | Log directory |
| `SHOW_PROGRESS` | `true` | Show progress bars |
| `ENABLE_TIMING_LOGS` | `true` | Log operation timings |

---

## 🎯 Common Configuration Scenarios

### Scenario 1: Smaller Chunks for Better Precision

```env
MAX_CHUNK_SIZE=600
TARGET_CHUNK_SIZE=500
OVERLAP_SIZE=100
```

### Scenario 2: Larger Chunks for More Context

```env
MAX_CHUNK_SIZE=2000
TARGET_CHUNK_SIZE=1500
OVERLAP_SIZE=300
```

### Scenario 3: Disable Document Summarization

```env
GENERATE_DOCUMENT_SUMMARY=false
```

### Scenario 4: More Aggressive Text Cleaning

```env
AGGRESSIVE_TEXT_CLEANING=true
REMOVE_TOC_PAGES=true
```

### Scenario 5: Production Settings

```env
LOG_LEVEL=INFO
API_DEBUG=false
ENABLE_TIMING_LOGS=false
SHOW_PROGRESS=false
ENABLE_FILE_LOGGING=true
```

### Scenario 6: Development Settings

```env
LOG_LEVEL=DEBUG
API_DEBUG=true
ENABLE_TIMING_LOGS=true
SHOW_PROGRESS=true
```

---

## 🐛 Troubleshooting

### Configuration Not Loading

**Problem:** Changes to .env not reflected

**Solution:**
```bash
# Make sure to restart the server
# Ctrl+C to stop, then:
uvicorn api:app --reload
```

### Wrong Collection Name

**Problem:** Using old collection name

**Check:**
```bash
python config.py | grep Collection
# Should show: Collection: knowvec_documents
```

**Fix:** Update QDRANT_COLLECTION in .env

### Hardcoded Values Still Appearing

**Problem:** Old code still using hardcoded values

**Verify:**
```bash
# Check api.py is using config
grep "config\." api.py

# Should see lines like:
#   qdrant_url=config.qdrant_url,
#   max_chunk_size=config.max_chunk_size,
```

---

## ✅ Verification Checklist

After implementing these changes, verify:

- [ ] `python config.py` shows all correct values
- [ ] Server starts without errors
- [ ] Log shows "Configuration loaded" messages
- [ ] Collection name matches .env setting
- [ ] Chunk sizes match .env settings
- [ ] Document upload works (test with the previously failing document)
- [ ] Summary generation respects .env setting
- [ ] Search results respect min_similarity_score from .env

---

## 📝 Summary of Changes

### Files Created:
1. **config.py** - Centralized configuration loader

### Files Modified:
1. **api.py** - Now uses config.py instead of hardcoded values
2. **metadata_aware_normalizer.py** - Fixed division by zero and all-pages-filtered bugs

### Files NOT Modified (but should work correctly):
- document_to_vector_service.py (already had env override logic)
- All other service files

---

## 🚀 Next Steps

1. **Test the configuration**
   ```bash
   python config.py
   ```

2. **Restart your server**
   ```bash
   uvicorn api:app --reload
   ```

3. **Retry document upload**
   - The previous error should now be fixed
   - All .env settings will be applied

4. **Customize as needed**
   - Edit .env to tune performance
   - Restart server to apply changes

---

**Status:** ✅ Configuration system complete and ready to use
**Date:** 2025-12-16
