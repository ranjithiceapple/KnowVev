# Document Summary Virtual Chunk Guide

## Overview

The Document Summary feature creates a special "virtual chunk" that contains a high-level overview of the entire document. This summary chunk is embedded alongside regular chunks, enabling:

- **Quick document overviews** - Get the gist without reading all chunks
- **Better semantic search** - Queries about document content return summaries first
- **Document-level retrieval** - Find documents by their overall topic
- **Hierarchical exploration** - Start with summary, drill down to specifics

## How It Works

### Pipeline Integration

The document summary is generated **between chunking and embedding** (Stage 3.5):

```
1. Extract text → 2. Normalize → 3. Chunk document
                                    ↓
                            [3.5 Generate Summary] ← NEW
                                    ↓
4. Prepare embeddings → 5. Store in vector DB
```

### Summary Generation Methods

Three methods available:

1. **Extractive** - Selects most representative existing chunks
2. **Abstractive** - Generates structured overview from document metadata
3. **Hybrid** - Combines both approaches (recommended)

## Configuration

### Enable in Service Config

```python
from document_to_vector_service import ServiceConfig, DocumentToVectorService

config = ServiceConfig(
    # ... other settings ...

    # Summary settings
    generate_document_summary=True,      # Enable/disable
    summary_max_length=2000,             # Max summary length in chars
    summary_method="hybrid",             # 'extractive', 'abstractive', 'hybrid'
)

service = DocumentToVectorService(config)
```

### Summary Methods Explained

#### Extractive Summary
Selects key excerpts:
- First chunk (introduction)
- Chunks with section titles
- Last chunk (conclusion)

```python
config = ServiceConfig(
    summary_method="extractive",
    summary_max_length=1500
)
```

#### Abstractive Summary
Generates structured overview:
```
Document: Technical Manual
Pages: 150
Sections: 12

Main Sections:
1. Introduction
2. System Architecture
3. API Reference
...

Overview:
This document describes...

Conclusion:
The system provides...
```

```python
config = ServiceConfig(
    summary_method="abstractive",
    summary_max_length=2000
)
```

#### Hybrid Summary (Recommended)
Combines structure + content:
```
=== Technical Manual ===
Total Pages: 150 | Total Sections: 12

Main Sections:
  • Introduction
  • System Architecture
  • API Reference

Content Summary:
Introduction: This manual provides comprehensive...
System Architecture: The system consists of...
Conclusion: Implementation guidelines...
```

```python
config = ServiceConfig(
    summary_method="hybrid",        # Best balance
    summary_max_length=2000
)
```

## Usage Examples

### Basic Usage

```python
from document_to_vector_service import DocumentToVectorService, ServiceConfig

# Configure with summary enabled
config = ServiceConfig(
    qdrant_url="http://localhost:6333",
    qdrant_collection="documents",
    generate_document_summary=True,
    summary_method="hybrid"
)

# Create service
service = DocumentToVectorService(config)

# Process document
result = service.process_document('document.pdf')

# Check summary was created
if result.has_summary:
    print(f"Summary generated: {result.summary_length} chars")
    print(f"Summary time: {result.summary_time:.2f}s")
```

### Query Document Summaries

```python
# Search for document overview
results = service.search(
    query="what is this document about",
    limit=5
)

# The summary chunk typically ranks high for overview queries
for result in results:
    payload = result['payload']

    # Check if this is a summary chunk
    is_summary = payload['chunk_id'].endswith('_SUMMARY')

    if is_summary:
        print("📄 DOCUMENT SUMMARY:")
        print(payload['text'])
        print(f"\nPages: {payload['page_range']}")
        print(f"Sections: {payload.get('section_title', 'N/A')}")
```

### Filter by Summary Chunks

```python
# Get only summary chunks
summaries = service.storage.filter_by_metadata({
    "section_title": "[DOCUMENT SUMMARY]"
})

for summary in summaries:
    payload = summary['payload']
    print(f"\nDocument: {payload['file_name']}")
    print(f"Summary: {payload['text'][:200]}...")
```

### Compare Multiple Documents

```python
# Get all document summaries
all_summaries = service.storage.filter_by_metadata({
    "section_title": "[DOCUMENT SUMMARY]"
})

print(f"Found {len(all_summaries)} documents\n")

for summary in all_summaries:
    payload = summary['payload']
    print(f"📄 {payload['file_name']}")
    print(f"   Pages: {payload['page_end']}")
    print(f"   Preview: {payload['text'][:150]}...\n")
```

## Summary Chunk Structure

### Metadata Fields

The summary virtual chunk includes special metadata:

```python
{
    "chunk_id": "doc-uuid_SUMMARY",           # Special _SUMMARY suffix
    "section_title": "[DOCUMENT SUMMARY]",    # Special marker
    "chunk_index": -1,                        # -1 indicates summary
    "page_number_start": 1,
    "page_number_end": <total_pages>,
    "heading_path": ["Document Overview"],
    "boundary_type": "section",

    # Content
    "text": "<full summary text>",

    # Standard metadata
    "doc_id": "...",
    "file_name": "...",
    "chunk_char_len": 1500,
    "chunk_word_count": 250,

    # Flags
    "contains_tables": false,
    "contains_code": false,
    "contains_bullets": true  # Often has section lists
}
```

### Identifying Summary Chunks

Multiple ways to identify summary chunks:

```python
# Method 1: Check chunk_id suffix
is_summary = chunk_id.endswith('_SUMMARY')

# Method 2: Check section_title
is_summary = section_title == "[DOCUMENT SUMMARY]"

# Method 3: Check chunk_index
is_summary = chunk_index == -1

# Method 4: Filter query
summaries = filter_by_metadata({
    "section_title": "[DOCUMENT SUMMARY]"
})
```

## Testing

### Test Script

```bash
# Basic test
python test_document_summary.py document.pdf

# Compare with/without summary
python test_document_summary.py document.pdf compare
```

### Manual Test

```python
from document_to_vector_service import ServiceConfig, DocumentToVectorService

# Test configuration
config = ServiceConfig(
    qdrant_collection="test_summary",
    generate_document_summary=True,
    summary_method="hybrid",
    summary_max_length=2000
)

service = DocumentToVectorService(config)
result = service.process_document('test.pdf')

# Verify summary
assert result.has_summary == True
assert result.summary_length > 0
assert result.summary_time > 0

print(f"✅ Summary generated: {result.summary_length} chars")
```

## Performance

### Overhead

Summary generation adds minimal overhead:

```
Without Summary:
  Chunking: 1.5s
  Embedding: 3.2s
  Total: 8.7s

With Summary:
  Chunking: 1.5s
  Summary: 0.3s  ← Added
  Embedding: 3.3s  ← Slightly higher (one more chunk)
  Total: 9.1s

Overhead: +0.4s (4.6%)
```

### Optimization Tips

1. **Adjust summary length** - Shorter summaries = faster processing
```python
summary_max_length=1000  # Instead of 2000
```

2. **Use extractive method** - Fastest method
```python
summary_method="extractive"  # Faster than hybrid
```

3. **Disable for small documents** - Not needed for short docs
```python
# Only enable for documents > 10 pages
if pages > 10:
    config.generate_document_summary = True
```

## Use Cases

### Use Case 1: Document Discovery

Find relevant documents by searching summaries:

```python
# Search across all documents
results = service.search(
    query="machine learning applications in healthcare",
    filters={"section_title": "[DOCUMENT SUMMARY]"},
    limit=10
)

# Get relevant documents
for result in results:
    print(f"Document: {result['payload']['file_name']}")
    print(f"Relevance: {result['score']:.2%}")
    print(f"Summary: {result['payload']['text'][:200]}...\n")
```

### Use Case 2: Chatbot Context

Provide document context to chatbot:

```python
# Get document summary for context
summary_results = service.search(
    query=user_question,
    filters={"section_title": "[DOCUMENT SUMMARY]"},
    limit=1
)

if summary_results:
    summary_text = summary_results[0]['payload']['text']

    # Use summary as context for detailed search
    context = f"Document overview: {summary_text}\n\n"

    # Now search for specific details
    detail_results = service.search(
        query=user_question,
        limit=5
    )

    # Combine summary + details for response
    full_context = context + "\n".join([r['payload']['text'] for r in detail_results])
```

### Use Case 3: Document Comparison

Compare multiple documents side-by-side:

```python
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

summaries = []
for doc in documents:
    result = service.process_document(doc)

    # Get the summary
    summary = service.storage.filter_by_metadata({
        "doc_id": result.doc_id,
        "section_title": "[DOCUMENT SUMMARY]"
    })[0]

    summaries.append({
        'file': doc,
        'summary': summary['payload']['text']
    })

# Display side-by-side
for s in summaries:
    print(f"\n{'='*80}")
    print(f"📄 {s['file']}")
    print(f"{'='*80}")
    print(s['summary'])
```

### Use Case 4: Content Recommendations

Suggest related documents:

```python
# User is reading document A
current_doc_summary = get_summary(doc_a_id)

# Find similar documents by comparing summaries
similar_docs = service.search(
    query=current_doc_summary['text'],
    filters={"section_title": "[DOCUMENT SUMMARY]"},
    limit=5
)

print("Related documents:")
for doc in similar_docs[1:]:  # Skip first (same doc)
    print(f"  • {doc['payload']['file_name']}")
    print(f"    Similarity: {doc['score']:.2%}")
```

## API Integration

### Add Summary to Search Response

Enhance search results with document summary:

```python
@app.get("/search_with_summary")
def search_with_summary(query: str, limit: int = 5):
    """Search and include document summary in results."""

    # Regular search
    results = service.search(query, limit=limit)

    # Enhance with summaries
    enhanced_results = []
    for result in results:
        doc_id = result['payload']['doc_id']

        # Get document summary
        summary = service.storage.filter_by_metadata({
            "doc_id": doc_id,
            "section_title": "[DOCUMENT SUMMARY]"
        })

        enhanced_results.append({
            "result": result,
            "document_summary": summary[0]['payload']['text'] if summary else None
        })

    return enhanced_results
```

### Summary-Only Endpoint

Create endpoint that returns only summaries:

```python
@app.get("/documents/summaries")
def get_all_summaries(limit: int = 100):
    """Get summaries of all documents."""

    summaries = service.storage.filter_by_metadata(
        filters={"section_title": "[DOCUMENT SUMMARY]"},
        limit=limit
    )

    return [
        {
            "doc_id": s['payload']['doc_id'],
            "file_name": s['payload']['file_name'],
            "pages": s['payload']['page_end'],
            "summary": s['payload']['text']
        }
        for s in summaries
    ]
```

## Troubleshooting

### Issue: Summary not generated

**Check:** Is summary enabled in config?
```python
config.generate_document_summary = True  # Must be True
```

### Issue: Summary too short

**Solution:** Increase max length
```python
config.summary_max_length = 3000  # Increase from 2000
```

### Issue: Summary quality poor

**Solution:** Try different method
```python
# Try each method
config.summary_method = "extractive"
config.summary_method = "abstractive"
config.summary_method = "hybrid"
```

### Issue: Can't find summary chunk

**Solution:** Use correct filter
```python
# Correct way to find summaries
summaries = service.storage.filter_by_metadata({
    "section_title": "[DOCUMENT SUMMARY]"
})

# Or check chunk_id suffix
is_summary = chunk['payload']['chunk_id'].endswith('_SUMMARY')
```

## Best Practices

1. **Use hybrid method** - Best balance of structure and content
2. **Set appropriate length** - 1500-2500 chars for most documents
3. **Filter summary chunks** - Use metadata to separate summaries from content
4. **Cache summaries** - Store summaries for quick access
5. **Update summaries** - Regenerate when documents are updated

## Summary

The Document Summary Virtual Chunk feature provides:

✅ **Automatic generation** - No manual summarization needed
✅ **Three methods** - Extractive, abstractive, hybrid
✅ **Seamless integration** - Works with existing pipeline
✅ **Minimal overhead** - ~0.3s per document
✅ **Better search** - Overview queries return summaries
✅ **Document discovery** - Find docs by high-level content
✅ **Hierarchical retrieval** - Start broad, drill down

Enable it in your pipeline for better document understanding and retrieval!
