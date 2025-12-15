# Metadata-Based Filtering Guide

This guide explains how to use metadata filtering in the KnowVec RAG pipeline.

## Available Metadata Fields

Your system stores rich metadata for each chunk:

### Document Identification
- `doc_id` - Unique document identifier
- `file_name` - Original file name
- `chunk_id` - Unique chunk identifier
- `embedding_id` - Embedding identifier

### Page Information
- `page_start` - Starting page number (integer)
- `page_end` - Ending page number (integer)
- `page_range` - Formatted page range string (e.g., "5-7")

### Section Hierarchy
- `section_title` - Section title/heading
- `heading_path` - List of hierarchical headings
- `heading_path_str` - String representation of heading hierarchy
- `hierarchy_level` - Depth in document hierarchy (integer)

### Chunk Position
- `chunk_index` - Position in document (integer)
- `total_chunks` - Total chunks in document (integer)

### Content Metrics
- `char_len` - Character count (integer)
- `word_count` - Word count (integer)
- `token_count` - Token count (integer)

### Content Type Flags (Boolean)
- `contains_tables` - Contains table data
- `contains_code` - Contains code blocks
- `contains_bullets` - Contains bullet lists
- `has_urls` - Contains URLs

### Boundary Information
- `boundary_type` - Type of boundary (string: "PAGE", "SECTION", "PARAGRAPH", "TABLE", "CODE_BLOCK", etc.)
- `has_overlap` - Whether chunk has overlap with adjacent chunks (boolean)

### Metadata
- `version` - Pipeline version
- `created_at` - ISO timestamp
- `pipeline` - Processing pipeline identifier

---

## Filter Types

### 1. Exact Match Filters

Match exact values:

```python
# Single value
{"doc_id": "document-123"}
{"file_name": "report.pdf"}
{"contains_tables": True}
{"boundary_type": "TABLE"}
```

### 2. Range Filters

Numeric range queries:

```python
# Greater than or equal (gte)
{"page_start": {"gte": 10}}

# Less than or equal (lte)
{"page_end": {"lte": 20}}

# Range (both)
{"page_start": {"gte": 10, "lte": 20}}

# Greater than (gt) / Less than (lt)
{"word_count": {"gt": 100, "lt": 500}}
```

### 3. Multiple Value Filters (OR logic)

Match any value in a list:

```python
# Match multiple documents
{"doc_id": ["doc-123", "doc-456", "doc-789"]}

# Match multiple boundary types
{"boundary_type": ["TABLE", "CODE_BLOCK"]}
```

### 4. Combined Filters (AND logic)

All filters are combined with AND logic:

```python
{
    "doc_id": "document-123",
    "page_start": {"gte": 5},
    "contains_code": True
}
```

---

## API Endpoints

### 1. Filtered Semantic Search

**Endpoint:** `POST /search/filtered`

Combines vector similarity with metadata filters.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "filters": {
    "doc_id": "document-123",
    "page_start": {"gte": 10, "lte": 20}
  },
  "limit": 10,
  "score_threshold": 0.7
}
```

**Response:**
```json
[
  {
    "id": "chunk-uuid",
    "score": 0.85,
    "text": "Machine learning is...",
    "metadata": {
      "doc_id": "document-123",
      "page_start": 15,
      "section_title": "Introduction to ML",
      ...
    }
  }
]
```

### 2. Metadata-Only Filtering

**Endpoint:** `POST /filter`

No vector search - pure metadata filtering.

**Request Body:**
```json
{
  "filters": {
    "contains_tables": true,
    "doc_id": "document-123"
  },
  "limit": 100,
  "offset": 0
}
```

**Response:**
```json
{
  "results": [...],
  "count": 42,
  "limit": 100,
  "offset": 0
}
```

---

## Common Use Cases

### Use Case 1: Search Within a Specific Document

```json
{
  "query": "machine learning algorithms",
  "filters": {
    "doc_id": "ml-textbook-2024"
  },
  "limit": 5
}
```

### Use Case 2: Search Within Page Range

```json
{
  "query": "neural networks",
  "filters": {
    "page_start": {"gte": 50, "lte": 100}
  },
  "limit": 10
}
```

### Use Case 3: Find All Tables in a Document

```json
{
  "filters": {
    "doc_id": "report-q4",
    "contains_tables": true
  },
  "limit": 50
}
```

### Use Case 4: Find All Code Examples

```json
{
  "filters": {
    "contains_code": true,
    "boundary_type": "CODE_BLOCK"
  },
  "limit": 20
}
```

### Use Case 5: Search Within Specific Section

```json
{
  "query": "implementation details",
  "filters": {
    "section_title": "Chapter 5: Implementation"
  },
  "limit": 5
}
```

### Use Case 6: Find Large Chunks Only

```json
{
  "filters": {
    "word_count": {"gte": 200}
  },
  "limit": 10
}
```

### Use Case 7: Search Multiple Documents

```json
{
  "query": "data privacy",
  "filters": {
    "doc_id": ["gdpr-guide", "privacy-policy", "compliance-doc"]
  },
  "limit": 10
}
```

### Use Case 8: Advanced Combined Filtering

```json
{
  "query": "database optimization",
  "filters": {
    "doc_id": "tech-manual",
    "page_start": {"gte": 100},
    "contains_code": true,
    "word_count": {"gte": 100, "lte": 500},
    "hierarchy_level": {"lte": 2}
  },
  "limit": 5,
  "score_threshold": 0.75
}
```

---

## Python Client Examples

### Example 1: Filtered Search

```python
import requests

url = "http://localhost:8000/search/filtered"
payload = {
    "query": "What is machine learning?",
    "filters": {
        "doc_id": "ml-textbook",
        "page_start": {"gte": 10, "lte": 50}
    },
    "limit": 5
}

response = requests.post(url, json=payload)
results = response.json()

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Page: {result['metadata']['page_range']}")
    print(f"Text: {result['text'][:200]}...\n")
```

### Example 2: Find All Tables

```python
url = "http://localhost:8000/filter"
payload = {
    "filters": {
        "contains_tables": True
    },
    "limit": 50
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Found {data['count']} chunks with tables")
for item in data['results']:
    print(f"- {item['metadata']['file_name']}, Page {item['metadata']['page_start']}")
```

### Example 3: Paginated Results

```python
def fetch_all_filtered_chunks(filters, page_size=100):
    """Fetch all matching chunks with pagination."""
    offset = 0
    all_results = []

    while True:
        response = requests.post(
            "http://localhost:8000/filter",
            json={
                "filters": filters,
                "limit": page_size,
                "offset": offset
            }
        )
        data = response.json()
        results = data['results']

        if not results:
            break

        all_results.extend(results)
        offset += page_size

    return all_results

# Usage
all_code_blocks = fetch_all_filtered_chunks({
    "contains_code": True,
    "doc_id": "python-tutorial"
})
print(f"Found {len(all_code_blocks)} code blocks")
```

---

## cURL Examples

### Filtered Search

```bash
curl -X POST http://localhost:8000/search/filtered \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "filters": {
      "doc_id": "ml-textbook",
      "page_start": {"gte": 10}
    },
    "limit": 5
  }'
```

### Metadata Filtering

```bash
curl -X POST http://localhost:8000/filter \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "contains_tables": true,
      "page_start": {"gte": 20, "lte": 50}
    },
    "limit": 20
  }'
```

---

## Filter Building Best Practices

### 1. Start Broad, Then Narrow

```python
# Bad: Too specific initially
{"doc_id": "doc-123", "page_start": {"gte": 45, "lte": 47}, "word_count": {"gte": 150, "lte": 160}}

# Good: Start broad, refine if needed
{"doc_id": "doc-123", "page_start": {"gte": 40, "lte": 50}}
```

### 2. Use Content Type Flags Effectively

```python
# Find all structured content
{"contains_tables": True}

# Find all technical content
{"contains_code": True}

# Find all reference sections (likely to have URLs)
{"has_urls": True}
```

### 3. Combine Semantic Search with Metadata

```python
# Good: Semantic meaning + structural filtering
{
    "query": "implementation details",
    "filters": {
        "contains_code": True,
        "hierarchy_level": {"gte": 2}  # Not in top-level sections
    }
}
```

### 4. Use Pagination for Large Result Sets

```python
# For metadata-only filtering, use offset/limit
{
    "filters": {"doc_id": "large-document"},
    "limit": 100,
    "offset": 0  # Then 100, 200, 300...
}
```

---

## Performance Tips

1. **Index Metadata Fields:** Qdrant automatically indexes payload fields for filtering
2. **Use Specific Filters:** More specific filters = faster queries
3. **Limit Results:** Use appropriate `limit` values
4. **Batch Processing:** For large datasets, use pagination
5. **Content Type Flags:** Boolean filters are very fast

---

## Advanced: Custom Metadata Filters

You can extend the metadata by modifying `embedding_preparation.py:166`:

```python
# Add custom metadata fields
payload['custom_field'] = some_value
payload['document_category'] = category
payload['confidence_score'] = score
```

Then filter on these new fields:

```python
{
    "filters": {
        "custom_field": "value",
        "confidence_score": {"gte": 0.8}
    }
}
```

---

## Troubleshooting

### No Results Found

- Check if your filters are too restrictive
- Verify field names match exactly (case-sensitive)
- Check if documents have been processed with metadata

### Slow Queries

- Reduce the `limit` value
- Use more specific filters
- Check Qdrant collection size with `/stats`

### Type Errors

- Ensure boolean fields use `true`/`false` (JSON) or `True`/`False` (Python)
- Use integers for numeric fields, not strings
- Range filters need `gte`, `lte`, `gt`, or `lt` keys

---

## Interactive API Documentation

Once your server is running, visit:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide interactive documentation and allow you to test the filtering endpoints directly in your browser.

---

## Summary

Metadata-based filtering in KnowVec provides:

✅ **Rich Metadata:** 25+ metadata fields per chunk
✅ **Flexible Filtering:** Exact match, ranges, multiple values
✅ **Combined Search:** Vector similarity + metadata filters
✅ **Pure Metadata:** Filter without semantic search
✅ **Pagination:** Handle large result sets efficiently
✅ **Boolean Flags:** Fast content type filtering
✅ **Hierarchical Structure:** Search by document structure

Start with simple filters and gradually combine them for precise results!
