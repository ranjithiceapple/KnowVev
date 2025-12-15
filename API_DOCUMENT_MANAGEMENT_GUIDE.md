# API Document Management Guide

## Overview

The API now includes comprehensive document management endpoints for:
- **Listing documents** - View all documents in the vector database
- **Getting document details** - View chunks and metadata for a specific document
- **Viewing embeddings** - Access the 384-dimensional vectors
- **Deleting documents** - Remove documents and their embeddings

## New Endpoints

### 1. List All Documents
**`GET /documents`**

Returns a list of all documents with basic metadata.

**Parameters:**
- `limit` (query, optional): Maximum documents to return (default: 1000, max: 10000)

**Example Request:**
```bash
curl http://localhost:8000/documents?limit=10
```

**Example Response:**
```json
[
  {
    "doc_id": "abc-123-xyz-789",
    "file_name": "technical_manual.pdf",
    "total_chunks": 45,
    "total_pages": 25,
    "has_summary": true,
    "created_at": "2024-01-15T10:30:00.123Z",
    "version": "1.0"
  },
  {
    "doc_id": "def-456-uvw-012",
    "file_name": "report_q4.pdf",
    "total_chunks": 32,
    "total_pages": 18,
    "has_summary": true,
    "created_at": "2024-01-16T14:20:00.456Z",
    "version": "1.0"
  }
]
```

---

### 2. Get Document Details
**`GET /documents/{doc_id}`**

Get detailed information about a specific document including all chunks.

**Parameters:**
- `doc_id` (path, required): Full document ID (UUID)
- `include_embeddings` (query, optional): Include 384-dim vectors (default: false)

**Example Request:**
```bash
curl "http://localhost:8000/documents/abc-123-xyz-789"
```

**Example Response:**
```json
{
  "doc_id": "abc-123-xyz-789",
  "file_name": "technical_manual.pdf",
  "total_chunks": 45,
  "total_pages": 25,
  "has_summary": true,
  "chunks": [
    {
      "id": "point-uuid-1",
      "chunk_id": "abc-123-xyz-789_chunk_0",
      "chunk_index": 0,
      "page_start": 1,
      "page_end": 2,
      "section_title": "Introduction",
      "char_len": 850,
      "word_count": 145,
      "contains_code": false,
      "contains_tables": false,
      "text_preview": "This manual provides comprehensive..."
    },
    {
      "id": "point-uuid-2",
      "chunk_id": "abc-123-xyz-789_SUMMARY",
      "chunk_index": -1,
      "page_start": 1,
      "page_end": 25,
      "section_title": "[DOCUMENT SUMMARY]",
      "char_len": 1847,
      "word_count": 310,
      "contains_code": false,
      "contains_tables": false,
      "text_preview": "=== Technical Manual === Total Pages: 25..."
    }
  ]
}
```

**With Embeddings:**
```bash
curl "http://localhost:8000/documents/abc-123-xyz-789?include_embeddings=true"
```

Response includes `vector` and `vector_dim` fields in each chunk.

---

### 3. Get Document Embeddings
**`GET /documents/{doc_id}/embeddings`**

Get all 384-dimensional embedding vectors for a document.

**Parameters:**
- `doc_id` (path, required): Full document ID

**Example Request:**
```bash
curl http://localhost:8000/documents/abc-123-xyz-789/embeddings
```

**Example Response:**
```json
{
  "doc_id": "abc-123-xyz-789",
  "total_chunks": 45,
  "vector_dimension": 384,
  "embeddings": [
    {
      "chunk_id": "abc-123-xyz-789_chunk_0",
      "chunk_index": 0,
      "vector": [0.123, -0.456, 0.789, ..., 0.321]  // 384 floats
    },
    {
      "chunk_id": "abc-123-xyz-789_chunk_1",
      "chunk_index": 1,
      "vector": [-0.234, 0.567, -0.890, ..., -0.432]
    }
  ]
}
```

⚠️ **Warning:** Response can be large (45 chunks × 384 dims × 4 bytes ≈ 69KB)

---

### 4. Delete Document
**`DELETE /documents/{doc_id}`**

Delete a document and all its embeddings from the vector database.

⚠️ **WARNING: This action is irreversible!**

**Parameters:**
- `doc_id` (path, required): Full document ID to delete

**Example Request:**
```bash
curl -X DELETE http://localhost:8000/documents/abc-123-xyz-789
```

**Example Response:**
```json
{
  "success": true,
  "doc_id": "abc-123-xyz-789",
  "file_name": "technical_manual.pdf",
  "chunks_deleted": 45,
  "message": "Document 'technical_manual.pdf' and 45 embeddings deleted successfully"
}
```

**Error Response (404 - Not Found):**
```json
{
  "detail": "Document not found: abc-123-xyz-789"
}
```

---

### 5. Bulk Delete Documents
**`POST /documents/bulk-delete`**

Delete multiple documents at once.

⚠️ **WARNING: This action is irreversible!**

**Request Body:**
```json
["doc-id-1", "doc-id-2", "doc-id-3"]
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/documents/bulk-delete \
  -H "Content-Type: application/json" \
  -d '["abc-123-xyz-789", "def-456-uvw-012", "ghi-789-rst-345"]'
```

**Example Response:**
```json
{
  "success": true,
  "total_requested": 3,
  "deleted": 2,
  "failed": 1,
  "results": [
    {
      "doc_id": "abc-123-xyz-789",
      "success": true,
      "chunks_deleted": 45
    },
    {
      "doc_id": "def-456-uvw-012",
      "success": true,
      "chunks_deleted": 32
    },
    {
      "doc_id": "ghi-789-rst-345",
      "success": false,
      "error": "Document not found"
    }
  ]
}
```

---

## Python Client Examples

### List All Documents

```python
import requests

response = requests.get("http://localhost:8000/documents")
documents = response.json()

print(f"Total documents: {len(documents)}\n")

for doc in documents:
    print(f"📄 {doc['file_name']}")
    print(f"   Doc ID: {doc['doc_id'][:8]}...")
    print(f"   Chunks: {doc['total_chunks']}, Pages: {doc['total_pages']}")
    print(f"   Summary: {'✓' if doc['has_summary'] else '✗'}")
    print()
```

### Get Document Details

```python
import requests

doc_id = "abc-123-xyz-789"
response = requests.get(f"http://localhost:8000/documents/{doc_id}")
details = response.json()

print(f"Document: {details['file_name']}")
print(f"Total Chunks: {details['total_chunks']}")
print(f"Total Pages: {details['total_pages']}\n")

print("Chunks:")
for chunk in details['chunks']:
    print(f"  [{chunk['chunk_index']}] {chunk['section_title']}")
    print(f"      Pages: {chunk['page_start']}-{chunk['page_end']}")
    print(f"      {chunk['text_preview']}")
    print()
```

### Get Embeddings

```python
import requests
import numpy as np

doc_id = "abc-123-xyz-789"
response = requests.get(f"http://localhost:8000/documents/{doc_id}/embeddings")
data = response.json()

print(f"Document: {doc_id}")
print(f"Vector Dimension: {data['vector_dimension']}")
print(f"Total Embeddings: {len(data['embeddings'])}\n")

# Analyze first embedding
first_embedding = data['embeddings'][0]
vector = np.array(first_embedding['vector'])

print(f"Chunk {first_embedding['chunk_index']}:")
print(f"  Vector shape: {vector.shape}")
print(f"  Vector norm: {np.linalg.norm(vector):.4f}")
print(f"  Mean: {vector.mean():.4f}")
print(f"  Std: {vector.std():.4f}")
```

### Delete Document

```python
import requests

doc_id = "abc-123-xyz-789"

# Confirm deletion
confirm = input(f"Delete document {doc_id}? (yes/no): ")

if confirm.lower() == 'yes':
    response = requests.delete(f"http://localhost:8000/documents/{doc_id}")
    result = response.json()

    if result['success']:
        print(f"✅ {result['message']}")
        print(f"   Chunks deleted: {result['chunks_deleted']}")
    else:
        print(f"❌ Deletion failed")
else:
    print("Deletion cancelled")
```

### Bulk Delete

```python
import requests

doc_ids = [
    "abc-123-xyz-789",
    "def-456-uvw-012",
    "ghi-789-rst-345"
]

# Confirm bulk deletion
print(f"About to delete {len(doc_ids)} documents:")
for doc_id in doc_ids:
    print(f"  - {doc_id}")

confirm = input("\nProceed with bulk deletion? (yes/no): ")

if confirm.lower() == 'yes':
    response = requests.post(
        "http://localhost:8000/documents/bulk-delete",
        json=doc_ids
    )
    result = response.json()

    print(f"\n✅ Bulk deletion complete:")
    print(f"   Requested: {result['total_requested']}")
    print(f"   Deleted: {result['deleted']}")
    print(f"   Failed: {result['failed']}\n")

    print("Results:")
    for item in result['results']:
        status = "✓" if item['success'] else "✗"
        print(f"  {status} {item['doc_id'][:8]}...", end="")

        if item['success']:
            print(f" - {item['chunks_deleted']} chunks deleted")
        else:
            print(f" - {item['error']}")
else:
    print("Bulk deletion cancelled")
```

---

## Use Cases

### Use Case 1: Document Inventory

Get a complete list of all documents:

```python
import requests
import pandas as pd

response = requests.get("http://localhost:8000/documents")
documents = response.json()

# Create DataFrame
df = pd.DataFrame(documents)

# Display summary
print(df[['file_name', 'total_chunks', 'total_pages', 'has_summary']])

# Statistics
print(f"\nTotal documents: {len(df)}")
print(f"Total chunks: {df['total_chunks'].sum()}")
print(f"Total pages: {df['total_pages'].sum()}")
print(f"Docs with summary: {df['has_summary'].sum()}")
```

### Use Case 2: Export Document Catalog

Export to CSV for record-keeping:

```python
import requests
import csv

response = requests.get("http://localhost:8000/documents")
documents = response.json()

with open('document_catalog.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'doc_id', 'file_name', 'total_chunks', 'total_pages',
        'has_summary', 'created_at', 'version'
    ])
    writer.writeheader()
    writer.writerows(documents)

print("✅ Catalog exported to document_catalog.csv")
```

### Use Case 3: Find Documents to Delete

Find old or large documents:

```python
import requests
from datetime import datetime, timedelta

response = requests.get("http://localhost:8000/documents")
documents = response.json()

# Find documents older than 30 days
thirty_days_ago = datetime.now() - timedelta(days=30)

old_docs = [
    doc for doc in documents
    if datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00')) < thirty_days_ago
]

print(f"Found {len(old_docs)} documents older than 30 days:\n")
for doc in old_docs:
    print(f"  - {doc['file_name']} ({doc['created_at']})")
    print(f"    Doc ID: {doc['doc_id']}")
    print(f"    Chunks: {doc['total_chunks']}\n")

# Optional: Delete them
if input("Delete these documents? (yes/no): ").lower() == 'yes':
    doc_ids = [doc['doc_id'] for doc in old_docs]
    response = requests.post(
        "http://localhost:8000/documents/bulk-delete",
        json=doc_ids
    )
    print(response.json())
```

### Use Case 4: Analyze Embeddings

Analyze embedding quality:

```python
import requests
import numpy as np

doc_id = "abc-123-xyz-789"
response = requests.get(f"http://localhost:8000/documents/{doc_id}/embeddings")
data = response.json()

# Convert to numpy array
embeddings_matrix = np.array([e['vector'] for e in data['embeddings']])

print(f"Document: {doc_id}")
print(f"Shape: {embeddings_matrix.shape}")
print(f"\nStatistics:")
print(f"  Mean norm: {np.linalg.norm(embeddings_matrix, axis=1).mean():.4f}")
print(f"  Min norm: {np.linalg.norm(embeddings_matrix, axis=1).min():.4f}")
print(f"  Max norm: {np.linalg.norm(embeddings_matrix, axis=1).max():.4f}")

# Cosine similarity between chunks
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings_matrix)

print(f"\nChunk Similarity:")
print(f"  Mean similarity: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean():.4f}")
print(f"  Min similarity: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].min():.4f}")
print(f"  Max similarity: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].max():.4f}")
```

---

## Interactive API Documentation

Visit these URLs when your server is running:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide:
- Interactive API testing
- Automatic request/response examples
- Schema documentation
- Try-it-out functionality

---

## Error Handling

### Common Errors

**404 - Document Not Found:**
```json
{
  "detail": "Document not found: abc-123-xyz"
}
```

**500 - Server Error:**
```json
{
  "detail": "Failed to connect to Qdrant"
}
```

**422 - Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["query", "limit"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error"
    }
  ]
}
```

### Error Handling in Python

```python
import requests

try:
    response = requests.get("http://localhost:8000/documents/invalid-id")
    response.raise_for_status()
    data = response.json()

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Document not found")
    elif e.response.status_code == 500:
        print("Server error")
    else:
        print(f"HTTP error: {e.response.status_code}")

except requests.exceptions.ConnectionError:
    print("Failed to connect to API server")

except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Performance Tips

1. **Pagination:** Use `limit` parameter for large collections
```python
# Get documents in batches
limit = 100
documents = requests.get(f"http://localhost:8000/documents?limit={limit}").json()
```

2. **Avoid Embeddings Unless Needed:** Don't include embeddings in detail requests unless necessary
```python
# Without embeddings (fast)
details = requests.get(f"http://localhost:8000/documents/{doc_id}").json()

# With embeddings (slower, larger response)
details = requests.get(f"http://localhost:8000/documents/{doc_id}?include_embeddings=true").json()
```

3. **Bulk Operations:** Use bulk delete for multiple documents
```python
# Efficient - single request
requests.post("/documents/bulk-delete", json=[id1, id2, id3])

# Inefficient - multiple requests
for doc_id in [id1, id2, id3]:
    requests.delete(f"/documents/{doc_id}")
```

---

## Summary

The document management API provides:

✅ **List all documents** - `GET /documents`
✅ **Get document details** - `GET /documents/{doc_id}`
✅ **View embeddings** - `GET /documents/{doc_id}/embeddings`
✅ **Delete document** - `DELETE /documents/{doc_id}`
✅ **Bulk delete** - `POST /documents/bulk-delete`
✅ **Comprehensive metadata** - Chunks, pages, summaries, timestamps
✅ **Interactive docs** - Swagger UI at `/docs`

Start managing your documents programmatically!
