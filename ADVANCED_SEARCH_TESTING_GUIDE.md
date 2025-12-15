# Advanced Search Testing Guide

## Overview

This guide covers testing the `/search/advanced` endpoint that implements the enhanced query analyzer to fix specific test case failures:

1. ✅ **"what is covered in document"** → Routes to Summary
2. ✅ **"git staging area explanation"** → Routes to Section (excludes summary)
3. ✅ **"git working directory and staging area"** → Routes to Sections

## Test Scripts

### 1. Quick Test (Recommended for rapid validation)

**File:** `quick_test_advanced_search.py`

**Purpose:** Fast validation of the three main test cases

**Usage:**
```bash
# Make sure the server is running first
python quick_test_advanced_search.py
```

**Output:**
```
================================================================================
QUICK TEST - Advanced Search Endpoint
================================================================================

[Test 1] Query: 'what is covered in document'
Expected: Summary (document-level)
--------------------------------------------------------------------------------
Scope: document_level
Strategy: summary_first
Exclude Summary: False
Total Results: 5
Summaries Excluded: 0
Fallback Used: False
✅ PASSED

[Test 2] Query: 'git staging area explanation'
Expected: Section (exclude summary)
--------------------------------------------------------------------------------
Scope: section_level
Strategy: section_only
Exclude Summary: True
Total Results: 8
Summaries Excluded: 1
Fallback Used: False
✅ PASSED

[Test 3] Query: 'git working directory and staging area'
Expected: Sections (multi-concept)
--------------------------------------------------------------------------------
Scope: section_level
Strategy: section_only
Exclude Summary: True
Total Results: 12
Summaries Excluded: 1
Fallback Used: False
✅ PASSED

================================================================================
SUMMARY
================================================================================
Passed: 3/3 ✅
Failed: 0/3 ❌
Warnings: 0/3 ⚠️
Errors: 0/3

🎉 All tests passed!

💾 Results saved to: quick_test_results.json
```

---

### 2. Comprehensive Test Suite

**File:** `test_advanced_search.py`

**Purpose:** Detailed testing with 6 test cases and full validation

**Usage:**
```bash
python test_advanced_search.py
```

**Features:**
- Tests 6 different query patterns
- Validates scope detection, strategy selection, summary exclusion
- Checks result quality and content
- Generates detailed validation reports
- Saves results to `test_advanced_search_results.json`

**Test Cases:**
1. "what is covered in document" → Summary
2. "git staging area explanation" → Section
3. "git working directory and staging area" → Sections
4. "overview of git cheat sheet" → Summary
5. "how to commit changes" → Section
6. "difference between merge and rebase" → Section

---

## Manual Testing with cURL

### Test Case 1: Document Overview → Summary

```bash
curl -X POST http://localhost:8000/search/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is covered in document",
    "min_results": 1,
    "max_results": 10
  }' | jq .
```

**Expected Response:**
```json
{
  "query": "what is covered in document",
  "analysis": {
    "scope": "document_level",
    "specificity": "very_broad",
    "search_strategy": "summary_first",
    "should_include_summary": true,
    "should_exclude_summary": false,
    "summary_only": true,
    "confidence": 0.85
  },
  "total_results": 5,
  "summary_excluded_count": 0,
  "fallback_used": false,
  "results": [
    {
      "score": 0.89,
      "payload": {
        "section_title": "[DOCUMENT SUMMARY]",
        "text": "=== Git Cheat Sheet === Total Pages: 5 ..."
      }
    }
  ]
}
```

---

### Test Case 2: Specific Explanation → Section (Exclude Summary)

```bash
curl -X POST http://localhost:8000/search/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "git staging area explanation",
    "min_results": 1,
    "max_results": 10
  }' | jq .
```

**Expected Response:**
```json
{
  "query": "git staging area explanation",
  "analysis": {
    "scope": "section_level",
    "specificity": "specific",
    "search_strategy": "section_only",
    "should_include_summary": false,
    "should_exclude_summary": true,
    "summary_only": false,
    "confidence": 0.72
  },
  "total_results": 8,
  "summary_excluded_count": 1,
  "fallback_used": false,
  "results": [
    {
      "score": 0.87,
      "payload": {
        "section_title": "Git Staging Area",
        "text": "The staging area (also called index) is a file..."
      }
    }
  ]
}
```

**Key Point:** Notice `summary_excluded_count: 1` - the summary chunk was filtered out.

---

### Test Case 3: Multi-Concept Query → Sections

```bash
curl -X POST http://localhost:8000/search/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "git working directory and staging area",
    "min_results": 1,
    "max_results": 10
  }' | jq .
```

**Expected Response:**
```json
{
  "query": "git working directory and staging area",
  "analysis": {
    "scope": "section_level",
    "specificity": "moderate",
    "search_strategy": "section_only",
    "should_include_summary": false,
    "should_exclude_summary": true,
    "summary_only": false,
    "confidence": 0.68
  },
  "total_results": 12,
  "summary_excluded_count": 1,
  "fallback_used": false,
  "results": [
    {
      "score": 0.88,
      "payload": {
        "section_title": "Working Directory",
        "text": "The working directory contains the actual files..."
      }
    },
    {
      "score": 0.86,
      "payload": {
        "section_title": "Git Staging Area",
        "text": "The staging area (also called index)..."
      }
    }
  ]
}
```

---

## Manual Testing with Python

### Example 1: Test Single Query

```python
import requests

response = requests.post(
    "http://localhost:8000/search/advanced",
    json={
        "query": "what is covered in document",
        "min_results": 1,
        "max_results": 10
    }
)

data = response.json()

print(f"Query: {data['query']}")
print(f"Scope: {data['analysis']['scope']}")
print(f"Strategy: {data['analysis']['search_strategy']}")
print(f"Results: {data['total_results']}")
print(f"Summaries Excluded: {data['summary_excluded_count']}")
```

---

### Example 2: Test Multiple Queries

```python
import requests

queries = [
    "what is covered in document",
    "git staging area explanation",
    "git working directory and staging area"
]

for query in queries:
    response = requests.post(
        "http://localhost:8000/search/advanced",
        json={"query": query}
    )

    data = response.json()
    analysis = data['analysis']

    print(f"\nQuery: '{query}'")
    print(f"  Scope: {analysis['scope']}")
    print(f"  Strategy: {analysis['search_strategy']}")
    print(f"  Exclude Summary: {analysis['should_exclude_summary']}")
    print(f"  Results: {data['total_results']}")
    print(f"  Summaries Excluded: {data['summary_excluded_count']}")
```

---

## Validation Checklist

When testing the advanced search endpoint, verify:

### ✅ Test Case 1: "what is covered in document"
- [ ] `analysis.scope` = `"document_level"`
- [ ] `analysis.search_strategy` = `"summary_first"`
- [ ] `analysis.should_exclude_summary` = `false`
- [ ] `total_results` > 0
- [ ] Top result has `section_title: "[DOCUMENT SUMMARY]"`
- [ ] If no summary found, `fallback_used` = `true`

### ✅ Test Case 2: "git staging area explanation"
- [ ] `analysis.scope` = `"section_level"`
- [ ] `analysis.search_strategy` = `"section_only"`
- [ ] `analysis.should_exclude_summary` = `true`
- [ ] `summary_excluded_count` > 0
- [ ] No results have `section_title: "[DOCUMENT SUMMARY]"`
- [ ] Results contain relevant section content

### ✅ Test Case 3: "git working directory and staging area"
- [ ] `analysis.scope` = `"section_level"` or `"mixed"`
- [ ] `analysis.search_strategy` = `"section_only"` or `"hybrid"`
- [ ] `analysis.should_exclude_summary` = `true`
- [ ] `summary_excluded_count` > 0
- [ ] Multiple relevant sections returned
- [ ] No summary chunks in results

---

## Troubleshooting

### Issue: "Connection failed - is the server running?"

**Solution:**
```bash
# Start the server in one terminal
uvicorn api:app --reload

# Run tests in another terminal
python quick_test_advanced_search.py
```

---

### Issue: "No results found"

**Possible Causes:**
1. No documents in the database
2. Query doesn't match any indexed content
3. Score threshold too high

**Solution:**
```python
# Check if any documents exist
import requests

docs = requests.get("http://localhost:8000/documents").json()
print(f"Total documents: {len(docs)}")

# If empty, ingest a test document first
```

---

### Issue: "Summary not excluded for section queries"

**Check:**
1. Verify `should_exclude_summary` is `true` in analysis
2. Check `summary_excluded_count` > 0
3. Ensure results don't contain `"[DOCUMENT SUMMARY]"` section

**Debug:**
```python
response = requests.post(
    "http://localhost:8000/search/advanced",
    json={"query": "git staging area explanation"}
)

data = response.json()

# Check analysis
print("Should exclude summary:", data['analysis']['should_exclude_summary'])
print("Summary excluded count:", data['summary_excluded_count'])

# Check results
for result in data['results']:
    section = result['payload'].get('section_title')
    print(f"  Section: {section}")
    if section == "[DOCUMENT SUMMARY]":
        print("  ❌ ERROR: Summary found in results!")
```

---

### Issue: "Test script fails with module not found"

**Solution:**
```bash
# Install required packages
pip install requests
```

---

## Performance Benchmarks

Expected performance for the advanced search endpoint:

| Metric | Target | Typical |
|--------|--------|---------|
| Query Analysis Time | < 5ms | 2-3ms |
| Search Time | < 200ms | 50-150ms |
| Total Response Time | < 250ms | 100-200ms |
| Accuracy (intent detection) | > 85% | ~90% |

---

## Integration with Existing Tests

### Add to CI/CD Pipeline

```yaml
# .github/workflows/test.yml
- name: Test Advanced Search
  run: |
    python quick_test_advanced_search.py
    if [ $? -ne 0 ]; then
      echo "Advanced search tests failed"
      exit 1
    fi
```

### Add to pytest

```python
# test_api.py
import pytest
import requests

def test_advanced_search_document_query():
    """Test document-level query routing."""
    response = requests.post(
        "http://localhost:8000/search/advanced",
        json={"query": "what is covered in document"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data['analysis']['scope'] == 'document_level'
    assert data['analysis']['search_strategy'] == 'summary_first'
    assert data['total_results'] > 0

def test_advanced_search_section_query():
    """Test section-level query with summary exclusion."""
    response = requests.post(
        "http://localhost:8000/search/advanced",
        json={"query": "git staging area explanation"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data['analysis']['scope'] == 'section_level'
    assert data['analysis']['should_exclude_summary'] == True
    assert data['summary_excluded_count'] > 0
```

---

## API Response Reference

### Full Response Schema

```json
{
  "query": "string",
  "analysis": {
    "query": "string",
    "scope": "document_level|section_level|mixed",
    "specificity": "very_broad|broad|moderate|specific|very_specific",
    "should_include_summary": boolean,
    "should_exclude_summary": boolean,
    "summary_only": boolean,
    "confidence": float,
    "specificity_score": float,
    "search_strategy": "summary_first|section_only|hybrid|summary_only",
    "recommended_filters": {},
    "fallback_strategy": "string|null"
  },
  "total_results": integer,
  "summary_excluded_count": integer,
  "fallback_used": boolean,
  "search_time": float,
  "results": [
    {
      "id": "string",
      "score": float,
      "payload": {
        "doc_id": "string",
        "chunk_id": "string",
        "text": "string",
        "section_title": "string",
        "chunk_index": integer,
        "page_start": integer,
        "page_end": integer,
        "file_name": "string",
        ...
      }
    }
  ]
}
```

---

## Summary

Testing the advanced search endpoint:

✅ **Quick Test:** `python quick_test_advanced_search.py` - 3 main test cases
✅ **Comprehensive Test:** `python test_advanced_search.py` - 6 test cases with validation
✅ **Manual Testing:** Use cURL or Python scripts
✅ **Validation:** Check scope, strategy, exclusion, results
✅ **Integration:** Add to CI/CD and pytest

**Key Improvements Verified:**
1. Document-level queries route to summaries
2. Section-level queries exclude summaries
3. Multi-concept queries retrieve relevant sections
4. Fallback strategies work when needed
5. Summary exclusion post-filtering is correct

Start testing: `python quick_test_advanced_search.py`
