# Quick Start: Testing Advanced Search

## 🚀 Run Tests in 3 Steps

### Step 1: Start the Server
```bash
uvicorn api:app --reload
```

### Step 2: Run Quick Tests (New Terminal)
```bash
python quick_test_advanced_search.py
```

### Step 3: Verify Results
✅ All 3 tests should pass:
- Test 1: "what is covered in document" → Summary
- Test 2: "git staging area explanation" → Section (no summary)
- Test 3: "git working directory and staging area" → Sections

---

## 📊 Expected Output

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
✅ PASSED

[Test 2] Query: 'git staging area explanation'
Expected: Section (exclude summary)
--------------------------------------------------------------------------------
Scope: section_level
Strategy: section_only
Exclude Summary: True
Total Results: 8
Summaries Excluded: 1
✅ PASSED

[Test 3] Query: 'git working directory and staging area'
Expected: Sections (multi-concept)
--------------------------------------------------------------------------------
Scope: section_level
Strategy: section_only
Exclude Summary: True
Total Results: 12
Summaries Excluded: 1
✅ PASSED

================================================================================
SUMMARY
================================================================================
Passed: 3/3 ✅

🎉 All tests passed!
```

---

## 🧪 Manual Testing

### Test with cURL

```bash
# Test 1: Document overview
curl -X POST http://localhost:8000/search/advanced \
  -H "Content-Type: application/json" \
  -d '{"query": "what is covered in document"}'

# Test 2: Specific explanation
curl -X POST http://localhost:8000/search/advanced \
  -H "Content-Type: application/json" \
  -d '{"query": "git staging area explanation"}'

# Test 3: Multi-concept
curl -X POST http://localhost:8000/search/advanced \
  -H "Content-Type: application/json" \
  -d '{"query": "git working directory and staging area"}'
```

### Test with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/search/advanced",
    json={"query": "what is covered in document"}
)

data = response.json()
print(f"Scope: {data['analysis']['scope']}")
print(f"Strategy: {data['analysis']['search_strategy']}")
print(f"Results: {data['total_results']}")
```

---

## 📝 What's Being Tested?

### Test Case 1: "what is covered in document"
- **Expected:** Routes to summary chunks
- **Strategy:** `summary_first` with fallback to sections
- **Should have:** Document-level scope, summary results

### Test Case 2: "git staging area explanation"
- **Expected:** Routes to section chunks, excludes summaries
- **Strategy:** `section_only` with post-filtering
- **Should have:** Section-level scope, no summary in results

### Test Case 3: "git working directory and staging area"
- **Expected:** Routes to multiple section chunks
- **Strategy:** `section_only` for multi-concept query
- **Should have:** Section-level scope, multiple relevant sections

---

## ⚠️ Troubleshooting

### "Connection Error"
→ Make sure server is running: `uvicorn api:app --reload`

### "No Results Found"
→ Check if documents exist: `curl http://localhost:8000/documents`
→ If empty, ingest a test document first

### Tests Fail
→ Check server logs for errors
→ Verify query_analyzer.py exists
→ Ensure api.py has /search/advanced endpoint

---

## 📚 More Information

- **Comprehensive Tests:** `python test_advanced_search.py`
- **Full Testing Guide:** `ADVANCED_SEARCH_TESTING_GUIDE.md`
- **Implementation Details:** `IMPLEMENTATION_SUMMARY.md`
- **API Documentation:** http://localhost:8000/docs

---

## ✅ Success Criteria

Tests pass if:
- ✅ Test 1 routes to `summary_first` strategy
- ✅ Test 2 routes to `section_only` and excludes summary
- ✅ Test 3 routes to sections and returns multiple results
- ✅ All tests show "PASSED ✅"
- ✅ Response time < 300ms

---

## 🎯 Next Steps After Testing

1. **If all tests pass:**
   - Integration complete ✅
   - Ready for production use
   - Consider adding more test documents

2. **If tests fail:**
   - Check troubleshooting section
   - Review server logs
   - Verify api.py and query_analyzer.py are correct

3. **For production:**
   - Monitor query patterns
   - Collect user feedback
   - Fine-tune patterns as needed

---

**Quick Command Reference:**
```bash
# Start server
uvicorn api:app --reload

# Run tests
python quick_test_advanced_search.py

# Comprehensive tests
python test_advanced_search.py

# Check documents
curl http://localhost:8000/documents

# Test single query
curl -X POST http://localhost:8000/search/advanced \
  -H "Content-Type: application/json" \
  -d '{"query": "your query here"}'
```
