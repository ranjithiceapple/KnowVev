# Section Name Preservation - Implementation Guide

## ✅ What Was Fixed

### Problem:
- Section names were stored as placeholders like `#`, `<<<HIERARCHY_L1>>>`, or synthetic labels
- Markdown headings included `#` symbols in section_title
- No preservation of exact heading text from original documents

### Solution Implemented:

#### ✅ STEP 1: Preserve Raw Heading Text During Extraction
**File:** `document_processor.py`
- DOCX files: Extract headings from paragraph styles (line 487)
- PDF files: Extract headings using pattern detection (line 355)
- Headings stored in `PageMetadata.headings` and `DocumentMetadata.section_titles`

#### ✅ STEP 2: Improve Heading Detection Logic
**File:** `enterprise_chunking_pipeline.py` (lines 133-156)

**NEW Pattern Structure:**
```python
# (regex_pattern, extractor_function, level)
(re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE),
 lambda m: m.group(1).strip(), 1)  # Extracts text WITHOUT #
```

**Supported Heading Formats:**
1. **ALL CAPS HEADINGS**
   - `INTRODUCTION` → `"INTRODUCTION"`

2. **Chapter/Section/Part Markers**
   - `Chapter 1: Getting Started` → `"Chapter 1: Getting Started"`
   - `Section 2.1: Configuration` → `"Section 2.1: Configuration"`

3. **Numbered Headings**
   - `1. Introduction` → `"1. Introduction"`
   - `2.1 Setup Guide` → `"2.1 Setup Guide"`

4. **Markdown Headings**
   - `# Main Title` → `"Main Title"` (NOT `"# Main Title"`)
   - `## Subsection` → `"Subsection"` (NOT `"## Subsection"`)

5. **Underlined Headings**
   ```
   Main Title
   ==========
   ```
   → `"Main Title"`

#### ✅ STEP 3: Never Store Placeholders
**File:** `enterprise_chunking_pipeline.py` (lines 214-216)

```python
# Never store placeholders or synthetic markers
if title.startswith('<<<') or title.startswith('#'):
    continue
```

- Filters out `<<<HIERARCHY_L1>>>`, `<<<HIERARCHY_L2>>>`, etc.
- Filters out improperly extracted markdown headings with `#`
- Sets `section_title=None` instead of using generic placeholders

#### ✅ STEP 4: Store BOTH Clean and Raw Section Names
**File:** `enterprise_chunking_pipeline.py` (lines 56-57)

**ChunkMetadata Fields:**
```python
section_title: Optional[str] = None  # Clean section name (primary field)
section_title_raw: Optional[str] = None  # Raw section name as appears in document
```

**Usage:**
- `section_title`: Clean text for search and display
- `section_title_raw`: Exact text as it appears in the original document
- Both fields are now populated during chunking (line 636-637)

---

## 🔍 How to Verify Section Name Preservation

### Method 1: Check Chunk Metadata in Database

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Get a few chunks
results = client.scroll(
    collection_name="knowvec_documents",
    limit=10
)

# Check section_title field
for point in results[0]:
    payload = point.payload
    print(f"Section: {payload.get('section_title')}")
    print(f"Section Raw: {payload.get('section_title_raw')}")
    print(f"Heading Path: {payload.get('heading_path')}")
    print("-" * 50)
```

### Method 2: Check During Document Upload

After uploading a document, check the API response:

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@your_document.docx" | jq '.chunks[].section_title'
```

**Expected Output:**
```
"Introduction"
"System Architecture"
"1.1 Core Components"
"2. Installation Guide"
null  # For chunks without clear section headings
```

**NOT this:**
```
"#"
"<<<HIERARCHY_L1>>>"
"# Introduction"  # Should NOT have # symbol
```

### Method 3: Test with Sample Document

Create a test document with these headings:

```markdown
# Main Introduction

This is the introduction section.

## 1.1 Getting Started

Steps to get started.

### Configuration Options

Details about configuration.
```

**Expected Section Titles:**
- `"Main Introduction"` (NOT `"# Main Introduction"`)
- `"1.1 Getting Started"` (NOT `"## 1.1 Getting Started"`)
- `"Configuration Options"` (NOT `"### Configuration Options"`)

---

## 📊 Examples of Correct Behavior

### Example 1: DOCX Document

**Original Document Headings:**
```
Heading 1: System Overview
Heading 2: Technical Architecture
Heading 3: Database Schema
```

**Stored in Qdrant:**
```json
{
  "section_title": "System Overview",
  "section_title_raw": "System Overview",
  "heading_path": ["System Overview"]
}
```

### Example 2: Markdown Document

**Original Markdown:**
```markdown
# User Authentication

## Login Flow

### Password Reset
```

**Stored in Qdrant:**
```json
[
  {
    "section_title": "User Authentication",
    "heading_path": ["User Authentication"]
  },
  {
    "section_title": "Login Flow",
    "heading_path": ["User Authentication", "Login Flow"]
  },
  {
    "section_title": "Password Reset",
    "heading_path": ["User Authentication", "Login Flow", "Password Reset"]
  }
]
```

### Example 3: Numbered Sections

**Original Document:**
```
1. Introduction
   1.1 Purpose
   1.2 Scope

2. Requirements
   2.1 Functional Requirements
```

**Stored in Qdrant:**
```json
[
  {
    "section_title": "1. Introduction",
    "heading_path": ["1. Introduction"]
  },
  {
    "section_title": "1.1 Purpose",
    "heading_path": ["1. Introduction", "1.1 Purpose"]
  },
  {
    "section_title": "2. Requirements",
    "heading_path": ["2. Requirements"]
  }
]
```

---

## 🚫 What Will NO LONGER Happen

### ❌ BEFORE (Bad):
```json
{
  "section_title": "#",
  "section_title": "<<<HIERARCHY_L1>>>",
  "section_title": "## Installation",  // Includes markdown symbols
  "section_title": "Section 1",  // Generic placeholder
}
```

### ✅ AFTER (Good):
```json
{
  "section_title": "Installation Guide",  // Actual heading text
  "section_title": "Getting Started",
  "section_title": "1.2 Configuration Options",
  "section_title": null,  // If no clear heading found
}
```

---

## 🧪 Testing Checklist

After implementing these changes:

- [ ] Upload a DOCX file with heading styles
  - Verify section_title shows actual heading text
  - Verify no `<<<HIERARCHY>>>` markers

- [ ] Upload a Markdown file
  - Verify section_title doesn't include `#` symbols
  - Verify heading hierarchy is preserved

- [ ] Upload a PDF with numbered sections
  - Verify section_title includes numbers (e.g., "2.1 Setup")
  - Verify subsection hierarchy is correct

- [ ] Check chunks without headings
  - Verify section_title is `null` (not a placeholder)

- [ ] Query by section name
  ```bash
  curl -X POST http://localhost:8000/search/filtered \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "filters": {"section_title": "Introduction"}}'
  ```

---

## 🔧 Troubleshooting

### Issue: Still seeing `#` in section_title

**Check:** Is the file markdown?
**Solution:** The new patterns should extract without `#`. Check if you restarted the server.

```bash
# Restart server to load new code
uvicorn api:app --reload
```

### Issue: Still seeing `<<<HIERARCHY>>>` markers

**Check:** Are you using an old version of the chunking pipeline?
**Solution:** The new code filters these out (line 214-216). Make sure changes were saved and server restarted.

### Issue: Section names are still `null` when they should have values

**Check:** Pattern matching might not recognize your heading format
**Debug:**
```python
# Test heading detection
from enterprise_chunking_pipeline import BoundaryDetector

detector = BoundaryDetector()
text = """
Your Heading Here
More text...
"""

boundaries = detector.find_section_boundaries(text)
print(boundaries)  # Should show extracted headings
```

---

## 📝 Summary

### Changes Made:

1. **enterprise_chunking_pipeline.py**
   - Removed synthetic `<<<HIERARCHY>>>` patterns
   - Updated patterns to extract clean heading text (without `#`)
   - Added `section_title_raw` field to ChunkMetadata
   - Updated `_chunk_by_sections` to preserve raw section names
   - Added placeholder filtering in `find_section_boundaries`

2. **document_processor.py**
   - Already extracting headings correctly from DOCX, PDF, TXT
   - No changes needed - already preserves raw heading text

### Files Modified:
- ✅ `enterprise_chunking_pipeline.py` - Main changes
- ✅ `SECTION_NAME_PRESERVATION_GUIDE.md` - This guide

### Result:
✅ Section names are now **exact text from original documents**
✅ No more `#`, `<<<HIERARCHY>>>`, or generic placeholders
✅ Both clean and raw section names stored
✅ Heading hierarchy properly preserved

---

**Status:** ✅ Complete - Ready for Testing
**Date:** 2025-12-16
