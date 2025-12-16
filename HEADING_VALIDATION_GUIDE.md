# Heading Validation & Hierarchy - Implementation Guide

## 🎯 MOST IMPORTANT: Correct Hierarchy Levels & Parent Section Inheritance

### Problem Solved:
Previously, heading hierarchy was not properly validated:
- ❌ No heading level tracking (H1, H2, H3, etc.)
- ❌ No parent-child relationship validation
- ❌ Broken hierarchy when headings skipped levels
- ❌ No direct parent section reference

### Solution Implemented:

## ✅ 1. CORRECT HIERARCHY LEVELS

### Heading Level Detection (Lines 133-176)

**Proper 6-Level Hierarchy:**
```
Level 1 (H1): # Title, Chapter X, Part X, 1. Title, ALL CAPS
Level 2 (H2): ## Title, Section X, 1.1 Title
Level 3 (H3): ### Title, 1.1.1 Title
Level 4 (H4): #### Title, 1.1.1.1 Title
Level 5 (H5): ##### Title
Level 6 (H6): ###### Title
```

**Pattern Detection:**
```python
# Markdown Headings - Exact level matching
# matches "### Subsection" as Level 3 (H3)
(re.compile(r'^###\s+(.+)$', re.MULTILINE), lambda m: m.group(1).strip(), 3)

# Numbered Headings - Count dots for level
"1. Title" → Level 1
"1.1 Title" → Level 2
"1.1.1 Title" → Level 3
"1.1.1.1 Title" → Level 4

# Chapter/Section/Part - Fixed levels
"Chapter 1: Introduction" → Level 1 (H1)
"Section 2.1: Setup" → Level 2 (H2)
"Part 1: Overview" → Level 1 (H1)
```

### Example: Document with Correct Levels

**Input Document:**
```markdown
# Introduction               ← Level 1 (H1)
This is the intro.

## Getting Started           ← Level 2 (H2)
Setup instructions.

### Installation             ← Level 3 (H3)
Install steps.

#### Windows Installation    ← Level 4 (H4)
Windows-specific steps.

## Configuration             ← Level 2 (H2)
Config details.

### Basic Config             ← Level 3 (H3)
Basic setup.
```

**Detected Levels:**
```json
[
  {"title": "Introduction", "level": 1},
  {"title": "Getting Started", "level": 2},
  {"title": "Installation", "level": 3},
  {"title": "Windows Installation", "level": 4},
  {"title": "Configuration", "level": 2},
  {"title": "Basic Config", "level": 3}
]
```

---

## ✅ 2. PARENT SECTION INHERITANCE

### Algorithm (Lines 633-644)

**Heading Stack Management:**
```python
heading_stack = []  # Stack of (title, level) tuples

for title, level in headings:
    # Remove all headings at same or deeper level
    heading_stack = [h for h in heading_stack if h[1] < level]

    # Add current heading
    heading_stack.append((title, level))

    # Build full path from root to current
    heading_path = [h[0] for h in heading_stack]

    # Get direct parent
    parent = heading_stack[-2][0] if len(heading_stack) > 1 else None
```

### Example: Hierarchy with Parent Inheritance

**Document Structure:**
```markdown
# Chapter 1: Introduction         ← H1
## 1.1 Overview                    ← H2 (parent: "Chapter 1")
### 1.1.1 Purpose                  ← H3 (parent: "1.1 Overview")
### 1.1.2 Scope                    ← H3 (parent: "1.1 Overview")
## 1.2 Background                  ← H2 (parent: "Chapter 1")
# Chapter 2: Architecture          ← H1 (no parent)
## 2.1 System Design               ← H2 (parent: "Chapter 2")
```

**Stored Hierarchy:**
```json
[
  {
    "section_title": "Chapter 1: Introduction",
    "heading_level": 1,
    "parent_section": null,
    "heading_path": ["Chapter 1: Introduction"]
  },
  {
    "section_title": "1.1 Overview",
    "heading_level": 2,
    "parent_section": "Chapter 1: Introduction",
    "heading_path": ["Chapter 1: Introduction", "1.1 Overview"]
  },
  {
    "section_title": "1.1.1 Purpose",
    "heading_level": 3,
    "parent_section": "1.1 Overview",
    "heading_path": ["Chapter 1: Introduction", "1.1 Overview", "1.1.1 Purpose"]
  },
  {
    "section_title": "1.1.2 Scope",
    "heading_level": 3,
    "parent_section": "1.1 Overview",
    "heading_path": ["Chapter 1: Introduction", "1.1 Overview", "1.1.2 Scope"]
  },
  {
    "section_title": "1.2 Background",
    "heading_level": 2,
    "parent_section": "Chapter 1: Introduction",
    "heading_path": ["Chapter 1: Introduction", "1.2 Background"]
  },
  {
    "section_title": "Chapter 2: Architecture",
    "heading_level": 1,
    "parent_section": null,
    "heading_path": ["Chapter 2: Architecture"]
  },
  {
    "section_title": "2.1 System Design",
    "heading_level": 2,
    "parent_section": "Chapter 2: Architecture",
    "heading_path": ["Chapter 2: Architecture", "2.1 System Design"]
  }
]
```

---

## 🔍 Hierarchy Validation Rules

### Rule 1: Level Sequence Validation
✅ **Valid:**
```
H1 → H2 → H3 → H2 → H3 → H1
```

❌ **Invalid (but now handled correctly):**
```
H1 → H3  # Skipped H2 (system maintains correct parent = H1)
```

### Rule 2: Parent Inheritance
When a heading appears:
1. **Remove** all headings at same or deeper level from stack
2. **Keep** all headings at shallower levels (these are parents)
3. **Add** current heading to stack
4. **Build** path from all headings in stack

### Rule 3: Direct Parent Reference
- **Level 1 (H1)**: parent_section = `null` (top level)
- **Level 2 (H2)**: parent_section = most recent H1
- **Level 3 (H3)**: parent_section = most recent H2
- **Level 4 (H4)**: parent_section = most recent H3
- And so on...

---

## 📊 New Metadata Fields

### ChunkMetadata Fields (Lines 56-60)

```python
@dataclass
class ChunkMetadata:
    # ... other fields ...

    # Section hierarchy (VALIDATED)
    section_title: Optional[str]      # Current section name
    section_title_raw: Optional[str]  # Exact as in document
    heading_path: List[str]           # Full hierarchy path
    heading_level: Optional[int]      # 1-6 (H1-H6)
    parent_section: Optional[str]     # Direct parent name
```

### Field Descriptions:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `section_title` | str | Clean current section name | `"1.2 Configuration"` |
| `section_title_raw` | str | Exact text from document | `"1.2 Configuration"` |
| `heading_path` | list | Full path from root | `["Chapter 1", "1.2 Configuration"]` |
| `heading_level` | int | Validated level (1-6) | `2` |
| `parent_section` | str | Direct parent name | `"Chapter 1"` |

---

## 🧪 Testing Hierarchy Validation

### Test 1: Basic Hierarchy

**Document:**
```markdown
# Main Title
## Subtitle
### Sub-subtitle
```

**Expected:**
```json
[
  {
    "section_title": "Main Title",
    "heading_level": 1,
    "parent_section": null,
    "heading_path": ["Main Title"]
  },
  {
    "section_title": "Subtitle",
    "heading_level": 2,
    "parent_section": "Main Title",
    "heading_path": ["Main Title", "Subtitle"]
  },
  {
    "section_title": "Sub-subtitle",
    "heading_level": 3,
    "parent_section": "Subtitle",
    "heading_path": ["Main Title", "Subtitle", "Sub-subtitle"]
  }
]
```

### Test 2: Level Skip (H1 → H3)

**Document:**
```markdown
# Main Title
### Deep Heading  ← Skipped H2
## Back to H2
```

**Expected:**
```json
[
  {
    "section_title": "Main Title",
    "heading_level": 1,
    "parent_section": null,
    "heading_path": ["Main Title"]
  },
  {
    "section_title": "Deep Heading",
    "heading_level": 3,
    "parent_section": "Main Title",  ← Parent is H1 (correct!)
    "heading_path": ["Main Title", "Deep Heading"]
  },
  {
    "section_title": "Back to H2",
    "heading_level": 2,
    "parent_section": "Main Title",
    "heading_path": ["Main Title", "Back to H2"]
  }
]
```

### Test 3: Multiple H1 Sections

**Document:**
```markdown
# Chapter 1
## Section 1.1
### Subsection 1.1.1

# Chapter 2  ← New H1 resets hierarchy
## Section 2.1
```

**Expected:**
```json
[
  {
    "section_title": "Chapter 1",
    "heading_level": 1,
    "parent_section": null,
    "heading_path": ["Chapter 1"]
  },
  {
    "section_title": "Section 1.1",
    "heading_level": 2,
    "parent_section": "Chapter 1",
    "heading_path": ["Chapter 1", "Section 1.1"]
  },
  {
    "section_title": "Subsection 1.1.1",
    "heading_level": 3,
    "parent_section": "Section 1.1",
    "heading_path": ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]
  },
  {
    "section_title": "Chapter 2",
    "heading_level": 1,
    "parent_section": null,  ← Hierarchy reset
    "heading_path": ["Chapter 2"]  ← New root
  },
  {
    "section_title": "Section 2.1",
    "heading_level": 2,
    "parent_section": "Chapter 2",
    "heading_path": ["Chapter 2", "Section 2.1"]
  }
]
```

---

## 🔍 Verification Queries

### Query 1: Check Heading Levels

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Get all chunks with heading levels
results = client.scroll(
    collection_name="knowvec_documents",
    limit=100
)

for point in results[0]:
    title = point.payload.get('section_title')
    level = point.payload.get('heading_level')
    parent = point.payload.get('parent_section')
    path = point.payload.get('heading_path')

    print(f"{'  ' * (level - 1 if level else 0)}{title} (L{level}, parent: {parent})")
    print(f"  Path: {' > '.join(path)}")
    print()
```

### Query 2: Find All H1 Sections

```bash
curl -X POST http://localhost:8000/search/filtered \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test",
    "filters": {"heading_level": 1}
  }' | jq '.[] | {section_title, heading_level, parent_section}'
```

### Query 3: Find Children of Specific Section

```python
# Find all sections with specific parent
results = client.scroll(
    collection_name="knowvec_documents",
    scroll_filter={
        "must": [
            {
                "key": "parent_section",
                "match": {"value": "Chapter 1: Introduction"}
            }
        ]
    }
)
```

---

## 📈 Benefits of Validated Hierarchy

### 1. Accurate Navigation
```python
# Can now query: "What are the subsections of Chapter 1?"
subsections = query_by_parent("Chapter 1: Introduction")

# Can now query: "Show me all H3 headings under Section 2.1"
deep_sections = query_by_parent_and_level("Section 2.1", 3)
```

### 2. Better Context Understanding
```python
# Chunk knows its full context
chunk = {
    "text": "Configuration details...",
    "section_title": "Database Configuration",
    "heading_path": ["Chapter 2: Setup", "2.1 Installation", "Database Configuration"],
    "parent_section": "2.1 Installation",
    "heading_level": 3
}

# LLM can understand: This is about Database Configuration,
# which is part of Installation (2.1), which is in Chapter 2: Setup
```

### 3. Structured Retrieval
```python
# Can retrieve in hierarchical order
# 1. Find matching H1
# 2. Get all its H2 children
# 3. Get all H3 children of those H2s
```

---

## ✅ Summary

### What Was Implemented:

1. **✅ Correct Hierarchy Levels (DONE)**
   - 6-level hierarchy (H1-H6)
   - Pattern-based level detection
   - Level stored in `heading_level` field

2. **✅ Parent Section Inheritance (DONE)**
   - Heading stack algorithm
   - Parent tracking in `parent_section` field
   - Full path in `heading_path` array

3. **✅ Validation Rules (DONE)**
   - Proper stack management
   - Level skip handling
   - Hierarchy reset on new H1

### New Fields:
- `heading_level`: int (1-6)
- `parent_section`: str (direct parent name)
- `heading_path`: list (full hierarchy)

### Files Modified:
- **enterprise_chunking_pipeline.py**
  - Updated patterns (lines 133-176)
  - Added hierarchy validation (lines 633-644)
  - Added new metadata fields (lines 56-60, 851-853)

---

**Status:** ✅ **COMPLETE - HIERARCHY VALIDATION IMPLEMENTED**
**Date:** 2025-12-16
