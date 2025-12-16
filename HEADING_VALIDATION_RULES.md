# Strict Heading Validation Rules

## 🎯 Purpose: Prevent False Positives

This document defines **exactly what qualifies as a section heading** to prevent incorrectly treating questions, exercises, code snippets, or inline text as section headings.

---

## ✅ POSITIVE RULES (Must Pass ALL)

A text is considered a section heading ONLY if it passes ALL these rules:

### Rule 1: Isolated Line
**The heading must be on its own line, not inline with other text.**

✅ **VALID:**
```
Introduction
This is the content...
```

❌ **INVALID:**
```
The Introduction section covers basic concepts.  ← "Introduction" is inline
```

### Rule 2: Reasonable Length
**Length must be between 3-100 characters.**

✅ **VALID:**
```
Introduction                    (12 chars)
System Architecture Overview    (28 chars)
```

❌ **INVALID:**
```
In                             (2 chars - too short)
This is a very long heading that goes on and on and includes multiple clauses and sub-clauses making it paragraph-like   (120+ chars - too long)
```

### Rule 3: Starts with Capital or Number
**Must start with uppercase letter or digit.**

✅ **VALID:**
```
Introduction
2.1 Configuration
Chapter 1: Overview
```

❌ **INVALID:**
```
introduction        (lowercase)
the basics          (lowercase article)
```

### Rule 4: Reasonable Word Count
**Between 1-15 words. Not too few, not too many.**

✅ **VALID:**
```
Introduction                           (1 word)
Getting Started Guide                  (3 words)
Advanced Configuration Options         (3 words)
```

❌ **INVALID:**
```
A                                      (Too short)
This is a very long title with many many words that really should be a paragraph instead of a heading   (16+ words)
```

---

## ❌ NEGATIVE RULES (Must Pass NONE)

A text is **NOT** a section heading if it matches ANY of these patterns:

### Negative Rule 1: Questions
**Text ending with ? or starting with question words is NOT a heading.**

❌ **NOT HEADINGS:**
```
What is Git?
How do I install?
Why use Python?
When should I commit?
Where is the config file?
```

✅ **VALID HEADING:**
```
Introduction to Git    (Statement, not question)
```

### Negative Rule 2: Exercises/Practice Prompts
**Exercise-related text is NOT a section heading.**

❌ **NOT HEADINGS:**
```
Exercise 1
Practice Problem
Try It Yourself
Hands-On Lab
Assignment 3
Homework Questions
Quiz
Test Yourself
Challenge
Activity: Build a Program
```

✅ **VALID HEADING:**
```
Advanced Techniques    (Section about techniques)
```

### Negative Rule 3: Fill-in-Blank or Multiple Choice
**Test questions are NOT headings.**

❌ **NOT HEADINGS:**
```
The ____ is used for version control
Git was created in ____
(A) True  (B) False
A) Option 1
```

### Negative Rule 4: Code Snippets
**Code syntax is NOT a heading.**

❌ **NOT HEADINGS:**
```
function()
get_user()
{variable}
array[]
user => data
object::method
if (condition)
```

✅ **VALID HEADING:**
```
Function Definitions    (About functions, not code itself)
```

### Negative Rule 5: List Items
**Bullet points or list markers are NOT headings.**

❌ **NOT HEADINGS:**
```
• First item
- Second item
* Third item
+ Fourth item
```

✅ **VALID HEADING:**
```
1. Introduction        (Numbered section)
1.1 Overview          (Numbered subsection)
```

### Negative Rule 6: URLs, Paths, Emails
**Technical identifiers are NOT headings.**

❌ **NOT HEADINGS:**
```
https://example.com/docs
/usr/local/bin
user@example.com
C:\Program Files\App
```

### Negative Rule 7: Numbers Only or Dates
**Pure numbers or dates are NOT headings.**

❌ **NOT HEADINGS:**
```
2024
12/25/2023
01-15-2024
42
```

✅ **VALID HEADING:**
```
2024 Release Notes     (Number + description)
Chapter 1              (Number + label)
```

### Negative Rule 8: All Lowercase
**All lowercase text (with exceptions) is NOT a heading.**

❌ **NOT HEADINGS:**
```
introduction
getting started
basic concepts
```

**Exceptions:**
- Markdown headings: `# introduction` ✓
- Numbered: `1. introduction` ✓

✅ **VALID HEADING:**
```
Introduction           (Proper capitalization)
Getting Started        (Title case)
```

### Negative Rule 9: Too Many Special Characters
**Text with excessive special characters (>3) is NOT a heading.**

❌ **NOT HEADINGS:**
```
!!URGENT!!
***ATTENTION***
<<< WARNING >>>
{{ config }}
```

✅ **VALID HEADING:**
```
Important Notice       (No excess special chars)
```

### Negative Rule 10: Starts with Conjunction
**Text starting with "and", "but", "or" etc. is NOT a heading (likely a continuation).**

❌ **NOT HEADINGS:**
```
And another thing
But wait there's more
Or alternatively
```

---

## 📊 Validation Examples

### Example Set 1: Valid Section Headings

```markdown
✅ Introduction
✅ Chapter 1: Getting Started
✅ 1.1 Installation Guide
✅ System Architecture
✅ Advanced Configuration
✅ API Documentation
✅ Database Schema
✅ User Management
```

### Example Set 2: Invalid - Questions

```markdown
❌ What is Docker?
❌ How do I configure this?
❌ Why use TypeScript?
❌ When to use async/await?
```

**Why:** Questions end with ? or start with question words.

### Example Set 3: Invalid - Exercises

```markdown
❌ Exercise 1: Create a Function
❌ Practice Problem
❌ Try It Yourself
❌ Hands-On Lab
❌ Assignment: Build an API
```

**Why:** Contains exercise/practice keywords.

### Example Set 4: Invalid - Code

```markdown
❌ getUserData()
❌ if (condition)
❌ array[index]
❌ object.method()
```

**Why:** Contains code syntax patterns.

### Example Set 5: Invalid - Lists

```markdown
❌ • First step
❌ - Second step
❌ * Third step
```

**Why:** Starts with bullet points.

### Example Set 6: Invalid - Too Short/Long

```markdown
❌ Hi
❌ A
❌ This is an extremely long heading that contains way too many words and should really be a paragraph instead of a section heading because headings should be concise
```

**Why:** Too short (<3 chars) or too long (>100 chars) or too many words (>15).

### Example Set 7: Invalid - Inline Text

```markdown
❌ "The Introduction section describes..."  ← "Introduction" is inline

✅ Introduction                              ← Isolated on its own line
   This section describes...
```

**Why:** Must be isolated, not embedded in sentence.

---

## 🔍 Special Cases

### Case 1: Numbered Sections vs. Numbered Lists

✅ **HEADING:**
```
1. Introduction
1.1 Overview
2. Architecture
2.1 System Design
```

❌ **NOT HEADING:**
```
1. First create a file
2. Then open the editor
3. Finally save the document
```

**Difference:** Headings are short, title-like. Lists are instructions/steps.

### Case 2: ALL CAPS

✅ **HEADING (if long enough):**
```
INTRODUCTION          (11+ chars)
SYSTEM OVERVIEW       (14 chars)
```

❌ **NOT HEADING:**
```
URGENT                (6 chars - too short for all caps)
```

### Case 3: Chapter/Section Keywords

✅ **HEADING:**
```
Chapter 1: Introduction
Section 2.1: Configuration
Part 1: Fundamentals
```

❌ **NOT HEADING:**
```
Chapter Summary Questions
Section Practice Problems
```

**Difference:** True chapter/section labels vs. exercise prompts.

---

## 🎯 Implementation Details

### Validation Function: `_is_valid_section_heading()`

**Location:** `enterprise_chunking_pipeline.py` (lines 215-329)

**Usage:**
```python
detector = BoundaryDetector()

# Test a heading
is_valid = detector._is_valid_section_heading(
    title="Introduction",
    match_text="# Introduction",
    surrounding_context="# Introduction\nThis is the content..."
)

if is_valid:
    print("✅ Valid section heading")
else:
    print("❌ Not a section heading")
```

### Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `title` | str | Extracted clean heading text |
| `match_text` | str | Original matched text with formatting |
| `surrounding_context` | str | Text around the match (for isolation check) |

### Returns:
- `True` if text passes ALL positive rules and NO negative rules
- `False` otherwise

---

## 📋 Validation Checklist

When processing a document, each potential heading is checked:

- [ ] ✅ Length 3-100 characters
- [ ] ✅ Word count 1-15 words
- [ ] ✅ Starts with capital letter or number
- [ ] ✅ Isolated on own line
- [ ] ❌ NOT a question (no ?)
- [ ] ❌ NOT exercise/practice prompt
- [ ] ❌ NOT fill-in-blank or multiple choice
- [ ] ❌ NOT code snippet
- [ ] ❌ NOT list item
- [ ] ❌ NOT URL, path, or email
- [ ] ❌ NOT pure number or date
- [ ] ❌ NOT all lowercase (except exceptions)
- [ ] ❌ NOT too many special characters
- [ ] ❌ NOT starting with conjunction

**Result:** Only if ALL checks pass, text is recognized as a section heading.

---

## 🧪 Testing

### Test Document:

```markdown
# Introduction
This is the intro.

## What is Git?
This looks like a heading but it's a question.

## Getting Started
This is a real section heading.

### Exercise 1: Install Git
This is an exercise, not a section.

### Installation Steps
This is a real subsection.

1. Clone the repo
2. Run npm install
3. Start the server

### Configuration Options
This is a real subsection.

function getUserData() {
  return data;
}
```

**Expected Headings Detected:**
```
✅ Introduction (Level 1)
❌ What is Git? (Question - rejected)
✅ Getting Started (Level 2)
❌ Exercise 1: Install Git (Exercise - rejected)
✅ Installation Steps (Level 3)
❌ 1. Clone the repo (List item - rejected)
✅ Configuration Options (Level 3)
❌ function getUserData() (Code - rejected)
```

---

## 📊 Benefits of Strict Validation

### 1. **Accurate Section Context**
```python
# BEFORE (without validation):
chunk = {
    "section_title": "What is Git?"  # Wrong!
}

# AFTER (with validation):
chunk = {
    "section_title": "Introduction to Git"  # Correct!
}
```

### 2. **Better Search Results**
Users searching for "Introduction" will find the actual section, not questions about it.

### 3. **Proper Hierarchy**
```
Introduction
  Getting Started  ← Real subsection
    Installation   ← Real sub-subsection

NOT:
Introduction
  What is Git?     ← This was incorrectly treated as subsection
    Exercise 1     ← This was incorrectly treated as sub-subsection
```

### 4. **Clean Metadata**
All `section_title` fields contain only real section names, not questions or exercises.

---

## ✅ Summary

### What Qualifies as a Section Heading:
1. **Isolated** on its own line
2. **3-100 characters** long
3. **Starts** with capital or number
4. **1-15 words**
5. **NOT** a question, exercise, code, list, URL, etc.

### Files Modified:
- **enterprise_chunking_pipeline.py**
  - Added `_is_valid_section_heading()` method (lines 215-329)
  - Updated `find_section_boundaries()` to use validation (lines 331-380)
  - 10 negative rules + 4 positive rules implemented

### Result:
✅ Only TRUE section headings are recognized
✅ Questions, exercises, code NOT treated as headings
✅ Proper section context maintained
✅ Clean, accurate metadata

---

**Status:** ✅ **STRICT VALIDATION IMPLEMENTED**
**Date:** 2025-12-16
