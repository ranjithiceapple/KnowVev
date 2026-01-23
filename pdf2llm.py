import os
import logging

# ------------------------------------------------------------------
# Configure environment for marker-pdf
# Suppress all verbose output (tqdm, transformers, surya, etc.)
# This prevents "[98B blob data]" spam in systemd journal
# ------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silence TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"  # Prevent 'sdpa' KeyError

# Disable tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"

# Disable huggingface_hub progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Disable transformers progress bars
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress verbose loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("surya").setLevel(logging.WARNING)
logging.getLogger("marker").setLevel(logging.WARNING)

import pymupdf4llm
import pymupdf as fitz
import pdfplumber
import pandas as pd
from pathlib import Path
import argparse
import sys
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Marker-pdf imports (optional, for advanced extraction)
# NOTE: marker-pdf 1.10.1+ has a completely different API
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    PdfConverter = None
    create_model_dict = None
    text_from_rendered = None


# ------------------------------------------------------------------
# Marker-PDF GLOBAL CACHED CONVERTER (Streamlit-compatible)
# ------------------------------------------------------------------

_MARKER_CONVERTER = None

def load_marker_converter(
    vram_gb=8,
    extract_images=False,
    output_format="markdown"
):
    """
    Canonical marker-pdf loader.
    EXACTLY matches the working Streamlit implementation.
    Cached globally for CLI / batch usage.
    """
    global _MARKER_CONVERTER

    if _MARKER_CONVERTER is not None:
        return _MARKER_CONVERTER

    # Environment configuration (same as Streamlit)
    os.environ.setdefault("TORCH_DEVICE", "cuda")
    os.environ.setdefault("INFERENCE_RAM", str(vram_gb))

    config = {
        "output_format": output_format,
        "extract_images": extract_images,
        "vram_per_task": vram_gb,
    }

    model_dict = create_model_dict()

    _MARKER_CONVERTER = PdfConverter(
        artifact_dict=model_dict,
        config=config
    )

    return _MARKER_CONVERTER



def roman_to_int(s):
    """Convert Roman numeral to integer"""
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev_value = 0
    for char in reversed(s.upper()):
        value = roman_values.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    return total


def is_roman_numeral(s):
    """Check if string is a valid Roman numeral"""
    pattern = r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(pattern, s.upper().strip()))


def detect_section_patterns(text):
    """Detect various section heading patterns"""
    text = text.strip()

    # Pattern 1: Roman numerals (I., II., III., etc.)
    if re.match(r'^[IVXLCDM]+\.?\s+[A-Z]', text):
        parts = text.split(None, 1)
        if parts and is_roman_numeral(parts[0].rstrip('.')):
            return True, 'roman_section'

    # Pattern 2: Abstract, Introduction, Conclusion, etc.
    special_sections = [
        r'^ABSTRACT\s*$',
        r'^Abstract\s*$',
        r'^INTRODUCTION\s*$',
        r'^Introduction\s*$',
        r'^CONCLUSION\s*$',
        r'^Conclusion\s*$',
        r'^REFERENCES\s*$',
        r'^References\s*$',
        r'^ACKNOWLEDGMENT',
        r'^Acknowledgment',
        r'^INDEX\s+TERMS?\s*$',
        r'^Index\s+Terms?\s*$',
        r'^KEYWORDS?\s*$',
        r'^Keywords?\s*$',
        r'^\d+\.\s+[A-Z][A-Za-z\s]+$',  # "1. Introduction"
        r'^[A-Z]+\s+[IVX]+\.?\s*$',  # "SECTION I"
    ]

    for pattern in special_sections:
        if re.match(pattern, text):
            return True, 'special_section'

    # Pattern 3: All caps short titles (likely headings)
    if text.isupper() and 3 < len(text) < 60 and len(text.split()) <= 8:
        return True, 'all_caps'

    # Pattern 4: Numbered sections (1.1, 2.3.4, etc.)
    if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', text):
        return True, 'numbered_section'

    return False, None


def extract_toc_structure(pdf_path):
    """Extract Table of Contents from PDF with hierarchy"""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    doc.close()

    if not toc:
        return []

    # TOC format: [[level, title, page_num], ...]
    toc_structure = []
    for entry in toc:
        level, title, page = entry
        toc_structure.append({
            'level': level,
            'title': title.strip(),
            'page': page,
            'is_toc_entry': True
        })

    return toc_structure


def analyze_font_sizes(pdf_path):
    """Analyze font sizes to identify potential headings"""
    doc = fitz.open(pdf_path)
    font_stats = defaultdict(lambda: {'count': 0, 'examples': []})

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = round(span.get("size", 0), 1)
                        text = span.get("text", "").strip()
                        font = span.get("font", "")

                        if text and size > 0:
                            font_stats[size]['count'] += len(text)
                            if len(font_stats[size]['examples']) < 5:
                                font_stats[size]['examples'].append({
                                    'text': text[:100],
                                    'page': page_num,
                                    'font': font
                                })

    doc.close()

    # Identify heading sizes (larger than body text)
    sizes = sorted(font_stats.keys(), reverse=True)
    if not sizes:
        return {'body_size': 10, 'heading_sizes': []}

    # Most common size is likely body text
    body_size = max(sizes, key=lambda s: font_stats[s]['count'])

    # Sizes larger than body are likely headings
    heading_sizes = [s for s in sizes if s > body_size]

    return {
        'body_size': body_size,
        'heading_sizes': heading_sizes,
        'font_stats': dict(font_stats),
        'all_sizes': sizes
    }


def extract_figures_and_captions(page):
    """Extract figures and their captions from a PDF page"""
    figures = []
    images = page.get_images()
    text_blocks = page.get_text("dict")["blocks"]

    for img_index, img in enumerate(images):
        xref = img[0]
        try:
            bbox = page.get_image_bbox(img)

            # Look for caption near the image (usually below)
            caption = None
            for block in text_blocks:
                if block.get("type") == 0:  # Text block
                    block_bbox = fitz.Rect(block["bbox"])
                    # Check if text is near image (within 50 points below)
                    if block_bbox[1] >= bbox[3] and block_bbox[1] - bbox[3] < 50:
                        text = " ".join([
                            span["text"] for line in block.get("lines", [])
                            for span in line.get("spans", [])
                        ]).strip()

                        # Check if it looks like a caption
                        if re.match(r'^(Figure|Fig\.?|Table)\s+\d+', text, re.IGNORECASE):
                            caption = text
                            break

            figures.append({
                'type': 'figure',
                'bbox': bbox,
                'caption': caption,
                'xref': xref
            })
        except:
            continue

    return figures


def normalize_line_breaks(text):
    """Fix line-break issues in extracted text"""
    # Fix hyphenated line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Fix broken words across lines (no hyphen)
    # Look for lowercase letter at end of line followed by lowercase at start of next
    text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1\2', text)

    # Preserve paragraph breaks (double newlines)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Single newlines within paragraphs become spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)

    return text


def extract_enhanced_pdf_structure(pdf_path):
    """
    Enhanced PDF extraction with proper heading detection, TOC usage,
    figure/table separation, and structural metadata.

    Fixes:
    1. Uses TOC for heading extraction
    2. Detects Roman numeral sections
    3. Uses font-size analysis for headings
    4. Filters false heading positives
    5. Preserves section hierarchy
    6. Promotes Abstract/Index Terms to sections
    7. Separates tables with clear boundaries
    8. Extracts figures and captions separately
    9. Includes structural metadata in chunks
    10. Maps pages to sections
    11. Normalizes line breaks properly
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    # Step 1: Extract TOC structure
    toc_structure = extract_toc_structure(pdf_path)
    toc_by_page = defaultdict(list)
    for entry in toc_structure:
        toc_by_page[entry['page']].append(entry)

    # Step 2: Analyze font sizes to identify headings
    font_analysis = analyze_font_sizes(pdf_path)
    heading_sizes = set(font_analysis['heading_sizes'])

    # Step 3: Extract content with structure
    content_blocks = []
    page_to_section = {}  # Map pages to their current section
    current_section = None
    current_section_hierarchy = []

    for page_num, page in enumerate(doc, 1):
        # Check if this page starts a new section from TOC
        if page_num in toc_by_page:
            for toc_entry in toc_by_page[page_num]:
                current_section = toc_entry['title']
                current_section_hierarchy = [toc_entry]

        page_to_section[page_num] = {
            'section': current_section,
            'hierarchy': list(current_section_hierarchy)
        }

        # Extract text with font information
        blocks = page.get_text("dict")["blocks"]

        # Extract figures and captions
        figures = extract_figures_and_captions(page)
        for fig in figures:
            content_blocks.append({
                'type': 'figure',
                'content': fig['caption'] or f"Figure on page {page_num}",
                'page': page_num,
                'y_position': fig['bbox'][1],
                'metadata': {
                    'pdf_file': pdf_path.name,
                    'page': page_num,
                    'block_type': 'figure',
                    'bbox': list(fig['bbox']),
                    'section': current_section,
                    'section_hierarchy': list(current_section_hierarchy)
                }
            })

        # Process text blocks
        for block in blocks:
            if block.get("type") != 0:  # Skip non-text blocks
                continue

            # Extract full block text
            block_text = ""
            block_size = None
            block_font = None

            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    if block_size is None:
                        block_size = round(span.get("size", 0), 1)
                        block_font = span.get("font", "")
                block_text += line_text + "\n"

            block_text = block_text.strip()
            if not block_text:
                continue

            # Normalize line breaks
            block_text = normalize_line_breaks(block_text)

            # Determine block type
            block_type = 'text'
            heading_level = None

            # Check if it's a heading based on multiple criteria
            is_heading = False
            heading_source = None

            # Criterion 1: Font size (larger than body)
            if block_size and block_size in heading_sizes:
                is_heading = True
                heading_source = 'font_size'
                # Map size to heading level (larger = higher level)
                size_rank = sorted(heading_sizes, reverse=True).index(block_size)
                heading_level = min(size_rank + 1, 6)

            # Criterion 2: Pattern matching (Roman numerals, special sections, etc.)
            is_pattern, pattern_type = detect_section_patterns(block_text)
            if is_pattern:
                is_heading = True
                heading_source = pattern_type
                # Abstract, Introduction, etc. are typically H1
                if pattern_type == 'special_section':
                    heading_level = 1
                    # Update current section for special sections
                    if any(keyword in block_text.upper() for keyword in ['ABSTRACT', 'INTRODUCTION', 'CONCLUSION', 'REFERENCES']):
                        current_section = block_text
                        current_section_hierarchy = [{'level': 1, 'title': block_text, 'page': page_num}]
                elif pattern_type == 'roman_section':
                    heading_level = 2
                elif pattern_type == 'numbered_section':
                    # Count dots to determine level (1.2.3 = level 3)
                    dots = block_text.split()[0].count('.')
                    heading_level = min(dots + 1, 6)
                else:
                    heading_level = heading_level or 2

            # Criterion 3: Check against TOC
            for toc_entry in toc_structure:
                if toc_entry['page'] == page_num and toc_entry['title'].lower() in block_text.lower()[:100]:
                    is_heading = True
                    heading_source = 'toc'
                    heading_level = toc_entry['level']
                    current_section = block_text
                    # Update hierarchy
                    current_section_hierarchy = [toc_entry]
                    break

            # Filter false positives
            # Skip if too long (likely not a heading)
            if is_heading and len(block_text) > 200:
                is_heading = False

            # Skip if it's just emphasis/bold inline text
            if is_heading and len(block_text.split()) > 15:
                is_heading = False

            if is_heading:
                block_type = 'heading'

            content_blocks.append({
                'type': block_type,
                'content': block_text,
                'page': page_num,
                'y_position': block["bbox"][1],
                'metadata': {
                    'pdf_file': pdf_path.name,
                    'page': page_num,
                    'block_type': block_type,
                    'font_size': block_size,
                    'font_name': block_font,
                    'heading_level': heading_level,
                    'heading_source': heading_source,
                    'section': current_section,
                    'section_hierarchy': list(current_section_hierarchy),
                    'bbox': block["bbox"]
                }
            })

    # Step 4: Extract tables with clear boundaries
    with pdfplumber.open(pdf_path) as pdf_plumber:
        table_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
        }

        for page_num, page in enumerate(pdf_plumber.pages, 1):
            tables = page.find_tables(table_settings=table_settings)
            if not tables:
                tables = page.find_tables()

            for table_num, table in enumerate(tables, 1):
                raw_data = table.extract()
                if not raw_data or len(raw_data) < 2:
                    continue

                try:
                    df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
                except ValueError:
                    df = pd.DataFrame(raw_data)

                # Create table with clear boundaries
                table_md = "<!-- TABLE START -->\n\n"
                table_md += df.to_markdown(index=False)
                table_md += "\n\n<!-- TABLE END -->"

                bbox = table.bbox
                content_blocks.append({
                    'type': 'table',
                    'content': table_md,
                    'page': page_num,
                    'y_position': bbox[1],
                    'metadata': {
                        'pdf_file': pdf_path.name,
                        'page': page_num,
                        'table_num': table_num,
                        'block_type': 'table',
                        'bbox': list(bbox),
                        'section': page_to_section[page_num]['section'],
                        'section_hierarchy': page_to_section[page_num]['hierarchy']
                    }
                })

    doc.close()

    # Sort by page and position
    content_blocks.sort(key=lambda x: (x['page'], x['y_position']))

    # Assign position indices
    for i, block in enumerate(content_blocks):
        block['position'] = i

    return {
        'content_blocks': content_blocks,
        'toc_structure': toc_structure,
        'page_to_section': page_to_section,
        'font_analysis': font_analysis
    }


def convert_blocks_to_markdown(content_blocks):
    """
    Convert enhanced content blocks to properly formatted Markdown.

    Args:
        content_blocks: List of content blocks from extract_enhanced_pdf_structure()

    Returns:
        Formatted markdown string
    """
    markdown_lines = []
    last_type = None

    for block in content_blocks:
        block_type = block['type']
        content = block['content']
        metadata = block.get('metadata', {})

        # Add spacing between different block types
        if last_type and last_type != block_type:
            markdown_lines.append("")

        if block_type == 'heading':
            # Format as markdown heading
            heading_level = metadata.get('heading_level', 2)
            prefix = '#' * heading_level
            markdown_lines.append(f"{prefix} {content}")
            markdown_lines.append("")  # Space after heading

        elif block_type == 'table':
            # Tables already have markers
            markdown_lines.append(content)
            markdown_lines.append("")

        elif block_type == 'figure':
            # Format figure with caption
            markdown_lines.append(f"**{content}**")
            markdown_lines.append("")

        else:
            # Regular text
            markdown_lines.append(content)

        last_type = block_type

    return '\n'.join(markdown_lines)


def convert_pdf_to_markdown_enhanced(pdf_path, output_path=None, include_metadata=False):
    """
    Convert PDF to Markdown using enhanced structure extraction.

    This function uses the improved extraction that:
    - Properly detects headings via TOC, font analysis, and patterns
    - Detects Roman numeral sections
    - Identifies Abstract, Introduction, and special sections
    - Separates tables and figures with clear boundaries
    - Normalizes line breaks correctly

    Args:
        pdf_path: Path to the input PDF file
        output_path: Path for the output markdown file (optional)
        include_metadata: If True, also save metadata JSON

    Returns:
        markdown_text: The converted markdown text
    """
    pdf_path = Path(pdf_path)

    # Validate input file
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")

    print(f"Converting PDF to Markdown (Enhanced): {pdf_path.name}")
    print(f"{'='*70}")

    try:
        # Extract with enhanced structure
        print("Extracting PDF structure...")
        result = extract_enhanced_pdf_structure(pdf_path)

        content_blocks = result['content_blocks']
        toc_structure = result['toc_structure']
        page_to_section = result['page_to_section']

        # Convert to markdown
        print("Converting to Markdown...")
        markdown_text = convert_blocks_to_markdown(content_blocks)

        # Determine output path
        if output_path is None:
            output_path = pdf_path.with_suffix('.md')
        else:
            output_path = Path(output_path)

        # Save markdown file
        print(f"Saving markdown to: {output_path.name}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        # Save metadata if requested
        if include_metadata:
            metadata_path = output_path.with_suffix('.metadata.json')
            metadata = {
                'toc_structure': toc_structure,
                'page_to_section': page_to_section,
                'total_blocks': len(content_blocks),
                'block_types': dict(Counter([b['type'] for b in content_blocks]))
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Metadata saved to: {metadata_path.name}")

        # Calculate statistics
        lines = len(markdown_text.split('\n'))
        words = len(markdown_text.split())
        chars = len(markdown_text)

        print(f"{'='*70}")
        print("Conversion Statistics:")
        print(f"  Content Blocks: {len(content_blocks):,}")
        print(f"  Headings: {len([b for b in content_blocks if b['type'] == 'heading']):,}")
        print(f"  Tables: {len([b for b in content_blocks if b['type'] == 'table']):,}")
        print(f"  Figures: {len([b for b in content_blocks if b['type'] == 'figure']):,}")
        print(f"  Lines: {lines:,}")
        print(f"  Words: {words:,}")
        print(f"  Characters: {chars:,}")
        print(f"  TOC Entries: {len(toc_structure):,}")
        print(f"{'='*70}")
        print(f"SUCCESS: Markdown saved to {output_path}")

        return markdown_text

    except Exception as e:
        print(f"ERROR: Failed to convert PDF")
        print(f"  {str(e)}")
        raise


def convert_pdf_with_marker(
    pdf_path,
    output_path=None,
    vram_gb=8,
    max_pages=None
):
    """
    Marker-PDF conversion using the EXACT Streamlit implementation.
    Cached, GPU-safe, CLI-ready.
    """
    if not MARKER_AVAILABLE:
        raise ImportError(
            "marker-pdf is not installed. Install with: pip install marker-pdf torch pillow"
        )

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF")

    print(f"Converting with marker-pdf (Streamlit-compatible): {pdf_path.name}")
    print("=" * 70)

    try:
        # Load cached marker converter (CRITICAL)
        converter = load_marker_converter(
            vram_gb=vram_gb,
            extract_images=False,
            output_format="markdown"
        )

        # Optional page limit
        if max_pages is not None:
            converter.config["max_pages"] = max_pages

        # Marker requires a REAL file path
        rendered = converter(str(pdf_path))

        markdown_text, _, metadata = text_from_rendered(rendered)

        # Output path
        if output_path is None:
            output_path = pdf_path.with_suffix(".md")
        else:
            output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        # Stats
        print("=" * 70)
        print("Marker-PDF Conversion Statistics:")
        print(f"  Pages: {max_pages or 'ALL'}")
        print(f"  Characters: {len(markdown_text):,}")
        print(f"  Words: {len(markdown_text.split()):,}")
        if metadata:
            print(f"  Metadata items: {len(metadata)}")
        print("=" * 70)
        print(f"SUCCESS: Markdown saved to {output_path}")

        return markdown_text

    except Exception:
        print("ERROR: marker-pdf conversion failed")
        import traceback
        traceback.print_exc()
        raise



def convert_pdf_hybrid(pdf_path, output_path=None, prefer_marker=True, fallback=True):
    """
    Hybrid PDF conversion using both marker-pdf and pymupdf4llm.

    This function combines the strengths of both libraries:
    - Marker-pdf: Better for complex layouts, multi-column, equations
    - pymupdf4llm: Faster, good for simple documents, more reliable fallback

    Args:
        pdf_path: Path to the input PDF file
        output_path: Path for the output markdown file (optional)
        prefer_marker: If True, try marker-pdf first (default: True)
        fallback: If True, fall back to pymupdf4llm if marker fails (default: True)

    Returns:
        markdown_text: The converted markdown text
    """
    pdf_path = Path(pdf_path)

    # Validate input file
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")

    print(f"Converting PDF to Markdown (Hybrid Mode): {pdf_path.name}")
    print(f"{'='*70}")

    primary_method = "marker" if prefer_marker else "pymupdf4llm"
    fallback_method = "pymupdf4llm" if prefer_marker else "marker"

    markdown_text = None
    method_used = None

    # Try primary method
    try:
        if primary_method == "marker":
            if not MARKER_AVAILABLE:
                print(f"WARNING: marker-pdf not available, using {fallback_method}")
                primary_method = fallback_method
            else:
                print(f"Attempting conversion with marker-pdf...")
                markdown_text = convert_pdf_with_marker(pdf_path, output_path)
                method_used = "marker-pdf"
        else:
            print(f"Attempting conversion with pymupdf4llm...")
            markdown_text = convert_pdf_to_markdown(pdf_path, output_path, page_chunks=False)
            method_used = "pymupdf4llm"

    except Exception as e:
        print(f"WARNING: {primary_method} conversion failed: {str(e)}")

        if fallback:
            print(f"Falling back to {fallback_method}...")
            try:
                if fallback_method == "marker" and MARKER_AVAILABLE:
                    markdown_text = convert_pdf_with_marker(pdf_path, output_path)
                    method_used = "marker-pdf (fallback)"
                else:
                    markdown_text = convert_pdf_to_markdown(pdf_path, output_path, page_chunks=False)
                    method_used = "pymupdf4llm (fallback)"
            except Exception as fallback_error:
                print(f"ERROR: Both methods failed")
                print(f"  Primary error: {str(e)}")
                print(f"  Fallback error: {str(fallback_error)}")
                raise
        else:
            raise

    if markdown_text:
        print(f"{'='*70}")
        print(f"SUCCESS: Conversion completed using {method_used}")
        print(f"{'='*70}")

    return markdown_text


def convert_pdf_to_markdown(pdf_path, output_path=None, page_chunks=False):
    """
    Convert PDF to Markdown using pymupdf4llm

    Args:
        pdf_path: Path to the input PDF file
        output_path: Path for the output markdown file (optional)
        page_chunks: If True, extract with page-level information

    Returns:
        markdown_text: The converted markdown text
    """
    pdf_path = Path(pdf_path)

    # Validate input file
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")

    print(f"Converting PDF to Markdown: {pdf_path.name}")
    print(f"{'='*70}")

    try:
        # Convert PDF to markdown
        if page_chunks:
            print("Extracting with page-level chunks...")
            page_data = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

            # Combine page chunks into single markdown
            markdown_text = ""
            for i, page in enumerate(page_data, 1):
                markdown_text += f"\n<!-- Page {i} -->\n\n"
                markdown_text += page['text']
                markdown_text += "\n\n"

            print(f"Extracted {len(page_data)} pages")
        else:
            print("Extracting as single document...")
            markdown_text = pymupdf4llm.to_markdown(str(pdf_path))

        # Determine output path
        if output_path is None:
            output_path = pdf_path.with_suffix('.md')
        else:
            output_path = Path(output_path)

        # Save markdown file
        print(f"Saving markdown to: {output_path.name}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        # Calculate statistics
        lines = len(markdown_text.split('\n'))
        words = len(markdown_text.split())
        chars = len(markdown_text)

        print(f"{'='*70}")
        print("Conversion Statistics:")
        print(f"  Lines: {lines:,}")
        print(f"  Words: {words:,}")
        print(f"  Characters: {chars:,}")
        print(f"{'='*70}")
        print(f"SUCCESS: Markdown saved to {output_path}")

        return markdown_text

    except Exception as e:
        print(f"ERROR: Failed to convert PDF")
        print(f"  {str(e)}")
        raise


def convert_pdf_to_text(pdf_path, output_path=None, page_separators=True):
    """
    Convert PDF to plain text using PyMuPDF

    Args:
        pdf_path: Path to the input PDF file
        output_path: Path for the output text file (optional)
        page_separators: If True, add page separators between pages

    Returns:
        text: The extracted plain text
    """
    pdf_path = Path(pdf_path)

    # Validate input file
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")

    print(f"Converting PDF to Text: {pdf_path.name}")
    print(f"{'='*70}")

    try:
        # Open PDF
        doc = fitz.open(str(pdf_path))
        text = ""

        print(f"Extracting text from {len(doc)} pages...")

        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()

            if page_separators:
                # Add page separator
                text += f"\n{'='*70}\n"
                text += f"PAGE {page_num + 1}\n"
                text += f"{'='*70}\n\n"

            text += page_text

            if not page_separators:
                text += "\n\n"

        doc.close()

        # Determine output path
        if output_path is None:
            output_path = pdf_path.with_suffix('.txt')
        else:
            output_path = Path(output_path)

        # Save text file
        print(f"Saving text to: {output_path.name}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # Calculate statistics
        lines = len(text.split('\n'))
        words = len(text.split())
        chars = len(text)

        print(f"{'='*70}")
        print("Conversion Statistics:")
        print(f"  Lines: {lines:,}")
        print(f"  Words: {words:,}")
        print(f"  Characters: {chars:,}")
        print(f"{'='*70}")
        print(f"SUCCESS: Text saved to {output_path}")

        return text

    except Exception as e:
        print(f"ERROR: Failed to convert PDF to text")
        print(f"  {str(e)}")
        raise


def extract_pdf_tables_as_markdown(pdf_file):
    """
    Extract tables from a PDF file with intelligent context association.
    Returns a list of dictionaries containing table data, context, and metadata.

    Args:
        pdf_file: Path to PDF file or file-like object

    Returns:
        List of dicts with keys: 'markdown', 'context_before', 'context_after', 
        'page', 'table_num', 'smart_name', 'full_content'
    """
    extracted_tables = []

    # Table detection settings for borderless tables
    table_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
    }

    # Handle file path or file object
    if isinstance(pdf_file, (str, Path)):
        pdf_path = pdf_file
        filename = Path(pdf_file).name
    else:
        # File-like object
        pdf_file.seek(0)
        pdf_path = pdf_file
        filename = getattr(pdf_file, 'name', 'document.pdf')

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                
                # --- GEOMETRIC CROPPING ---
                # Crop top 50px (header) and bottom 50px (footer) to reduce noise
                width = page.width
                height = page.height
                
                # Safety check: Only crop if page is tall enough
                if height > 150:
                    cropped_page = page.crop((0, 50, width, height - 50))
                    crop_offset = 50
                else:
                    cropped_page = page
                    crop_offset = 0

                # Find tables on the CLEAN cropped page
                tables = cropped_page.find_tables(table_settings=table_settings)

                # Fallback if no tables detected with custom settings
                if not tables:
                    tables = cropped_page.find_tables()

                # Extract text lines from ORIGINAL page for context
                text_objects = page.extract_text_lines()

                for table_num, table in enumerate(tables, 1):
                    raw_data = table.extract()

                    # Skip invalid tables
                    if not raw_data or len(raw_data) < 2:
                        continue

                    # Convert to DataFrame and Markdown
                    try:
                        # First row as header
                        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
                    except ValueError:
                        # Fallback if column mismatch
                        df = pd.DataFrame(raw_data)

                    markdown_table = df.to_markdown(index=False)

                    # --- CONTEXT EXTRACTION ---
                    # Map table coordinates back to original page
                    t_bbox = table.bbox
                    abs_top = t_bbox[1] + crop_offset  # Add crop offset
                    abs_bottom = t_bbox[3] + crop_offset

                    context_before = []
                    context_after = []
                    smart_name = f"Table {table_num} on Page {page_num}"
                    
                    # Hunt for context within 150px of table
                    for line in text_objects:
                        # Text above the table
                        if line['bottom'] < abs_top and (abs_top - line['bottom']) < 150:
                            context_before.append(line['text'])
                        
                        # Text below the table
                        if line['top'] > abs_bottom and (line['top'] - abs_bottom) < 150:
                            context_after.append(line['text'])
                    
                    # --- SMART NAMING ---
                    # Use the last line before the table as name if it's short
                    if context_before:
                        last_line = context_before[-1].strip()
                        if len(last_line) < 60 and len(last_line) > 3:
                            smart_name = last_line
                        else:
                            snippet = last_line[:50].replace("\n", " ") + "..."
                            smart_name = f"Table under: '{snippet}'"
                    
                    # Get last 3 lines before and first 3 lines after
                    full_context_before = "\n".join(context_before[-3:]) if context_before else ""
                    full_context_after = "\n".join(context_after[:3]) if context_after else ""

                    # --- BUILD RICH CONTENT ---
                    full_content = f"""SOURCE FILE: {filename}
PAGE: {page_num}
TABLE REFERENCE: {smart_name}

--- CONTEXT BEFORE ---
{full_context_before}

--- TABLE DATA ---
{markdown_table}

--- CONTEXT AFTER ---
{full_context_after}
"""

                    # Store everything
                    table_data = {
                        'markdown': markdown_table,
                        'context_before': full_context_before,
                        'context_after': full_context_after,
                        'page': page_num,
                        'table_num': table_num,
                        'smart_name': smart_name,
                        'full_content': full_content,
                        'filename': filename,
                        'bbox': {
                            'top': abs_top,
                            'bottom': abs_bottom,
                            'left': t_bbox[0],
                            'right': t_bbox[2]
                        }
                    }
                    
                    extracted_tables.append(table_data)

    except Exception as e:
        print(f"Warning: Error extracting tables from PDF: {str(e)}")
        return []

    return extracted_tables


def extract_pdf_with_structure(pdf_path):
    """
    Extract PDF content preserving the order of text and tables.
    Returns a unified list of content blocks with metadata for RAG.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of content blocks, each with:
        {
            'type': 'text'|'table'|'heading',
            'content': str,
            'page': int,
            'position': int (order on page),
            'metadata': dict
        }
    """
    content_blocks = []
    pdf_path = Path(pdf_path)

    # Table detection settings
    table_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
    }

    try:
        # Open with both pdfplumber for tables and pymupdf for text
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Get tables with their bounding boxes
                tables = page.find_tables(table_settings=table_settings)
                if not tables:
                    tables = page.find_tables()

                # Extract table regions
                table_regions = []
                table_data_map = {}
                for table_num, table in enumerate(tables, 1):
                    bbox = table.bbox  # (x0, y0, x1, y1)
                    raw_data = table.extract()

                    if raw_data and len(raw_data) >= 2:
                        # Convert to markdown
                        try:
                            df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
                        except ValueError:
                            df = pd.DataFrame(raw_data)

                        table_md = df.to_markdown(index=False)
                        table_regions.append(bbox)
                        table_data_map[bbox] = {
                            'markdown': table_md,
                            'table_num': table_num,
                            'raw_data': raw_data
                        }

                # Extract text with layout preservation
                text = page.extract_text()
                if not text:
                    continue

                # Split text into blocks (paragraphs)
                text_blocks = [b.strip() for b in text.split('\n\n') if b.strip()]

                # Get text lines with positions for ordering
                text_lines = page.extract_text_lines()

                # Create content blocks in order
                position = 0

                # Process text blocks
                for block in text_blocks:
                    if not block:
                        continue

                    # Find the Y position of this block
                    block_lines = [l for l in text_lines if l['text'] in block]
                    y_pos = block_lines[0]['top'] if block_lines else 0

                    # Determine block type
                    block_type = 'text'
                    if block.startswith('#'):
                        block_type = 'heading'
                    elif len(block.split()) < 10 and block.isupper():
                        block_type = 'heading'

                    content_blocks.append({
                        'type': block_type,
                        'content': block,
                        'page': page_num,
                        'position': position,
                        'y_position': y_pos,
                        'metadata': {
                            'pdf_file': pdf_path.name,
                            'page': page_num,
                            'block_type': block_type
                        }
                    })
                    position += 1

                # Add tables to content blocks
                for bbox, table_info in table_data_map.items():
                    y_pos = bbox[1]  # Top of table

                    content_blocks.append({
                        'type': 'table',
                        'content': table_info['markdown'],
                        'page': page_num,
                        'position': position,
                        'y_position': y_pos,
                        'metadata': {
                            'pdf_file': pdf_path.name,
                            'page': page_num,
                            'table_num': table_info['table_num'],
                            'block_type': 'table',
                            'bbox': bbox
                        }
                    })
                    position += 1

        # Sort all blocks by page and y_position to preserve document order
        content_blocks.sort(key=lambda x: (x['page'], x['y_position']))

        # Re-assign position numbers after sorting
        for i, block in enumerate(content_blocks):
            block['position'] = i

    except Exception as e:
        print(f"Error extracting PDF with structure: {str(e)}")
        return []

    return content_blocks


def extract_headings(markdown_text):
    """
    Extract all headings from markdown text with hierarchy

    Returns:
        List of heading dictionaries with level, text, and line number
    """
    headings = []
    lines = markdown_text.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Match markdown headings (# H1, ## H2, etc.)
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({
                'level': level,
                'text': text,
                'line_number': line_num,
                'type': f'H{level}'
            })

    return headings


def build_heading_hierarchy(headings):
    """
    Build a hierarchical structure of headings

    Returns:
        Nested dictionary representing document structure
    """
    hierarchy = []
    stack = []

    for heading in headings:
        level = heading['level']

        # Pop stack until we find parent level
        while stack and stack[-1]['level'] >= level:
            stack.pop()

        # Create heading node
        node = {
            'text': heading['text'],
            'level': heading['level'],
            'line': heading['line_number'],
            'children': []
        }

        # Add to parent or root
        if stack:
            stack[-1]['children'].append(node)
        else:
            hierarchy.append(node)

        stack.append(node)

    return hierarchy


def extract_pdf_metadata(pdf_path):
    """
    Extract comprehensive metadata from PDF using PyMuPDF

    Returns:
        Dictionary with PDF metadata
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    # Get PDF version (handle API differences)
    try:
        pdf_version = doc.pdf_version() if callable(getattr(doc, 'pdf_version', None)) else getattr(doc, 'pdf_version', 'Unknown')
    except:
        # For older/newer versions that don't have pdf_version
        try:
            pdf_version = f"1.{doc.pdf_catalog().get('/Version', 'Unknown')}"
        except:
            pdf_version = "Unknown"

    metadata = {
        'filename': pdf_path.name,
        'file_size_bytes': pdf_path.stat().st_size,
        'file_size_mb': round(pdf_path.stat().st_size / (1024 * 1024), 2),
        'page_count': len(doc),
        'pdf_version': pdf_version,
        'is_encrypted': doc.is_encrypted,
        'is_pdf': doc.is_pdf,
        'pdf_metadata': doc.metadata,
        'toc': doc.get_toc(),
        'pages': []
    }

    # Extract per-page metadata
    for page_num, page in enumerate(doc, 1):
        try:
            page_info = {
                'page_number': page_num,
                'width': round(page.rect.width, 2),
                'height': round(page.rect.height, 2),
                'rotation': page.rotation,
                'image_count': len(page.get_images()),
                'link_count': len(page.get_links()),
                'text_length': len(page.get_text())
            }
        except Exception as e:
            # Fallback for pages that can't be fully analyzed
            page_info = {
                'page_number': page_num,
                'width': 0,
                'height': 0,
                'rotation': 0,
                'image_count': 0,
                'link_count': 0,
                'text_length': 0,
                'error': str(e)
            }
        metadata['pages'].append(page_info)

    # Calculate statistics (with safe division)
    total_images = sum(p.get('image_count', 0) for p in metadata['pages'])
    total_links = sum(p.get('link_count', 0) for p in metadata['pages'])
    total_text = sum(p.get('text_length', 0) for p in metadata['pages'])
    page_count = len(metadata['pages'])

    metadata['statistics'] = {
        'total_images': total_images,
        'total_links': total_links,
        'total_text_length': total_text,
        'avg_images_per_page': round(total_images / page_count, 2) if page_count > 0 else 0,
        'avg_text_per_page': round(total_text / page_count, 2) if page_count > 0 else 0
    }

    doc.close()
    return metadata


def extract_document_structure(markdown_text):
    """
    Analyze markdown structure and extract patterns

    Returns:
        Dictionary with structure analysis
    """
    lines = markdown_text.split('\n')

    structure = {
        'total_lines': len(lines),
        'non_empty_lines': len([l for l in lines if l.strip()]),
        'paragraphs': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
        'code_blocks': len(re.findall(r'```', markdown_text)) // 2,
        'lists': {
            'unordered': len(re.findall(r'^\s*[-*+]\s+', markdown_text, re.MULTILINE)),
            'ordered': len(re.findall(r'^\s*\d+\.\s+', markdown_text, re.MULTILINE))
        },
        'links': len(re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', markdown_text)),
        'images': len(re.findall(r'!\[([^\]]*)\]\(([^\)]+)\)', markdown_text)),
        'bold_text': len(re.findall(r'\*\*([^*]+)\*\*', markdown_text)),
        'italic_text': len(re.findall(r'\*([^*]+)\*', markdown_text)),
        'tables': len(re.findall(r'\|[^\n]+\|', markdown_text)),
        'blockquotes': len(re.findall(r'^>\s+', markdown_text, re.MULTILINE))
    }

    # Word statistics
    words = markdown_text.split()
    structure['word_count'] = len(words)
    structure['char_count'] = len(markdown_text)
    structure['avg_word_length'] = round(sum(len(w) for w in words) / len(words), 2) if words else 0

    return structure


def analyze_content_patterns(markdown_text):
    """
    Detect common patterns and content types in the document

    Returns:
        Dictionary with pattern analysis
    """
    patterns = {
        'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', markdown_text),
        'urls': re.findall(r'https?://[^\s\)]+', markdown_text),
        'phone_numbers': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', markdown_text),
        'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', markdown_text),
        'numbers': re.findall(r'\b\d+\.?\d*\b', markdown_text),
    }

    # Count occurrences
    analysis = {
        'email_count': len(patterns['emails']),
        'url_count': len(patterns['urls']),
        'phone_count': len(patterns['phone_numbers']),
        'date_count': len(patterns['dates']),
        'number_count': len(patterns['numbers'])
    }

    # Most common words (excluding common stop words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'being'}
    words = [w.lower().strip('.,!?;:()[]{}') for w in markdown_text.split()]
    words = [w for w in words if w and w not in stop_words and len(w) > 3]
    word_freq = Counter(words)
    analysis['top_words'] = word_freq.most_common(10)

    return analysis


def generate_metadata_report(pdf_path, markdown_text, output_path=None):
    """
    Generate comprehensive metadata report for PDF

    Args:
        pdf_path: Path to PDF file
        markdown_text: Converted markdown text
        output_path: Optional path for metadata JSON file

    Returns:
        Complete metadata dictionary
    """
    print(f"\n{'='*70}")
    print("EXTRACTING METADATA AND DOCUMENT STRUCTURE")
    print(f"{'='*70}\n")

    # Extract all metadata
    print("Extracting PDF metadata...")
    pdf_metadata = extract_pdf_metadata(pdf_path)

    print("Analyzing headings...")
    headings = extract_headings(markdown_text)
    heading_hierarchy = build_heading_hierarchy(headings)

    print("Analyzing document structure...")
    structure = extract_document_structure(markdown_text)

    print("Detecting content patterns...")
    patterns = analyze_content_patterns(markdown_text)

    # Compile complete report
    report = {
        'extraction_date': datetime.now().isoformat(),
        'source_file': str(pdf_path),
        'pdf_metadata': pdf_metadata,
        'headings': {
            'count': len(headings),
            'by_level': dict(Counter(h['level'] for h in headings)),
            'list': headings,
            'hierarchy': heading_hierarchy
        },
        'structure': structure,
        'content_patterns': patterns
    }

    # Save to JSON if output path provided
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nMetadata saved to: {output_path}")

    return report


def print_metadata_summary(report):
    """
    Print a human-readable summary of the metadata report
    """
    print(f"\n{'='*70}")
    print("METADATA SUMMARY")
    print(f"{'='*70}\n")

    # PDF Info
    pdf = report['pdf_metadata']
    print(f"PDF Information:")
    print(f"  File: {pdf['filename']}")
    print(f"  Size: {pdf['file_size_mb']} MB")
    print(f"  Pages: {pdf['page_count']}")
    print(f"  Version: {pdf['pdf_version']}")
    print(f"  Encrypted: {pdf['is_encrypted']}")

    # PDF Document Metadata
    if pdf['pdf_metadata']:
        print(f"\nDocument Properties:")
        meta = pdf['pdf_metadata']
        if meta.get('title'):
            print(f"  Title: {meta['title']}")
        if meta.get('author'):
            print(f"  Author: {meta['author']}")
        if meta.get('subject'):
            print(f"  Subject: {meta['subject']}")
        if meta.get('creator'):
            print(f"  Creator: {meta['creator']}")

    # Table of Contents
    if pdf['toc']:
        print(f"\nTable of Contents: {len(pdf['toc'])} entries")

    # Statistics
    stats = pdf['statistics']
    print(f"\nPDF Statistics:")
    print(f"  Total Images: {stats['total_images']}")
    print(f"  Total Links: {stats['total_links']}")
    print(f"  Avg Images/Page: {stats['avg_images_per_page']}")

    # Headings
    headings = report['headings']
    print(f"\nHeadings: {headings['count']} total")
    for level, count in sorted(headings['by_level'].items()):
        print(f"  H{level}: {count}")

    # Structure
    struct = report['structure']
    print(f"\nDocument Structure:")
    print(f"  Lines: {struct['total_lines']:,}")
    print(f"  Words: {struct['word_count']:,}")
    print(f"  Characters: {struct['char_count']:,}")
    print(f"  Paragraphs: {struct['paragraphs']}")
    print(f"  Code Blocks: {struct['code_blocks']}")
    print(f"  Lists: {struct['lists']['unordered']} unordered, {struct['lists']['ordered']} ordered")
    print(f"  Links: {struct['links']}")
    print(f"  Images: {struct['images']}")
    print(f"  Tables: {struct['tables']}")

    # Content Patterns
    patterns = report['content_patterns']
    print(f"\nContent Patterns Detected:")
    print(f"  Emails: {patterns['email_count']}")
    print(f"  URLs: {patterns['url_count']}")
    print(f"  Phone Numbers: {patterns['phone_count']}")
    print(f"  Dates: {patterns['date_count']}")

    if patterns['top_words']:
        print(f"\nTop 10 Words:")
        for word, count in patterns['top_words']:
            print(f"  {word}: {count}")

    # Heading Preview
    if headings['list']:
        print(f"\nFirst 5 Headings:")
        for i, h in enumerate(headings['list'][:5], 1):
            indent = "  " * (h['level'] - 1)
            print(f"  {indent}H{h['level']}: {h['text']}")

    print(f"\n{'='*70}")


def batch_convert_pdfs(pdf_directory, output_directory=None, page_chunks=False, output_format='md'):
    """
    Convert all PDFs in a directory to Markdown or Text

    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory for output files (optional)
        page_chunks: If True, extract with page-level information
        output_format: Output format - 'md' for Markdown or 'txt' for Plain Text
    """
    pdf_dir = Path(pdf_directory)

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise ValueError(f"Invalid directory: {pdf_directory}")

    # Find all PDF files
    pdf_files = list(pdf_dir.glob('*.pdf'))

    if not pdf_files:
        print(f"No PDF files found in: {pdf_directory}")
        return

    format_name = "Markdown" if output_format == 'md' else "Text"
    print(f"\nFound {len(pdf_files)} PDF files")
    print(f"Converting to {format_name} format")
    print(f"{'='*70}\n")

    # Setup output directory
    if output_directory:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = pdf_dir

    # Convert each PDF
    success_count = 0
    failed_files = []

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Converting: {pdf_file.name}")

        try:
            # Determine output file extension
            output_ext = '.md' if output_format == 'md' else '.txt'
            output_file = output_dir / pdf_file.with_suffix(output_ext).name

            # Call appropriate conversion function
            if output_format == 'txt':
                convert_pdf_to_text(pdf_file, output_file, page_chunks)
            else:
                convert_pdf_to_markdown(pdf_file, output_file, page_chunks)

            success_count += 1
        except Exception as e:
            print(f"  FAILED: {str(e)}")
            failed_files.append(pdf_file.name)

    # Summary
    print(f"\n{'='*70}")
    print(f"BATCH CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"  Successful: {success_count}/{len(pdf_files)}")

    if failed_files:
        print(f"  Failed files:")
        for fname in failed_files:
            print(f"    - {fname}")


def main():
    """
    Command-line interface for PDF to Markdown/Text conversion
    """
    parser = argparse.ArgumentParser(
        description='Convert PDF files to Markdown or Plain Text using pymupdf4llm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert single PDF to markdown (fast, default)
  python pdf2llm.py document.pdf

  # Convert using marker-pdf (best quality, ML-based)
  python pdf2llm.py document.pdf --method marker

  # Convert using hybrid mode (tries marker, falls back to pymupdf4llm)
  python pdf2llm.py document.pdf --method hybrid

  # Convert using enhanced extraction (better heading detection, TOC usage)
  python pdf2llm.py document.pdf --method enhanced

  # Convert to plain text
  python pdf2llm.py document.pdf --format txt

  # Convert with custom output name
  python pdf2llm.py document.pdf -o output.md

  # Marker with page limit and custom batch size
  python pdf2llm.py document.pdf --method marker --max-pages 10 --batch-multiplier 4

  # Enhanced extraction with metadata
  python pdf2llm.py document.pdf --method enhanced --metadata

  # Extract only metadata (no conversion)
  python pdf2llm.py document.pdf --metadata-only

  # Batch convert all PDFs in directory
  python pdf2llm.py --batch ./pdfs/ -o ./markdown/

Extraction Methods:
  1. pymupdf4llm (default): Fast, lightweight, good for simple documents
  2. enhanced: Uses TOC, font analysis, better structure detection
  3. marker: ML-based, best for complex layouts, equations, multi-column
  4. hybrid: Tries marker first, falls back to pymupdf4llm (recommended)

Enhanced Mode Features:
  - Uses PDF Table of Contents for accurate heading detection
  - Detects Roman numeral sections (I., II., III.)
  - Identifies special sections (Abstract, Introduction, Conclusion, etc.)
  - Font-size analysis for heading detection
  - Filters false heading positives
  - Preserves section hierarchy
  - Separates tables and figures with clear boundaries
  - Normalizes line breaks correctly
  - Rich structural metadata in chunks

Marker-PDF Features:
  - Advanced ML-based layout detection
  - Multi-column document support
  - Mathematical equation extraction
  - Better table recognition and formatting
  - Improved reading order detection
  - Figure and caption association
  - Handles complex academic papers excellently
        '''
    )

    parser.add_argument(
        'input',
        nargs='?',
        help='Input PDF file or directory (for batch mode)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output file or directory (for batch mode)'
    )

    parser.add_argument(
        '--format',
        choices=['md', 'txt'],
        default='md',
        help='Output format: md (Markdown) or txt (Plain Text). Default: md'
    )

    parser.add_argument(
        '--page-chunks',
        action='store_true',
        help='Extract with page-level markers (md) or separators (txt)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch convert all PDFs in input directory'
    )

    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Extract and save metadata (headings, structure, patterns)'
    )

    parser.add_argument(
        '--metadata-only',
        action='store_true',
        help='Extract only metadata without converting to markdown'
    )

    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Use enhanced extraction with better heading detection, TOC usage, and structure preservation'
    )

    parser.add_argument(
        '--method',
        choices=['pymupdf4llm', 'enhanced', 'marker', 'hybrid'],
        default=None,
        help='Extraction method: pymupdf4llm (default, fast), enhanced (better structure), marker (advanced ML-based), hybrid (tries marker then falls back)'
    )

    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Maximum number of pages to process (marker method only)'
    )



    args = parser.parse_args()

    # Show help if no input provided
    if not args.input:
        parser.print_help()
        sys.exit(1)

    try:
        if args.metadata_only:
            # Extract metadata only (no markdown conversion)
            print("Extracting metadata only...")
            pdf_path = Path(args.input)

            # First convert to get markdown for analysis
            markdown_text = pymupdf4llm.to_markdown(str(pdf_path))

            # Generate metadata report
            metadata_output = args.output or pdf_path.with_suffix('.metadata.json')
            report = generate_metadata_report(pdf_path, markdown_text, metadata_output)
            print_metadata_summary(report)

        elif args.batch:
            # Batch conversion mode
            batch_convert_pdfs(
                args.input,
                args.output,
                args.page_chunks,
                args.format
            )
        else:
            # Single file conversion
            extraction_method = None  # Initialize for metadata check later

            if args.format == 'txt':
                # Convert to plain text
                text = convert_pdf_to_text(
                    args.input,
                    args.output,
                    args.page_chunks
                )
                content_text = text
            else:
                # Determine extraction method
                # Priority: --method flag > --enhanced flag > default
                if args.method:
                    extraction_method = args.method
                elif args.enhanced:
                    extraction_method = 'enhanced'
                else:
                    extraction_method = 'pymupdf4llm'

                # Convert to markdown based on method
                if extraction_method == 'marker':
                    # Use marker-pdf
                    if not MARKER_AVAILABLE:
                        print("ERROR: marker-pdf is not installed.")
                        print("Install it with: pip install marker-pdf torch pillow")
                        sys.exit(1)
                    markdown_text = convert_pdf_with_marker(
                        pdf_path=args.input,
                        output_path=args.output,
                        vram_gb=8,
                        max_pages=args.max_pages
                    )

                elif extraction_method == 'hybrid':
                    # Use hybrid approach
                    markdown_text = convert_pdf_hybrid(
                        args.input,
                        args.output,
                        prefer_marker=True,
                        fallback=True
                    )
                elif extraction_method == 'enhanced':
                    # Use enhanced extraction
                    markdown_text = convert_pdf_to_markdown_enhanced(
                        args.input,
                        args.output,
                        include_metadata=args.metadata
                    )
                else:
                    # Use standard pymupdf4llm extraction
                    markdown_text = convert_pdf_to_markdown(
                        args.input,
                        args.output,
                        args.page_chunks
                    )
                content_text = markdown_text

            # Extract metadata if requested (for non-enhanced/marker mode)
            if args.metadata and extraction_method not in ['enhanced', 'marker']:
                pdf_path = Path(args.input)
                metadata_output = pdf_path.with_suffix('.metadata.json')
                report = generate_metadata_report(pdf_path, content_text, metadata_output)
                print_metadata_summary(report)

    except KeyboardInterrupt:
        print("\n\nConversion cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
