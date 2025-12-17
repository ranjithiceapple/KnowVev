"""
Debug script to trace heading detection through the pipeline.
Run this to see where headings are being lost.
"""

from document_processor import extract_text_from_document
from metadata_aware_normalizer import normalize_with_metadata, NormalizationConfig
from enterprise_chunking_pipeline import chunk_with_normalization, ChunkingConfig
import json


def debug_heading_pipeline(file_path: str):
    """
    Trace headings through the complete pipeline.
    """
    print("=" * 80)
    print("HEADING DETECTION DEBUG")
    print("=" * 80)

    # Step 1: Extract document
    print("\n[STEP 1] Extracting document...")
    extraction_result = extract_text_from_document(file_path, extract_metadata=True)

    print(f"Total pages: {extraction_result.metadata.total_pages}")
    print(f"All headings found: {len(extraction_result.metadata.all_headings)}")
    print(f"Sections: {len(extraction_result.metadata.sections)}")

    if extraction_result.metadata.all_headings:
        print("\nHeadings detected during extraction:")
        for i, heading in enumerate(extraction_result.metadata.all_headings[:10], 1):
            print(f"  {i}. {heading}")
    else:
        print("\n⚠️  NO HEADINGS DETECTED during extraction!")

    # Check first page
    if extraction_result.pages:
        first_page = extraction_result.pages[0]
        print(f"\nFirst page headings: {first_page.headings}")
        print(f"First 500 chars of page text:")
        print(first_page.text[:500])
        print("...")

    # Check extraction result text
    print(f"\nFirst 500 chars of extraction_result.text:")
    print(extraction_result.text[:500])
    print("...")

    # Step 2: Normalize
    print("\n" + "=" * 80)
    print("[STEP 2] Normalizing text...")

    norm_config = NormalizationConfig(
        protect_headings=True,
        preserve_hierarchy=True,
        add_page_markers=True
    )

    normalized_text, page_results, stats = normalize_with_metadata(
        extraction_result,
        norm_config
    )

    print(f"Normalization complete")
    print(f"Pages processed: {len(page_results)}")

    # Check if headings are in normalized text
    print(f"\nFirst 500 chars of normalized text:")
    print(normalized_text[:500])
    print("...")

    # Check for markdown headings
    import re
    markdown_headings = re.findall(r'^#{1,6}\s+.+$', normalized_text, re.MULTILINE)
    print(f"\nMarkdown headings found in normalized text: {len(markdown_headings)}")
    if markdown_headings:
        for i, heading in enumerate(markdown_headings[:10], 1):
            print(f"  {i}. {heading}")
    else:
        print("  ⚠️  NO MARKDOWN HEADINGS in normalized text!")

    # Step 3: Chunk
    print("\n" + "=" * 80)
    print("[STEP 3] Chunking...")

    chunk_config = ChunkingConfig(
        max_chunk_size=1000,
        target_chunk_size=500
    )

    chunks = chunk_with_normalization(
        extraction_result,
        normalized_text,
        chunk_config
    )

    print(f"Total chunks: {len(chunks)}")

    # Check chunks with section info
    chunks_with_sections = [c for c in chunks if c.section_title]
    print(f"Chunks with section titles: {len(chunks_with_sections)}")

    if chunks_with_sections:
        print("\nFirst 5 chunks with sections:")
        for chunk in chunks_with_sections[:5]:
            print(f"  - Section: {chunk.section_title}")
            print(f"    Heading path: {chunk.heading_path}")
            print(f"    Level: {chunk.hierarchy_level}")
    else:
        print("  ⚠️  NO CHUNKS have section information!")

    # Show first few chunks
    print("\nFirst 3 chunks (section info):")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i}:")
        print(f"  Section: {chunk.section_title or '(none)'}")
        print(f"  Heading path: {chunk.heading_path}")
        print(f"  Text preview: {chunk.text[:100]}...")

    # Step 4: Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    issues = []

    if not extraction_result.metadata.all_headings:
        issues.append("❌ No headings detected during extraction - TextPatternDetector patterns may not match this document's heading style")

    if not markdown_headings:
        issues.append("❌ No markdown headings in normalized text - enrichment or hierarchy injection failed")

    if not chunks_with_sections:
        issues.append("❌ No chunks have section info - chunker's BoundaryDetector couldn't find sections")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ Pipeline working correctly!")

    return {
        "headings_extracted": len(extraction_result.metadata.all_headings),
        "markdown_in_normalized": len(markdown_headings),
        "chunks_with_sections": len(chunks_with_sections),
        "total_chunks": len(chunks)
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python debug_heading_detection.py <pdf_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    results = debug_heading_pipeline(file_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(results, indent=2))
