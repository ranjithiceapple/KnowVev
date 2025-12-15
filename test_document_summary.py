"""
Test Document Summary Virtual Chunk

This script demonstrates the document summary feature that creates
a virtual chunk containing a high-level overview of the document.
"""

import sys
import json
from pathlib import Path
from document_to_vector_service import DocumentToVectorService, ServiceConfig
from logger_config import get_logger

logger = get_logger(__name__)


def test_document_summary(file_path: str, collection_name: str = "test_summary"):
    """
    Test document summary generation.

    Args:
        file_path: Path to document
        collection_name: Qdrant collection name
    """
    logger.info("=" * 80)
    logger.info("TESTING DOCUMENT SUMMARY VIRTUAL CHUNK")
    logger.info("=" * 80)
    logger.info(f"Document: {file_path}")
    logger.info(f"Collection: {collection_name}\n")

    # Configure service with summary enabled
    config = ServiceConfig(
        qdrant_url="http://localhost:6333",
        qdrant_collection=collection_name,
        embedding_model_name="all-MiniLM-L6-v2",
        vector_size=384,

        # Chunking settings
        max_chunk_size=1000,
        target_chunk_size=500,
        enable_overlap=True,
        overlap_size=100,

        # Summary settings
        generate_document_summary=True,
        summary_max_length=2000,
        summary_method="hybrid",  # or 'extractive', 'abstractive'

        # Other settings
        deduplicate_chunks=True
    )

    # Create service
    logger.info("Initializing DocumentToVectorService...")
    service = DocumentToVectorService(config)

    # Process document
    logger.info(f"\nProcessing document: {Path(file_path).name}")
    result = service.process_document(file_path)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING RESULTS")
    logger.info("=" * 80)

    if result.success:
        logger.info("✅ SUCCESS!")
        logger.info(f"\nDocument: {result.file_name}")
        logger.info(f"Doc ID: {result.doc_id}")

        logger.info(f"\n📊 Statistics:")
        logger.info(f"  Pages extracted: {result.pages_extracted}")
        logger.info(f"  Chunks created: {result.chunks_created}")
        logger.info(f"  Unique chunks: {result.unique_chunks}")
        logger.info(f"  Vectors stored: {result.vectors_stored}")

        logger.info(f"\n📝 Summary:")
        logger.info(f"  Has summary: {result.has_summary}")
        logger.info(f"  Summary length: {result.summary_length} chars")

        logger.info(f"\n⏱️ Timing:")
        logger.info(f"  Extraction: {result.extraction_time:.2f}s")
        logger.info(f"  Normalization: {result.normalization_time:.2f}s")
        logger.info(f"  Chunking: {result.chunking_time:.2f}s")
        logger.info(f"  Summary: {result.summary_time:.2f}s")
        logger.info(f"  Embedding: {result.embedding_time:.2f}s")
        logger.info(f"  Storage: {result.storage_time:.2f}s")
        logger.info(f"  Total: {result.total_time:.2f}s")

        # Query the summary chunk
        logger.info("\n" + "=" * 80)
        logger.info("QUERYING SUMMARY CHUNK")
        logger.info("=" * 80)

        # Search for the document by name to get the summary
        search_results = service.search(
            query=f"overview of {result.file_name}",
            limit=5
        )

        if search_results:
            # The first result should be the summary (highest similarity)
            summary_result = search_results[0]
            payload = summary_result.get('payload', {})

            logger.info(f"\n📄 Summary Chunk Found:")
            logger.info(f"  Chunk ID: {payload.get('chunk_id', 'N/A')}")
            logger.info(f"  Section: {payload.get('section_title', 'N/A')}")
            logger.info(f"  Pages: {payload.get('page_range', 'N/A')}")
            logger.info(f"  Score: {summary_result.get('score', 0):.3f}")
            logger.info(f"\n  Summary Text:")
            logger.info(f"  {'-' * 76}")

            summary_text = payload.get('text', '')
            # Display first 500 chars
            if len(summary_text) > 500:
                logger.info(f"  {summary_text[:500]}...")
                logger.info(f"  ... ({len(summary_text)} total chars)")
            else:
                logger.info(f"  {summary_text}")
            logger.info(f"  {'-' * 76}")

        # Test semantic search with summary
        logger.info("\n" + "=" * 80)
        logger.info("SEMANTIC SEARCH TEST")
        logger.info("=" * 80)

        test_queries = [
            "what is this document about",
            "give me an overview",
            "document summary"
        ]

        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            results = service.search(query=query, limit=3)

            for i, r in enumerate(results, 1):
                p = r.get('payload', {})
                is_summary = p.get('chunk_id', '').endswith('_SUMMARY')
                chunk_type = "SUMMARY" if is_summary else "REGULAR"

                logger.info(f"  [{i}] [{chunk_type}] Score: {r.get('score', 0):.3f}")
                logger.info(f"      Section: {p.get('section_title', 'N/A')}")
                logger.info(f"      Text: {p.get('text', '')[:100]}...")

    else:
        logger.error("❌ FAILED!")
        logger.error(f"Error: {result.error_message}")
        logger.error(f"Stage: {result.error_stage}")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)

    return result


def compare_with_without_summary(file_path: str):
    """
    Compare processing with and without summary generation.

    Args:
        file_path: Path to document
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: WITH vs WITHOUT SUMMARY")
    logger.info("=" * 80)

    # Test WITHOUT summary
    logger.info("\n[1/2] Processing WITHOUT summary...")
    config_no_summary = ServiceConfig(
        qdrant_collection="test_no_summary",
        generate_document_summary=False
    )
    service_no_summary = DocumentToVectorService(config_no_summary)
    result_no_summary = service_no_summary.process_document(file_path)

    # Test WITH summary
    logger.info("\n[2/2] Processing WITH summary...")
    config_with_summary = ServiceConfig(
        qdrant_collection="test_with_summary",
        generate_document_summary=True,
        summary_method="hybrid"
    )
    service_with_summary = DocumentToVectorService(config_with_summary)
    result_with_summary = service_with_summary.process_document(file_path)

    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)

    logger.info(f"\n{'Metric':<30} {'Without Summary':<20} {'With Summary':<20}")
    logger.info("-" * 70)

    logger.info(f"{'Chunks created':<30} {result_no_summary.chunks_created:<20} {result_with_summary.chunks_created:<20}")
    logger.info(f"{'Unique chunks':<30} {result_no_summary.unique_chunks:<20} {result_with_summary.unique_chunks:<20}")
    logger.info(f"{'Vectors stored':<30} {result_no_summary.vectors_stored:<20} {result_with_summary.vectors_stored:<20}")
    logger.info(f"{'Has summary':<30} {result_no_summary.has_summary!s:<20} {result_with_summary.has_summary!s:<20}")
    logger.info(f"{'Summary length':<30} {result_no_summary.summary_length:<20} {result_with_summary.summary_length:<20}")

    logger.info(f"\n{'Processing time':<30} {'Without Summary':<20} {'With Summary':<20}")
    logger.info("-" * 70)
    logger.info(f"{'Chunking':<30} {result_no_summary.chunking_time:.2f}s{'':<16} {result_with_summary.chunking_time:.2f}s")
    logger.info(f"{'Summary generation':<30} {'-':<20} {result_with_summary.summary_time:.2f}s")
    logger.info(f"{'Embedding':<30} {result_no_summary.embedding_time:.2f}s{'':<16} {result_with_summary.embedding_time:.2f}s")
    logger.info(f"{'Total':<30} {result_no_summary.total_time:.2f}s{'':<16} {result_with_summary.total_time:.2f}s")

    overhead_time = result_with_summary.total_time - result_no_summary.total_time
    overhead_pct = (overhead_time / result_no_summary.total_time * 100) if result_no_summary.total_time > 0 else 0

    logger.info(f"\n⏱️ Summary generation overhead: +{overhead_time:.2f}s ({overhead_pct:.1f}%)")


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python test_document_summary.py <document_path> [mode]")
        print("\nModes:")
        print("  test      - Test summary generation (default)")
        print("  compare   - Compare with and without summary")
        print("\nExamples:")
        print("  python test_document_summary.py ./document.pdf")
        print("  python test_document_summary.py ./document.pdf compare")
        sys.exit(1)

    file_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "test"

    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    if mode == "compare":
        compare_with_without_summary(file_path)
    else:
        test_document_summary(file_path)


if __name__ == "__main__":
    main()
