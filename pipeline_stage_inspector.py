"""
Pipeline Stage Inspector

Saves output of each pipeline stage to separate text files for inspection.

Stages:
1. Raw Extraction (after document_processor)
2. After Heading Enrichment
3. After Normalization
4. After Chunking (before embeddings)
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from document_processor import extract_text_from_document
from metadata_aware_normalizer import normalize_with_metadata, NormalizationConfig
from enterprise_chunking_pipeline import chunk_with_normalization, ChunkingConfig
from logger_config import get_logger

logger = get_logger(__name__)


class PipelineStageInspector:
    """
    Inspect and save outputs from each pipeline stage.
    """

    def __init__(self, output_dir: str = "pipeline_debug"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def inspect_pipeline(self, file_path: str) -> Dict[str, str]:
        """
        Run document through pipeline and save each stage.

        Returns:
            Dictionary of stage names to output file paths
        """
        file_name = Path(file_path).stem
        output_files = {}

        print("=" * 80)
        print(f"PIPELINE STAGE INSPECTOR")
        print(f"Document: {file_path}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)

        # ================================================================
        # STAGE 1: RAW EXTRACTION
        # ================================================================
        print("\n[STAGE 1] RAW EXTRACTION")
        print("-" * 80)

        extraction_result = extract_text_from_document(
            file_path,
            extract_metadata=True
        )

        # Save raw extraction
        stage1_file = self.output_dir / f"{file_name}_stage1_raw_extraction.txt"
        self._save_stage1(stage1_file, extraction_result)
        output_files['stage1_raw_extraction'] = str(stage1_file)
        print(f"✓ Saved: {stage1_file}")

        # Save metadata separately
        stage1_meta = self.output_dir / f"{file_name}_stage1_metadata.json"
        self._save_metadata(stage1_meta, extraction_result.metadata)
        output_files['stage1_metadata'] = str(stage1_meta)
        print(f"✓ Saved: {stage1_meta}")

        # ================================================================
        # STAGE 2: AFTER HEADING ENRICHMENT
        # ================================================================
        print("\n[STAGE 2] AFTER HEADING ENRICHMENT")
        print("-" * 80)
        print("(Enrichment happens during extraction)")

        # The extraction_result.text already has enriched headings
        stage2_file = self.output_dir / f"{file_name}_stage2_enriched_text.txt"
        self._save_stage2(stage2_file, extraction_result)
        output_files['stage2_enriched'] = str(stage2_file)
        print(f"✓ Saved: {stage2_file}")

        # ================================================================
        # STAGE 3: AFTER NORMALIZATION
        # ================================================================
        print("\n[STAGE 3] AFTER NORMALIZATION")
        print("-" * 80)

        norm_config = NormalizationConfig(
            # Structural
            normalize_line_breaks=True,
            remove_hyphen_line_breaks=True,
            collapse_whitespace=True,
            unicode_normalize=True,

            # Noise removal
            remove_urls=True,
            remove_page_numbers=True,
            remove_headers_footers=True,
            remove_toc_pages=True,

            # Protection
            protect_headings=True,
            protect_tables=True,
            protect_code_blocks=True,

            # Advanced
            detect_multi_column=True,
            preserve_hierarchy=True,
            add_page_markers=True,
        )

        normalized_text, page_results, norm_stats = normalize_with_metadata(
            extraction_result,
            norm_config
        )

        # Save normalized text
        stage3_file = self.output_dir / f"{file_name}_stage3_normalized.txt"
        self._save_stage3(stage3_file, normalized_text, page_results, norm_stats)
        output_files['stage3_normalized'] = str(stage3_file)
        print(f"✓ Saved: {stage3_file}")

        # Save normalization stats
        stage3_stats = self.output_dir / f"{file_name}_stage3_stats.json"
        with open(stage3_stats, 'w', encoding='utf-8') as f:
            json.dump(norm_stats, f, indent=2)
        output_files['stage3_stats'] = str(stage3_stats)
        print(f"✓ Saved: {stage3_stats}")

        # ================================================================
        # STAGE 4: AFTER CHUNKING (BEFORE EMBEDDINGS)
        # ================================================================
        print("\n[STAGE 4] AFTER CHUNKING (BEFORE EMBEDDINGS)")
        print("-" * 80)

        chunk_config = ChunkingConfig(
            max_chunk_size=1000,
            target_chunk_size=500,
            enable_overlap=True,
            overlap_size=100,
            respect_page_boundaries=True,
            keep_tables_intact=True,
        )

        chunks = chunk_with_normalization(
            extraction_result,
            normalized_text,
            chunk_config
        )

        # Save chunks
        stage4_file = self.output_dir / f"{file_name}_stage4_chunks.txt"
        self._save_stage4(stage4_file, chunks)
        output_files['stage4_chunks'] = str(stage4_file)
        print(f"✓ Saved: {stage4_file}")

        # Save chunk metadata
        stage4_meta = self.output_dir / f"{file_name}_stage4_chunk_metadata.json"
        self._save_chunk_metadata(stage4_meta, chunks)
        output_files['stage4_metadata'] = str(stage4_meta)
        print(f"✓ Saved: {stage4_meta}")

        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "=" * 80)
        print("PIPELINE INSPECTION COMPLETE")
        print("=" * 80)
        print(f"\nTotal pages: {extraction_result.metadata.total_pages}")
        print(f"Headings detected: {len(extraction_result.metadata.all_headings)}")
        print(f"Sections: {len(extraction_result.metadata.sections)}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Chunks with sections: {sum(1 for c in chunks if c.section_title)}")

        print(f"\nOutput files saved to: {self.output_dir}")
        for stage, path in output_files.items():
            print(f"  {stage}: {path}")

        return output_files

    def _save_stage1(self, filepath: Path, extraction_result):
        """Save Stage 1: Raw extraction output."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 1: RAW EXTRACTION\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"File: {extraction_result.metadata.file_name}\n")
            f.write(f"Type: {extraction_result.metadata.file_type}\n")
            f.write(f"Pages: {extraction_result.metadata.total_pages}\n")
            f.write(f"Total chars: {extraction_result.metadata.total_chars:,}\n")
            f.write(f"Total words: {extraction_result.metadata.total_words:,}\n\n")

            f.write("-" * 80 + "\n")
            f.write("EXTRACTED TEXT (RAW)\n")
            f.write("-" * 80 + "\n\n")

            f.write(extraction_result.text)

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("END OF RAW EXTRACTION\n")
            f.write("=" * 80 + "\n")

    def _save_metadata(self, filepath: Path, metadata):
        """Save extraction metadata as JSON."""
        meta_dict = {
            'file_name': metadata.file_name,
            'file_type': metadata.file_type,
            'total_pages': metadata.total_pages,
            'total_chars': metadata.total_chars,
            'total_words': metadata.total_words,
            'extraction_date': metadata.extraction_date,
            'all_headings': metadata.all_headings,
            'sections': metadata.sections,
            'total_sections': metadata.total_sections,
            'has_toc': metadata.has_toc,
            'toc_page_numbers': metadata.toc_page_numbers,
            'all_urls': metadata.all_urls[:20],  # First 20 URLs
            'all_emails': metadata.all_emails[:20],  # First 20 emails
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(meta_dict, f, indent=2)

    def _save_stage2(self, filepath: Path, extraction_result):
        """Save Stage 2: After heading enrichment."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 2: AFTER HEADING ENRICHMENT\n")
            f.write("=" * 80 + "\n\n")

            f.write("Headings should now be in markdown format (##)\n")
            f.write(f"Total headings detected: {len(extraction_result.metadata.all_headings)}\n\n")

            if extraction_result.metadata.all_headings:
                f.write("Detected headings:\n")
                for i, heading in enumerate(extraction_result.metadata.all_headings[:20], 1):
                    f.write(f"  {i}. {heading}\n")
                f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("ENRICHED TEXT (with markdown headings)\n")
            f.write("-" * 80 + "\n\n")

            f.write(extraction_result.text)

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("END OF ENRICHED TEXT\n")
            f.write("=" * 80 + "\n")

    def _save_stage3(self, filepath: Path, normalized_text: str, page_results, stats):
        """Save Stage 3: After normalization."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 3: AFTER NORMALIZATION\n")
            f.write("=" * 80 + "\n\n")

            f.write("Normalization Statistics:\n")
            f.write(f"  Pages processed: {stats['total_pages_processed']}\n")
            f.write(f"  Original chars: {stats['original_chars']:,}\n")
            f.write(f"  Normalized chars: {stats['normalized_chars']:,}\n")
            f.write(f"  Reduction: {stats['char_reduction_percent']:.1f}%\n")
            f.write(f"  URLs removed: {stats['total_urls_removed']}\n")
            f.write(f"  Protected elements: {stats['total_protected_elements']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("NORMALIZED TEXT (cleaned, with page markers)\n")
            f.write("-" * 80 + "\n\n")

            f.write(normalized_text)

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("END OF NORMALIZED TEXT\n")
            f.write("=" * 80 + "\n")

    def _save_stage4(self, filepath: Path, chunks):
        """Save Stage 4: Chunks before embeddings."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 4: CHUNKS (BEFORE EMBEDDINGS)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total chunks: {len(chunks)}\n")
            chunks_with_sections = sum(1 for c in chunks if c.section_title)
            f.write(f"Chunks with sections: {chunks_with_sections}\n")
            f.write(f"Chunks without sections: {len(chunks) - chunks_with_sections}\n\n")

            f.write("=" * 80 + "\n\n")

            for i, chunk in enumerate(chunks, 1):
                f.write(f"CHUNK {i}/{len(chunks)}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Chunk ID: {chunk.chunk_id}\n")
                f.write(f"Page range: {chunk.page_number_start}-{chunk.page_number_end}\n")
                f.write(f"Section title: {chunk.section_title or '(none)'}\n")
                f.write(f"Heading path: {chunk.heading_path}\n")
                f.write(f"Hierarchy level: {chunk.hierarchy_level}\n")
                f.write(f"Boundary type: {chunk.boundary_type}\n")
                f.write(f"Char length: {chunk.char_length}\n")
                f.write(f"Word count: {chunk.word_count}\n")
                f.write(f"Has overlap: {chunk.has_overlap}\n")
                f.write(f"Contains tables: {chunk.contains_tables}\n")
                f.write(f"Contains code: {chunk.contains_code}\n")
                f.write("\nTEXT:\n")
                f.write(chunk.text)
                f.write("\n\n" + "=" * 80 + "\n\n")

            f.write("END OF CHUNKS\n")
            f.write("=" * 80 + "\n")

    def _save_chunk_metadata(self, filepath: Path, chunks):
        """Save chunk metadata as JSON."""
        chunk_metadata = []

        for chunk in chunks:
            chunk_metadata.append({
                'chunk_id': chunk.chunk_id,
                'chunk_index': chunk.chunk_index,
                'page_start': chunk.page_number_start,
                'page_end': chunk.page_number_end,
                'section_title': chunk.section_title,
                'heading_path': chunk.heading_path,
                'heading_path_str': chunk.heading_path_str,
                'hierarchy_level': chunk.hierarchy_level,
                'char_length': chunk.char_length,
                'word_count': chunk.word_count,
                'boundary_type': chunk.boundary_type,
                'has_overlap': chunk.has_overlap,
                'contains_tables': chunk.contains_tables,
                'contains_code': chunk.contains_code,
                'contains_bullets': chunk.contains_bullets,
                'text_preview': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'total_chunks': len(chunks),
                'chunks_with_sections': sum(1 for c in chunks if c.section_title),
                'chunks': chunk_metadata
            }, f, indent=2)


def inspect_document(file_path: str, output_dir: str = "pipeline_debug"):
    """
    Convenience function to inspect a document.

    Args:
        file_path: Path to document
        output_dir: Directory to save debug outputs

    Returns:
        Dictionary of output file paths
    """
    inspector = PipelineStageInspector(output_dir)
    return inspector.inspect_pipeline(file_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 80)
        print("Pipeline Stage Inspector")
        print("=" * 80)
        print("\nUsage: python pipeline_stage_inspector.py <document_path> [output_dir]")
        print("\nExample:")
        print("  python pipeline_stage_inspector.py document.pdf")
        print("  python pipeline_stage_inspector.py document.pdf my_debug_output")
        print("\nThis will create text files showing output at each pipeline stage:")
        print("  1. Raw extraction")
        print("  2. After heading enrichment")
        print("  3. After normalization")
        print("  4. After chunking (before embeddings)")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "pipeline_debug"

    # Check file exists
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Run inspection
    output_files = inspect_document(file_path, output_dir)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nCheck the output files in: {output_dir}/")
    print("\nRecommended inspection order:")
    print("  1. stage1_raw_extraction.txt - See raw extracted text")
    print("  2. stage1_metadata.json - See detected headings/sections")
    print("  3. stage2_enriched_text.txt - Verify markdown formatting")
    print("  4. stage3_normalized.txt - Check normalization preserved headings")
    print("  5. stage4_chunks.txt - Verify chunks have section info")
    print("  6. stage4_chunk_metadata.json - Quick overview of all chunks")
