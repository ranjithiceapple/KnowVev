"""
Document Summarizer Module

Generates document-level summaries from chunks to create a "virtual chunk"
that provides high-level overview of the entire document.

Strategies:
1. Extractive: Select most representative sentences/chunks
2. Abstractive: Generate summary using first/last chunks + headings
3. Hierarchical: Combine section summaries
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
from logger_config import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentSummary:
    """
    Represents a document-level summary.
    """
    doc_id: str
    file_name: str
    summary_text: str
    summary_method: str  # 'extractive', 'abstractive', 'hierarchical'

    # Source information
    total_chunks: int
    total_chars: int
    total_pages: int

    # Summary metadata
    summary_length: int
    compression_ratio: float  # summary_length / total_chars

    # Key information extracted
    document_title: Optional[str] = None
    main_sections: List[str] = None
    key_topics: List[str] = None

    # Timestamps
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.main_sections is None:
            self.main_sections = []
        if self.key_topics is None:
            self.key_topics = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'doc_id': self.doc_id,
            'file_name': self.file_name,
            'summary_text': self.summary_text,
            'summary_method': self.summary_method,
            'total_chunks': self.total_chunks,
            'total_chars': self.total_chars,
            'total_pages': self.total_pages,
            'summary_length': self.summary_length,
            'compression_ratio': self.compression_ratio,
            'document_title': self.document_title,
            'main_sections': self.main_sections,
            'key_topics': self.key_topics,
            'created_at': self.created_at
        }


class DocumentSummarizer:
    """
    Generates document summaries from chunks.

    Creates a "virtual chunk" that represents the entire document.
    """

    def __init__(self, max_summary_length: int = 2000, method: str = 'hybrid'):
        """
        Initialize summarizer.

        Args:
            max_summary_length: Maximum length of summary in characters
            method: Summarization method ('extractive', 'abstractive', 'hybrid')
        """
        self.max_summary_length = max_summary_length
        self.method = method
        logger.info(f"DocumentSummarizer initialized - Max length: {max_summary_length}, Method: {method}")

    def generate_summary(
        self,
        chunks: List[Any],
        extraction_result: Any,
        doc_id: str,
        file_name: str
    ) -> DocumentSummary:
        """
        Generate document summary from chunks.

        Args:
            chunks: List of ChunkMetadata objects
            extraction_result: ExtractionResult from document_processor
            doc_id: Document ID
            file_name: File name

        Returns:
            DocumentSummary object
        """
        logger.info(f"Generating document summary for {file_name} ({len(chunks)} chunks)")

        # Calculate document statistics
        total_chars = sum(chunk.chunk_char_len for chunk in chunks)
        total_pages = extraction_result.metadata.total_pages if hasattr(extraction_result, 'metadata') else len(chunks)

        # Extract document structure
        document_title = self._extract_title(chunks, file_name)
        main_sections = self._extract_sections(chunks)

        # Generate summary based on method
        if self.method == 'extractive':
            summary_text = self._extractive_summary(chunks)
        elif self.method == 'abstractive':
            summary_text = self._abstractive_summary(chunks, document_title, main_sections)
        else:  # hybrid
            summary_text = self._hybrid_summary(chunks, document_title, main_sections)

        # Ensure summary doesn't exceed max length
        if len(summary_text) > self.max_summary_length:
            summary_text = summary_text[:self.max_summary_length - 3] + "..."
            logger.debug(f"Summary truncated to {self.max_summary_length} characters")

        # Extract key topics
        key_topics = self._extract_key_topics(chunks)

        # Create summary object
        summary = DocumentSummary(
            doc_id=doc_id,
            file_name=file_name,
            summary_text=summary_text,
            summary_method=self.method,
            total_chunks=len(chunks),
            total_chars=total_chars,
            total_pages=total_pages,
            summary_length=len(summary_text),
            compression_ratio=len(summary_text) / total_chars if total_chars > 0 else 0,
            document_title=document_title,
            main_sections=main_sections,
            key_topics=key_topics
        )

        logger.info(
            f"Summary generated - Length: {summary.summary_length} chars, "
            f"Compression: {summary.compression_ratio:.1%}"
        )

        return summary

    def _extract_title(self, chunks: List[Any], file_name: str) -> str:
        """Extract document title from chunks or filename."""
        # Try to find title from first chunk with section title
        for chunk in chunks[:5]:  # Check first 5 chunks
            if hasattr(chunk, 'section_title') and chunk.section_title:
                # Clean up title
                title = chunk.section_title
                # Remove common prefixes
                title = re.sub(r'^(Chapter|Section|Part)\s+\d+[:\s]*', '', title, flags=re.IGNORECASE)
                if len(title) > 5:  # Valid title
                    return title

        # Fallback to filename without extension
        return file_name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()

    def _extract_sections(self, chunks: List[Any]) -> List[str]:
        """Extract main section headings from chunks."""
        sections = []
        seen = set()

        for chunk in chunks:
            if hasattr(chunk, 'section_title') and chunk.section_title:
                title = chunk.section_title.strip()
                # Only add unique, substantial titles
                if title and title not in seen and len(title) > 3:
                    sections.append(title)
                    seen.add(title)

                    # Limit to top 10 sections
                    if len(sections) >= 10:
                        break

        return sections

    def _extract_key_topics(self, chunks: List[Any]) -> List[str]:
        """Extract key topics from section titles and headings."""
        topics = set()

        for chunk in chunks:
            # Extract from section titles
            if hasattr(chunk, 'section_title') and chunk.section_title:
                # Extract meaningful words (3+ chars)
                words = re.findall(r'\b[A-Z][a-z]{2,}\b', chunk.section_title)
                topics.update(words[:3])

            # Extract from heading path
            if hasattr(chunk, 'heading_path') and chunk.heading_path:
                for heading in chunk.heading_path[:2]:  # Top 2 levels
                    words = re.findall(r'\b[A-Z][a-z]{2,}\b', heading)
                    topics.update(words[:2])

        return sorted(list(topics))[:15]  # Top 15 topics

    def _extractive_summary(self, chunks: List[Any]) -> str:
        """
        Extractive summarization: Select most representative chunks.

        Strategy:
        - First chunk (introduction)
        - Chunks with section titles (key sections)
        - Last chunk (conclusion)
        """
        summary_parts = []
        current_length = 0

        # 1. Add first chunk (introduction)
        if chunks:
            first_text = self._clean_text(chunks[0].normalized_text)
            if first_text:
                summary_parts.append(f"[Introduction]\n{first_text[:400]}")
                current_length += len(summary_parts[-1])

        # 2. Add chunks with section titles (key sections)
        section_chunks = [c for c in chunks if hasattr(c, 'section_title') and c.section_title]

        for chunk in section_chunks:
            if current_length >= self.max_summary_length * 0.8:
                break

            section_text = self._clean_text(chunk.normalized_text)
            if section_text:
                excerpt = section_text[:300]
                summary_parts.append(f"[{chunk.section_title}]\n{excerpt}")
                current_length += len(summary_parts[-1])

        # 3. Add last chunk (conclusion) if space available
        if chunks and current_length < self.max_summary_length * 0.9:
            last_text = self._clean_text(chunks[-1].normalized_text)
            if last_text:
                summary_parts.append(f"[Conclusion]\n{last_text[:300]}")

        return "\n\n".join(summary_parts)

    def _abstractive_summary(self, chunks: List[Any], title: str, sections: List[str]) -> str:
        """
        Abstractive summarization: Generate overview from structure.

        Strategy:
        - Use document title
        - List main sections
        - Include first and last chunk excerpts
        """
        summary_parts = []

        # Document overview
        summary_parts.append(f"Document: {title}")

        # Add document statistics
        total_pages = max((c.page_number_end for c in chunks), default=0)
        summary_parts.append(f"Pages: {total_pages}")
        summary_parts.append(f"Sections: {len(sections)}")

        # Main sections
        if sections:
            summary_parts.append("\nMain Sections:")
            for i, section in enumerate(sections[:8], 1):
                summary_parts.append(f"{i}. {section}")

        # Content overview from first chunk
        if chunks:
            first_text = self._clean_text(chunks[0].normalized_text)
            if first_text:
                summary_parts.append(f"\nOverview:\n{first_text[:500]}")

        # Key points from last chunk
        if len(chunks) > 1:
            last_text = self._clean_text(chunks[-1].normalized_text)
            if last_text:
                summary_parts.append(f"\nConclusion:\n{last_text[:300]}")

        return "\n".join(summary_parts)

    def _hybrid_summary(self, chunks: List[Any], title: str, sections: List[str]) -> str:
        """
        Hybrid summarization: Combine abstractive structure with extractive content.

        Best of both worlds approach.
        """
        summary_parts = []

        # 1. Document header (abstractive)
        summary_parts.append(f"=== {title} ===\n")

        total_pages = max((c.page_number_end for c in chunks), default=0)
        summary_parts.append(f"Total Pages: {total_pages} | Total Sections: {len(sections)}\n")

        # 2. Main sections list (abstractive)
        if sections:
            summary_parts.append("Main Sections:")
            for section in sections[:6]:
                summary_parts.append(f"  • {section}")
            summary_parts.append("")

        # 3. Content excerpts (extractive)
        summary_parts.append("Content Summary:\n")

        # First chunk
        if chunks:
            first_text = self._clean_text(chunks[0].normalized_text)
            if first_text:
                summary_parts.append(f"Introduction: {first_text[:400]}...\n")

        # Middle sections (sample 2-3 key sections)
        section_chunks = [c for c in chunks[1:-1] if hasattr(c, 'section_title') and c.section_title]
        for chunk in section_chunks[:3]:
            text = self._clean_text(chunk.normalized_text)
            if text:
                summary_parts.append(f"{chunk.section_title}: {text[:250]}...\n")

        # Last chunk
        if len(chunks) > 1:
            last_text = self._clean_text(chunks[-1].normalized_text)
            if last_text:
                summary_parts.append(f"Conclusion: {last_text[:300]}...")

        return "\n".join(summary_parts)

    def _clean_text(self, text: str) -> str:
        """Clean text for summary."""
        if not text:
            return ""

        # Remove page markers
        text = re.sub(r'<<<PAGE_\d+>>>', '', text)

        # Remove hierarchy markers
        text = re.sub(r'<<<HIERARCHY_L\d+>>>', '', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def create_summary_chunk(self, summary: DocumentSummary) -> Dict[str, Any]:
        """
        Create a virtual chunk from the document summary.

        This chunk can be added to the regular chunks for embedding.

        Returns:
            Dictionary representing a virtual chunk
        """
        from enterprise_chunking_pipeline import ChunkMetadata, BoundaryType

        # Create a special chunk for the summary
        summary_chunk = ChunkMetadata(
            doc_id=summary.doc_id,
            file_name=summary.file_name,
            chunk_id=f"{summary.doc_id}_SUMMARY",
            page_number_start=1,
            page_number_end=summary.total_pages,
            section_title="[DOCUMENT SUMMARY]",
            heading_path=["Document Overview"],
            chunk_index=-1,  # Special index to indicate summary
            total_chunks=summary.total_chunks + 1,  # +1 for the summary itself
            chunk_char_len=summary.summary_length,
            chunk_word_count=len(summary.summary_text.split()),
            boundary_type=BoundaryType.SECTION.value,
            normalized_text=summary.summary_text,
            contains_tables=False,
            contains_code=False,
            contains_bullets=len(summary.main_sections) > 0,
        )

        logger.info(f"Created summary virtual chunk: {summary_chunk.chunk_id}")

        return summary_chunk


def generate_document_summary(
    chunks: List[Any],
    extraction_result: Any,
    doc_id: str,
    file_name: str,
    max_length: int = 2000,
    method: str = 'hybrid'
) -> DocumentSummary:
    """
    Convenience function to generate document summary.

    Args:
        chunks: List of ChunkMetadata objects
        extraction_result: ExtractionResult from document_processor
        doc_id: Document ID
        file_name: File name
        max_length: Maximum summary length
        method: Summarization method

    Returns:
        DocumentSummary object
    """
    summarizer = DocumentSummarizer(max_length, method)
    return summarizer.generate_summary(chunks, extraction_result, doc_id, file_name)


# Example usage
if __name__ == "__main__":
    print("Document Summarizer Module")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✅ Extractive summarization (select key chunks)")
    print("  ✅ Abstractive summarization (generate overview)")
    print("  ✅ Hybrid summarization (combine both)")
    print("  ✅ Virtual chunk creation for embedding")
    print("\nUsage:")
    print("""
from document_summarizer import generate_document_summary, DocumentSummarizer

# Generate summary
summary = generate_document_summary(
    chunks=chunks,
    extraction_result=extraction_result,
    doc_id=doc_id,
    file_name=file_name,
    max_length=2000,
    method='hybrid'
)

# Create virtual chunk
summarizer = DocumentSummarizer()
summary_chunk = summarizer.create_summary_chunk(summary)

# Add to chunks for embedding
all_chunks = [summary_chunk] + chunks
""")
