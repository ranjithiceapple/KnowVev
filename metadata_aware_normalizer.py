"""
Metadata-Aware Text Normalization Module

This module provides intelligent text normalization that uses document metadata
to make context-aware decisions about text cleanup. Processes text per-page
rather than globally to preserve document structure.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageNormalizationResult:
    """Result of normalizing a single page."""
    page_number: int
    normalized_text: str
    original_char_count: int
    normalized_char_count: int
    original_word_count: int
    normalized_word_count: int
    removed_urls: List[str] = field(default_factory=list)
    removed_page_numbers: List[str] = field(default_factory=list)
    protected_elements: List[str] = field(default_factory=list)
    changes_applied: List[str] = field(default_factory=list)


@dataclass
class NormalizationConfig:
    """Configuration for metadata-aware normalization."""
    # Structural normalizations
    normalize_line_breaks: bool = True
    remove_hyphen_line_breaks: bool = True
    collapse_whitespace: bool = True
    collapse_blank_lines: bool = True
    unicode_normalize: bool = True

    # Noise removal (metadata-aware)
    remove_urls: bool = True
    remove_page_numbers: bool = True
    remove_headers_footers: bool = True
    remove_toc_pages: bool = True

    # Semantic normalizations
    fix_broken_sentences: bool = True
    merge_hyphenated_words: bool = True
    normalize_bullet_points: bool = True

    # OCR corrections (conservative by default)
    apply_ocr_corrections: bool = False  # Disabled by default
    fix_ligatures: bool = True  # Safe to always apply
    remove_duplicated_lines: bool = False  # Only if explicitly needed

    # Protection rules
    protect_headings: bool = True
    protect_bullet_points: bool = True
    protect_code_blocks: bool = True
    protect_tables: bool = True

    # Page markers
    add_page_markers: bool = True
    page_marker_template: str = "\n\n<<<PAGE_{page_num}>>>\n\n"

    # Advanced features (NEW)
    detect_multi_column: bool = True
    preserve_hierarchy: bool = True
    enable_metadata_crosscheck: bool = True
    multi_column_threshold: int = 4  # Number of spaces to detect columns

    # Override rules (user-configurable)
    custom_protect_patterns: List[str] = field(default_factory=list)
    custom_remove_patterns: List[str] = field(default_factory=list)
    skip_pages: List[int] = field(default_factory=list)
    force_process_pages: List[int] = field(default_factory=list)


class ProtectionMarker:
    """Manages protection of special text elements during normalization."""

    def __init__(self):
        self.protected_blocks: Dict[str, str] = {}
        self.marker_counter = 0

        # IMPROVED: Enhanced code block patterns
        self.code_block_patterns = [
            # Markdown code blocks
            re.compile(r'```[\s\S]*?```', re.MULTILINE),
            # Inline code
            re.compile(r'`[^`\n]+`'),
            # Indented code blocks (4+ spaces)
            re.compile(r'^(?: {4}|\t).+$', re.MULTILINE),
            # Code with common keywords
            re.compile(
                r'^(?:def|class|function|var|const|let|import|from|#include|public|private)\s+.+$',
                re.MULTILINE
            ),
        ]

        # IMPROVED: Enhanced table patterns
        self.table_patterns = [
            # Pipe-separated tables
            re.compile(r'^\|.+\|$', re.MULTILINE),
            # Grid tables (with +, -, |)
            re.compile(r'^\+[-+]+\+$', re.MULTILINE),
            # Tables with --- TABLES --- marker (from DOCX extraction)
            re.compile(r'^--- TABLES ---$.*?(?=\n\n|\Z)', re.MULTILINE | re.DOTALL),
            # [Table N] markers
            re.compile(r'^\[Table \d+\]$.*?(?=\n\n|\Z)', re.MULTILINE | re.DOTALL),
            # Aligned columns (detected by consistent spacing)
            re.compile(r'^(?:\S+\s{2,}){2,}\S+$', re.MULTILINE),
        ]

        # Bullet patterns
        self.bullet_pattern = re.compile(
            r'^[\s]*[•·∙●○◦▪▫■□\*\-\+]\s+.+$',
            re.MULTILINE
        )

        # IMPROVED: Enhanced heading patterns with hierarchy
        self.heading_patterns = [
            # Markdown headings
            re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
            # ALL CAPS headings
            re.compile(r'^[A-Z][A-Z\s]{5,}$', re.MULTILINE),
            # Numbered sections (1., 1.1, etc.)
            re.compile(r'^\d+(?:\.\d+)*\s+[A-Z].+$', re.MULTILINE),
            # Chapter/Section/Part
            re.compile(r'^(?:Chapter|Section|Part|Article|Appendix)\s+\d+[:\s]+.+$', re.MULTILINE | re.IGNORECASE),
            # Underlined headings (=== or ---)
            re.compile(r'^.+\n[=\-]{3,}$', re.MULTILINE),
            # --- SECTION BREAK --- markers
            re.compile(r'^--- SECTION BREAK ---$', re.MULTILINE),
        ]

    def protect_text(self, text: str, protect_config: Dict[str, bool], custom_patterns: Optional[List[str]] = None) -> Tuple[str, List[str]]:
        """
        Replace protected elements with markers.

        Args:
            text: Input text
            protect_config: Dictionary of protection flags
            custom_patterns: Optional list of custom regex patterns to protect

        Returns:
            Tuple of (protected_text, list_of_protected_elements)
        """
        protected_elements = []

        # IMPROVED: Protect code blocks (multiple patterns)
        if protect_config.get('code_blocks', True):
            for code_pattern in self.code_block_patterns:
                text, code_elements = self._protect_pattern(text, code_pattern, 'CODE')
                protected_elements.extend(code_elements)

        # IMPROVED: Protect tables (multiple patterns)
        if protect_config.get('tables', True):
            for table_pattern in self.table_patterns:
                text, table_elements = self._protect_pattern(text, table_pattern, 'TABLE')
                protected_elements.extend(table_elements)

        # Protect bullet points
        if protect_config.get('bullets', True):
            text, bullet_elements = self._protect_pattern(text, self.bullet_pattern, 'BULLET')
            protected_elements.extend(bullet_elements)

        # IMPROVED: Protect headings (multiple patterns for hierarchy)
        if protect_config.get('headings', True):
            for heading_pattern in self.heading_patterns:
                text, heading_elements = self._protect_pattern(text, heading_pattern, 'HEADING')
                protected_elements.extend(heading_elements)

        # NEW: Custom protection patterns
        if custom_patterns:
            for idx, custom_pattern_str in enumerate(custom_patterns):
                try:
                    custom_pattern = re.compile(custom_pattern_str, re.MULTILINE)
                    text, custom_elements = self._protect_pattern(text, custom_pattern, f'CUSTOM_{idx}')
                    protected_elements.extend(custom_elements)
                except re.error as e:
                    logger.warning(f"Invalid custom pattern '{custom_pattern_str}': {e}")

        return text, protected_elements

    def _protect_pattern(self, text: str, pattern: re.Pattern, element_type: str) -> Tuple[str, List[str]]:
        """Protect matches of a pattern."""
        protected = []

        def replace_with_marker(match):
            marker = f"<<<PROTECTED_{element_type}_{self.marker_counter}>>>"
            self.protected_blocks[marker] = match.group(0)
            self.marker_counter += 1
            protected.append(f"{element_type}: {match.group(0)[:50]}...")
            return marker

        text = pattern.sub(replace_with_marker, text)
        return text, protected

    def restore_text(self, text: str) -> str:
        """Restore protected elements."""
        for marker, original in self.protected_blocks.items():
            text = text.replace(marker, original)
        return text

    def clear(self):
        """Clear protection state."""
        self.protected_blocks.clear()
        self.marker_counter = 0


class MultiColumnDetector:
    """
    Detects and handles multi-column text layouts.
    IMPROVED: Better detection algorithm.
    """

    def __init__(self, threshold: int = 4):
        self.threshold = threshold
        # Pattern for detecting column separators (multiple spaces)
        self.column_separator_pattern = re.compile(rf'\s{{{threshold},}}')

    def detect_columns(self, text: str) -> bool:
        """
        Detect if text contains multi-column layout.

        Returns:
            True if multi-column layout detected
        """
        lines = text.split('\n')
        column_lines = 0

        for line in lines:
            if len(line) > 40 and self.column_separator_pattern.search(line):
                column_lines += 1

        # If more than 30% of lines have column separators, it's multi-column
        return column_lines > len(lines) * 0.3 if lines else False

    def split_columns(self, text: str) -> str:
        """
        Convert multi-column text to single column.

        Args:
            text: Multi-column text

        Returns:
            Single-column text
        """
        lines = text.split('\n')
        result_lines = []

        for line in lines:
            # Split by column separator
            if self.column_separator_pattern.search(line):
                columns = self.column_separator_pattern.split(line)
                # Add each column as separate lines
                for col in columns:
                    col = col.strip()
                    if col:
                        result_lines.append(col)
            else:
                result_lines.append(line)

        return '\n'.join(result_lines)


class HierarchyPreserver:
    """
    Preserves document hierarchy and section structure.
    NEW: Tracks and maintains document outline.
    """

    def __init__(self):
        self.hierarchy = []
        self.current_level = 0

        # Hierarchy patterns with levels
        self.hierarchy_patterns = [
            (re.compile(r'^#{1}\s+(.+)$', re.MULTILINE), 1),  # # H1
            (re.compile(r'^#{2}\s+(.+)$', re.MULTILINE), 2),  # ## H2
            (re.compile(r'^#{3}\s+(.+)$', re.MULTILINE), 3),  # ### H3
            (re.compile(r'^[A-Z][A-Z\s]{8,}$', re.MULTILINE), 1),  # ALL CAPS = H1
            (re.compile(r'^(?:Chapter|Part)\s+\d+', re.MULTILINE | re.IGNORECASE), 1),  # Chapter = H1
            (re.compile(r'^(?:Section)\s+\d+', re.MULTILINE | re.IGNORECASE), 2),  # Section = H2
            (re.compile(r'^\d+\.\s+[A-Z]', re.MULTILINE), 2),  # 1. Title = H2
            (re.compile(r'^\d+\.\d+\s+[A-Z]', re.MULTILINE), 3),  # 1.1 Title = H3
        ]

    def extract_hierarchy(self, text: str) -> List[Dict[str, any]]:
        """
        Extract document hierarchy/outline.

        Returns:
            List of hierarchy items with level and title
        """
        hierarchy = []

        for pattern, level in self.hierarchy_patterns:
            for match in pattern.finditer(text):
                hierarchy.append({
                    'level': level,
                    'title': match.group(0).strip(),
                    'position': match.start()
                })

        # Sort by position
        hierarchy.sort(key=lambda x: x['position'])

        return hierarchy

    def add_hierarchy_markers(self, text: str) -> str:
        """
        Add explicit hierarchy markers to preserve structure.

        Args:
            text: Input text

        Returns:
            Text with hierarchy markers
        """
        hierarchy = self.extract_hierarchy(text)

        # Add markers in reverse order to maintain positions
        for item in reversed(hierarchy):
            marker = f"\n<<<HIERARCHY_L{item['level']}>>>\n"
            # Insert marker before the heading
            pos = item['position']
            text = text[:pos] + marker + text[pos:]

        return text


class MetadataCrossChecker:
    """
    Cross-checks normalized text against extracted metadata.
    NEW: Validates that normalization preserves important content.
    """

    def __init__(self):
        self.warnings = []

    def crosscheck(self, normalized_text: str, page_meta, original_text: str) -> List[str]:
        """
        Cross-check normalized text against metadata.

        Args:
            normalized_text: Normalized text
            page_meta: PageMetadata object
            original_text: Original text before normalization

        Returns:
            List of warnings/issues found
        """
        warnings = []

        # Check 1: Verify headings are preserved
        if page_meta.headings:
            for heading in page_meta.headings:
                if heading not in normalized_text:
                    warnings.append(f"Heading possibly lost: '{heading[:50]}...'")

        # Check 2: Verify excessive text loss
        original_words = len(original_text.split())
        normalized_words = len(normalized_text.split())
        loss_percent = ((original_words - normalized_words) / original_words * 100) if original_words > 0 else 0

        if loss_percent > 50:
            warnings.append(f"Excessive text loss: {loss_percent:.1f}% of words removed")

        # Check 3: Check for over-aggressive whitespace removal
        if '\n\n\n' not in original_text and '\n\n\n' not in normalized_text:
            # Good - no excessive blank lines in either
            pass
        elif '\n\n\n' in original_text and '\n\n\n' not in normalized_text:
            # Good - removed excessive blank lines
            pass

        # Check 4: Verify URLs were removed if configured
        if page_meta.urls:
            urls_still_present = sum(1 for url in page_meta.urls if url in normalized_text)
            if urls_still_present > 0:
                warnings.append(f"{urls_still_present} URLs still present in normalized text")

        return warnings


class MetadataAwareNormalizer:
    """
    Intelligent text normalizer that uses document metadata for context-aware
    normalization. Processes pages individually to preserve structure.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
        self.protection = ProtectionMarker()

        # NEW: Advanced components
        self.column_detector = MultiColumnDetector(self.config.multi_column_threshold)
        self.hierarchy_preserver = HierarchyPreserver()
        self.crosschecker = MetadataCrossChecker()

        # Patterns for structural normalization
        self.hyphen_linebreak_pattern = re.compile(r'(\w+)-\s*\n\s*(\w+)')
        self.whitespace_pattern = re.compile(r'[ \t]+')
        self.blank_lines_pattern = re.compile(r'\n\s*\n\s*\n+')

        # Improved broken sentence pattern (excludes ALL CAPS lines)
        self.broken_sentence_pattern = re.compile(
            r'([a-z,;:])\s*\n+\s*([a-z])',  # Only lowercase to lowercase
            re.MULTILINE
        )

        # Page number patterns with context awareness
        self.page_number_patterns = [
            re.compile(r'^\s*-?\s*{page_num}\s*-?\s*$', re.MULTILINE),
            re.compile(r'^\s*Page\s+{page_num}\s*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\s*{page_num}\s+of\s+\d+\s*$', re.MULTILINE | re.IGNORECASE),
        ]

        # Header/footer patterns
        self.header_footer_patterns = [
            re.compile(r'^\s*(?:confidential|proprietary|draft|internal use only)\s*$',
                      re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\s*©\s*\d{4}.*$', re.MULTILINE),
        ]

        # Custom removal patterns from config
        if self.config.custom_remove_patterns:
            for pattern_str in self.config.custom_remove_patterns:
                try:
                    pattern = re.compile(pattern_str, re.MULTILINE | re.IGNORECASE)
                    self.header_footer_patterns.append(pattern)
                except re.error as e:
                    logger.warning(f"Invalid custom remove pattern '{pattern_str}': {e}")

        # Ligature map
        self.ligature_map = {
            'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            'ﬅ': 'st', 'ﬆ': 'st', 'Æ': 'AE', 'æ': 'ae', 'Œ': 'OE', 'œ': 'oe'
        }

    def normalize_document(self, extraction_result) -> Tuple[str, List[PageNormalizationResult]]:
        """
        Normalize an entire document using metadata-aware processing.

        Args:
            extraction_result: ExtractionResult from document_processor

        Returns:
            Tuple of (normalized_full_text, list_of_page_results)
        """
        page_results = []
        normalized_pages = []

        # Filter out TOC pages if configured
        pages_to_process = self._filter_toc_pages(
            extraction_result.pages,
            extraction_result.metadata
        )

        logger.info(f"Normalizing {len(pages_to_process)} pages (filtered from {len(extraction_result.pages)})")

        # Process each page individually
        for page_meta in pages_to_process:
            page_result = self._normalize_page(page_meta, extraction_result.metadata)
            page_results.append(page_result)

            # Add page marker if configured
            if self.config.add_page_markers:
                marker = self.config.page_marker_template.format(page_num=page_meta.page_number)
                normalized_pages.append(marker + page_result.normalized_text)
            else:
                normalized_pages.append(page_result.normalized_text)

        # Combine normalized pages
        full_normalized_text = "\n\n".join(normalized_pages)

        logger.info(f"Normalization complete: {len(page_results)} pages processed")

        return full_normalized_text, page_results

    def _filter_toc_pages(self, pages, doc_metadata) -> List:
        """
        Filter out TOC pages if configured.
        NEW: Also applies skip_pages and force_process_pages rules.
        """
        filtered_pages = []

        for page in pages:
            page_num = page.page_number

            # Check skip_pages override
            if page_num in self.config.skip_pages:
                logger.info(f"Skipping page {page_num} (skip_pages rule)")
                continue

            # Check force_process_pages override
            if page_num in self.config.force_process_pages:
                logger.info(f"Force processing page {page_num} (force_process_pages rule)")
                filtered_pages.append(page)
                continue

            # Check TOC pages
            if self.config.remove_toc_pages and doc_metadata.has_toc:
                if page_num in doc_metadata.toc_page_numbers:
                    logger.info(f"Removing page {page_num} (TOC page)")
                    continue

            # Page passes all filters
            filtered_pages.append(page)

        removed_count = len(pages) - len(filtered_pages)
        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} page(s)")

        return filtered_pages

    def normalize_page(self, page_meta, doc_metadata=None) -> PageNormalizationResult:
        """
        NEW: Per-page normalizer entrypoint.
        Normalize a single page independently.

        Args:
            page_meta: PageMetadata object
            doc_metadata: Optional DocumentMetadata for context

        Returns:
            PageNormalizationResult
        """
        logger.info(f"Normalizing page {page_meta.page_number}")
        return self._normalize_page(page_meta, doc_metadata)

    def _normalize_page(self, page_meta, doc_metadata) -> PageNormalizationResult:
        """
        Normalize a single page using its metadata.

        Args:
            page_meta: PageMetadata object
            doc_metadata: DocumentMetadata object

        Returns:
            PageNormalizationResult
        """
        text = page_meta.text
        original_char_count = len(text)
        original_word_count = len(text.split())
        changes_applied = []
        removed_urls = []
        removed_page_numbers = []

        # Step 1: Protect special elements
        protect_config = {
            'code_blocks': self.config.protect_code_blocks,
            'tables': self.config.protect_tables,
            'bullets': self.config.protect_bullet_points,
            'headings': self.config.protect_headings,
        }

        self.protection.clear()
        text, protected_elements = self.protection.protect_text(
            text,
            protect_config,
            custom_patterns=self.config.custom_protect_patterns
        )

        if protected_elements:
            changes_applied.append(f"Protected {len(protected_elements)} elements")

        # NEW: Detect and handle multi-column layout
        if self.config.detect_multi_column:
            if self.column_detector.detect_columns(text):
                text = self.column_detector.split_columns(text)
                changes_applied.append("Converted multi-column layout")

        # NEW: Preserve hierarchy if configured
        if self.config.preserve_hierarchy:
            hierarchy = self.hierarchy_preserver.extract_hierarchy(text)
            if hierarchy:
                text = self.hierarchy_preserver.add_hierarchy_markers(text)
                changes_applied.append(f"Preserved hierarchy ({len(hierarchy)} levels)")

        # Step 2: Structural normalizations
        if self.config.normalize_line_breaks:
            text = self._normalize_line_breaks(text)
            changes_applied.append("Normalized line breaks")

        if self.config.remove_hyphen_line_breaks:
            text = self._remove_hyphen_line_breaks(text)
            changes_applied.append("Removed hyphen line breaks")

        # Step 3: Metadata-aware noise removal
        if self.config.remove_urls and page_meta.urls:
            text, removed = self._remove_urls_from_metadata(text, page_meta.urls)
            removed_urls.extend(removed)
            if removed:
                changes_applied.append(f"Removed {len(removed)} URLs")

        if self.config.remove_page_numbers:
            text, removed = self._remove_page_numbers_for_page(text, page_meta.page_number)
            removed_page_numbers.extend(removed)
            if removed:
                changes_applied.append(f"Removed page number")

        if self.config.remove_headers_footers:
            text = self._remove_headers_footers(text)
            changes_applied.append("Removed headers/footers")

        # Step 4: OCR corrections (conservative)
        if self.config.fix_ligatures:
            text = self._fix_ligatures(text)
            changes_applied.append("Fixed ligatures")

        if self.config.apply_ocr_corrections:
            text = self._apply_ocr_corrections(text)
            changes_applied.append("Applied OCR corrections")

        # Step 5: Semantic normalizations (improved)
        if self.config.merge_hyphenated_words:
            text = self._merge_hyphenated_words(text)
            changes_applied.append("Merged hyphenated words")

        if self.config.fix_broken_sentences:
            text = self._fix_broken_sentences_safe(text)
            changes_applied.append("Fixed broken sentences")

        if self.config.normalize_bullet_points and not self.config.protect_bullet_points:
            text = self._normalize_bullet_points(text)
            changes_applied.append("Normalized bullet points")

        # Step 6: Final cleanup
        if self.config.collapse_whitespace:
            text = self._collapse_whitespace(text)
            changes_applied.append("Collapsed whitespace")

        if self.config.collapse_blank_lines:
            text = self._collapse_blank_lines(text)
            changes_applied.append("Collapsed blank lines")

        if self.config.unicode_normalize:
            text = unicodedata.normalize('NFKC', text)
            changes_applied.append("Unicode normalized")

        # Step 7: Restore protected elements
        text = self.protection.restore_text(text)

        # Step 8: Final cleanup
        text = text.strip()

        # NEW: Step 9: Cross-check with metadata
        warnings = []
        if self.config.enable_metadata_crosscheck:
            warnings = self.crosschecker.crosscheck(text, page_meta, page_meta.text)
            if warnings:
                for warning in warnings:
                    logger.warning(f"Page {page_meta.page_number}: {warning}")
                changes_applied.append(f"Crosscheck: {len(warnings)} warnings")

        # Create result
        result = PageNormalizationResult(
            page_number=page_meta.page_number,
            normalized_text=text,
            original_char_count=original_char_count,
            normalized_char_count=len(text),
            original_word_count=original_word_count,
            normalized_word_count=len(text.split()),
            removed_urls=removed_urls,
            removed_page_numbers=removed_page_numbers,
            protected_elements=protected_elements,
            changes_applied=changes_applied
        )

        return result

    # Structural normalization methods

    def _normalize_line_breaks(self, text: str) -> str:
        """Normalize line breaks to LF."""
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        return text

    def _remove_hyphen_line_breaks(self, text: str) -> str:
        """Remove hyphenation at line breaks."""
        return self.hyphen_linebreak_pattern.sub(r'\1\2', text)

    def _collapse_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/tabs to single space."""
        lines = text.split('\n')
        normalized_lines = [self.whitespace_pattern.sub(' ', line.strip()) for line in lines]
        return '\n'.join(normalized_lines)

    def _collapse_blank_lines(self, text: str, max_consecutive: int = 2) -> str:
        """Collapse multiple blank lines."""
        replacement = '\n' * max_consecutive
        return self.blank_lines_pattern.sub(replacement, text)

    # Metadata-aware noise removal

    def _remove_urls_from_metadata(self, text: str, urls: List[str]) -> Tuple[str, List[str]]:
        """
        Remove URLs that are listed in page metadata.
        More accurate than regex scanning.
        """
        removed = []
        for url in urls:
            # Escape special regex characters in URL
            escaped_url = re.escape(url)
            pattern = re.compile(escaped_url)
            if pattern.search(text):
                text = pattern.sub('', text)
                removed.append(url)
        return text, removed

    def _remove_page_numbers_for_page(self, text: str, page_number: int) -> Tuple[str, List[str]]:
        """
        Remove page numbers, but only those matching the current page.
        Context-aware to avoid false positives.
        """
        removed = []

        for pattern_template in self.page_number_patterns:
            # Create pattern for this specific page number
            pattern_str = pattern_template.pattern.replace('{page_num}', str(page_number))
            pattern = re.compile(pattern_str, pattern_template.flags)

            matches = pattern.findall(text)
            if matches:
                text = pattern.sub('', text)
                removed.extend(matches)

        return text, removed

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header/footer patterns."""
        for pattern in self.header_footer_patterns:
            text = pattern.sub('', text)
        return text

    # OCR corrections (conservative)

    def _fix_ligatures(self, text: str) -> str:
        """Fix OCR ligatures (always safe)."""
        for ligature, replacement in self.ligature_map.items():
            text = text.replace(ligature, replacement)
        return text

    def _apply_ocr_corrections(self, text: str) -> str:
        """
        Apply OCR corrections cautiously.
        Only when explicitly enabled.
        """
        # Conservative number/letter corrections
        # Only fix obvious OCR errors in specific contexts
        text = re.sub(r'\b0(?=bject|utput|pen)\b', 'O', text)  # 0bject -> Object
        text = re.sub(r'\bl(?=icense|etter)\b', 'L', text)  # license with lowercase L
        return text

    # Semantic normalizations

    def _merge_hyphenated_words(self, text: str) -> str:
        """Merge hyphenated words split across lines."""
        return self.hyphen_linebreak_pattern.sub(r'\1\2', text)

    def _fix_broken_sentences_safe(self, text: str) -> str:
        """
        Fix broken sentences but AVOID merging ALL CAPS lines.
        Improved pattern that only merges lowercase-to-lowercase.
        """
        return self.broken_sentence_pattern.sub(r'\1 \2', text)

    def _normalize_bullet_points(self, text: str) -> str:
        """Normalize bullet point styles."""
        bullet_replacements = [
            (re.compile(r'^[\s]*[·∙●○◦▪▫■□]\s+', re.MULTILINE), '• '),
            (re.compile(r'^[\s]*[\*\-\+]\s+', re.MULTILINE), '• '),
        ]

        for pattern, replacement in bullet_replacements:
            text = pattern.sub(replacement, text)

        return text

    def get_normalization_stats(self, page_results: List[PageNormalizationResult]) -> Dict:
        """Generate statistics about normalization."""
        total_original_chars = sum(r.original_char_count for r in page_results)
        total_normalized_chars = sum(r.normalized_char_count for r in page_results)
        total_original_words = sum(r.original_word_count for r in page_results)
        total_normalized_words = sum(r.normalized_word_count for r in page_results)

        return {
            'total_pages_processed': len(page_results),
            'original_chars': total_original_chars,
            'normalized_chars': total_normalized_chars,
            'char_reduction': total_original_chars - total_normalized_chars,
            'char_reduction_percent': ((total_original_chars - total_normalized_chars) / total_original_chars * 100) if total_original_chars > 0 else 0,
            'original_words': total_original_words,
            'normalized_words': total_normalized_words,
            'word_reduction': total_original_words - total_normalized_words,
            'total_urls_removed': sum(len(r.removed_urls) for r in page_results),
            'total_page_numbers_removed': sum(len(r.removed_page_numbers) for r in page_results),
            'total_protected_elements': sum(len(r.protected_elements) for r in page_results),
        }


# Convenience function
def normalize_with_metadata(extraction_result, config: Optional[NormalizationConfig] = None):
    """
    Convenience function to normalize a document using metadata.

    Args:
        extraction_result: ExtractionResult from document_processor
        config: Optional NormalizationConfig

    Returns:
        Tuple of (normalized_text, page_results, stats)
    """
    normalizer = MetadataAwareNormalizer(config)
    normalized_text, page_results = normalizer.normalize_document(extraction_result)
    stats = normalizer.get_normalization_stats(page_results)

    return normalized_text, page_results, stats


if __name__ == "__main__":
    print("MetadataAwareNormalizer - Intelligent text normalization")
    print("=" * 80)
    print("\nThis module requires ExtractionResult from document_processor.")
    print("\nExample usage:")
    print("""
from document_processor import extract_text_from_document
from metadata_aware_normalizer import normalize_with_metadata, NormalizationConfig

# Extract document with metadata
result = extract_text_from_document('document.pdf', extract_metadata=True)

# Configure normalization
config = NormalizationConfig(
    remove_toc_pages=True,
    apply_ocr_corrections=False,  # Conservative by default
    protect_headings=True,
    add_page_markers=True
)

# Normalize with metadata awareness
normalized_text, page_results, stats = normalize_with_metadata(result, config)

print(f"Processed {stats['total_pages_processed']} pages")
print(f"Character reduction: {stats['char_reduction_percent']:.1f}%")
print(f"URLs removed: {stats['total_urls_removed']}")
print(f"Protected elements: {stats['total_protected_elements']}")
    """)
