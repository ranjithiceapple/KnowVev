"""
Text Normalization and Cleanup Module

This module provides comprehensive text normalization and cleanup functionality
organized into specialized classes by category.
"""

import re
import unicodedata
from typing import List, Set, Tuple, Optional
from collections import defaultdict


class StructuralNormalizer:
    """
    Handles structural text normalization tasks including line breaks,
    whitespace, and unicode normalization.
    """

    def __init__(self):
        # Pattern for hyphenated line breaks
        self.hyphen_linebreak_pattern = re.compile(r'(\w+)-\s*\n\s*(\w+)')
        # Pattern for multiple whitespace
        self.whitespace_pattern = re.compile(r'[ \t]+')
        # Pattern for multiple blank lines
        self.blank_lines_pattern = re.compile(r'\n\s*\n\s*\n+')

    def normalize_line_breaks(self, text: str) -> str:
        """
        Normalize different types of line breaks to consistent newlines.
        Converts Windows (CRLF) and Mac (CR) line endings to Unix (LF).

        Args:
            text: Input text with mixed line endings

        Returns:
            Text with normalized line breaks
        """
        # Convert Windows CRLF to LF
        text = text.replace('\r\n', '\n')
        # Convert Mac CR to LF
        text = text.replace('\r', '\n')
        return text

    def remove_hyphen_line_breaks(self, text: str) -> str:
        """
        Remove hyphenation at line breaks, merging split words.
        Example: "exam-\nple" becomes "example"

        Args:
            text: Input text with hyphenated line breaks

        Returns:
            Text with merged hyphenated words
        """
        return self.hyphen_linebreak_pattern.sub(r'\1\2', text)

    def collapse_whitespace(self, text: str) -> str:
        """
        Collapse multiple spaces and tabs into single spaces.
        Preserves newlines.

        Args:
            text: Input text with excessive whitespace

        Returns:
            Text with collapsed whitespace
        """
        # Split by lines to preserve line structure
        lines = text.split('\n')
        # Collapse whitespace in each line
        normalized_lines = [self.whitespace_pattern.sub(' ', line) for line in lines]
        return '\n'.join(normalized_lines)

    def collapse_blank_lines(self, text: str, max_consecutive: int = 2) -> str:
        """
        Collapse multiple consecutive blank lines into a maximum number.

        Args:
            text: Input text with multiple blank lines
            max_consecutive: Maximum number of consecutive blank lines to keep

        Returns:
            Text with collapsed blank lines
        """
        replacement = '\n' * max_consecutive
        return self.blank_lines_pattern.sub(replacement, text)

    def unicode_normalize(self, text: str, form: str = 'NFKC') -> str:
        """
        Normalize unicode characters using specified normalization form.
        NFKC is recommended for most text processing (compatibility decomposition + composition).

        Args:
            text: Input text with various unicode representations
            form: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')

        Returns:
            Unicode-normalized text
        """
        return unicodedata.normalize(form, text)

    def normalize_all(self, text: str) -> str:
        """
        Apply all structural normalizations in sequence.

        Args:
            text: Input text

        Returns:
            Fully normalized text
        """
        text = self.normalize_line_breaks(text)
        text = self.remove_hyphen_line_breaks(text)
        text = self.collapse_whitespace(text)
        text = self.collapse_blank_lines(text)
        text = self.unicode_normalize(text)
        return text


class NoiseRemover:
    """
    Removes common noise patterns from documents including headers, footers,
    page numbers, URLs, and table of contents artifacts.
    """

    def __init__(self):
        # Page number patterns
        self.page_number_patterns = [
            re.compile(r'^\s*-?\s*\d+\s*-?\s*$', re.MULTILINE),  # Standalone page numbers
            re.compile(r'^\s*Page\s+\d+\s*$', re.MULTILINE | re.IGNORECASE),  # "Page 123"
            re.compile(r'^\s*\d+\s+of\s+\d+\s*$', re.MULTILINE | re.IGNORECASE),  # "1 of 10"
        ]

        # Header/footer patterns
        self.header_footer_patterns = [
            re.compile(r'^\s*(?:confidential|proprietary|draft|internal use only)\s*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\s*©\s*\d{4}.*$', re.MULTILINE),  # Copyright lines
            re.compile(r'^\s*\d{4}\s*©.*$', re.MULTILINE),  # Copyright lines (year first)
        ]

        # URL patterns
        self.url_pattern = re.compile(
            r'(?:https?://|www\.)[^\s]+|'  # HTTP/HTTPS URLs
            r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b'  # Domain-based URLs
        )

        # TOC patterns
        self.toc_patterns = [
            re.compile(r'^.{5,}\.{3,}\s*\d+\s*$', re.MULTILINE),  # "Chapter Title ... 123"
            re.compile(r'^\d+\.\d+\s+.{5,}\s+\.{3,}\s*\d+\s*$', re.MULTILINE),  # "1.2 Title ... 45"
            re.compile(r'^.{5,}\s{3,}\d+\s*$', re.MULTILINE),  # "Chapter Title    123"
        ]

    def remove_headers_footers(self, text: str, custom_patterns: Optional[List[str]] = None) -> str:
        """
        Remove common header and footer patterns.

        Args:
            text: Input text
            custom_patterns: Optional list of regex patterns for custom headers/footers

        Returns:
            Text with headers/footers removed
        """
        for pattern in self.header_footer_patterns:
            text = pattern.sub('', text)

        # Apply custom patterns if provided
        if custom_patterns:
            for custom_pattern in custom_patterns:
                text = re.sub(custom_pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

        return text

    def remove_page_numbers(self, text: str) -> str:
        """
        Remove standalone page numbers from text.

        Args:
            text: Input text with page numbers

        Returns:
            Text with page numbers removed
        """
        for pattern in self.page_number_patterns:
            text = pattern.sub('', text)
        return text

    def remove_urls(self, text: str, replace_with: str = '') -> str:
        """
        Remove URLs from text.

        Args:
            text: Input text containing URLs
            replace_with: String to replace URLs with (default: empty string)

        Returns:
            Text with URLs removed
        """
        return self.url_pattern.sub(replace_with, text)

    def remove_toc_artifacts(self, text: str) -> str:
        """
        Remove table of contents artifacts like dot leaders and page references.

        Args:
            text: Input text with TOC artifacts

        Returns:
            Text with TOC artifacts removed
        """
        for pattern in self.toc_patterns:
            text = pattern.sub('', text)
        return text

    def remove_all(self, text: str, custom_patterns: Optional[List[str]] = None) -> str:
        """
        Apply all noise removal operations.

        Args:
            text: Input text
            custom_patterns: Optional list of custom patterns for headers/footers

        Returns:
            Text with all noise removed
        """
        text = self.remove_headers_footers(text, custom_patterns)
        text = self.remove_page_numbers(text)
        text = self.remove_urls(text)
        text = self.remove_toc_artifacts(text)
        return text


class SemanticNormalizer:
    """
    Handles semantic text normalization including sentence reconstruction,
    hyphenated word merging, column cleanup, and bullet point normalization.
    """

    def __init__(self):
        # Pattern for broken sentences (sentence ending without punctuation)
        self.broken_sentence_pattern = re.compile(r'([a-z,])\n+([A-Z])')
        # Pattern for hyphenated words at line breaks
        self.hyphenated_word_pattern = re.compile(r'(\w+)-\s*\n+\s*(\w+)')
        # Pattern for multiple columns (detected by multiple spaces)
        self.column_pattern = re.compile(r'\s{4,}')
        # Pattern for various bullet point styles
        self.bullet_patterns = [
            (re.compile(r'^\s*[•·∙●○◦▪▫■□]\s*'), '• '),  # Unicode bullets
            (re.compile(r'^\s*[\*\-\+]\s+'), '• '),  # ASCII bullets
            (re.compile(r'^\s*\d+[\.\)]\s+'), lambda m: '• '),  # Numbered lists
            (re.compile(r'^\s*[a-z][\.\)]\s+', re.IGNORECASE), '• '),  # Lettered lists
        ]
        # Pattern for section titles (all caps, numbered sections)
        self.section_title_pattern = re.compile(
            r'^(?:'
            r'(?:[A-Z][A-Z\s]{5,})|'  # ALL CAPS TITLES
            r'(?:(?:Chapter|Section|Part|Article)\s+\d+[:\s]+[A-Z][^.!?]*)|'  # Chapter 1: Title
            r'(?:\d+\.\s+[A-Z][^.!?]*)'  # 1. Title
            r')$',
            re.MULTILINE
        )

    def fix_broken_sentences(self, text: str) -> str:
        """
        Fix sentences broken across lines without proper punctuation.
        Merges lines where a lowercase letter is followed by an uppercase letter.

        Args:
            text: Input text with broken sentences

        Returns:
            Text with fixed sentences
        """
        # Replace line break between lowercase and uppercase with space
        return self.broken_sentence_pattern.sub(r'\1 \2', text)

    def merge_hyphenated_words(self, text: str) -> str:
        """
        Merge hyphenated words split across lines.
        Example: "under-\nstand" becomes "understand"

        Args:
            text: Input text with hyphenated words

        Returns:
            Text with merged hyphenated words
        """
        return self.hyphenated_word_pattern.sub(r'\1\2', text)

    def column_detection_cleanup(self, text: str) -> str:
        """
        Clean up multi-column text layouts by converting to single column.
        Detects columns by multiple consecutive spaces.

        Args:
            text: Input text with column layout

        Returns:
            Single-column text
        """
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Split by large gaps (potential columns)
            if self.column_pattern.search(line):
                # Replace column separators with newlines
                parts = self.column_pattern.split(line)
                cleaned_lines.extend([part.strip() for part in parts if part.strip()])
            else:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def normalize_bullet_points(self, text: str) -> str:
        """
        Normalize various bullet point styles to consistent format.

        Args:
            text: Input text with various bullet styles

        Returns:
            Text with normalized bullet points
        """
        lines = text.split('\n')
        normalized_lines = []

        for line in lines:
            normalized_line = line
            # Try each bullet pattern
            for pattern, replacement in self.bullet_patterns:
                if pattern.match(line):
                    if callable(replacement):
                        normalized_line = pattern.sub(replacement, line, count=1)
                    else:
                        normalized_line = pattern.sub(replacement, line, count=1)
                    break
            normalized_lines.append(normalized_line)

        return '\n'.join(normalized_lines)

    def preserve_section_titles(self, text: str) -> str:
        """
        Identify and preserve section titles by adding markers.
        Adds double newlines before and after section titles for clarity.

        Args:
            text: Input text

        Returns:
            Text with preserved section titles
        """
        lines = text.split('\n')
        preserved_lines = []

        for i, line in enumerate(lines):
            if self.section_title_pattern.match(line.strip()):
                # Add spacing around section titles
                if i > 0 and preserved_lines and preserved_lines[-1].strip():
                    preserved_lines.append('')
                preserved_lines.append(line)
                if i < len(lines) - 1:
                    preserved_lines.append('')
            else:
                preserved_lines.append(line)

        return '\n'.join(preserved_lines)

    def normalize_all(self, text: str) -> str:
        """
        Apply all semantic normalizations in sequence.

        Args:
            text: Input text

        Returns:
            Semantically normalized text
        """
        text = self.merge_hyphenated_words(text)
        text = self.fix_broken_sentences(text)
        text = self.column_detection_cleanup(text)
        text = self.normalize_bullet_points(text)
        text = self.preserve_section_titles(text)
        return text


class OCRCorrector:
    """
    Corrects common OCR (Optical Character Recognition) errors including
    duplicated lines, ligature issues, character confusion, and artifacts.
    """

    def __init__(self):
        # Common OCR ligatures that need fixing
        self.ligature_map = {
            'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            'ﬅ': 'st', 'ﬆ': 'st', 'Ꜳ': 'AA', 'ꜳ': 'aa', 'Æ': 'AE', 'æ': 'ae',
            'Œ': 'OE', 'œ': 'oe'
        }

        # Common OCR character confusions
        self.ocr_confusion_map = {
            # Number/letter confusions
            r'\b0(?=[a-zA-Z])': 'O',  # 0 -> O before letters
            r'(?<=[a-zA-Z])0\b': 'O',  # 0 -> O after letters
            r'\b1(?=[a-zA-Z])': 'I',  # 1 -> I before letters (context-dependent)
            r'\b5(?=[a-zA-Z])': 'S',  # 5 -> S before letters
            r'\b8(?=[a-zA-Z])': 'B',  # 8 -> B before letters
            r'(?<=[a-zA-Z])1(?=[a-zA-Z])': 'l',  # 1 -> l between letters
        }

        # Weird OCR characters to remove
        self.weird_char_pattern = re.compile(
            r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]|'  # Control characters
            r'[■□●○▪▫](?![•\s])|'  # Box/bullet characters not used as bullets
            r'[‗‾⁓∼∽⁀⁔〰]|'  # Various line artifacts
            r'[┌┐└┘├┤┬┴┼│─═║╔╗╚╝╠╣╦╩╬]'  # Box drawing characters
        )

    def remove_duplicated_ocr_lines(self, text: str, similarity_threshold: float = 0.9) -> str:
        """
        Remove duplicated OCR lines that appear consecutively.
        OCR sometimes produces duplicate lines due to scanning artifacts.

        Args:
            text: Input text with potential duplicates
            similarity_threshold: Threshold for considering lines as duplicates (0-1)

        Returns:
            Text with duplicate lines removed
        """
        lines = text.split('\n')
        if not lines:
            return text

        deduplicated = [lines[0]]

        for i in range(1, len(lines)):
            current = lines[i].strip()
            previous = deduplicated[-1].strip()

            # Skip if current line is empty
            if not current:
                deduplicated.append(lines[i])
                continue

            # Calculate similarity using simple character overlap
            if previous:
                similarity = self._calculate_similarity(current, previous)
                if similarity < similarity_threshold:
                    deduplicated.append(lines[i])
            else:
                deduplicated.append(lines[i])

        return '\n'.join(deduplicated)

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple character-level similarity between two strings."""
        if not str1 or not str2:
            return 0.0

        # Simple character overlap ratio
        set1, set2 = set(str1.lower()), set(str2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def fix_ligatures(self, text: str) -> str:
        """
        Fix OCR ligatures by replacing them with standard character sequences.

        Args:
            text: Input text with ligatures

        Returns:
            Text with ligatures replaced
        """
        for ligature, replacement in self.ligature_map.items():
            text = text.replace(ligature, replacement)
        return text

    def fix_ocr_confusion(self, text: str) -> str:
        """
        Fix common OCR character confusions between numbers and letters.

        Args:
            text: Input text with OCR confusion

        Returns:
            Text with fixed character confusion
        """
        for pattern, replacement in self.ocr_confusion_map.items():
            text = re.sub(pattern, replacement, text)
        return text

    def remove_weird_chars(self, text: str) -> str:
        """
        Remove weird OCR artifacts and control characters.

        Args:
            text: Input text with weird characters

        Returns:
            Text with artifacts removed
        """
        return self.weird_char_pattern.sub('', text)

    def correct_all(self, text: str) -> str:
        """
        Apply all OCR corrections in sequence.

        Args:
            text: Input text

        Returns:
            OCR-corrected text
        """
        text = self.remove_duplicated_ocr_lines(text)
        text = self.fix_ligatures(text)
        text = self.fix_ocr_confusion(text)
        text = self.remove_weird_chars(text)
        return text


class DomainNormalizer:
    """
    Handles domain-specific normalization including disclaimer removal
    and date/number standardization.
    """

    def __init__(self):
        # Common disclaimer patterns
        self.disclaimer_patterns = [
            re.compile(
                r'(?:^|\n)\s*(?:disclaimer|legal notice|confidentiality notice)[:\s]*'
                r'.{0,500}?(?=\n\n|\n[A-Z]|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
            re.compile(
                r'(?:^|\n)\s*this (?:document|email|message) is (?:confidential|proprietary)'
                r'.{0,300}?(?=\n\n|\n[A-Z]|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
            re.compile(
                r'(?:^|\n)\s*(?:please consider the environment|think before you print)'
                r'.{0,200}?(?=\n\n|\n[A-Z]|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
        ]

        # Date patterns for normalization
        self.date_patterns = [
            # MM/DD/YYYY or DD/MM/YYYY
            (re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'), r'\1/\2/\3'),
            # Month DD, YYYY
            (re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE), r'\1 \2, \3'),
            # DD Month YYYY
            (re.compile(r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', re.IGNORECASE), r'\1 \2 \3'),
        ]

        # Number patterns for normalization
        self.number_patterns = [
            # Thousands separator normalization (1,234.56 or 1.234,56)
            (re.compile(r'\b(\d{1,3}(?:,\d{3})+)(?:\.\d+)?\b'), 'thousands_comma'),
            (re.compile(r'\b(\d{1,3}(?:\.\d{3})+)(?:,\d+)?\b'), 'thousands_period'),
        ]

    def remove_disclaimers(self, text: str, custom_patterns: Optional[List[str]] = None) -> str:
        """
        Remove common legal disclaimers and confidentiality notices.

        Args:
            text: Input text with disclaimers
            custom_patterns: Optional list of custom disclaimer regex patterns

        Returns:
            Text with disclaimers removed
        """
        for pattern in self.disclaimer_patterns:
            text = pattern.sub('', text)

        # Apply custom patterns if provided
        if custom_patterns:
            for custom_pattern in custom_patterns:
                text = re.sub(custom_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

        return text

    def normalize_dates(self, text: str, target_format: str = 'standard') -> str:
        """
        Normalize date formats to a consistent standard.

        Args:
            text: Input text with various date formats
            target_format: Target format ('standard' keeps readable format)

        Returns:
            Text with normalized dates
        """
        for pattern, replacement in self.date_patterns:
            text = pattern.sub(replacement, text)
        return text

    def normalize_numbers(self, text: str, target_format: str = 'comma') -> str:
        """
        Normalize number formats (thousands separators, decimals).

        Args:
            text: Input text with various number formats
            target_format: 'comma' for US style (1,234.56) or 'period' for EU style (1.234,56)

        Returns:
            Text with normalized numbers
        """
        # This is a simplified version - full implementation would need more context
        # to avoid false positives with decimals vs thousands separators
        return text

    def normalize_all(self, text: str, custom_disclaimer_patterns: Optional[List[str]] = None) -> str:
        """
        Apply all domain normalizations in sequence.

        Args:
            text: Input text
            custom_disclaimer_patterns: Optional custom patterns for disclaimers

        Returns:
            Domain-normalized text
        """
        text = self.remove_disclaimers(text, custom_disclaimer_patterns)
        text = self.normalize_dates(text)
        text = self.normalize_numbers(text)
        return text


class TextNormalizationPipeline:
    """
    Comprehensive text normalization pipeline that combines all normalizers.
    Provides convenient interface for applying normalizations in proper sequence.
    """

    def __init__(self):
        self.structural = StructuralNormalizer()
        self.noise = NoiseRemover()
        self.semantic = SemanticNormalizer()
        self.ocr = OCRCorrector()
        self.domain = DomainNormalizer()

    def normalize(
        self,
        text: str,
        apply_structural: bool = True,
        apply_noise_removal: bool = True,
        apply_semantic: bool = True,
        apply_ocr_correction: bool = True,
        apply_domain: bool = True,
        custom_header_patterns: Optional[List[str]] = None,
        custom_disclaimer_patterns: Optional[List[str]] = None
    ) -> str:
        """
        Apply comprehensive text normalization pipeline.

        Args:
            text: Input text to normalize
            apply_structural: Apply structural normalizations
            apply_noise_removal: Apply noise removal
            apply_semantic: Apply semantic normalizations
            apply_ocr_correction: Apply OCR corrections
            apply_domain: Apply domain-specific normalizations
            custom_header_patterns: Custom patterns for headers/footers
            custom_disclaimer_patterns: Custom patterns for disclaimers

        Returns:
            Fully normalized text
        """
        if apply_structural:
            text = self.structural.normalize_all(text)

        if apply_noise_removal:
            text = self.noise.remove_all(text, custom_header_patterns)

        if apply_ocr_correction:
            text = self.ocr.correct_all(text)

        if apply_semantic:
            text = self.semantic.normalize_all(text)

        if apply_domain:
            text = self.domain.normalize_all(text, custom_disclaimer_patterns)

        # Final cleanup
        text = text.strip()

        return text


# Example usage
if __name__ == "__main__":
    # Sample text with various issues
    sample_text = """
    Page 1

    CHAPTER 1: INTRODUCTION

    This is an exam-
    ple of text that has been scanned.  It  contains   multiple    spaces.




    Some sentences are broken
    across lines without proper spacing.The OCR also created 0CR errors with
    numb3rs and l3tters.

    • Bullet point one
    * Bullet point two
    - Bullet point three

    Visit our website at https://example.com for more info.

    Page 2

    DISCLAIMER: This document is confidential and proprietary.

    The ligature ﬁ and ﬂ need to be ﬁxed.

    Date: 12/31/2023

    © 2023 Company Name. All rights reserved.
    """

    print("=" * 80)
    print("ORIGINAL TEXT:")
    print("=" * 80)
    print(sample_text)

    # Test individual normalizers
    print("\n" + "=" * 80)
    print("STRUCTURAL NORMALIZATION:")
    print("=" * 80)
    structural = StructuralNormalizer()
    text_structural = structural.normalize_all(sample_text)
    print(text_structural)

    print("\n" + "=" * 80)
    print("NOISE REMOVAL:")
    print("=" * 80)
    noise = NoiseRemover()
    text_noise = noise.remove_all(text_structural)
    print(text_noise)

    print("\n" + "=" * 80)
    print("OCR CORRECTION:")
    print("=" * 80)
    ocr = OCRCorrector()
    text_ocr = ocr.correct_all(text_noise)
    print(text_ocr)

    print("\n" + "=" * 80)
    print("SEMANTIC NORMALIZATION:")
    print("=" * 80)
    semantic = SemanticNormalizer()
    text_semantic = semantic.normalize_all(text_ocr)
    print(text_semantic)

    print("\n" + "=" * 80)
    print("DOMAIN NORMALIZATION:")
    print("=" * 80)
    domain = DomainNormalizer()
    text_domain = domain.normalize_all(text_semantic)
    print(text_domain)

    print("\n" + "=" * 80)
    print("FULL PIPELINE:")
    print("=" * 80)
    pipeline = TextNormalizationPipeline()
    normalized_text = pipeline.normalize(sample_text)
    print(normalized_text)
