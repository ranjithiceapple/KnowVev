"""
Enhanced Query Analyzer

Analyzes queries for:
1. Intent classification (summary vs section-level)
2. Specificity detection (broad vs specific)
3. Summary eligibility
4. Fallback strategies
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from logger_config import get_logger

logger = get_logger(__name__)


class QueryScope(Enum):
    """Query scope level."""
    DOCUMENT_LEVEL = "document_level"  # Overview, summary, entire document
    SECTION_LEVEL = "section_level"    # Specific topics, concepts
    MIXED = "mixed"                     # Both document and section


class QuerySpecificity(Enum):
    """How specific the query is."""
    VERY_BROAD = "very_broad"      # "what is covered", "overview"
    BROAD = "broad"                # "summary of concepts"
    MODERATE = "moderate"          # "explain git staging"
    SPECIFIC = "specific"          # "git staging area explanation"
    VERY_SPECIFIC = "very_specific" # "difference between staging and working directory"


@dataclass
class QueryAnalysis:
    """Complete query analysis result."""
    query: str
    scope: QueryScope
    specificity: QuerySpecificity

    # Summary routing
    should_include_summary: bool
    should_exclude_summary: bool
    summary_only: bool

    # Confidence and scoring
    confidence: float
    specificity_score: float

    # Search strategy
    search_strategy: str  # 'summary_first', 'section_only', 'hybrid', 'section_first'
    recommended_filters: Dict[str, Any]
    fallback_strategy: Optional[str] = None

    # Pattern matches
    patterns_matched: List[str] = None

    def __post_init__(self):
        if self.patterns_matched is None:
            self.patterns_matched = []


class EnhancedQueryAnalyzer:
    """
    Enhanced query analyzer with improved intent detection and routing.

    Fixes issues:
    1. "what is covered in document" → Summary
    2. "git staging area explanation" → Section (not summary)
    3. Better hybrid strategies for mixed queries
    """

    def __init__(self):
        """Initialize analyzer with patterns."""
        self._setup_patterns()
        logger.info("EnhancedQueryAnalyzer initialized")

    def _setup_patterns(self):
        """Setup detection patterns."""

        # Document-level patterns (STRONG indicators for summary)
        self.document_patterns = {
            'overview': [
                re.compile(r'\boverview\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+is\s+covered\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+does\s+(?:this|the)\s+document\s+cover\b', re.IGNORECASE),
                re.compile(r'\bcontent\s+of\s+(?:this|the)\s+document\b', re.IGNORECASE),
                re.compile(r'\bdocument\s+(?:summary|overview|contents)\b', re.IGNORECASE),
            ],
            'summary': [
                re.compile(r'\bsummar(?:y|ize)\b', re.IGNORECASE),
                re.compile(r'\bkey\s+points\b', re.IGNORECASE),
                re.compile(r'\bmain\s+(?:points|topics|concepts)\b', re.IGNORECASE),
                re.compile(r'\bhighlights?\b', re.IGNORECASE),
                re.compile(r'\bin\s+brief\b', re.IGNORECASE),
            ],
            'general': [
                re.compile(r'\bwhat\s+is\s+this\s+about\b', re.IGNORECASE),
                re.compile(r'\btell\s+me\s+about\s+(?:this|the)\s+document\b', re.IGNORECASE),
                re.compile(r'\bexplain\s+(?:this|the)\s+document\b', re.IGNORECASE),
            ],
            'list_all': [
                re.compile(r'\blist\s+all\b', re.IGNORECASE),
                re.compile(r'\bshow\s+(?:me\s+)?all\b', re.IGNORECASE),
                re.compile(r'\ball\s+(?:topics|concepts|sections)\b', re.IGNORECASE),
            ]
        }

        # Section-level patterns (indicators for specific content)
        self.section_patterns = {
            'explanation': [
                re.compile(r'\bexplain(?:ation)?\s+(?:of\s+)?(?!(?:this|the)\s+document)\w+', re.IGNORECASE),
                re.compile(r'\bhow\s+(?:does|do)\s+\w+\s+work', re.IGNORECASE),
                re.compile(r'\bwhat\s+(?:is|are)\s+\w+(?:\s+and\s+\w+)?\s*\??$', re.IGNORECASE),
            ],
            'specific_topic': [
                re.compile(r'\b(?:about|regarding)\s+\w+', re.IGNORECASE),
                re.compile(r'\bdetails?\s+(?:on|about|of)\b', re.IGNORECASE),
                re.compile(r'\bspecific(?:ally)?\s+about\b', re.IGNORECASE),
            ],
            'comparison': [
                re.compile(r'\bdifference\s+between\b', re.IGNORECASE),
                re.compile(r'\bcompare\b', re.IGNORECASE),
                re.compile(r'\bvs\b|\bversus\b', re.IGNORECASE),
            ],
            'multi_concept': [
                re.compile(r'\b\w+\s+(?:and|or)\s+\w+', re.IGNORECASE),
            ]
        }

        # Specificity indicators
        self.specificity_indicators = {
            'very_broad': ['overview', 'covered', 'about', 'document', 'summary', 'general', 'introduction'],
            'broad': ['all', 'list', 'show me', 'what are', 'types of', 'kinds of'],
            'specific': ['explanation', 'details', 'specifically', 'how does', 'why does', 'when to'],
            'very_specific': ['difference between', 'compare', 'vs', 'step by step', 'example of']
        }

        # Summary exclusion patterns (queries that should NOT use summary)
        self.summary_exclusion_patterns = [
            re.compile(r'\bexplain(?:ation)?\s+(?:of\s+)?\w+\s+\w+', re.IGNORECASE),  # "explanation of staging area"
            re.compile(r'\bhow\s+(?:to|do)\b', re.IGNORECASE),                        # "how to use staging"
            re.compile(r'\bsteps?\s+(?:to|for)\b', re.IGNORECASE),                    # "steps to commit"
            re.compile(r'\bdetailed?\b', re.IGNORECASE),                              # "detailed guide"
            re.compile(r'\bin-depth\b', re.IGNORECASE),                               # "in-depth explanation"
            re.compile(r'\bexample(?:s)?\s+of\b', re.IGNORECASE),                     # "examples of"
            re.compile(r'\bcode\s+(?:for|example)\b', re.IGNORECASE),                 # "code for"
        ]

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query to determine optimal search strategy.

        Args:
            query: User query string

        Returns:
            QueryAnalysis with routing recommendations
        """
        if not query or not query.strip():
            return self._default_analysis(query)

        query = query.strip()
        query_lower = query.lower()

        logger.debug(f"Analyzing query: '{query}'")

        # Step 1: Detect scope (document vs section)
        scope, doc_score, sec_score = self._detect_scope(query, query_lower)

        # Step 2: Detect specificity
        specificity, spec_score = self._detect_specificity(query, query_lower)

        # Step 3: Determine summary routing
        should_include_summary, should_exclude_summary, summary_only = self._determine_summary_routing(
            query, query_lower, scope, specificity, doc_score, sec_score
        )

        # Step 4: Determine search strategy
        search_strategy, recommended_filters, fallback = self._determine_strategy(
            scope, specificity, should_include_summary, should_exclude_summary, summary_only
        )

        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(doc_score, sec_score, spec_score)

        analysis = QueryAnalysis(
            query=query,
            scope=scope,
            specificity=specificity,
            should_include_summary=should_include_summary,
            should_exclude_summary=should_exclude_summary,
            summary_only=summary_only,
            confidence=confidence,
            specificity_score=spec_score,
            search_strategy=search_strategy,
            recommended_filters=recommended_filters,
            fallback_strategy=fallback
        )

        logger.info(
            f"Query analyzed - Scope: {scope.value}, Specificity: {specificity.value}, "
            f"Strategy: {search_strategy}, Confidence: {confidence:.2f}"
        )

        return analysis

    def _detect_scope(self, query: str, query_lower: str) -> Tuple[QueryScope, float, float]:
        """Detect query scope (document vs section level)."""
        doc_score = 0.0
        sec_score = 0.0

        # Check document-level patterns
        for category, patterns in self.document_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    if category == 'overview':
                        doc_score += 1.0
                    elif category == 'summary':
                        doc_score += 0.9
                    elif category == 'general':
                        doc_score += 0.7
                    elif category == 'list_all':
                        doc_score += 0.6

        # Check section-level patterns
        for category, patterns in self.section_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    if category == 'explanation':
                        sec_score += 0.8
                    elif category == 'specific_topic':
                        sec_score += 0.7
                    elif category == 'comparison':
                        sec_score += 0.9
                    elif category == 'multi_concept':
                        sec_score += 0.5

        # Determine scope
        if doc_score > sec_score + 0.3:
            scope = QueryScope.DOCUMENT_LEVEL
        elif sec_score > doc_score + 0.3:
            scope = QueryScope.SECTION_LEVEL
        else:
            scope = QueryScope.MIXED

        return scope, doc_score, sec_score

    def _detect_specificity(self, query: str, query_lower: str) -> Tuple[QuerySpecificity, float]:
        """Detect how specific the query is."""
        word_count = len(query.split())

        # Count specificity indicators
        very_broad_count = sum(1 for word in self.specificity_indicators['very_broad'] if word in query_lower)
        broad_count = sum(1 for word in self.specificity_indicators['broad'] if word in query_lower)
        specific_count = sum(1 for word in self.specificity_indicators['specific'] if word in query_lower)
        very_specific_count = sum(1 for word in self.specificity_indicators['very_specific'] if word in query_lower)

        # Calculate specificity score (0.0 = very broad, 1.0 = very specific)
        spec_score = 0.5  # Default

        if very_broad_count > 0:
            spec_score -= 0.3
        if broad_count > 0:
            spec_score -= 0.15
        if specific_count > 0:
            spec_score += 0.2
        if very_specific_count > 0:
            spec_score += 0.3

        # Adjust based on word count (more words = more specific)
        if word_count <= 3:
            spec_score -= 0.1
        elif word_count >= 8:
            spec_score += 0.15

        # Clamp score
        spec_score = max(0.0, min(1.0, spec_score))

        # Determine specificity level
        if spec_score < 0.2:
            specificity = QuerySpecificity.VERY_BROAD
        elif spec_score < 0.4:
            specificity = QuerySpecificity.BROAD
        elif spec_score < 0.6:
            specificity = QuerySpecificity.MODERATE
        elif spec_score < 0.8:
            specificity = QuerySpecificity.SPECIFIC
        else:
            specificity = QuerySpecificity.VERY_SPECIFIC

        return specificity, spec_score

    def _determine_summary_routing(
        self,
        query: str,
        query_lower: str,
        scope: QueryScope,
        specificity: QuerySpecificity,
        doc_score: float,
        sec_score: float
    ) -> Tuple[bool, bool, bool]:
        """Determine if/how to use summary chunks."""

        # Check for summary exclusion patterns
        should_exclude = any(pattern.search(query) for pattern in self.summary_exclusion_patterns)

        # Document-level queries should use summary
        if scope == QueryScope.DOCUMENT_LEVEL:
            return True, False, True  # include, not exclude, summary only

        # Very broad queries should include summary
        if specificity in [QuerySpecificity.VERY_BROAD, QuerySpecificity.BROAD]:
            if not should_exclude:
                return True, False, False  # include summary, but not only

        # Specific queries should exclude summary
        if specificity in [QuerySpecificity.SPECIFIC, QuerySpecificity.VERY_SPECIFIC]:
            return False, True, False  # don't include, exclude, not only

        # Moderate specificity with strong section indicators
        if specificity == QuerySpecificity.MODERATE:
            if sec_score > 0.5:
                return False, True, False  # section-focused query
            else:
                return True, False, False  # can include summary

        # Mixed scope
        if scope == QueryScope.MIXED:
            return True, False, False  # include both

        # Default: include summary if no exclusion
        return not should_exclude, should_exclude, False

    def _determine_strategy(
        self,
        scope: QueryScope,
        specificity: QuerySpecificity,
        should_include_summary: bool,
        should_exclude_summary: bool,
        summary_only: bool
    ) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Determine search strategy and filters."""

        filters = {}
        fallback = None

        # Summary only
        if summary_only:
            strategy = 'summary_first'
            filters = {"section_title": "[DOCUMENT SUMMARY]"}
            fallback = 'section_fallback'  # If no summary results, try sections

        # Exclude summary (section only)
        elif should_exclude_summary:
            strategy = 'section_only'
            # Negative filter to exclude summaries
            # Note: Qdrant doesn't support NOT filters directly, so we handle this in post-processing
            fallback = None

        # Include summary but not exclusively
        elif should_include_summary:
            if specificity in [QuerySpecificity.VERY_BROAD, QuerySpecificity.BROAD]:
                strategy = 'summary_first'
                fallback = 'section_fallback'
            else:
                strategy = 'hybrid'
                fallback = None

        # Default to hybrid
        else:
            strategy = 'hybrid'
            fallback = None

        return strategy, filters, fallback

    def _calculate_confidence(self, doc_score: float, sec_score: float, spec_score: float) -> float:
        """Calculate overall confidence in the analysis."""

        # Higher difference = higher confidence
        score_diff = abs(doc_score - sec_score)

        # Extreme specificity scores also indicate confidence
        spec_confidence = abs(spec_score - 0.5) * 2  # 0.0 at 0.5, 1.0 at extremes

        # Combine
        confidence = (score_diff + spec_confidence) / 2

        # Clamp
        return max(0.3, min(1.0, confidence))

    def _default_analysis(self, query: str) -> QueryAnalysis:
        """Default analysis for empty/invalid queries."""
        return QueryAnalysis(
            query=query,
            scope=QueryScope.MIXED,
            specificity=QuerySpecificity.MODERATE,
            should_include_summary=True,
            should_exclude_summary=False,
            summary_only=False,
            confidence=0.5,
            specificity_score=0.5,
            search_strategy='hybrid',
            recommended_filters={},
            fallback_strategy=None
        )


def analyze_query(query: str) -> QueryAnalysis:
    """Convenience function to analyze a query."""
    analyzer = EnhancedQueryAnalyzer()
    return analyzer.analyze(query)


# Testing
if __name__ == "__main__":
    print("Enhanced Query Analyzer - Test Cases")
    print("=" * 80)

    test_queries = [
        "overview of git cheat sheet",
        "what is covered in document",
        "summary of git commands",
        "git staging area explanation",
        "git working directory and staging area",
        "how to commit changes",
        "difference between merge and rebase",
        "what is git",
        "list all git commands",
        "detailed explanation of branching",
    ]

    analyzer = EnhancedQueryAnalyzer()

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        analysis = analyzer.analyze(query)
        print(f"  Scope: {analysis.scope.value}")
        print(f"  Specificity: {analysis.specificity.value}")
        print(f"  Strategy: {analysis.search_strategy}")
        print(f"  Summary only: {analysis.summary_only}")
        print(f"  Exclude summary: {analysis.should_exclude_summary}")
        print(f"  Confidence: {analysis.confidence:.2f}")
        if analysis.recommended_filters:
            print(f"  Filters: {analysis.recommended_filters}")
        if analysis.fallback_strategy:
            print(f"  Fallback: {analysis.fallback_strategy}")
