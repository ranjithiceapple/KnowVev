"""
Query Intent Classification Module

Classifies user queries into different intents to optimize search and retrieval:
- Factual questions (what, who, when, where)
- How-to questions
- Definition/explanation
- Comparison
- Code/technical
- Summary/overview
- Troubleshooting
- Opinion/recommendation

Uses pattern-based and keyword-based classification for fast, deterministic results.
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from logger_config import get_logger

logger = get_logger(__name__)


class QueryIntent(Enum):
    """Query intent types."""
    FACTUAL = "factual"                    # Who, what, when, where questions
    HOW_TO = "how_to"                      # How-to, tutorial, guide
    DEFINITION = "definition"              # What is, define, explain
    COMPARISON = "comparison"              # Compare, vs, difference
    CODE_TECHNICAL = "code_technical"      # Code examples, API, syntax
    SUMMARY = "summary"                    # Overview, summarize, list
    TROUBLESHOOTING = "troubleshooting"    # Error, fix, problem, issue
    RECOMMENDATION = "recommendation"      # Best, should, recommend
    PROCEDURAL = "procedural"              # Steps, process, workflow
    CONCEPTUAL = "conceptual"              # Why, concept, theory
    GENERAL = "general"                    # Catch-all for unclear intent


@dataclass
class IntentClassification:
    """Result of intent classification."""
    primary_intent: QueryIntent
    confidence: float  # 0.0 to 1.0
    secondary_intents: List[Tuple[QueryIntent, float]] = None
    keywords: List[str] = None
    patterns_matched: List[str] = None

    # Search strategy recommendations
    recommended_filters: Dict[str, Any] = None
    recommended_limit: int = 10
    recommended_score_threshold: float = 0.5

    def __post_init__(self):
        if self.secondary_intents is None:
            self.secondary_intents = []
        if self.keywords is None:
            self.keywords = []
        if self.patterns_matched is None:
            self.patterns_matched = []
        if self.recommended_filters is None:
            self.recommended_filters = {}


class QueryIntentClassifier:
    """
    Classifies query intent using pattern matching and keyword detection.

    Fast, deterministic, and interpretable classification suitable for production.
    """

    def __init__(self):
        """Initialize classifier with patterns and keywords."""
        self._setup_patterns()
        self._setup_keywords()
        logger.info("QueryIntentClassifier initialized")

    def _setup_patterns(self):
        """Setup regex patterns for intent detection."""
        self.patterns = {
            QueryIntent.FACTUAL: [
                re.compile(r'\b(what|who|when|where|which)\s+(is|are|was|were)\b', re.IGNORECASE),
                re.compile(r'\b(tell me|show me)\s+(about|the)\b', re.IGNORECASE),
                re.compile(r'\blist\s+(all|the)\b', re.IGNORECASE),
            ],

            QueryIntent.HOW_TO: [
                re.compile(r'\bhow\s+(to|do|can|does)\b', re.IGNORECASE),
                re.compile(r'\b(guide|tutorial|walkthrough)\b', re.IGNORECASE),
                re.compile(r'\b(step by step|steps to)\b', re.IGNORECASE),
                re.compile(r'\b(teach|learn|show me how)\b', re.IGNORECASE),
            ],

            QueryIntent.DEFINITION: [
                re.compile(r'\bwhat\s+(is|are)\s+(a|an|the)?\s*\w+\??$', re.IGNORECASE),
                re.compile(r'\b(define|definition|meaning of)\b', re.IGNORECASE),
                re.compile(r'\bexplain\s+(what|the)\b', re.IGNORECASE),
                re.compile(r'\bmeans\b', re.IGNORECASE),
            ],

            QueryIntent.COMPARISON: [
                re.compile(r'\b(compare|comparison|vs|versus)\b', re.IGNORECASE),
                re.compile(r'\b(difference|different)\s+between\b', re.IGNORECASE),
                re.compile(r'\b(better|worse)\s+than\b', re.IGNORECASE),
                re.compile(r'\bwhich\s+(is|are)\s+(better|best|faster)\b', re.IGNORECASE),
            ],

            QueryIntent.CODE_TECHNICAL: [
                re.compile(r'\b(code|syntax|api|function|method|class)\b', re.IGNORECASE),
                re.compile(r'\b(implement|implementation|example code)\b', re.IGNORECASE),
                re.compile(r'\b(snippet|sample|demo)\b', re.IGNORECASE),
                re.compile(r'\b(import|export|endpoint|query)\b', re.IGNORECASE),
            ],

            QueryIntent.SUMMARY: [
                re.compile(r'\b(summary|summarize|overview)\b', re.IGNORECASE),
                re.compile(r'\b(brief|briefly|in short)\b', re.IGNORECASE),
                re.compile(r'\b(key points|main points|highlights)\b', re.IGNORECASE),
                re.compile(r'\bgive me (an overview|a summary)\b', re.IGNORECASE),
            ],

            QueryIntent.TROUBLESHOOTING: [
                re.compile(r'\b(error|exception|bug|issue|problem)\b', re.IGNORECASE),
                re.compile(r'\b(fix|solve|resolve|debug)\b', re.IGNORECASE),
                re.compile(r'\b(not working|doesn\'t work|won\'t work|fails)\b', re.IGNORECASE),
                re.compile(r'\b(troubleshoot|diagnose)\b', re.IGNORECASE),
                re.compile(r'\bwhy\s+(is|does|doesn\'t|won\'t)\b', re.IGNORECASE),
            ],

            QueryIntent.RECOMMENDATION: [
                re.compile(r'\b(best|better|recommend|suggestion)\b', re.IGNORECASE),
                re.compile(r'\b(should I|which one|what\'s the best)\b', re.IGNORECASE),
                re.compile(r'\b(prefer|preferable|advised)\b', re.IGNORECASE),
                re.compile(r'\b(good|ideal|optimal)\s+(for|to)\b', re.IGNORECASE),
            ],

            QueryIntent.PROCEDURAL: [
                re.compile(r'\b(process|procedure|workflow|pipeline)\b', re.IGNORECASE),
                re.compile(r'\b(steps|phases|stages)\b', re.IGNORECASE),
                re.compile(r'\bfirst.*then.*finally\b', re.IGNORECASE),
                re.compile(r'\b(sequence|order|flow)\b', re.IGNORECASE),
            ],

            QueryIntent.CONCEPTUAL: [
                re.compile(r'\bwhy\s+(is|does|do|are)\b', re.IGNORECASE),
                re.compile(r'\b(concept|theory|principle|idea)\b', re.IGNORECASE),
                re.compile(r'\b(understand|understanding|rationale)\b', re.IGNORECASE),
                re.compile(r'\b(behind|underlying|fundamental)\b', re.IGNORECASE),
            ],
        }

    def _setup_keywords(self):
        """Setup keyword lists for intent detection."""
        self.keywords = {
            QueryIntent.FACTUAL: [
                'what', 'who', 'when', 'where', 'which', 'list',
                'show', 'tell', 'display', 'give', 'provide'
            ],

            QueryIntent.HOW_TO: [
                'how', 'guide', 'tutorial', 'walkthrough', 'steps',
                'procedure', 'instructions', 'teach', 'learn'
            ],

            QueryIntent.DEFINITION: [
                'define', 'definition', 'meaning', 'means', 'explain',
                'explanation', 'clarify', 'describe'
            ],

            QueryIntent.COMPARISON: [
                'compare', 'comparison', 'vs', 'versus', 'difference',
                'different', 'similar', 'similarity', 'better', 'worse'
            ],

            QueryIntent.CODE_TECHNICAL: [
                'code', 'syntax', 'api', 'function', 'method', 'class',
                'implement', 'implementation', 'snippet', 'example',
                'sample', 'demo', 'library', 'framework'
            ],

            QueryIntent.SUMMARY: [
                'summary', 'summarize', 'overview', 'brief', 'briefly',
                'key points', 'main points', 'highlights', 'abstract'
            ],

            QueryIntent.TROUBLESHOOTING: [
                'error', 'exception', 'bug', 'issue', 'problem',
                'fix', 'solve', 'resolve', 'debug', 'troubleshoot',
                'not working', "doesn't work", 'fails', 'failed'
            ],

            QueryIntent.RECOMMENDATION: [
                'best', 'better', 'recommend', 'recommendation', 'suggest',
                'suggestion', 'should', 'prefer', 'preferable', 'advised',
                'good', 'ideal', 'optimal'
            ],

            QueryIntent.PROCEDURAL: [
                'process', 'procedure', 'workflow', 'pipeline', 'steps',
                'phases', 'stages', 'sequence', 'order', 'flow'
            ],

            QueryIntent.CONCEPTUAL: [
                'why', 'concept', 'theory', 'principle', 'idea',
                'understand', 'understanding', 'rationale', 'reason',
                'behind', 'underlying', 'fundamental'
            ],
        }

    def classify(self, query: str) -> IntentClassification:
        """
        Classify query intent.

        Args:
            query: User query string

        Returns:
            IntentClassification with primary intent, confidence, and recommendations
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to classifier")
            return IntentClassification(
                primary_intent=QueryIntent.GENERAL,
                confidence=1.0
            )

        query = query.strip()
        logger.debug(f"Classifying query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        # Score each intent
        intent_scores = {}
        intent_matches = {}

        for intent in QueryIntent:
            if intent == QueryIntent.GENERAL:
                continue  # Handle separately as fallback

            score = 0.0
            matches = []

            # Pattern matching (higher weight)
            if intent in self.patterns:
                for pattern in self.patterns[intent]:
                    if pattern.search(query):
                        score += 0.4
                        matches.append(f"pattern:{pattern.pattern[:30]}")

            # Keyword matching
            if intent in self.keywords:
                query_lower = query.lower()
                for keyword in self.keywords[intent]:
                    if keyword in query_lower:
                        score += 0.15
                        matches.append(f"keyword:{keyword}")

            if score > 0:
                intent_scores[intent] = score
                intent_matches[intent] = matches

        # Determine primary intent
        if not intent_scores:
            primary_intent = QueryIntent.GENERAL
            confidence = 1.0
            patterns_matched = []
        else:
            # Get top intent
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            primary_intent, primary_score = sorted_intents[0]

            # Normalize confidence
            max_possible_score = 2.0  # Approximate max
            confidence = min(primary_score / max_possible_score, 1.0)

            # Get secondary intents
            secondary_intents = [
                (intent, min(score / max_possible_score, 1.0))
                for intent, score in sorted_intents[1:3]
                if score > 0.2
            ]

            patterns_matched = intent_matches.get(primary_intent, [])

        logger.info(
            f"Classified query as {primary_intent.value} "
            f"(confidence: {confidence:.2f})"
        )

        # Build classification result
        classification = IntentClassification(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents if 'secondary_intents' in locals() else [],
            patterns_matched=patterns_matched
        )

        # Add search strategy recommendations
        self._add_search_recommendations(classification, query)

        return classification

    def _add_search_recommendations(self, classification: IntentClassification, query: str):
        """Add search strategy recommendations based on intent."""
        intent = classification.primary_intent

        # Intent-specific recommendations
        if intent == QueryIntent.SUMMARY:
            # Prefer summary chunks
            classification.recommended_filters = {
                "section_title": "[DOCUMENT SUMMARY]"
            }
            classification.recommended_limit = 3
            classification.recommended_score_threshold = 0.6

        elif intent == QueryIntent.CODE_TECHNICAL:
            # Prefer code-heavy chunks
            classification.recommended_filters = {
                "contains_code": True
            }
            classification.recommended_limit = 8
            classification.recommended_score_threshold = 0.5

        elif intent == QueryIntent.HOW_TO:
            # Prefer procedural content
            classification.recommended_limit = 10
            classification.recommended_score_threshold = 0.5

        elif intent == QueryIntent.DEFINITION:
            # Prefer beginning chunks and summaries
            classification.recommended_limit = 5
            classification.recommended_score_threshold = 0.6

        elif intent == QueryIntent.TROUBLESHOOTING:
            # Cast wider net for troubleshooting
            classification.recommended_limit = 15
            classification.recommended_score_threshold = 0.4

        elif intent == QueryIntent.COMPARISON:
            # Need multiple perspectives
            classification.recommended_limit = 12
            classification.recommended_score_threshold = 0.5

        elif intent == QueryIntent.FACTUAL:
            # Precise answers needed
            classification.recommended_limit = 5
            classification.recommended_score_threshold = 0.65

        else:
            # Default recommendations
            classification.recommended_limit = 10
            classification.recommended_score_threshold = 0.5


def classify_query_intent(query: str) -> IntentClassification:
    """
    Convenience function to classify query intent.

    Args:
        query: User query string

    Returns:
        IntentClassification object
    """
    classifier = QueryIntentClassifier()
    return classifier.classify(query)


# Example usage and testing
if __name__ == "__main__":
    print("Query Intent Classifier")
    print("=" * 80)

    # Test queries
    test_queries = [
        "What is machine learning?",
        "How to implement neural networks in Python",
        "Compare TensorFlow vs PyTorch",
        "Fix AttributeError in my code",
        "Show me API documentation",
        "Why does gradient descent work?",
        "Best practices for data preprocessing",
        "List all available functions",
        "Give me a summary of this document",
        "Steps to deploy a model",
    ]

    classifier = QueryIntentClassifier()

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        classification = classifier.classify(query)
        print(f"  Intent: {classification.primary_intent.value}")
        print(f"  Confidence: {classification.confidence:.2f}")
        if classification.secondary_intents:
            print(f"  Secondary: {[(i.value, f'{c:.2f}') for i, c in classification.secondary_intents]}")
        print(f"  Recommended limit: {classification.recommended_limit}")
        print(f"  Recommended threshold: {classification.recommended_score_threshold}")
        if classification.recommended_filters:
            print(f"  Recommended filters: {classification.recommended_filters}")
