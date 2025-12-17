"""
Enhanced Query Analyzer - Production-Ready Intent Detection

Architecture:
1. Rule-first: Fast pattern matching for common queries (90% of traffic)
2. LLM-assisted: Accurate fallback for complex queries (10% of traffic)
3. Filter mapping: Convert intent → Qdrant filters

Performance:
- Rule-based: <5ms latency, 0 cost
- LLM-assisted: ~200ms latency, ~$0.0001 per query
"""

import re
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
from logger_config import get_logger

from intent_schema import (
    Intent, IntentType, Entity, EntityType, DetectionMethod,
    build_qdrant_filter, INTENT_EXAMPLES
)

logger = get_logger(__name__)


class RuleEngine:
    """
    Fast, deterministic intent detection using pattern matching.
    Handles ~90% of queries without LLM.
    """

    def __init__(self):
        # Intent patterns (ordered by priority)
        self.intent_patterns = [
            # NAVIGATE (highest priority - very specific)
            (IntentType.NAVIGATE, [
                r'(?:go to|jump to|navigate to|show me)\s+(?:page|section|chapter)\s+(\S+)',
                r'(?:page|section|chapter)\s+(\d+(?:\.\d+)*)',
            ], 0.95),

            # FILTER (high priority - explicit filtering)
            (IntentType.FILTER, [
                r'(?:show|list|find|get)\s+(?:all|only)?\s*(?:files?|documents?)',
                r'(?:from|in|created|uploaded|modified)\s+(?:\d{4}|today|yesterday|last\s+\w+)',
                r'(?:PDF|DOCX|TXT)\s+(?:files?|documents?)',
                r'(?:by|from|written by|authored by)\s+\w+',
            ], 0.90),

            # LIST (medium-high priority)
            (IntentType.LIST, [
                r'^(?:list|show|display)\s+(?:all|the|available)',
                r'(?:what|which)\s+(?:files?|documents?|sections?)',
            ], 0.85),

            # QUESTION (medium priority - starts with question words)
            (IntentType.QUESTION, [
                r'^(?:what|who|when|where|why|how)\s+',
                r'^(?:can you|could you|would you)\s+(?:explain|tell|describe)',
                r'\?$',  # Ends with question mark
            ], 0.80),

            # SUMMARIZE (medium priority)
            (IntentType.SUMMARIZE, [
                r'(?:summarize|summary of|give me a summary)',
                r'(?:key points|main ideas|highlights)',
                r'(?:tl;?dr|too long)',
            ], 0.85),

            # COMPARE (medium priority)
            (IntentType.COMPARE, [
                r'(?:compare|difference between|vs|versus)',
                r'(?:how does .+ differ from|contrast)',
            ], 0.85),

            # EXTRACT (medium priority)
            (IntentType.EXTRACT, [
                r'(?:extract|get|find)\s+(?:all|the)?\s*(?:numbers?|dates?|names?|emails?)',
                r'(?:pull out|retrieve)\s+',
            ], 0.80),

            # SEARCH (lowest priority - catch-all)
            (IntentType.SEARCH, [
                r'.+',  # Matches anything
            ], 0.70),
        ]

        # Entity extraction patterns
        self.entity_patterns = {
            EntityType.FILE_NAME: [
                r'file\s+(?:named|called)?\s*["\']?([a-zA-Z0-9_\-\.]+\.(pdf|docx|txt))["\']?',
                r'document\s+["\']?([a-zA-Z0-9_\-\.]+\.(pdf|docx|txt))["\']?',
            ],

            EntityType.FILE_TYPE: [
                r'\b(PDF|DOCX|TXT)\b',
                r'\.(pdf|docx|txt)\s+files?',
            ],

            EntityType.SECTION: [
                r'(?:section|chapter|part)\s+(\d+(?:\.\d+)*)',
                r'(?:section|chapter|part)\s+["\']?([^"\']+)["\']?',
            ],

            EntityType.PAGE_NUMBER: [
                r'page\s+(\d+)',
                r'on\s+page\s+(\d+)',
            ],

            EntityType.DATE_RANGE: [
                r'(?:from|since|after)\s+(\d{4})-(\d{2})-(\d{2})',
                r'in\s+(\d{4})',
                r'(today|yesterday|last\s+(?:week|month|year))',
            ],

            EntityType.DOCUMENT_TYPE: [
                r'\b(contract|report|manual|specification|proposal|invoice|receipt)\b',
            ],

            EntityType.NUMBER: [
                r'\b(\d+(?:\.\d+)?)\b',
            ],
        }

    def detect_intent(self, query: str) -> Tuple[IntentType, float]:
        """
        Detect intent using pattern matching.

        Returns:
            (intent_type, confidence)
        """
        query_lower = query.lower().strip()

        for intent_type, patterns, base_confidence in self.intent_patterns:
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return intent_type, base_confidence

        return IntentType.UNKNOWN, 0.0

    def extract_entities(self, query: str) -> List[Entity]:
        """
        Extract entities using pattern matching.

        Returns:
            List of extracted entities
        """
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    value = self._normalize_entity_value(entity_type, match)
                    if value:
                        entities.append(Entity(
                            type=entity_type,
                            value=value,
                            confidence=0.95,  # High confidence for regex matches
                            raw_text=match.group(0)
                        ))

        return entities

    def _normalize_entity_value(self, entity_type: EntityType, match: re.Match) -> Any:
        """Normalize extracted entity value."""

        if entity_type == EntityType.FILE_NAME:
            return match.group(1)

        elif entity_type == EntityType.FILE_TYPE:
            return match.group(1).lower()

        elif entity_type == EntityType.SECTION:
            return match.group(1)

        elif entity_type == EntityType.PAGE_NUMBER:
            return int(match.group(1))

        elif entity_type == EntityType.DATE_RANGE:
            # Parse date ranges
            if match.group(0) == "today":
                today = datetime.now().date()
                return {"start": today.isoformat(), "end": today.isoformat()}
            elif match.group(0) == "yesterday":
                yesterday = (datetime.now() - timedelta(days=1)).date()
                return {"start": yesterday.isoformat(), "end": yesterday.isoformat()}
            elif "last week" in match.group(0):
                end = datetime.now().date()
                start = end - timedelta(days=7)
                return {"start": start.isoformat(), "end": end.isoformat()}
            elif "last month" in match.group(0):
                end = datetime.now().date()
                start = end - timedelta(days=30)
                return {"start": start.isoformat(), "end": end.isoformat()}
            elif len(match.groups()) >= 3:  # Full date
                year, month, day = match.groups()[:3]
                date = f"{year}-{month}-{day}"
                return {"start": date, "end": date}
            elif len(match.groups()) == 1 and match.group(1).isdigit():  # Year only
                year = match.group(1)
                return {"start": f"{year}-01-01", "end": f"{year}-12-31"}

        elif entity_type == EntityType.DOCUMENT_TYPE:
            return match.group(1).lower()

        elif entity_type == EntityType.NUMBER:
            try:
                return float(match.group(1))
            except ValueError:
                return match.group(1)

        return match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)


class LLMAssistant:
    """
    LLM-powered intent detection for complex queries.
    Used as fallback when rules can't confidently classify.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize LLM assistant.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Model to use (gpt-4o-mini is fast and cheap)
        """
        self.api_key = api_key
        self.model = model
        self.client = None

        # Try to import OpenAI
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"LLM assistant initialized with model: {model}")
        except ImportError:
            logger.warning("OpenAI not installed. LLM fallback disabled. Install: pip install openai")
        except Exception as e:
            logger.warning(f"LLM assistant initialization failed: {e}")

    def analyze_query(self, query: str) -> Tuple[Intent, str]:
        """
        Analyze query using LLM.

        Returns:
            (Intent object, reasoning)
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized")

        prompt = self._build_prompt(query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert query intent classifier for a RAG system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Parse result into Intent object
            intent = self._parse_llm_response(query, result)
            reasoning = result.get("reasoning", "No reasoning provided")

            return intent, reasoning

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise

    def _build_prompt(self, query: str) -> str:
        """
        Build production-ready LLM prompt.

        CRITICAL: This prompt must return valid JSON matching our schema.
        """
        return f"""Analyze this user query for a document search system and classify its intent.

Query: "{query}"

Return a JSON object with this EXACT structure:
{{
  "intent_type": "<one of: search, filter, question, summarize, compare, extract, list, navigate, unknown>",
  "confidence": <float 0.0-1.0>,
  "normalized_query": "<simplified query without stop words>",
  "key_terms": ["term1", "term2", ...],
  "entities": [
    {{
      "type": "<one of: file_name, file_type, date_range, section, page_number, author, topic, keyword, document_type, number, boolean>",
      "value": "<extracted value>",
      "confidence": <float 0.0-1.0>,
      "raw_text": "<original text>"
    }}
  ],
  "search_type": "<one of: hybrid, semantic, keyword, metadata>",
  "limit": <integer, suggested number of results>,
  "rerank": <boolean, whether to rerank results>,
  "reasoning": "<brief explanation of classification>"
}}

Intent Type Guidelines:
- search: General information retrieval (e.g., "machine learning algorithms")
- filter: Metadata-based filtering (e.g., "PDFs from 2024")
- question: Specific questions (e.g., "What is the main conclusion?")
- summarize: Request for summary (e.g., "summarize this document")
- compare: Compare multiple items (e.g., "difference between X and Y")
- extract: Pull specific data (e.g., "extract all dates")
- list: List available items (e.g., "show all files")
- navigate: Go to specific location (e.g., "go to page 5")

Entity Extraction:
- file_name: Specific document names
- file_type: pdf, docx, txt
- date_range: {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
- section: Chapter/section identifiers
- page_number: Specific page numbers
- document_type: contract, report, manual, etc.

Search Type Guidelines:
- hybrid: Mix of semantic + keyword (default)
- semantic: Pure vector search for meaning
- keyword: Exact term matching
- metadata: Pure filtering without semantic search

Be precise. Return ONLY valid JSON."""

    def _parse_llm_response(self, query: str, result: Dict[str, Any]) -> Intent:
        """Parse LLM JSON response into Intent object."""

        # Parse entities
        entities = []
        for entity_data in result.get("entities", []):
            entities.append(Entity(
                type=EntityType(entity_data["type"]),
                value=entity_data["value"],
                confidence=entity_data.get("confidence", 0.8),
                raw_text=entity_data.get("raw_text")
            ))

        # Build Qdrant filters from entities
        qdrant_filters = build_qdrant_filter(entities) if entities else None

        return Intent(
            intent_type=IntentType(result["intent_type"]),
            confidence=result["confidence"],
            detection_method=DetectionMethod.LLM_ASSISTED,
            original_query=query,
            normalized_query=result.get("normalized_query", query),
            key_terms=result.get("key_terms", []),
            entities=entities,
            qdrant_filters=qdrant_filters,
            search_type=result.get("search_type", "hybrid"),
            limit=result.get("limit", 10),
            rerank=result.get("rerank", False),
            llm_reasoning=result.get("reasoning")
        )


class EnhancedQueryAnalyzer:
    """
    Production-ready query analyzer with rule-first, LLM-assisted architecture.

    Performance:
    - Rule-based: ~90% of queries, <5ms, $0
    - LLM fallback: ~10% of queries, ~200ms, ~$0.0001

    Cost optimization:
    - 1M queries/month
    - 900K via rules = $0
    - 100K via LLM = $10
    - Total: $10/month for 1M queries
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        confidence_threshold: float = 0.75
    ):
        """
        Initialize analyzer.

        Args:
            use_llm: Enable LLM fallback
            llm_api_key: OpenAI API key
            llm_model: LLM model name
            confidence_threshold: Minimum confidence to trust rule-based detection
        """
        logger.info("Initializing EnhancedQueryAnalyzer")

        self.rule_engine = RuleEngine()
        self.llm_assistant = None
        self.use_llm = use_llm
        self.confidence_threshold = confidence_threshold

        if use_llm:
            try:
                self.llm_assistant = LLMAssistant(api_key=llm_api_key, model=llm_model)
                logger.info("LLM fallback enabled")
            except Exception as e:
                logger.warning(f"LLM fallback disabled: {e}")
                self.use_llm = False

        logger.info(f"Analyzer ready (LLM: {self.use_llm}, threshold: {confidence_threshold})")

    def analyze(self, query: str) -> Intent:
        """
        Analyze query using rule-first, LLM-assisted approach.

        Flow:
        1. Try rule-based detection (fast)
        2. If confidence >= threshold → return
        3. Else → fallback to LLM (accurate)
        4. Build Qdrant filters from entities

        Args:
            query: User query string

        Returns:
            Intent object with classification and filters
        """
        start_time = time.time()

        # Step 1: Rule-based detection (fast path)
        intent_type, rule_confidence = self.rule_engine.detect_intent(query)
        entities = self.rule_engine.extract_entities(query)

        # Check if we trust the rule-based result
        if rule_confidence >= self.confidence_threshold:
            # High confidence - use rule-based result
            logger.info(f"Rule-based detection: {intent_type.value} ({rule_confidence:.2f})")

            # Build filters
            qdrant_filters = build_qdrant_filter(entities) if entities else None

            # Extract key terms (simple tokenization)
            key_terms = [word.lower() for word in query.split() if len(word) > 2]

            intent = Intent(
                intent_type=intent_type,
                confidence=rule_confidence,
                detection_method=DetectionMethod.RULE_BASED,
                original_query=query,
                normalized_query=query.lower().strip(),
                key_terms=key_terms,
                entities=entities,
                qdrant_filters=qdrant_filters,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        else:
            # Low confidence - fallback to LLM
            if self.use_llm and self.llm_assistant:
                logger.info(f"Rule confidence {rule_confidence:.2f} < {self.confidence_threshold}, using LLM fallback")

                try:
                    intent, reasoning = self.llm_assistant.analyze_query(query)
                    intent.processing_time_ms = (time.time() - start_time) * 1000
                    logger.info(f"LLM detection: {intent.intent_type.value} ({intent.confidence:.2f})")

                except Exception as e:
                    logger.error(f"LLM fallback failed: {e}, using rule-based result")
                    # Fall back to rule result
                    intent = Intent(
                        intent_type=intent_type,
                        confidence=rule_confidence,
                        detection_method=DetectionMethod.FALLBACK,
                        original_query=query,
                        normalized_query=query.lower().strip(),
                        key_terms=[],
                        entities=entities,
                        qdrant_filters=build_qdrant_filter(entities),
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
            else:
                # LLM not available, use rule result anyway
                logger.warning(f"Low confidence {rule_confidence:.2f} but LLM unavailable")
                qdrant_filters = build_qdrant_filter(entities) if entities else None
                key_terms = [word.lower() for word in query.split() if len(word) > 2]

                intent = Intent(
                    intent_type=intent_type,
                    confidence=rule_confidence,
                    detection_method=DetectionMethod.FALLBACK,
                    original_query=query,
                    normalized_query=query.lower().strip(),
                    key_terms=key_terms,
                    entities=entities,
                    qdrant_filters=qdrant_filters,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

        # Log result
        logger.debug(f"Query analysis complete: {intent.intent_type.value}, "
                    f"confidence={intent.confidence:.2f}, "
                    f"method={intent.detection_method.value}, "
                    f"time={intent.processing_time_ms:.1f}ms")

        return intent


# Demo
if __name__ == "__main__":
    import sys

    # Initialize analyzer (LLM disabled for demo)
    analyzer = EnhancedQueryAnalyzer(use_llm=False, confidence_threshold=0.75)

    # Test queries
    test_queries = [
        "machine learning algorithms",
        "show me all PDF files from 2024",
        "What are the key findings in section 3?",
        "summarize the research report",
        "go to page 5",
        "extract all dates from the contract",
    ]

    print("Enhanced Query Analyzer Demo")
    print("=" * 80)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 80)

        intent = analyzer.analyze(query)
        print(intent.to_json())
        print(f"\nFilters: {intent.qdrant_filters}")
