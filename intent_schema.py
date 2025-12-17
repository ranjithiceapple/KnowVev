"""
Intent JSON Schema for RAG Query Analysis

Production-ready intent classification with:
- Rule-first detection (fast, deterministic)
- LLM-assisted fallback (accurate, flexible)
- Qdrant filter mapping
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


class IntentType(Enum):
    """Primary intent types for user queries."""
    SEARCH = "search"                    # General semantic search
    FILTER = "filter"                    # Metadata-based filtering
    QUESTION = "question"                # Specific question answering
    SUMMARIZE = "summarize"              # Document summarization
    COMPARE = "compare"                  # Compare multiple documents/sections
    EXTRACT = "extract"                  # Extract specific information
    LIST = "list"                        # List documents/sections
    NAVIGATE = "navigate"                # Navigate to specific section/page
    UNKNOWN = "unknown"                  # Cannot determine intent


class EntityType(Enum):
    """Entity types that can be extracted from queries."""
    FILE_NAME = "file_name"              # Specific document name
    FILE_TYPE = "file_type"              # PDF, DOCX, TXT
    DATE_RANGE = "date_range"            # Date/time constraints
    SECTION = "section"                  # Document section/chapter
    PAGE_NUMBER = "page_number"          # Specific page
    AUTHOR = "author"                    # Document author/creator
    TOPIC = "topic"                      # Subject matter
    KEYWORD = "keyword"                  # Specific terms to match
    DOCUMENT_TYPE = "document_type"      # Contract, report, manual, etc.
    NUMBER = "number"                    # Numeric values
    BOOLEAN = "boolean"                  # Yes/no, true/false


class ConfidenceLevel(Enum):
    """Confidence in intent classification."""
    HIGH = "high"          # >0.8 confidence
    MEDIUM = "medium"      # 0.5-0.8 confidence
    LOW = "low"            # 0.3-0.5 confidence
    UNCERTAIN = "uncertain" # <0.3 confidence


class DetectionMethod(Enum):
    """How the intent was detected."""
    RULE_BASED = "rule_based"        # Pattern/regex matching
    LLM_ASSISTED = "llm_assisted"    # LLM classification
    HYBRID = "hybrid"                # Combination of both
    FALLBACK = "fallback"            # Default/unknown


@dataclass
class Entity:
    """Extracted entity from query."""
    type: EntityType
    value: Any
    confidence: float = 1.0
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'value': self.value,
            'confidence': self.confidence,
            'raw_text': self.raw_text
        }


@dataclass
class Intent:
    """Complete intent analysis result."""

    # Primary classification
    intent_type: IntentType
    confidence: float
    detection_method: DetectionMethod

    # Query understanding
    original_query: str
    normalized_query: str
    key_terms: List[str] = field(default_factory=list)

    # Extracted entities
    entities: List[Entity] = field(default_factory=list)

    # Qdrant filter generation
    qdrant_filters: Optional[Dict[str, Any]] = None

    # Search parameters
    search_type: str = "hybrid"  # "hybrid", "semantic", "keyword"
    limit: int = 10
    score_threshold: Optional[float] = None
    rerank: bool = False

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: Optional[float] = None

    # LLM reasoning (if used)
    llm_reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'intent_type': self.intent_type.value,
            'confidence': self.confidence,
            'detection_method': self.detection_method.value,
            'original_query': self.original_query,
            'normalized_query': self.normalized_query,
            'key_terms': self.key_terms,
            'entities': [e.to_dict() for e in self.entities],
            'qdrant_filters': self.qdrant_filters,
            'search_type': self.search_type,
            'limit': self.limit,
            'score_threshold': self.score_threshold,
            'rerank': self.rerank,
            'timestamp': self.timestamp,
            'processing_time_ms': self.processing_time_ms,
            'llm_reasoning': self.llm_reasoning
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def get_confidence_level(self) -> ConfidenceLevel:
        """Get confidence level classification."""
        if self.confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence > 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN


# Filter mapping helpers

def entity_to_qdrant_filter(entity: Entity) -> Optional[Dict[str, Any]]:
    """
    Convert a single entity to Qdrant filter syntax.

    Returns:
        Qdrant filter dict or None if not filterable
    """
    if entity.type == EntityType.FILE_NAME:
        return {
            "key": "file_name",
            "match": {"value": entity.value}
        }

    elif entity.type == EntityType.FILE_TYPE:
        return {
            "key": "file_type",
            "match": {"value": entity.value.lower()}
        }

    elif entity.type == EntityType.DATE_RANGE:
        # Expecting entity.value = {"start": "2024-01-01", "end": "2024-12-31"}
        filters = []
        if "start" in entity.value:
            filters.append({
                "key": "extraction_date",
                "range": {"gte": entity.value["start"]}
            })
        if "end" in entity.value:
            filters.append({
                "key": "extraction_date",
                "range": {"lte": entity.value["end"]}
            })
        return filters if filters else None

    elif entity.type == EntityType.SECTION:
        return {
            "key": "section_title",
            "match": {"text": entity.value}
        }

    elif entity.type == EntityType.PAGE_NUMBER:
        return {
            "key": "page_number_start",
            "match": {"value": entity.value}
        }

    elif entity.type == EntityType.DOCUMENT_TYPE:
        return {
            "key": "document_type",
            "match": {"value": entity.value.lower()}
        }

    elif entity.type == EntityType.AUTHOR:
        return {
            "key": "author",
            "match": {"text": entity.value}
        }

    return None


def build_qdrant_filter(entities: List[Entity], combinator: str = "must") -> Optional[Dict[str, Any]]:
    """
    Build complete Qdrant filter from entities.

    Args:
        entities: List of extracted entities
        combinator: "must" (AND) or "should" (OR)

    Returns:
        Qdrant filter dict in proper syntax
    """
    filters = []

    for entity in entities:
        entity_filter = entity_to_qdrant_filter(entity)
        if entity_filter:
            if isinstance(entity_filter, list):
                filters.extend(entity_filter)
            else:
                filters.append(entity_filter)

    if not filters:
        return None

    if len(filters) == 1:
        return filters[0]

    # Multiple filters - combine with must/should
    return {
        combinator: filters
    }


# Example intent JSONs

INTENT_EXAMPLES = {
    "search": {
        "intent_type": "search",
        "confidence": 0.95,
        "detection_method": "rule_based",
        "original_query": "machine learning algorithms",
        "normalized_query": "machine learning algorithms",
        "key_terms": ["machine", "learning", "algorithms"],
        "entities": [],
        "qdrant_filters": None,
        "search_type": "hybrid",
        "limit": 10
    },

    "filter": {
        "intent_type": "filter",
        "confidence": 0.98,
        "detection_method": "rule_based",
        "original_query": "show me all PDF files from 2024",
        "normalized_query": "PDF files 2024",
        "key_terms": ["PDF", "2024"],
        "entities": [
            {
                "type": "file_type",
                "value": "pdf",
                "confidence": 1.0,
                "raw_text": "PDF"
            },
            {
                "type": "date_range",
                "value": {"start": "2024-01-01", "end": "2024-12-31"},
                "confidence": 0.9,
                "raw_text": "2024"
            }
        ],
        "qdrant_filters": {
            "must": [
                {"key": "file_type", "match": {"value": "pdf"}},
                {"key": "extraction_date", "range": {"gte": "2024-01-01"}},
                {"key": "extraction_date", "range": {"lte": "2024-12-31"}}
            ]
        },
        "search_type": "metadata",
        "limit": 100
    },

    "question": {
        "intent_type": "question",
        "confidence": 0.92,
        "detection_method": "hybrid",
        "original_query": "What are the key findings in the research report?",
        "normalized_query": "key findings research report",
        "key_terms": ["key", "findings", "research", "report"],
        "entities": [
            {
                "type": "document_type",
                "value": "research report",
                "confidence": 0.85,
                "raw_text": "research report"
            }
        ],
        "qdrant_filters": {
            "key": "document_type",
            "match": {"value": "research report"}
        },
        "search_type": "hybrid",
        "limit": 5,
        "rerank": True
    },

    "navigate": {
        "intent_type": "navigate",
        "confidence": 0.88,
        "detection_method": "rule_based",
        "original_query": "go to section 3.2",
        "normalized_query": "section 3.2",
        "key_terms": ["section", "3.2"],
        "entities": [
            {
                "type": "section",
                "value": "3.2",
                "confidence": 1.0,
                "raw_text": "section 3.2"
            }
        ],
        "qdrant_filters": {
            "key": "section_title",
            "match": {"text": "3.2"}
        },
        "search_type": "metadata",
        "limit": 1
    }
}


if __name__ == "__main__":
    # Demo: Create and display intent examples
    print("Intent Schema Examples")
    print("=" * 80)

    for intent_name, intent_data in INTENT_EXAMPLES.items():
        print(f"\n{intent_name.upper()} Intent:")
        print(json.dumps(intent_data, indent=2))
