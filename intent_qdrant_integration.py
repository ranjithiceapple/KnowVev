"""
Complete Integration: Intent Analysis → Qdrant Search

Shows the full production flow from query → intent → filters → search → results
"""

from typing import List, Dict, Any, Optional
from enhanced_query_analyzer import EnhancedQueryAnalyzer
from intent_schema import IntentType, Intent
from logger_config import get_logger

logger = get_logger(__name__)


class IntentAwareSearch:
    """
    Intent-aware search that automatically applies filters and search strategies.
    """

    def __init__(
        self,
        qdrant_storage,
        embedding_model,
        analyzer: Optional[EnhancedQueryAnalyzer] = None
    ):
        """
        Initialize intent-aware search.

        Args:
            qdrant_storage: QdrantStorage instance
            embedding_model: SentenceTransformer model for embeddings
            analyzer: EnhancedQueryAnalyzer (created if not provided)
        """
        self.storage = qdrant_storage
        self.embedding_model = embedding_model
        self.analyzer = analyzer or EnhancedQueryAnalyzer(use_llm=True)

        logger.info("IntentAwareSearch initialized")

    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute intent-aware search.

        Flow:
        1. Analyze query → detect intent + extract entities
        2. Build Qdrant filters from entities
        3. Choose search strategy based on intent
        4. Execute search with appropriate parameters
        5. Return results with intent metadata

        Args:
            query: User query
            **kwargs: Override search parameters

        Returns:
            {
                "intent": Intent object (serialized),
                "results": List of search results,
                "metadata": Search metadata
            }
        """
        # Step 1: Analyze query
        intent = self.analyzer.analyze(query)

        logger.info(f"Query intent: {intent.intent_type.value} (confidence: {intent.confidence:.2f})")

        # Step 2: Determine search strategy based on intent
        if intent.intent_type == IntentType.FILTER or intent.intent_type == IntentType.LIST:
            # Pure metadata filtering (no vector search needed)
            results = self._filter_only_search(intent, **kwargs)

        elif intent.intent_type == IntentType.NAVIGATE:
            # Direct navigation (precise match)
            results = self._navigate_search(intent, **kwargs)

        else:
            # Semantic/hybrid search (most common)
            results = self._semantic_search(intent, **kwargs)

        # Step 3: Return results with intent metadata
        return {
            "intent": intent.to_dict(),
            "results": results,
            "metadata": {
                "query": query,
                "result_count": len(results),
                "search_strategy": self._get_strategy_name(intent),
                "filters_applied": intent.qdrant_filters is not None,
                "processing_time_ms": intent.processing_time_ms
            }
        }

    def _semantic_search(self, intent: Intent, **kwargs) -> List[Dict]:
        """
        Execute semantic search with optional filters.

        This is the main search path for most queries.
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [intent.normalized_query]
        )[0].tolist()

        # Build search parameters
        search_params = {
            "query_vector": query_embedding,
            "limit": kwargs.get("limit", intent.limit),
            "score_threshold": kwargs.get("score_threshold", intent.score_threshold),
        }

        # Add filters if present
        if intent.qdrant_filters:
            search_params["filters"] = self._convert_to_qdrant_filter(intent.qdrant_filters)

        logger.debug(f"Semantic search with params: {search_params}")

        # Execute search
        results = self.storage.search(**search_params)

        # Rerank if requested
        if intent.rerank and len(results) > 0:
            results = self._rerank_results(results, intent.original_query)

        return results

    def _filter_only_search(self, intent: Intent, **kwargs) -> List[Dict]:
        """
        Execute metadata-only filtering (no vector search).

        Used for LIST and FILTER intents.
        """
        if not intent.qdrant_filters:
            logger.warning("Filter intent but no filters extracted, returning empty")
            return []

        filters = self._convert_to_qdrant_filter(intent.qdrant_filters)

        logger.debug(f"Filter-only search with filters: {filters}")

        # Use scroll or filter_by_metadata for pure filtering
        results = self.storage.filter_by_metadata(
            filters=filters,
            limit=kwargs.get("limit", intent.limit)
        )

        return results

    def _navigate_search(self, intent: Intent, **kwargs) -> List[Dict]:
        """
        Execute navigation to specific location.

        Returns the most relevant single result.
        """
        if not intent.qdrant_filters:
            logger.warning("Navigate intent but no target extracted")
            return []

        filters = self._convert_to_qdrant_filter(intent.qdrant_filters)

        # Navigation typically returns 1 result
        results = self.storage.filter_by_metadata(
            filters=filters,
            limit=1
        )

        return results

    def _convert_to_qdrant_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert our filter format to Qdrant's exact syntax.

        Our format:
        {
            "must": [
                {"key": "file_type", "match": {"value": "pdf"}},
                {"key": "page_number", "range": {"gte": 1, "lte": 10}}
            ]
        }

        Qdrant format:
        {
            "must": [
                {"key": "file_type", "match": {"value": "pdf"}},
                {"key": "page_number", "range": {"gte": 1, "lte": 10}}
            ]
        }

        (They're already compatible! But this function allows for conversion if needed)
        """
        return filters

    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Rerank results using cross-encoder or other reranking model.

        For production, integrate a reranking model like:
        - cross-encoder/ms-marco-MiniLM-L-12-v2
        - BAAI/bge-reranker-large
        """
        # TODO: Implement reranking
        logger.debug("Reranking requested but not implemented yet")
        return results

    def _get_strategy_name(self, intent: Intent) -> str:
        """Get human-readable search strategy name."""
        if intent.intent_type in [IntentType.FILTER, IntentType.LIST]:
            return "metadata_filtering"
        elif intent.intent_type == IntentType.NAVIGATE:
            return "precise_navigation"
        else:
            if intent.qdrant_filters:
                return "semantic_with_filters"
            else:
                return "pure_semantic"


# ============================================================================
# COMPLETE EXAMPLES: Intent → Qdrant Filter Mappings
# ============================================================================

INTEGRATION_EXAMPLES = {
    "Example 1: Simple Search": {
        "query": "machine learning algorithms",
        "intent": {
            "intent_type": "search",
            "confidence": 0.95,
            "entities": [],
            "qdrant_filters": None
        },
        "qdrant_search": {
            "query_vector": "[0.123, 0.456, ...]",  # Embedding
            "limit": 10,
            "score_threshold": None,
            "filters": None  # No filters - pure semantic search
        },
        "strategy": "pure_semantic"
    },

    "Example 2: Filter by File Type": {
        "query": "show me all PDF files",
        "intent": {
            "intent_type": "filter",
            "confidence": 0.98,
            "entities": [
                {"type": "file_type", "value": "pdf"}
            ],
            "qdrant_filters": {
                "key": "file_type",
                "match": {"value": "pdf"}
            }
        },
        "qdrant_search": {
            "filters": {
                "must": [
                    {"key": "file_type", "match": {"value": "pdf"}}
                ]
            },
            "limit": 100
        },
        "strategy": "metadata_filtering"
    },

    "Example 3: Semantic Search with File Type Filter": {
        "query": "machine learning algorithms in PDF documents",
        "intent": {
            "intent_type": "search",
            "confidence": 0.92,
            "entities": [
                {"type": "file_type", "value": "pdf"}
            ],
            "qdrant_filters": {
                "key": "file_type",
                "match": {"value": "pdf"}
            }
        },
        "qdrant_search": {
            "query_vector": "[0.123, 0.456, ...]",
            "limit": 10,
            "filters": {
                "must": [
                    {"key": "file_type", "match": {"value": "pdf"}}
                ]
            }
        },
        "strategy": "semantic_with_filters"
    },

    "Example 4: Date Range Filter": {
        "query": "documents from 2024",
        "intent": {
            "intent_type": "filter",
            "confidence": 0.95,
            "entities": [
                {
                    "type": "date_range",
                    "value": {"start": "2024-01-01", "end": "2024-12-31"}
                }
            ],
            "qdrant_filters": {
                "must": [
                    {"key": "extraction_date", "range": {"gte": "2024-01-01"}},
                    {"key": "extraction_date", "range": {"lte": "2024-12-31"}}
                ]
            }
        },
        "qdrant_search": {
            "filters": {
                "must": [
                    {"key": "extraction_date", "range": {"gte": "2024-01-01"}},
                    {"key": "extraction_date", "range": {"lte": "2024-12-31"}}
                ]
            },
            "limit": 100
        },
        "strategy": "metadata_filtering"
    },

    "Example 5: Multiple Filters (AND)": {
        "query": "PDF research reports from 2024",
        "intent": {
            "intent_type": "filter",
            "confidence": 0.93,
            "entities": [
                {"type": "file_type", "value": "pdf"},
                {"type": "document_type", "value": "research report"},
                {"type": "date_range", "value": {"start": "2024-01-01", "end": "2024-12-31"}}
            ],
            "qdrant_filters": {
                "must": [
                    {"key": "file_type", "match": {"value": "pdf"}},
                    {"key": "document_type", "match": {"value": "research report"}},
                    {"key": "extraction_date", "range": {"gte": "2024-01-01"}},
                    {"key": "extraction_date", "range": {"lte": "2024-12-31"}}
                ]
            }
        },
        "qdrant_search": {
            "filters": {
                "must": [
                    {"key": "file_type", "match": {"value": "pdf"}},
                    {"key": "document_type", "match": {"value": "research report"}},
                    {"key": "extraction_date", "range": {"gte": "2024-01-01"}},
                    {"key": "extraction_date", "range": {"lte": "2024-12-31"}}
                ]
            },
            "limit": 50
        },
        "strategy": "metadata_filtering"
    },

    "Example 6: Navigate to Section": {
        "query": "go to section 3.2",
        "intent": {
            "intent_type": "navigate",
            "confidence": 0.96,
            "entities": [
                {"type": "section", "value": "3.2"}
            ],
            "qdrant_filters": {
                "key": "section_title",
                "match": {"text": "3.2"}
            }
        },
        "qdrant_search": {
            "filters": {
                "must": [
                    {"key": "section_title", "match": {"text": "3.2"}}
                ]
            },
            "limit": 1
        },
        "strategy": "precise_navigation"
    },

    "Example 7: Question with Context Filter": {
        "query": "What are the main conclusions in the 2024 annual report?",
        "intent": {
            "intent_type": "question",
            "confidence": 0.88,
            "entities": [
                {"type": "document_type", "value": "annual report"},
                {"type": "date_range", "value": {"start": "2024-01-01", "end": "2024-12-31"}}
            ],
            "qdrant_filters": {
                "must": [
                    {"key": "document_type", "match": {"value": "annual report"}},
                    {"key": "extraction_date", "range": {"gte": "2024-01-01"}},
                    {"key": "extraction_date", "range": {"lte": "2024-12-31"}}
                ]
            }
        },
        "qdrant_search": {
            "query_vector": "[0.123, ...]",  # Embedding of "main conclusions"
            "filters": {
                "must": [
                    {"key": "document_type", "match": {"value": "annual report"}},
                    {"key": "extraction_date", "range": {"gte": "2024-01-01"}},
                    {"key": "extraction_date", "range": {"lte": "2024-12-31"}}
                ]
            },
            "limit": 5,
            "rerank": True
        },
        "strategy": "semantic_with_filters"
    }
}


if __name__ == "__main__":
    import json

    print("\n" + "=" * 80)
    print("INTENT → QDRANT FILTER MAPPING EXAMPLES")
    print("=" * 80)

    for title, example in INTEGRATION_EXAMPLES.items():
        print(f"\n{title}")
        print("-" * 80)
        print(f"Query: \"{example['query']}\"")
        print(f"\nIntent Type: {example['intent']['intent_type']}")
        print(f"Confidence: {example['intent']['confidence']}")
        print(f"\nExtracted Entities:")
        for entity in example['intent'].get('entities', []):
            print(f"  - {entity['type']}: {entity['value']}")

        print(f"\nQdrant Search Parameters:")
        print(json.dumps(example['qdrant_search'], indent=2))

        print(f"\nStrategy: {example['strategy']}")
        print()
