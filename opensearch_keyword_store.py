"""
OpenSearch Keyword Store for Hybrid Search

This module provides functionality to store and search keywords in OpenSearch
for hybrid (keyword + vector) search capabilities.

Features:
- Index creation with optimized mappings for keyword search
- Keyword extraction from chunks (headings, clauses, metadata, summaries)
- Bulk indexing for efficient document ingestion
- Multiple search strategies (BM25, fuzzy, phrase, multi-match)
- Faceted search support
- Integration with enterprise_chunking_pipeline

Usage:
    from opensearch_keyword_store import OpenSearchKeywordStore
    from enterprise_chunking_pipeline import chunk_for_hybrid_search

    # Initialize store
    store = OpenSearchKeywordStore(host="localhost", port=9200)

    # Create index
    store.create_index("my_documents")

    # Index chunks from hybrid chunking
    hybrid_chunks = chunk_for_hybrid_search(extraction_result)
    store.index_hybrid_chunks("my_documents", hybrid_chunks, doc_id="doc_001")

    # Search
    results = store.keyword_search("my_documents", "machine learning")
"""

import re
import json
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

try:
    from opensearchpy import OpenSearch, helpers
    from opensearchpy.exceptions import NotFoundError, RequestError
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    print("Warning: opensearch-py not installed. Install with: pip install opensearch-py")

from logger_config import get_logger

logger = get_logger(__name__)


class SearchStrategy(Enum):
    """Available search strategies for keyword search."""
    BM25 = "bm25"                    # Standard BM25 ranking
    FUZZY = "fuzzy"                  # Fuzzy matching for typos
    PHRASE = "phrase"                # Exact phrase matching
    MULTI_MATCH = "multi_match"      # Search across multiple fields
    WILDCARD = "wildcard"            # Wildcard pattern matching
    BOOL_COMBINED = "bool_combined"  # Combined boolean query


@dataclass
class KeywordDocument:
    """
    Document structure for OpenSearch keyword indexing.
    Optimized for keyword search and faceted filtering.

    CRITICAL FIELDS FOR CITATIONS:
    - source: file_name for citation display
    - title/document_title: document title for display
    - document_name: human readable document name
    - page: page number for citation
    - section: section name for citation
    """
    # Identifiers
    doc_id: str
    chunk_id: str
    chunk_type: str  # content, heading, clause, metadata, summary

    # Searchable text fields
    text: str                           # Main searchable text
    text_normalized: str                # Lowercase normalized text
    keywords: List[str]                 # Extracted keywords

    # === CRITICAL FOR CITATIONS ===
    source: str = ""                    # Original filename
    title: str = ""                     # Document title
    document_title: str = ""            # Document title (alias)
    document_name: str = ""             # Clean document name
    page: int = 1                       # Page number for citation
    section: str = ""                   # Section for citation

    # Heading/Section information
    section_title: Optional[str] = None
    heading_path: List[str] = None
    heading_level: Optional[int] = None
    parent_section: Optional[str] = None

    # Source tracking
    page_number_start: int = 1
    page_number_end: int = 1
    file_name: str = ""

    # Metadata for filtering
    chunk_index: int = 0
    total_chunks: int = 0
    char_count: int = 0
    word_count: int = 0

    # Content flags
    contains_tables: bool = False
    contains_code: bool = False
    contains_bullets: bool = False

    # Project scoping (for multi-tenancy)
    project_id: Optional[str] = None
    schema_version: Optional[str] = None

    # Timestamps
    indexed_at: str = None

    # Reference to parent chunk (for clause/heading chunks)
    parent_chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for indexing."""
        data = asdict(self)
        if data['indexed_at'] is None:
            data['indexed_at'] = datetime.now().isoformat()
        if data['heading_path'] is None:
            data['heading_path'] = []
        return data


class KeywordExtractor:
    """
    Extracts keywords from text for indexing and search.
    Uses multiple extraction strategies.
    """

    def __init__(self):
        # Common English stopwords
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he',
            'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than',
            'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
            'if', 'else', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'once'
        }

        # Technical term patterns
        self.technical_patterns = [
            re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'),  # CamelCase
            re.compile(r'\b[a-z]+_[a-z_]+\b'),               # snake_case
            re.compile(r'\b[A-Z]{2,}\b'),                     # ACRONYMS
            re.compile(r'\b\d+(?:\.\d+)+\b'),                 # Version numbers
        ]

    def extract_keywords(
        self,
        text: str,
        max_keywords: int = 50,
        min_word_length: int = 3,
        include_phrases: bool = True
    ) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            min_word_length: Minimum word length to consider
            include_phrases: Include multi-word phrases

        Returns:
            List of extracted keywords
        """
        keywords = set()

        # Extract single words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text)
        for word in words:
            word_lower = word.lower()
            if (len(word) >= min_word_length and
                word_lower not in self.stopwords):
                keywords.add(word_lower)

        # Extract technical terms (preserve case)
        for pattern in self.technical_patterns:
            matches = pattern.findall(text)
            keywords.update(matches)

        # Extract phrases (2-3 word combinations)
        if include_phrases:
            phrases = self._extract_phrases(text)
            keywords.update(phrases)

        # Extract capitalized proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for noun in proper_nouns:
            if len(noun) >= min_word_length:
                keywords.add(noun)

        # Limit and sort
        keyword_list = list(keywords)[:max_keywords]
        return sorted(keyword_list, key=lambda x: (-len(x), x))

    def _extract_phrases(self, text: str, max_phrases: int = 20) -> List[str]:
        """Extract meaningful phrases from text."""
        phrases = set()

        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)

        for sentence in sentences:
            words = sentence.split()

            # Extract 2-word phrases
            for i in range(len(words) - 1):
                w1 = words[i].lower().strip('.,;:!?"\'')
                w2 = words[i + 1].lower().strip('.,;:!?"\'')

                if (w1 not in self.stopwords and
                    w2 not in self.stopwords and
                    len(w1) >= 3 and len(w2) >= 3):
                    phrases.add(f"{w1} {w2}")

            # Extract 3-word phrases
            for i in range(len(words) - 2):
                w1 = words[i].lower().strip('.,;:!?"\'')
                w2 = words[i + 1].lower().strip('.,;:!?"\'')
                w3 = words[i + 2].lower().strip('.,;:!?"\'')

                if (w1 not in self.stopwords and
                    w3 not in self.stopwords and
                    len(w1) >= 3 and len(w3) >= 3):
                    phrases.add(f"{w1} {w2} {w3}")

        return list(phrases)[:max_phrases]

    def extract_from_heading(self, heading: str) -> List[str]:
        """
        Extract keywords from heading text.
        Headings get special treatment - all words are potentially important.
        """
        keywords = []

        # Split heading into words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', heading)

        for word in words:
            if len(word) >= 2:  # Lower threshold for headings
                keywords.append(word.lower())
                # Also keep original case for proper nouns
                if word[0].isupper():
                    keywords.append(word)

        # Add the full heading as a phrase
        keywords.append(heading.strip())

        return list(set(keywords))


class OpenSearchKeywordStore:
    """
    OpenSearch client for keyword-based document storage and search.
    Provides hybrid search capabilities when combined with vector search.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False,
        verify_certs: bool = True,
        ssl_show_warn: bool = True
    ):
        """
        Initialize OpenSearch connection.

        Args:
            host: OpenSearch host
            port: OpenSearch port
            username: Optional username for authentication
            password: Optional password for authentication
            use_ssl: Use SSL/TLS connection
            verify_certs: Verify SSL certificates
            ssl_show_warn: Show SSL warnings
        """
        if not OPENSEARCH_AVAILABLE:
            raise ImportError(
                "opensearch-py is required. Install with: pip install opensearch-py"
            )

        self.host = host
        self.port = port
        self.keyword_extractor = KeywordExtractor()

        # Build connection config
        auth = (username, password) if username and password else None

        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=ssl_show_warn,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )

        logger.info(f"OpenSearch client initialized: {host}:{port}")

    def get_index_mapping(self) -> Dict[str, Any]:
        """
        Get the index mapping optimized for keyword search.

        Returns:
            Index mapping configuration
        """
        return {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1
                },
                "analysis": {
                    "analyzer": {
                        "keyword_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding", "keyword_stemmer"]
                        },
                        "search_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    },
                    "filter": {
                        "keyword_stemmer": {
                            "type": "stemmer",
                            "language": "english"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # Identifiers
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},

                    # === CRITICAL FOR CITATIONS ===
                    "source": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
                    },
                    "document_title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
                    },
                    "document_name": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
                    },
                    "page": {"type": "integer"},
                    "section": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
                    },

                    # Main searchable text fields
                    "text": {
                        "type": "text",
                        "analyzer": "keyword_analyzer",
                        "search_analyzer": "search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 512},
                            "raw": {"type": "text"}
                        }
                    },
                    "text_normalized": {
                        "type": "text",
                        "analyzer": "standard"
                    },

                    # Keywords array for exact matching
                    "keywords": {
                        "type": "keyword"
                    },

                    # Section/Heading fields
                    "section_title": {
                        "type": "text",
                        "analyzer": "keyword_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "heading_path": {
                        "type": "keyword"
                    },
                    "heading_level": {"type": "integer"},
                    "parent_section": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },

                    # Source tracking
                    "page_number_start": {"type": "integer"},
                    "page_number_end": {"type": "integer"},
                    "file_name": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },

                    # Metadata
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "char_count": {"type": "integer"},
                    "word_count": {"type": "integer"},

                    # Content flags
                    "contains_tables": {"type": "boolean"},
                    "contains_code": {"type": "boolean"},
                    "contains_bullets": {"type": "boolean"},

                    # Project scoping (for multi-tenancy)
                    "project_id": {"type": "keyword"},
                    "schema_version": {"type": "keyword"},

                    # Timestamps
                    "indexed_at": {"type": "date"},

                    # Reference
                    "parent_chunk_id": {"type": "keyword"}
                }
            }
        }

    def create_index(
        self,
        index_name: str,
        delete_if_exists: bool = False
    ) -> bool:
        """
        Create an index with keyword-optimized mappings.

        Args:
            index_name: Name of the index to create
            delete_if_exists: Delete existing index if it exists

        Returns:
            True if successful
        """
        try:
            # Check if index exists
            if self.client.indices.exists(index=index_name):
                if delete_if_exists:
                    logger.warning(f"Deleting existing index: {index_name}")
                    self.client.indices.delete(index=index_name)
                else:
                    logger.info(f"Index already exists: {index_name}")
                    return True

            # Create index with mappings
            mapping = self.get_index_mapping()
            self.client.indices.create(index=index_name, body=mapping)

            logger.info(f"Created index: {index_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise

    def delete_index(self, index_name: str) -> bool:
        """Delete an index."""
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logger.info(f"Deleted index: {index_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            raise

    def chunk_to_keyword_doc(self, chunk, doc_id: str = None) -> KeywordDocument:
        """
        Convert a ChunkMetadata object to KeywordDocument for indexing.

        Args:
            chunk: ChunkMetadata from enterprise_chunking_pipeline
            doc_id: Optional override for doc_id

        Returns:
            KeywordDocument ready for indexing
        """
        text = chunk.normalized_text or ""

        # Extract keywords based on chunk type
        if chunk.chunk_type == "heading":
            keywords = self.keyword_extractor.extract_from_heading(text)
        else:
            keywords = self.keyword_extractor.extract_keywords(text)

        # Clean file_name to get document_name (remove UUID prefix if present)
        file_name = chunk.file_name or ""
        document_name = file_name
        if '_' in document_name and len(document_name.split('_')[0]) == 36:
            document_name = '_'.join(document_name.split('_')[1:])

        # Get document_title from chunk metadata if available
        document_title = getattr(chunk, 'document_title', None) or document_name

        return KeywordDocument(
            doc_id=doc_id or chunk.doc_id,
            chunk_id=chunk.chunk_id,
            chunk_type=chunk.chunk_type,
            text=text,
            text_normalized=text.lower(),
            keywords=keywords,
            # CRITICAL FOR CITATIONS
            source=file_name,
            title=document_title,
            document_title=document_title,
            document_name=document_name,
            page=chunk.page_number_start,
            section=chunk.section_title or "",
            # Heading/Section
            section_title=chunk.section_title,
            heading_path=chunk.heading_path or [],
            heading_level=chunk.heading_level,
            parent_section=chunk.parent_section,
            # Source tracking
            page_number_start=chunk.page_number_start,
            page_number_end=chunk.page_number_end,
            file_name=file_name,
            # Metadata
            chunk_index=chunk.chunk_index,
            total_chunks=chunk.total_chunks,
            char_count=chunk.chunk_char_len,
            word_count=chunk.chunk_word_count,
            contains_tables=chunk.contains_tables,
            contains_code=chunk.contains_code,
            contains_bullets=chunk.contains_bullets,
            parent_chunk_id=chunk.parent_chunk_id,
            # Project scoping
            project_id=getattr(chunk, 'project_id', None),
            schema_version=getattr(chunk, 'schema_version', None)
        )

    def index_document(
        self,
        index_name: str,
        doc: KeywordDocument
    ) -> Dict[str, Any]:
        """
        Index a single keyword document.

        Args:
            index_name: Target index
            doc: KeywordDocument to index

        Returns:
            Index response
        """
        try:
            response = self.client.index(
                index=index_name,
                id=doc.chunk_id,
                body=doc.to_dict(),
                refresh=True
            )
            logger.debug(f"Indexed document: {doc.chunk_id}")
            return response
        except Exception as e:
            logger.error(f"Failed to index document {doc.chunk_id}: {e}")
            raise

    def bulk_index(
        self,
        index_name: str,
        documents: List[KeywordDocument],
        batch_size: int = 500
    ) -> Dict[str, int]:
        """
        Bulk index multiple documents.

        Args:
            index_name: Target index
            documents: List of KeywordDocuments
            batch_size: Documents per batch

        Returns:
            Statistics dict with success/failure counts
        """
        stats = {"success": 0, "failed": 0, "total": len(documents)}

        def generate_actions():
            for doc in documents:
                yield {
                    "_index": index_name,
                    "_id": doc.chunk_id,
                    "_source": doc.to_dict()
                }

        try:
            success, failed = helpers.bulk(
                self.client,
                generate_actions(),
                chunk_size=batch_size,
                raise_on_error=False,
                raise_on_exception=False
            )

            stats["success"] = success
            stats["failed"] = len(failed) if isinstance(failed, list) else 0

            logger.info(
                f"Bulk indexed {stats['success']}/{stats['total']} documents "
                f"({stats['failed']} failed)"
            )

        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            raise

        return stats

    def index_hybrid_chunks(
        self,
        index_name: str,
        hybrid_chunks: Dict[str, List],
        doc_id: str = None,
        chunk_types: List[str] = None
    ) -> Dict[str, int]:
        """
        Index chunks from hybrid chunking pipeline.

        Args:
            index_name: Target index
            hybrid_chunks: Dict from generate_hybrid_chunks() with keys:
                          'content', 'heading', 'clause', 'metadata', 'summary'
            doc_id: Optional override for doc_id
            chunk_types: Optional list of chunk types to index.
                        If None, indexes all types.

        Returns:
            Statistics dict
        """
        # Default to all chunk types
        if chunk_types is None:
            chunk_types = ['content', 'heading', 'clause', 'metadata', 'summary']

        all_docs = []

        for chunk_type in chunk_types:
            chunks = hybrid_chunks.get(chunk_type, [])
            for chunk in chunks:
                keyword_doc = self.chunk_to_keyword_doc(chunk, doc_id)
                all_docs.append(keyword_doc)

        logger.info(f"Indexing {len(all_docs)} documents from hybrid chunks")
        return self.bulk_index(index_name, all_docs)

    # =========================================================================
    # DELETE METHODS
    # =========================================================================

    def delete_document(
        self,
        index_name: str,
        doc_id: str,
        chunk_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Delete all chunks associated with a document.

        When a document is deleted from the system, this method removes
        all associated keyword chunks from OpenSearch.

        Args:
            index_name: Index to delete from
            doc_id: Document ID to delete
            chunk_types: Optional list of chunk types to delete.
                        If None, deletes all chunk types.

        Returns:
            Dict with deletion statistics:
            - deleted: Number of documents deleted
            - failed: Number of failures
            - total: Total documents matched
        """
        try:
            # Build delete query
            query = {
                "bool": {
                    "must": [
                        {"term": {"doc_id": doc_id}}
                    ]
                }
            }

            # Add chunk type filter if specified
            if chunk_types:
                query["bool"]["filter"] = [
                    {"terms": {"chunk_type": chunk_types}}
                ]

            # First, count how many documents will be deleted
            count_response = self.client.count(
                index=index_name,
                body={"query": query}
            )
            total_to_delete = count_response.get("count", 0)

            if total_to_delete == 0:
                logger.info(f"No documents found for doc_id: {doc_id}")
                return {"deleted": 0, "failed": 0, "total": 0}

            # Perform delete by query
            response = self.client.delete_by_query(
                index=index_name,
                body={"query": query},
                refresh=True,
                conflicts="proceed"
            )

            deleted = response.get("deleted", 0)
            failed = len(response.get("failures", []))

            logger.info(
                f"Deleted {deleted}/{total_to_delete} chunks for doc_id: {doc_id}"
            )

            return {
                "deleted": deleted,
                "failed": failed,
                "total": total_to_delete,
                "doc_id": doc_id
            }

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise

    def delete_chunk(
        self,
        index_name: str,
        chunk_id: str
    ) -> bool:
        """
        Delete a single chunk by its chunk_id.

        Args:
            index_name: Index to delete from
            chunk_id: Chunk ID to delete

        Returns:
            True if deleted successfully, False if not found
        """
        try:
            response = self.client.delete(
                index=index_name,
                id=chunk_id,
                refresh=True
            )

            result = response.get("result", "")
            if result == "deleted":
                logger.info(f"Deleted chunk: {chunk_id}")
                return True
            else:
                logger.warning(f"Chunk not found: {chunk_id}")
                return False

        except NotFoundError:
            logger.warning(f"Chunk not found: {chunk_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            raise

    def delete_chunks_bulk(
        self,
        index_name: str,
        chunk_ids: List[str]
    ) -> Dict[str, int]:
        """
        Delete multiple chunks by their chunk_ids.

        Args:
            index_name: Index to delete from
            chunk_ids: List of chunk IDs to delete

        Returns:
            Dict with deletion statistics
        """
        if not chunk_ids:
            return {"deleted": 0, "failed": 0, "total": 0}

        try:
            # Build bulk delete actions
            actions = []
            for chunk_id in chunk_ids:
                actions.append({
                    "_op_type": "delete",
                    "_index": index_name,
                    "_id": chunk_id
                })

            # Execute bulk delete
            success, failed = helpers.bulk(
                self.client,
                actions,
                raise_on_error=False,
                raise_on_exception=False,
                refresh=True
            )

            failed_count = len(failed) if isinstance(failed, list) else 0

            logger.info(
                f"Bulk deleted {success}/{len(chunk_ids)} chunks "
                f"({failed_count} failed)"
            )

            return {
                "deleted": success,
                "failed": failed_count,
                "total": len(chunk_ids)
            }

        except Exception as e:
            logger.error(f"Bulk delete failed: {e}")
            raise

    def delete_by_file_name(
        self,
        index_name: str,
        file_name: str
    ) -> Dict[str, Any]:
        """
        Delete all chunks associated with a specific file name.

        Useful when you want to delete by file name instead of doc_id.

        Args:
            index_name: Index to delete from
            file_name: File name to match

        Returns:
            Dict with deletion statistics
        """
        try:
            query = {
                "bool": {
                    "should": [
                        {"term": {"file_name.keyword": file_name}},
                        {"match": {"file_name": file_name}}
                    ],
                    "minimum_should_match": 1
                }
            }

            # Count documents
            count_response = self.client.count(
                index=index_name,
                body={"query": query}
            )
            total_to_delete = count_response.get("count", 0)

            if total_to_delete == 0:
                logger.info(f"No documents found for file: {file_name}")
                return {"deleted": 0, "failed": 0, "total": 0}

            # Delete by query
            response = self.client.delete_by_query(
                index=index_name,
                body={"query": query},
                refresh=True,
                conflicts="proceed"
            )

            deleted = response.get("deleted", 0)
            failed = len(response.get("failures", []))

            logger.info(
                f"Deleted {deleted}/{total_to_delete} chunks for file: {file_name}"
            )

            return {
                "deleted": deleted,
                "failed": failed,
                "total": total_to_delete,
                "file_name": file_name
            }

        except Exception as e:
            logger.error(f"Failed to delete by file name {file_name}: {e}")
            raise

    def delete_by_chunk_type(
        self,
        index_name: str,
        doc_id: str,
        chunk_type: str
    ) -> Dict[str, Any]:
        """
        Delete chunks of a specific type for a document.

        Useful for selectively removing certain chunk types
        (e.g., removing old summary chunks before regenerating).

        Args:
            index_name: Index to delete from
            doc_id: Document ID
            chunk_type: Chunk type to delete (content, heading, clause, metadata, summary)

        Returns:
            Dict with deletion statistics
        """
        return self.delete_document(
            index_name,
            doc_id,
            chunk_types=[chunk_type]
        )

    def delete_all_documents(
        self,
        index_name: str,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Delete ALL documents from an index.

        WARNING: This is a destructive operation!

        Args:
            index_name: Index to clear
            confirm: Must be True to proceed

        Returns:
            Dict with deletion statistics
        """
        if not confirm:
            raise ValueError(
                "Must set confirm=True to delete all documents. "
                "This is a destructive operation!"
            )

        try:
            # Count total documents
            count_response = self.client.count(index=index_name)
            total = count_response.get("count", 0)

            if total == 0:
                logger.info(f"Index {index_name} is already empty")
                return {"deleted": 0, "failed": 0, "total": 0}

            # Delete all documents
            response = self.client.delete_by_query(
                index=index_name,
                body={"query": {"match_all": {}}},
                refresh=True,
                conflicts="proceed"
            )

            deleted = response.get("deleted", 0)
            failed = len(response.get("failures", []))

            logger.warning(
                f"Deleted ALL {deleted}/{total} documents from index: {index_name}"
            )

            return {
                "deleted": deleted,
                "failed": failed,
                "total": total
            }

        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            raise

    def check_document_exists(
        self,
        index_name: str,
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Check if a document exists in the index and get its chunk count.

        Args:
            index_name: Index to check
            doc_id: Document ID to look for

        Returns:
            Dict with exists flag and chunk counts by type
        """
        try:
            # Count total chunks for this doc_id
            query = {"term": {"doc_id": doc_id}}

            count_response = self.client.count(
                index=index_name,
                body={"query": query}
            )
            total_count = count_response.get("count", 0)

            if total_count == 0:
                return {
                    "exists": False,
                    "doc_id": doc_id,
                    "total_chunks": 0,
                    "chunks_by_type": {}
                }

            # Get counts by chunk type
            agg_response = self.client.search(
                index=index_name,
                body={
                    "query": query,
                    "size": 0,
                    "aggs": {
                        "chunk_types": {
                            "terms": {"field": "chunk_type", "size": 10}
                        }
                    }
                }
            )

            chunks_by_type = {}
            buckets = agg_response.get("aggregations", {}).get("chunk_types", {}).get("buckets", [])
            for bucket in buckets:
                chunks_by_type[bucket["key"]] = bucket["doc_count"]

            return {
                "exists": True,
                "doc_id": doc_id,
                "total_chunks": total_count,
                "chunks_by_type": chunks_by_type
            }

        except Exception as e:
            logger.error(f"Failed to check document existence: {e}")
            raise

    # =========================================================================
    # SEARCH METHODS
    # =========================================================================

    def keyword_search(
        self,
        index_name: str,
        query: str,
        strategy: SearchStrategy = SearchStrategy.MULTI_MATCH,
        size: int = 10,
        chunk_types: List[str] = None,
        page_filter: Optional[int] = None,
        section_filter: Optional[str] = None,
        highlight: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search with specified strategy.

        Args:
            index_name: Index to search
            query: Search query
            strategy: Search strategy to use
            size: Maximum results
            chunk_types: Filter by chunk types
            page_filter: Filter by page number
            section_filter: Filter by section title
            highlight: Include highlighted matches

        Returns:
            List of search results with scores
        """
        # Build the query based on strategy
        if strategy == SearchStrategy.BM25:
            search_query = self._build_bm25_query(query)
        elif strategy == SearchStrategy.FUZZY:
            search_query = self._build_fuzzy_query(query)
        elif strategy == SearchStrategy.PHRASE:
            search_query = self._build_phrase_query(query)
        elif strategy == SearchStrategy.MULTI_MATCH:
            search_query = self._build_multi_match_query(query)
        elif strategy == SearchStrategy.WILDCARD:
            search_query = self._build_wildcard_query(query)
        elif strategy == SearchStrategy.BOOL_COMBINED:
            search_query = self._build_bool_combined_query(query)
        else:
            search_query = self._build_multi_match_query(query)

        # Add filters
        filters = []

        if chunk_types:
            filters.append({"terms": {"chunk_type": chunk_types}})

        if page_filter:
            filters.append({
                "range": {
                    "page_number_start": {"lte": page_filter},
                    "page_number_end": {"gte": page_filter}
                }
            })

        if section_filter:
            filters.append({
                "match": {"section_title": section_filter}
            })

        # Combine query with filters
        if filters:
            final_query = {
                "bool": {
                    "must": search_query,
                    "filter": filters
                }
            }
        else:
            final_query = search_query

        # Build search body
        body = {
            "query": final_query,
            "size": size,
            "_source": True
        }

        # Add highlighting
        if highlight:
            body["highlight"] = {
                "fields": {
                    "text": {"fragment_size": 150, "number_of_fragments": 3},
                    "section_title": {},
                    "keywords": {}
                },
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"]
            }

        try:
            response = self.client.search(index=index_name, body=body)
            return self._format_search_results(response)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _build_bm25_query(self, query: str) -> Dict:
        """Build standard BM25 query."""
        return {
            "match": {
                "text": {
                    "query": query,
                    "operator": "or"
                }
            }
        }

    def _build_fuzzy_query(self, query: str) -> Dict:
        """Build fuzzy query for typo tolerance."""
        return {
            "match": {
                "text": {
                    "query": query,
                    "fuzziness": "AUTO",
                    "prefix_length": 2
                }
            }
        }

    def _build_phrase_query(self, query: str) -> Dict:
        """Build phrase query for exact matching."""
        return {
            "match_phrase": {
                "text": {
                    "query": query,
                    "slop": 2
                }
            }
        }

    def _build_multi_match_query(self, query: str) -> Dict:
        """Build multi-match query across multiple fields."""
        return {
            "multi_match": {
                "query": query,
                "fields": [
                    "text^2",
                    "text.raw",
                    "section_title^3",
                    "keywords^2",
                    "heading_path"
                ],
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO"
            }
        }

    def _build_wildcard_query(self, query: str) -> Dict:
        """Build wildcard query for pattern matching."""
        # Add wildcards if not present
        if '*' not in query and '?' not in query:
            query = f"*{query}*"

        return {
            "wildcard": {
                "text.keyword": {
                    "value": query,
                    "case_insensitive": True
                }
            }
        }

    def _build_bool_combined_query(self, query: str) -> Dict:
        """Build combined boolean query with multiple strategies."""
        return {
            "bool": {
                "should": [
                    # Exact phrase match (highest boost)
                    {
                        "match_phrase": {
                            "text": {
                                "query": query,
                                "boost": 3
                            }
                        }
                    },
                    # Multi-match across fields
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["text^2", "section_title^2", "keywords"],
                            "type": "best_fields",
                            "boost": 2
                        }
                    },
                    # Fuzzy match
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "fuzziness": "AUTO",
                                "boost": 1
                            }
                        }
                    },
                    # Keyword exact match
                    {
                        "terms": {
                            "keywords": query.lower().split(),
                            "boost": 1.5
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }

    def _format_search_results(self, response: Dict) -> List[Dict[str, Any]]:
        """Format OpenSearch response into clean results."""
        results = []

        hits = response.get("hits", {}).get("hits", [])

        for hit in hits:
            result = {
                "chunk_id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"],
                "highlights": hit.get("highlight", {})
            }
            results.append(result)

        return results

    def search_by_section(
        self,
        index_name: str,
        section_title: str,
        size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks by section title.

        Args:
            index_name: Index to search
            section_title: Section title to search for
            size: Maximum results

        Returns:
            List of matching chunks
        """
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"section_title": {"query": section_title, "boost": 2}}},
                        {"match": {"heading_path": section_title}}
                    ]
                }
            },
            "size": size
        }

        response = self.client.search(index=index_name, body=body)
        return self._format_search_results(response)

    def search_headings_only(
        self,
        index_name: str,
        query: str,
        size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search only heading chunks (useful for TOC-style queries).

        Args:
            index_name: Index to search
            query: Search query
            size: Maximum results

        Returns:
            List of matching heading chunks
        """
        return self.keyword_search(
            index_name,
            query,
            strategy=SearchStrategy.MULTI_MATCH,
            size=size,
            chunk_types=["heading"]
        )

    def get_document_sections(
        self,
        index_name: str,
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all sections/headings for a document (table of contents).

        Args:
            index_name: Index to search
            doc_id: Document ID

        Returns:
            List of heading chunks sorted by page number
        """
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"doc_id": doc_id}},
                        {"term": {"chunk_type": "heading"}}
                    ]
                }
            },
            "sort": [
                {"page_number_start": "asc"},
                {"heading_level": "asc"}
            ],
            "size": 100
        }

        response = self.client.search(index=index_name, body=body)
        return self._format_search_results(response)

    def faceted_search(
        self,
        index_name: str,
        query: str,
        facets: List[str] = None,
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Perform faceted search with aggregations.

        Args:
            index_name: Index to search
            query: Search query
            facets: List of fields to aggregate on
            size: Maximum results

        Returns:
            Dict with results and facet counts
        """
        if facets is None:
            facets = ["chunk_type", "section_title.keyword", "file_name.keyword"]

        body = {
            "query": self._build_multi_match_query(query),
            "size": size,
            "aggs": {}
        }

        # Add aggregations for each facet
        for facet in facets:
            facet_name = facet.replace(".", "_")
            body["aggs"][facet_name] = {
                "terms": {
                    "field": facet,
                    "size": 20
                }
            }

        response = self.client.search(index=index_name, body=body)

        return {
            "results": self._format_search_results(response),
            "facets": response.get("aggregations", {})
        }

    def get_stats(self, index_name: str) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = self.client.indices.stats(index=index_name)
            count = self.client.count(index=index_name)

            return {
                "document_count": count["count"],
                "index_size": stats["indices"][index_name]["total"]["store"]["size_in_bytes"],
                "index_size_human": stats["indices"][index_name]["total"]["store"]["size"]
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_keyword_store(
    host: str = "localhost",
    port: int = 9200,
    username: str = None,
    password: str = None
) -> OpenSearchKeywordStore:
    """
    Convenience function to create an OpenSearch keyword store.

    Args:
        host: OpenSearch host
        port: OpenSearch port
        username: Optional username
        password: Optional password

    Returns:
        Configured OpenSearchKeywordStore instance
    """
    return OpenSearchKeywordStore(
        host=host,
        port=port,
        username=username,
        password=password
    )


def index_document_for_hybrid_search(
    store: OpenSearchKeywordStore,
    index_name: str,
    extraction_result,
    doc_id: str = None
) -> Dict[str, int]:
    """
    Index a document for hybrid search using the chunking pipeline.

    Args:
        store: OpenSearchKeywordStore instance
        index_name: Target index name
        extraction_result: ExtractionResult from document_processor
        doc_id: Optional document ID

    Returns:
        Indexing statistics
    """
    from enterprise_chunking_pipeline import (
        EnterpriseChunkingPipeline,
        ChunkingConfig
    )

    # Configure for hybrid search
    config = ChunkingConfig(
        max_chunk_size=1000,
        enable_overlap=True,
        generate_heading_chunks=True,
        generate_clause_chunks=True,
        generate_metadata_chunks=True,
        generate_summary_chunks=True
    )

    # Generate chunks
    pipeline = EnterpriseChunkingPipeline(config)
    hybrid_chunks = pipeline.generate_hybrid_chunks(extraction_result, doc_id)

    # Index chunks
    return store.index_hybrid_chunks(index_name, hybrid_chunks, doc_id)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("OpenSearch Keyword Store for Hybrid Search")
    print("=" * 60)
    print("\nFeatures:")
    print("  - Index creation with keyword-optimized mappings")
    print("  - Keyword extraction from chunks")
    print("  - Multiple search strategies (BM25, fuzzy, phrase, multi-match)")
    print("  - Faceted search support")
    print("  - Integration with enterprise_chunking_pipeline")
    print("\n  DELETE OPERATIONS:")
    print("  - delete_document()       - Delete all chunks for a document by doc_id")
    print("  - delete_chunk()          - Delete a single chunk by chunk_id")
    print("  - delete_chunks_bulk()    - Bulk delete multiple chunks")
    print("  - delete_by_file_name()   - Delete all chunks by file name")
    print("  - delete_by_chunk_type()  - Delete specific chunk types for a document")
    print("  - delete_all_documents()  - Delete ALL documents (requires confirmation)")
    print("  - check_document_exists() - Check if document exists and get chunk counts")
    print("\nUsage example:")
    print("""
from opensearch_keyword_store import (
    OpenSearchKeywordStore,
    SearchStrategy,
    index_document_for_hybrid_search
)
from document_processor import extract_text_from_document
from enterprise_chunking_pipeline import chunk_for_hybrid_search

# 1. Initialize store
store = OpenSearchKeywordStore(host="localhost", port=9200)

# 2. Create index
store.create_index("my_documents", delete_if_exists=True)

# 3. Extract and chunk document
result = extract_text_from_document('document.pdf', extract_metadata=True)
hybrid_chunks = chunk_for_hybrid_search(result)

# 4. Index chunks
stats = store.index_hybrid_chunks("my_documents", hybrid_chunks)
print(f"Indexed {stats['success']} chunks")

# 5. Search with different strategies

# BM25 search
results = store.keyword_search(
    "my_documents",
    "machine learning",
    strategy=SearchStrategy.BM25
)

# Fuzzy search (typo tolerant)
results = store.keyword_search(
    "my_documents",
    "machien lerning",  # typo
    strategy=SearchStrategy.FUZZY
)

# Phrase search
results = store.keyword_search(
    "my_documents",
    "neural network architecture",
    strategy=SearchStrategy.PHRASE
)

# Multi-match search (searches all fields)
results = store.keyword_search(
    "my_documents",
    "deep learning",
    strategy=SearchStrategy.MULTI_MATCH
)

# Filter by chunk type
results = store.keyword_search(
    "my_documents",
    "introduction",
    chunk_types=["heading", "summary"]
)

# Search headings only (TOC-style)
headings = store.search_headings_only("my_documents", "chapter")

# Get document sections (table of contents)
toc = store.get_document_sections("my_documents", "doc_001")

# Faceted search
faceted = store.faceted_search(
    "my_documents",
    "data analysis",
    facets=["chunk_type", "section_title.keyword"]
)

# Print results
for result in results:
    print(f"Score: {result['score']:.2f}")
    print(f"Chunk Type: {result['source']['chunk_type']}")
    print(f"Section: {result['source'].get('section_title', 'N/A')}")
    print(f"Text: {result['source']['text'][:200]}...")
    print("-" * 40)

# ============================================
# DELETE OPERATIONS
# ============================================

# Check if document exists before deletion
doc_info = store.check_document_exists("my_documents", "doc_001")
print(f"Document exists: {doc_info['exists']}")
print(f"Total chunks: {doc_info['total_chunks']}")
print(f"Chunks by type: {doc_info['chunks_by_type']}")

# Delete all chunks for a document
delete_result = store.delete_document("my_documents", "doc_001")
print(f"Deleted {delete_result['deleted']} chunks for doc_001")

# Delete a single chunk
store.delete_chunk("my_documents", "doc_001_chunk_0001")

# Bulk delete multiple chunks
store.delete_chunks_bulk("my_documents", [
    "doc_001_chunk_0001",
    "doc_001_chunk_0002",
    "doc_001_chunk_0003"
])

# Delete by file name
store.delete_by_file_name("my_documents", "document.pdf")

# Delete specific chunk type for a document
# (useful for regenerating summaries)
store.delete_by_chunk_type("my_documents", "doc_001", "summary")

# Delete ALL documents (requires confirmation)
# WARNING: This is destructive!
# store.delete_all_documents("my_documents", confirm=True)
    """)
