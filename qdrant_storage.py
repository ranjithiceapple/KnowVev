"""
Qdrant Vector Storage Module (Updated for Qdrant Client 1.7+)

Handles:
- Collection creation
- Full-text indexing (supported index type)
- Vector storage
- Metadata filtering
- Vector search
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import uuid
import time
from logger_config import get_logger

# -------------------------------------------------------------------
# IMPORTS — Updated for Qdrant Client 1.7+
# -------------------------------------------------------------------
logger = get_logger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
        TextIndexParams,
        TextIndexType,
    )
    QDRANT_AVAILABLE = True
    logger.info("Qdrant client imported successfully")
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not installed. Install with: pip install qdrant-client")


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: int = 60

    collection_name: str = "documents"
    vector_size: int = 384
    distance_metric: str = "Cosine"

    enable_full_text_index: bool = True

    batch_size: int = 100
    parallel: int = 1


# -------------------------------------------------------------------
# COLLECTION MANAGER
# -------------------------------------------------------------------
class QdrantCollectionManager:
    def __init__(self, config: QdrantConfig):
        logger.info("Initializing QdrantCollectionManager")
        if not QDRANT_AVAILABLE:
            logger.error("Qdrant client not available - cannot initialize QdrantCollectionManager")
            raise ImportError("Qdrant client not installed. Install with: pip install qdrant-client")

        self.config = config
        logger.info(
            f"Connecting to Qdrant - URL: {config.url}, Collection: {config.collection_name}, "
            f"Vector size: {config.vector_size}, Distance: {config.distance_metric}"
        )

        try:
            self.client = QdrantClient(
                url=config.url,
                api_key=config.api_key,
                timeout=config.timeout
            )
            logger.info("Successfully connected to Qdrant server")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant server at {config.url}: {str(e)}", exc_info=True)
            raise

    def create_collection(self, recreate: bool = False, on_disk_payload: bool = True):
        start_time = time.time()
        collection = self.config.collection_name
        logger.info(f"Creating collection '{collection}' (recreate={recreate}, on_disk_payload={on_disk_payload})")

        try:
            logger.debug("Checking if collection exists")
            exists = any(c.name == collection for c in self.client.get_collections().collections)
            logger.debug(f"Collection exists: {exists}")

            if exists:
                if not recreate:
                    logger.info(f"Collection '{collection}' already exists - skipping creation")
                    return False

                logger.warning(f"Deleting existing collection '{collection}' due to recreate=True")
                self.client.delete_collection(collection)
                logger.info(f"Collection '{collection}' deleted successfully")

            dist_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            distance = dist_map.get(self.config.distance_metric, Distance.COSINE)
            logger.debug(f"Using distance metric: {self.config.distance_metric} -> {distance}")

            logger.info(
                f"Creating collection '{collection}' with vector_size={self.config.vector_size}, "
                f"distance={self.config.distance_metric}"
            )
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=distance,
                    on_disk=False
                ),
                on_disk_payload=on_disk_payload
            )

            duration = time.time() - start_time
            logger.info(f"Collection '{collection}' created successfully in {duration:.2f}s")

            self._create_indexes()
            return True

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Failed to create collection '{collection}' after {duration:.2f}s: {str(e)}",
                exc_info=True
            )
            raise

    def _create_indexes(self):
        collection = self.config.collection_name
        logger.info(f"Starting index creation for collection '{collection}'")

        if self.config.enable_full_text_index:
            logger.debug("Full-text indexing is enabled")
            try:
                logger.info("Creating full-text index on 'text' field")
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name="text",
                    field_schema=TextIndexParams(
                        type=TextIndexType.TEXT,
                        tokenizer="word",
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True
                    ),
                )
                logger.info("Full-text index created successfully on 'text' field")
            except Exception as e:
                logger.warning(f"Failed to create full-text index: {e}", exc_info=True)
        else:
            logger.debug("Full-text indexing is disabled")

        logger.info("Index setup complete")

    def collection_info(self) -> Dict:
        collection = self.config.collection_name
        logger.debug(f"Fetching info for collection '{collection}'")
        try:
            info = self.client.get_collection(collection)
            result = {
                "name": collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
            logger.info(
                f"Collection info retrieved - Name: {collection}, "
                f"Vectors: {info.vectors_count}, Points: {info.points_count}, Status: {info.status}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to load collection info for '{collection}': {e}", exc_info=True)
            return {}

    def delete_collection(self):
        collection = self.config.collection_name
        logger.warning(f"Deleting collection '{collection}'")
        try:
            self.client.delete_collection(collection)
            logger.warning(f"Collection '{collection}' deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection}': {e}", exc_info=True)
            raise


# -------------------------------------------------------------------
# STORAGE LAYER
# -------------------------------------------------------------------
class QdrantStorage:
    def __init__(self, config: QdrantConfig):
        logger.info("Initializing QdrantStorage")
        if not QDRANT_AVAILABLE:
            logger.error("Qdrant client not available - cannot initialize QdrantStorage")
            raise ImportError("Qdrant client not installed.")

        self.config = config
        logger.info(f"Initializing QdrantStorage with URL: {config.url}, Collection: {config.collection_name}")

        try:
            self.client = QdrantClient(
                url=config.url,
                api_key=config.api_key,
                timeout=config.timeout
            )
            logger.info("QdrantClient initialized successfully")

            self.collection_manager = QdrantCollectionManager(config)
            logger.info("QdrantStorage initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize QdrantStorage: {str(e)}", exc_info=True)
            raise

    def store_embeddings(self, embedding_records: List, embeddings: List[List[float]], show_progress: bool = False) -> Dict:
        start_time = time.time()
        logger.info(f"Starting embedding storage - Records: {len(embedding_records)}, Embeddings: {len(embeddings)}")

        if len(embedding_records) != len(embeddings):
            logger.error(
                f"Record count mismatch - Records: {len(embedding_records)}, Embeddings: {len(embeddings)}"
            )
            raise ValueError("Record count does not match embedding count")

        collection = self.config.collection_name
        batch_size = self.config.batch_size
        logger.info(f"Storing {len(embedding_records)} embeddings into collection '{collection}' (batch_size={batch_size})")

        total_uploaded = 0
        total_failed = 0
        num_batches = (len(embedding_records) + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(embedding_records), batch_size), 1):
            batch_start_time = time.time()
            batch_records = embedding_records[i:i+batch_size]
            batch_vectors = embeddings[i:i+batch_size]

            logger.debug(
                f"Processing batch {batch_num}/{num_batches} - "
                f"Records: {len(batch_records)}, Offset: {i}"
            )

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=rec.to_qdrant_payload()
                )
                for rec, vec in zip(batch_records, batch_vectors)
            ]

            try:
                self.client.upsert(collection_name=collection, points=points)
                total_uploaded += len(points)
                batch_duration = time.time() - batch_start_time
                logger.info(
                    f"Batch {batch_num}/{num_batches} uploaded - "
                    f"Points: {len(points)}, Total: {total_uploaded}/{len(embedding_records)}, "
                    f"Duration: {batch_duration:.2f}s"
                )
            except Exception as e:
                total_failed += len(points)
                batch_duration = time.time() - batch_start_time
                logger.error(
                    f"Batch {batch_num}/{num_batches} upload failed - "
                    f"Points: {len(points)}, Duration: {batch_duration:.2f}s, Error: {str(e)}",
                    exc_info=True
                )

        total_duration = time.time() - start_time
        result = {"uploaded": total_uploaded, "failed": total_failed, "total": len(embedding_records)}
        logger.info(
            f"Embedding storage complete - Uploaded: {total_uploaded}, Failed: {total_failed}, "
            f"Total: {len(embedding_records)}, Duration: {total_duration:.2f}s"
        )

        return result

    def search(self, query_vector: List[float], limit: int = 10, filters: Optional[Dict] = None, score_threshold: Optional[float] = None,):
        start_time = time.time()
        logger.info(
            f"Vector search started - Limit: {limit}, Score threshold: {score_threshold}, "
            f"Filters: {bool(filters)}, Vector dim: {len(query_vector)}"
        )

        try:
            f = self._build_filter(filters)
            logger.debug(f"Built filter: {f}")

            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                query_filter=f,
                with_vectors=False,
                score_threshold=score_threshold,
            )

            duration = time.time() - start_time
            formatted_results = [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
            logger.info(
                f"Vector search completed - Results: {len(formatted_results)}, Duration: {duration:.3f}s"
            )

            return formatted_results

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Vector search failed - Duration: {duration:.3f}s, Error: {str(e)}",
                exc_info=True
            )
            raise

    def search_by_text(self, query_text: str, limit: int = 10, filters: Optional[Dict] = None):
        start_time = time.time()
        logger.info(
            f"Text search started - Query: '{query_text[:100]}{'...' if len(query_text) > 100 else ''}', "
            f"Limit: {limit}, Filters: {bool(filters)}"
        )

        try:
            from qdrant_client.models import MatchText

            conditions = [
                FieldCondition(
                    key="text",
                    match=MatchText(text=query_text)
                )
            ]
            logger.debug("Created text match condition")

            if filters:
                add = self._build_filter(filters)
                if add:
                    conditions.extend(add.must)
                    logger.debug(f"Added {len(add.must)} filter conditions")

            query_filter = Filter(must=conditions)

            records, _ = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            duration = time.time() - start_time
            results = [{"id": r.id, "payload": r.payload} for r in records]
            logger.info(
                f"Text search completed - Results: {len(results)}, Duration: {duration:.3f}s"
            )

            return results

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Text search failed - Query: '{query_text[:50]}', Duration: {duration:.3f}s, Error: {str(e)}",
                exc_info=True
            )
            raise

    def filter_by_metadata(self, filters: Dict, limit: int = 100, offset: int = 0):
        start_time = time.time()
        logger.info(f"Filtering by metadata - Filters: {filters}, Limit: {limit}, Offset: {offset}")

        try:
            f = self._build_filter(filters)
            logger.debug(f"Built metadata filter: {f}")

            records, _ = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=f,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            duration = time.time() - start_time
            results = [{"id": r.id, "payload": r.payload} for r in records]
            logger.info(
                f"Metadata filter completed - Results: {len(results)}, Duration: {duration:.3f}s"
            )

            return results

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Metadata filter failed - Duration: {duration:.3f}s, Error: {str(e)}",
                exc_info=True
            )
            raise

    def _build_filter(self, filters: Dict) -> Optional[Filter]:
        if not filters:
            logger.debug("No filters provided, returning None")
            return None

        logger.debug(f"Building filter from: {filters}")
        conditions = []

        for key, value in filters.items():
            if isinstance(value, dict):
                conditions.append(
                    FieldCondition(
                        key=key,
                        range=Range(
                            gte=value.get("gte"),
                            lte=value.get("lte"),
                            gt=value.get("gt"),
                            lt=value.get("lt"),
                        ),
                    )
                )
                logger.debug(f"Added range condition for key '{key}'")
            elif isinstance(value, list):
                for v in value:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
                logger.debug(f"Added {len(value)} match conditions for key '{key}'")
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                logger.debug(f"Added match condition for key '{key}' with value '{value}'")

        logger.debug(f"Built filter with {len(conditions)} conditions")
        return Filter(must=conditions)

    def update_metadata(self, point_id: str, metadata: Dict):
        logger.info(f"Updating metadata for point '{point_id}' - Fields: {list(metadata.keys())}")
        try:
            self.client.set_payload(
                collection_name=self.config.collection_name,
                payload=metadata,
                points=[point_id]
            )
            logger.info(f"Metadata updated successfully for point '{point_id}'")
        except Exception as e:
            logger.error(
                f"Failed to update metadata for point '{point_id}': {str(e)}",
                exc_info=True
            )
            raise

    def delete_by_filter(self, filters: Dict):
        logger.warning(f"Deleting points by filter - Filters: {filters}")
        try:
            f = self._build_filter(filters)
            result = self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=f
            )
            logger.warning(f"Delete by filter completed - Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Delete by filter failed: {str(e)}", exc_info=True)
            raise

    def get_by_id(self, point_id: str):
        logger.debug(f"Retrieving point by ID: {point_id}")
        try:
            results = self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[point_id],
                with_vectors=True,
                with_payload=True
            )
            if results:
                r = results[0]
                logger.info(f"Point retrieved successfully - ID: {point_id}")
                return {"id": r.id, "vector": r.vector, "payload": r.payload}
            else:
                logger.warning(f"No point found with ID: {point_id}")
                return None
        except Exception as e:
            logger.error(f"Retrieve failed for point '{point_id}': {str(e)}", exc_info=True)
            return None


# -------------------------------------------------------------------
# Helper Functions (OUTSIDE THE CLASS — CORRECT)
# -------------------------------------------------------------------

def setup_qdrant_collection(url="http://localhost:6333",
                            collection_name="documents",
                            vector_size=384,
                            recreate=False):
    logger.info(
        f"Setting up Qdrant collection - URL: {url}, Collection: {collection_name}, "
        f"Vector size: {vector_size}, Recreate: {recreate}"
    )
    try:
        config = QdrantConfig(
            url=url,
            collection_name=collection_name,
            vector_size=vector_size
        )
        manager = QdrantCollectionManager(config)
        manager.create_collection(recreate=recreate)
        logger.info(f"Qdrant collection '{collection_name}' setup complete")
        return manager
    except Exception as e:
        logger.error(f"Failed to setup Qdrant collection: {str(e)}", exc_info=True)
        raise


def store_embeddings_in_qdrant(embedding_records, embeddings,
                               url="http://localhost:6333",
                               collection_name="documents",
                               vector_size=384):
    logger.info(
        f"Storing embeddings in Qdrant - URL: {url}, Collection: {collection_name}, "
        f"Records: {len(embedding_records)}, Embeddings: {len(embeddings)}"
    )
    try:
        config = QdrantConfig(
            url=url,
            collection_name=collection_name,
            vector_size=vector_size
        )
        storage = QdrantStorage(config)
        storage.collection_manager.create_collection(recreate=False)
        result = storage.store_embeddings(embedding_records, embeddings)
        logger.info(f"Embeddings stored successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to store embeddings in Qdrant: {str(e)}", exc_info=True)
        raise
