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

# -------------------------------------------------------------------
# IMPORTS — Updated for Qdrant Client 1.7+
# -------------------------------------------------------------------
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
except ImportError:
    QDRANT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Qdrant client not installed. Install with: pip install qdrant-client")

logger = logging.getLogger(__name__)


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
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not installed. Install with: pip install qdrant-client")

        self.config = config
        self.client = QdrantClient(
            url=config.url,
            api_key=config.api_key,
            timeout=config.timeout
        )

    def create_collection(self, recreate: bool = False, on_disk_payload: bool = True):
        collection = self.config.collection_name

        exists = any(c.name == collection for c in self.client.get_collections().collections)

        if exists:
            if not recreate:
                logger.info(f"Collection '{collection}' already exists.")
                return False

            logger.warning(f"Deleting existing collection '{collection}'")
            self.client.delete_collection(collection)

        dist_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        distance = dist_map.get(self.config.distance_metric, Distance.COSINE)

        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=self.config.vector_size,
                distance=distance,
                on_disk=False
            ),
            on_disk_payload=on_disk_payload
        )

        logger.info(f"Collection '{collection}' created")
        self._create_indexes()
        return True

    def _create_indexes(self):
        collection = self.config.collection_name

        if self.config.enable_full_text_index:
            try:
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
                logger.info("Full-text index created on 'text'")
            except Exception as e:
                logger.warning(f"Failed to create full-text index: {e}")

        logger.info("Index setup complete")

    def collection_info(self) -> Dict:
        try:
            info = self.client.get_collection(self.config.collection_name)
            return {
                "name": self.config.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Failed to load collection info: {e}")
            return {}

    def delete_collection(self):
        logger.warning(f"Deleting collection '{self.config.collection_name}'")
        self.client.delete_collection(self.config.collection_name)


# -------------------------------------------------------------------
# STORAGE LAYER
# -------------------------------------------------------------------
class QdrantStorage:
    def __init__(self, config: QdrantConfig):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not installed.")

        self.config = config
        self.client = QdrantClient(
            url=config.url,
            api_key=config.api_key,
            timeout=config.timeout
        )
        self.collection_manager = QdrantCollectionManager(config)

    def store_embeddings(self, embedding_records: List, embeddings: List[List[float]], show_progress: bool = False) -> Dict:
        if len(embedding_records) != len(embeddings):
            raise ValueError("Record count does not match embedding count")

        if show_progress:
            logger.info(f"Storing {len(embedding_records)} embeddings into Qdrant")


        collection = self.config.collection_name
        batch_size = self.config.batch_size

        total_uploaded = 0
        total_failed = 0

        for i in range(0, len(embedding_records), batch_size):
            batch_records = embedding_records[i:i+batch_size]
            batch_vectors = embeddings[i:i+batch_size]

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
                logger.info(f"Uploaded {total_uploaded}/{len(embedding_records)}")
            except Exception as e:
                total_failed += len(points)
                logger.error(f"Batch upload failed: {e}")

        return {"uploaded": total_uploaded, "failed": total_failed, "total": len(embedding_records)}

    def search(self, query_vector: List[float], limit: int = 10, filters: Optional[Dict] = None, score_threshold: Optional[float] = None,):
        f = self._build_filter(filters)

        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=f,
            with_vectors=False,
            score_threshold=score_threshold,
        )

        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    def search_by_text(self, query_text: str, limit: int = 10, filters: Optional[Dict] = None):
        from qdrant_client.models import MatchText

        conditions = [
            FieldCondition(
                key="text",
                match=MatchText(text=query_text)
            )
        ]

        if filters:
            add = self._build_filter(filters)
            if add:
                conditions.extend(add.must)

        query_filter = Filter(must=conditions)

        records, _ = self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return [{"id": r.id, "payload": r.payload} for r in records]

    def filter_by_metadata(self, filters: Dict, limit: int = 100, offset: int = 0):
        f = self._build_filter(filters)

        records, _ = self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=f,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        return [{"id": r.id, "payload": r.payload} for r in records]

    def _build_filter(self, filters: Dict) -> Optional[Filter]:
        if not filters:
            return None

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
            elif isinstance(value, list):
                for v in value:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions)

    def update_metadata(self, point_id: str, metadata: Dict):
        self.client.set_payload(
            collection_name=self.config.collection_name,
            payload=metadata,
            points=[point_id]
        )

    def delete_by_filter(self, filters: Dict):
        f = self._build_filter(filters)
        return self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=f
        )

    def get_by_id(self, point_id: str):
        try:
            results = self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[point_id],
                with_vectors=True,
                with_payload=True
            )
            if results:
                r = results[0]
                return {"id": r.id, "vector": r.vector, "payload": r.payload}
        except Exception as e:
            logger.error(f"Retrieve failed: {e}")
        return None


# -------------------------------------------------------------------
# Helper Functions (OUTSIDE THE CLASS — CORRECT)
# -------------------------------------------------------------------

def setup_qdrant_collection(url="http://localhost:6333",
                            collection_name="documents",
                            vector_size=384,
                            recreate=False):
    config = QdrantConfig(
        url=url,
        collection_name=collection_name,
        vector_size=vector_size
    )
    manager = QdrantCollectionManager(config)
    manager.create_collection(recreate=recreate)
    return manager


def store_embeddings_in_qdrant(embedding_records, embeddings,
                               url="http://localhost:6333",
                               collection_name="documents",
                               vector_size=384):
    config = QdrantConfig(
        url=url,
        collection_name=collection_name,
        vector_size=vector_size
    )
    storage = QdrantStorage(config)
    storage.collection_manager.create_collection(recreate=False)
    return storage.store_embeddings(embedding_records, embeddings)
