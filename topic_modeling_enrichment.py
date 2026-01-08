"""
Topic Modeling as Non-Blocking Enrichment Stage

Position: AFTER chunk storage, does NOT block ingestion completion.
Trigger: Threshold-based or scheduled batch processing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# INGESTION LIFECYCLE POSITION
# ============================================================================

"""
Pipeline Stages:
================

1. Extract         → artifacts/doc_id/00_raw/
2. Normalize       → artifacts/doc_id/01_normalized/
3. Chunk           → artifacts/doc_id/02_chunks/
4. Embed           → artifacts/doc_id/03_embeddings/
5. Store in Qdrant → ✓ INGESTION COMPLETE ✓
   ↓
   [User can search immediately]
   ↓
6. Topic Modeling  → artifacts/doc_id/04_enrichment/topics.json (ASYNC)
   ↓
7. Update Qdrant   → Add topic metadata to existing chunks


Key Points:
- Stages 1-5: BLOCKING (required for ingestion)
- Stage 6-7: NON-BLOCKING (enrichment, runs async)
- User can search after stage 5
- Topics enhance search but aren't required
"""


# ============================================================================
# TOPIC MODEL VERSIONING
# ============================================================================

class TopicModelVersion(str, Enum):
    """Topic model versions"""
    V1_LDA = "v1_lda"               # LDA baseline
    V2_BERTOPIC = "v2_bertopic"     # BERTopic with embeddings
    V3_CUSTOM = "v3_custom"         # Custom model


@dataclass
class TopicModelMetadata:
    """Topic model metadata"""
    model_id: str                   # "topic_model_v2_20240115"
    version: TopicModelVersion
    trained_at: str
    num_topics: int
    num_documents: int              # Training corpus size

    # Model parameters
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Performance
    coherence_score: Optional[float] = None
    perplexity: Optional[float] = None

    # Training data
    chunk_ids_trained_on: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'model_id': self.model_id,
            'version': self.version.value,
            'trained_at': self.trained_at,
            'num_topics': self.num_topics,
            'num_documents': self.num_documents,
            'model_params': self.model_params,
            'coherence_score': self.coherence_score,
            'perplexity': self.perplexity,
            'chunk_ids_trained_on': self.chunk_ids_trained_on
        }


# ============================================================================
# TOPIC ASSIGNMENTS
# ============================================================================

@dataclass
class TopicAssignment:
    """Topic assignment for a chunk"""
    chunk_id: str
    doc_id: str

    # Primary topic
    primary_topic_id: int
    primary_topic_label: str
    primary_topic_score: float      # Confidence (0-1)

    # Secondary topics
    secondary_topics: List[Dict[str, Any]] = field(default_factory=list)
    # [{'topic_id': 2, 'label': 'Machine Learning', 'score': 0.3}, ...]

    # Model version
    model_id: str = ""
    assigned_at: str = ""

    def to_dict(self) -> dict:
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'primary_topic_id': self.primary_topic_id,
            'primary_topic_label': self.primary_topic_label,
            'primary_topic_score': self.primary_topic_score,
            'secondary_topics': self.secondary_topics,
            'model_id': self.model_id,
            'assigned_at': self.assigned_at
        }


@dataclass
class TopicDefinition:
    """Topic metadata"""
    topic_id: int
    label: str                      # "Machine Learning"
    keywords: List[str]             # ["learning", "model", "training", ...]
    description: str                # Human-readable description
    num_chunks: int = 0             # Chunks assigned to this topic


# ============================================================================
# TRIGGER CONDITIONS
# ============================================================================

@dataclass
class TopicModelingTrigger:
    """When to trigger topic modeling"""

    # Threshold-based triggers
    min_chunks_for_training: int = 100      # Need at least N chunks
    chunks_since_last_train: int = 0        # Chunks added since last run
    retrain_threshold: int = 500            # Retrain after N new chunks

    # Schedule-based triggers
    schedule_enabled: bool = True
    schedule_cron: str = "0 2 * * *"       # 2 AM daily

    # Manual trigger
    manual_trigger: bool = False

    def should_trigger(self, total_chunks: int, new_chunks: int) -> bool:
        """Check if topic modeling should run"""

        # Manual override
        if self.manual_trigger:
            return True

        # Minimum chunks not met
        if total_chunks < self.min_chunks_for_training:
            return False

        # Threshold reached
        if new_chunks >= self.retrain_threshold:
            return True

        # Schedule (checked externally)
        return False


# ============================================================================
# TOPIC MODELING PIPELINE
# ============================================================================

class TopicModelingPipeline:
    """
    Async topic modeling enrichment.
    Runs AFTER chunks are stored in Qdrant.
    """

    def __init__(self, artifact_base_dir: str = "./artifacts"):
        from pathlib import Path
        self.artifact_base_dir = Path(artifact_base_dir)

    async def run_topic_modeling(
        self,
        doc_ids: List[str],
        model_version: TopicModelVersion = TopicModelVersion.V2_BERTOPIC,
        num_topics: int = 10
    ) -> TopicModelMetadata:
        """
        Run topic modeling on chunks from multiple documents.

        Steps:
        1. Load chunks from Qdrant (already embedded)
        2. Train topic model
        3. Assign topics to chunks
        4. Store topic assignments
        5. Update Qdrant metadata
        """

        # 1. Load chunks
        chunks = await self._load_chunks_from_qdrant(doc_ids)
        print(f"Loaded {len(chunks)} chunks for topic modeling")

        # 2. Train model
        model, topics = await self._train_model(chunks, model_version, num_topics)

        # 3. Assign topics
        assignments = await self._assign_topics(chunks, model, topics)

        # 4. Store artifacts
        model_metadata = await self._store_topic_artifacts(
            model, topics, assignments, model_version
        )

        # 5. Update Qdrant (non-blocking)
        await self._update_qdrant_with_topics(assignments)

        return model_metadata

    async def _load_chunks_from_qdrant(self, doc_ids: List[str]) -> List[dict]:
        """Load chunks from Qdrant"""
        # Query Qdrant for all chunks in doc_ids
        chunks = []

        # Pseudo-code (replace with actual Qdrant client)
        # for doc_id in doc_ids:
        #     results = qdrant.scroll(
        #         collection_name="documents",
        #         scroll_filter={'doc_id': doc_id}
        #     )
        #     chunks.extend(results)

        # For now, return mock
        return [
            {
                'chunk_id': f'chunk_{i}',
                'doc_id': doc_ids[0],
                'text': f'Sample text {i}',
                'embedding': [0.1] * 1536
            }
            for i in range(100)
        ]

    async def _train_model(
        self, chunks: List[dict],
        model_version: TopicModelVersion,
        num_topics: int
    ) -> tuple:
        """Train topic model"""

        texts = [c['text'] for c in chunks]

        if model_version == TopicModelVersion.V2_BERTOPIC:
            # Use embeddings already in chunks
            embeddings = [c['embedding'] for c in chunks]

            # Train BERTopic
            from bertopic import BERTopic
            model = BERTopic(nr_topics=num_topics)
            topics, probs = model.fit_transform(texts, embeddings)

            # Extract topic definitions
            topic_defs = []
            for topic_id in set(topics):
                if topic_id == -1:  # Outlier topic
                    continue

                topic_info = model.get_topic(topic_id)
                keywords = [word for word, _ in topic_info[:10]]

                topic_defs.append(TopicDefinition(
                    topic_id=topic_id,
                    label=f"Topic {topic_id}",
                    keywords=keywords,
                    description=", ".join(keywords[:5]),
                    num_chunks=sum(1 for t in topics if t == topic_id)
                ))

            return model, topic_defs

        else:
            # Fallback: LDA
            return None, []

    async def _assign_topics(
        self, chunks: List[dict], model, topics: List[TopicDefinition]
    ) -> List[TopicAssignment]:
        """Assign topics to chunks"""

        assignments = []
        model_id = f"topic_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get predictions
        texts = [c['text'] for c in chunks]
        embeddings = [c['embedding'] for c in chunks]

        topic_ids, probs = model.transform(texts, embeddings)

        for i, chunk in enumerate(chunks):
            primary_topic_id = topic_ids[i]
            primary_score = float(probs[i][primary_topic_id]) if probs is not None else 1.0

            # Find topic definition
            topic_def = next(
                (t for t in topics if t.topic_id == primary_topic_id),
                None
            )

            if topic_def:
                assignment = TopicAssignment(
                    chunk_id=chunk['chunk_id'],
                    doc_id=chunk['doc_id'],
                    primary_topic_id=primary_topic_id,
                    primary_topic_label=topic_def.label,
                    primary_topic_score=primary_score,
                    model_id=model_id,
                    assigned_at=datetime.now().isoformat()
                )
                assignments.append(assignment)

        return assignments

    async def _store_topic_artifacts(
        self, model, topics: List[TopicDefinition],
        assignments: List[TopicAssignment],
        model_version: TopicModelVersion
    ) -> TopicModelMetadata:
        """Store topic model artifacts"""

        model_id = f"topic_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store model metadata
        metadata = TopicModelMetadata(
            model_id=model_id,
            version=model_version,
            trained_at=datetime.now().isoformat(),
            num_topics=len(topics),
            num_documents=len(assignments)
        )

        # Store artifacts
        artifacts_dir = self.artifact_base_dir / "topic_models" / model_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        import json
        (artifacts_dir / "metadata.json").write_text(
            json.dumps(metadata.to_dict(), indent=2)
        )

        # Save topic definitions
        (artifacts_dir / "topics.json").write_text(
            json.dumps([t.__dict__ for t in topics], indent=2)
        )

        # Save assignments (per document)
        assignments_by_doc = {}
        for assignment in assignments:
            doc_id = assignment.doc_id
            if doc_id not in assignments_by_doc:
                assignments_by_doc[doc_id] = []
            assignments_by_doc[doc_id].append(assignment.to_dict())

        for doc_id, doc_assignments in assignments_by_doc.items():
            doc_artifact_dir = self.artifact_base_dir / doc_id / "04_enrichment"
            doc_artifact_dir.mkdir(parents=True, exist_ok=True)

            (doc_artifact_dir / "topics.json").write_text(
                json.dumps({
                    'model_id': model_id,
                    'assignments': doc_assignments
                }, indent=2)
            )

        return metadata

    async def _update_qdrant_with_topics(self, assignments: List[TopicAssignment]):
        """Update Qdrant chunks with topic metadata"""

        # Batch update Qdrant payloads
        # for assignment in assignments:
        #     qdrant.set_payload(
        #         collection_name="documents",
        #         points=[assignment.chunk_id],
        #         payload={
        #             'topic_id': assignment.primary_topic_id,
        #             'topic_label': assignment.primary_topic_label,
        #             'topic_score': assignment.primary_topic_score,
        #             'topic_model_id': assignment.model_id
        #         }
        #     )

        print(f"Updated {len(assignments)} chunks in Qdrant with topics")


# ============================================================================
# TRIGGER SCHEDULER
# ============================================================================

class TopicModelingScheduler:
    """Schedule and trigger topic modeling jobs"""

    def __init__(self):
        self.trigger_config = TopicModelingTrigger()
        self.pipeline = TopicModelingPipeline()

    async def check_and_trigger(self, qdrant_client) -> bool:
        """
        Check if topic modeling should run.
        Called periodically (e.g., every hour).
        """

        # Get corpus stats
        total_chunks = await self._get_total_chunks(qdrant_client)
        new_chunks = await self._get_chunks_without_topics(qdrant_client)

        if self.trigger_config.should_trigger(total_chunks, new_chunks):
            print(f"Triggering topic modeling: {new_chunks} new chunks")

            # Get all doc IDs
            doc_ids = await self._get_all_doc_ids(qdrant_client)

            # Run async
            await self.pipeline.run_topic_modeling(doc_ids)

            return True

        return False

    async def _get_total_chunks(self, qdrant_client) -> int:
        """Count total chunks in Qdrant"""
        # result = qdrant_client.count(collection_name="documents")
        # return result.count
        return 1000  # Mock

    async def _get_chunks_without_topics(self, qdrant_client) -> int:
        """Count chunks without topic assignments"""
        # result = qdrant_client.count(
        #     collection_name="documents",
        #     count_filter={'must_not': [{'key': 'topic_id', 'match': {'any': [...]}}]}
        # )
        return 150  # Mock

    async def _get_all_doc_ids(self, qdrant_client) -> List[str]:
        """Get all unique doc IDs"""
        return ["doc1", "doc2", "doc3"]  # Mock


# ============================================================================
# RETRIEVAL & UI USAGE
# ============================================================================

"""
How Retrieval Uses Topics:
===========================

1. Topic Filtering
------------------
GET /search?query="machine learning"&topic_id=5

qdrant.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter={
        'must': [
            {'key': 'topic_id', 'match': {'value': 5}}
        ]
    }
)


2. Topic Boosting
-----------------
# Boost results from specific topics
results = qdrant.search(...)
for result in results:
    if result['payload']['topic_id'] == preferred_topic:
        result['score'] *= 1.2  # Boost


3. Topic Diversification
-------------------------
# Show results from different topics
results_by_topic = {}
for result in results:
    topic = result['payload']['topic_id']
    if topic not in results_by_topic:
        results_by_topic[topic] = []
    results_by_topic[topic].append(result)

# Take top N from each topic
diverse_results = []
for topic_results in results_by_topic.values():
    diverse_results.extend(topic_results[:2])


4. Topic Discovery
-------------------
# "Show me what topics this document covers"
GET /documents/{doc_id}/topics

{
  "topics": [
    {
      "topic_id": 5,
      "label": "Machine Learning",
      "num_chunks": 42,
      "percentage": 35.0
    },
    {
      "topic_id": 8,
      "label": "Neural Networks",
      "num_chunks": 28,
      "percentage": 23.3
    }
  ]
}


How UI Uses Topics:
===================

1. Topic Facets (Sidebar)
--------------------------
Search Results
├── Machine Learning (42)
├── Neural Networks (28)
├── Data Processing (15)
└── Statistics (8)

[Click to filter by topic]


2. Document Overview
--------------------
Document: research_paper.pdf
Topics:
  ███████████░░░░░░░░░ Machine Learning (35%)
  ████████░░░░░░░░░░░░ Neural Networks (23%)
  ████░░░░░░░░░░░░░░░░ Data Processing (12%)


3. Related Documents
--------------------
# Find documents with similar topic distributions
current_doc_topics = [5, 8, 12]
similar_docs = find_docs_with_topics(current_doc_topics)


4. Topic Timeline
-----------------
# Show how topics evolve across document
Page 1-10:  [Topic 5: Introduction]
Page 11-25: [Topic 8: Methodology]
Page 26-40: [Topic 12: Results]


5. Cross-Reference
------------------
# "Other sections about Machine Learning"
GET /documents/{doc_id}/chunks?topic_id=5

Returns all chunks in same doc with same topic


Artifacts Produced:
===================

artifacts/topic_models/{model_id}/
├── metadata.json           # Model version, training info
├── topics.json             # Topic definitions, keywords
└── model.pkl               # Serialized model (optional)

artifacts/{doc_id}/04_enrichment/
└── topics.json             # Topic assignments for this doc

Qdrant Payload Update:
{
  'chunk_id': 'doc::chunk::5',
  'text': '...',
  'page_start': 5,

  # Topic metadata (added after enrichment)
  'topic_id': 5,
  'topic_label': 'Machine Learning',
  'topic_score': 0.87,
  'topic_model_id': 'topic_model_20240115_120000'
}
"""


if __name__ == "__main__":
    print("Topic Modeling as Non-Blocking Enrichment")
    print("=" * 70)
    print("\nLifecycle Position:")
    print("  1-5: Core ingestion (BLOCKING)")
    print("  6-7: Topic modeling (NON-BLOCKING, ASYNC)")
    print("\nTriggers:")
    print("  - Threshold: 500 new chunks")
    print("  - Schedule: Daily at 2 AM")
    print("  - Manual: On-demand")
    print("\nArtifacts:")
    print("  - Model metadata: artifacts/topic_models/{model_id}/")
    print("  - Assignments: artifacts/{doc_id}/04_enrichment/topics.json")
    print("  - Qdrant update: Add topic_id to chunk payloads")
