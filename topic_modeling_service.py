"""
Topic Modeling Service for KnowVec

Implements NMF (Non-negative Matrix Factorization) topic modeling with:
- Document-level and chunk-level topic assignment
- Collection-wide global topic model
- Automatic model retraining based on collection growth
- Efficient sparse matrix operations

Features:
- Hybrid topic assignment (document + chunk level)
- Lazy model initialization
- Background retraining
- Topic label generation
- Model persistence with versioning
"""

import os
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import time
import json
from logger_config import get_logger

logger = get_logger(__name__)

# Scikit-learn imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn loaded successfully for topic modeling")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class TopicConfig:
    """Configuration for topic modeling"""
    enabled: bool = True
    n_topics: int = 10
    max_features: int = 5000
    min_df: int = 2
    max_df: float = 0.85
    ngram_range: Tuple[int, int] = (1, 2)
    model_dir: str = "models/topics"
    retrain_threshold_docs: int = 100
    retrain_threshold_pct: float = 0.20
    random_state: int = 42
    nmf_init: str = 'nndsvd'
    nmf_solver: str = 'cd'
    nmf_max_iter: int = 200
    top_keywords_per_topic: int = 10


@dataclass
class TopicAssignment:
    """Topic assignment for a chunk or document"""
    topic_id: int
    topic_label: str
    confidence: float
    keywords: List[str] = field(default_factory=list)
    distribution: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ModelMetadata:
    """Metadata about the topic model"""
    version: str
    created_at: str
    n_topics: int
    n_documents: int
    n_training_samples: int
    model_params: Dict[str, Any] = field(default_factory=dict)
    topic_labels: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class NMFTopicModel:
    """
    NMF-based topic model for document and chunk analysis.

    Uses TF-IDF vectorization followed by NMF decomposition.
    """

    def __init__(self, config: TopicConfig):
        logger.info("Initializing NMFTopicModel")
        self.config = config
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.nmf_model: Optional[NMF] = None
        self.topic_labels: Dict[int, str] = {}
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for topic modeling. Install with: pip install scikit-learn")

    def train(self, texts: List[str]) -> None:
        """
        Train NMF topic model on a collection of texts.

        Args:
            texts: List of text documents to train on
        """
        start_time = time.time()
        logger.info(f"Training NMF topic model on {len(texts)} texts...")

        if len(texts) < self.config.n_topics:
            logger.warning(
                f"Number of texts ({len(texts)}) is less than n_topics ({self.config.n_topics}). "
                f"Reducing n_topics to {len(texts)}."
            )
            self.config.n_topics = max(2, len(texts))

        # Step 1: TF-IDF vectorization
        logger.debug("Creating TF-IDF vectorizer")
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            ngram_range=self.config.ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )

        logger.debug("Fitting TF-IDF vectorizer and transforming texts")
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

        logger.info(
            f"TF-IDF matrix shape: {X.shape}, "
            f"Features: {len(self.feature_names)}, "
            f"Sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])):.2%}"
        )

        # Step 2: NMF decomposition
        logger.debug(f"Initializing NMF model (n_components={self.config.n_topics})")
        self.nmf_model = NMF(
            n_components=self.config.n_topics,
            init=self.config.nmf_init,
            solver=self.config.nmf_solver,
            max_iter=self.config.nmf_max_iter,
            random_state=self.config.random_state,
            alpha_W=0.1,  # Regularization
            alpha_H=0.1,
            l1_ratio=0.5
        )

        logger.debug("Fitting NMF model")
        self.nmf_model.fit(X)

        # Step 3: Generate topic labels
        logger.debug("Generating topic labels")
        self.topic_labels = self._generate_topic_labels()

        self.is_trained = True
        duration = time.time() - start_time

        logger.info(
            f"NMF model trained successfully - "
            f"Topics: {self.config.n_topics}, "
            f"Reconstruction error: {self.nmf_model.reconstruction_err_:.4f}, "
            f"Duration: {duration:.2f}s"
        )

        # Log topic summaries
        for topic_id, label in self.topic_labels.items():
            keywords = self._get_top_keywords(topic_id, n=5)
            logger.info(f"  Topic {topic_id}: {label} | Keywords: {', '.join(keywords)}")

    def transform(self, texts: List[str]) -> List[TopicAssignment]:
        """
        Assign topics to a list of texts.

        Args:
            texts: List of texts to assign topics to

        Returns:
            List of TopicAssignment objects
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calling transform()")

        if not texts:
            return []

        # Vectorize texts
        X = self.vectorizer.transform(texts)

        # Transform with NMF
        H = self.nmf_model.transform(X)

        # Convert to TopicAssignment objects
        results = []
        for row in H:
            topic_id = int(row.argmax())
            confidence = float(row[topic_id])

            results.append(TopicAssignment(
                topic_id=topic_id,
                topic_label=self.topic_labels.get(topic_id, f"Topic_{topic_id}"),
                confidence=confidence,
                keywords=self._get_top_keywords(topic_id),
                distribution=row.tolist()
            ))

        return results

    def get_dominant_topic(self, text: str) -> TopicAssignment:
        """
        Get the dominant topic for a single text.

        Args:
            text: Text to analyze

        Returns:
            TopicAssignment for the dominant topic
        """
        results = self.transform([text])
        return results[0] if results else TopicAssignment(
            topic_id=-1,
            topic_label="[UNKNOWN]",
            confidence=0.0
        )

    def _generate_topic_labels(self) -> Dict[int, str]:
        """
        Generate human-readable labels for topics based on top keywords.

        Returns:
            Dictionary mapping topic_id to label string
        """
        labels = {}

        for topic_id in range(self.config.n_topics):
            top_keywords = self._get_top_keywords(topic_id, n=3)

            # Create label from top keywords
            label = " & ".join([word.title() for word in top_keywords[:2]])
            labels[topic_id] = label

        return labels

    def _get_top_keywords(self, topic_id: int, n: int = None) -> List[str]:
        """
        Get top N keywords for a topic.

        Args:
            topic_id: Topic index
            n: Number of keywords (default from config)

        Returns:
            List of top keywords
        """
        if n is None:
            n = self.config.top_keywords_per_topic

        if not self.is_trained or self.nmf_model is None:
            return []

        # Get topic vector
        topic_vector = self.nmf_model.components_[topic_id]

        # Get indices of top features
        top_indices = topic_vector.argsort()[-n:][::-1]

        # Map to feature names
        keywords = [self.feature_names[i] for i in top_indices]

        return keywords

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model (without extension)
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save model components
        model_data = {
            'vectorizer': self.vectorizer,
            'nmf_model': self.nmf_model,
            'topic_labels': self.topic_labels,
            'feature_names': self.feature_names,
            'config': asdict(self.config),
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, f"{path}.pkl")
        logger.info(f"Model saved to {path}.pkl")

    def load(self, path: str) -> bool:
        """
        Load model from disk.

        Args:
            path: Path to model file (without extension)

        Returns:
            True if loaded successfully
        """
        try:
            model_data = joblib.load(f"{path}.pkl")

            self.vectorizer = model_data['vectorizer']
            self.nmf_model = model_data['nmf_model']
            self.topic_labels = model_data['topic_labels']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']

            # Update config from loaded model
            loaded_config = model_data['config']
            self.config.n_topics = loaded_config['n_topics']

            logger.info(f"Model loaded from {path}.pkl (n_topics={self.config.n_topics})")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from {path}.pkl: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            'is_trained': self.is_trained,
            'n_topics': self.config.n_topics,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'topic_labels': self.topic_labels,
            'reconstruction_error': float(self.nmf_model.reconstruction_err_) if self.is_trained else None
        }


class TopicModelManager:
    """
    Manages topic model lifecycle including creation, retraining, and topic assignment.

    Handles:
    - Lazy model initialization
    - Periodic retraining based on collection growth
    - Hybrid topic assignment (document + chunk level)
    - Model persistence and versioning
    """

    def __init__(self, config: TopicConfig, qdrant_storage):
        logger.info("Initializing TopicModelManager")
        self.config = config
        self.qdrant_storage = qdrant_storage
        self.current_model: Optional[NMFTopicModel] = None
        self.model_path = Path(self.config.model_dir) / "current_model"
        self.metadata_path = Path(self.config.model_dir) / "model_metadata.json"
        self.last_trained_doc_count = 0

        # Create model directory
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

        # Try to load existing model
        self._load_or_create_model()

        logger.info("TopicModelManager initialized successfully")

    def _load_or_create_model(self) -> None:
        """Load existing model or prepare for creation on first document"""
        if self.model_path.with_suffix('.pkl').exists():
            logger.info("Loading existing topic model...")
            self.current_model = NMFTopicModel(self.config)
            if self.current_model.load(str(self.model_path)):
                self._load_metadata()
                logger.info(
                    f"Topic model loaded successfully - "
                    f"Topics: {self.current_model.config.n_topics}, "
                    f"Last trained on: {self.last_trained_doc_count} documents"
                )
            else:
                logger.warning("Failed to load model, will create new on first document")
                self.current_model = None
        else:
            logger.info("No existing model found, will create on first document")
            self.current_model = None

    def _load_metadata(self) -> None:
        """Load model metadata"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.last_trained_doc_count = metadata.get('n_documents', 0)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            self.last_trained_doc_count = 0

    def _save_metadata(self, n_documents: int, n_training_samples: int) -> None:
        """Save model metadata"""
        metadata = ModelMetadata(
            version=datetime.now().strftime("v%Y%m%d_%H%M%S"),
            created_at=datetime.now().isoformat(),
            n_topics=self.config.n_topics,
            n_documents=n_documents,
            n_training_samples=n_training_samples,
            model_params={
                'max_features': self.config.max_features,
                'min_df': self.config.min_df,
                'max_df': self.config.max_df,
                'nmf_init': self.config.nmf_init,
                'nmf_solver': self.config.nmf_solver
            },
            topic_labels=self.current_model.topic_labels if self.current_model else {}
        )

        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            logger.info(f"Model metadata saved: {metadata.version}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_or_create_model(self, texts: List[str]) -> NMFTopicModel:
        """
        Get current model or create new one if needed.

        Args:
            texts: Training texts (used only if model needs creation)

        Returns:
            Current topic model
        """
        if self.current_model is None:
            logger.info("Creating initial topic model...")
            self.current_model = NMFTopicModel(self.config)
            self.current_model.train(texts)

            # Save model
            self.current_model.save(str(self.model_path))
            self._save_metadata(n_documents=1, n_training_samples=len(texts))

            logger.info("Initial topic model created and saved")

        return self.current_model

    def should_retrain(self) -> bool:
        """
        Check if model should be retrained based on collection growth.

        Returns:
            True if retraining is recommended
        """
        try:
            # Get current collection stats
            stats = self.qdrant_storage.collection_manager.collection_info()
            current_doc_count = stats.get('points_count', 0)

            # Check if we've crossed the threshold
            docs_added = current_doc_count - self.last_trained_doc_count

            if docs_added >= self.config.retrain_threshold_docs:
                logger.info(
                    f"Retrain threshold reached: {docs_added} docs added "
                    f"(threshold: {self.config.retrain_threshold_docs})"
                )
                return True

            # Check percentage growth
            if self.last_trained_doc_count > 0:
                growth_rate = docs_added / self.last_trained_doc_count
                if growth_rate >= self.config.retrain_threshold_pct:
                    logger.info(
                        f"Retrain threshold reached: {growth_rate:.1%} growth "
                        f"(threshold: {self.config.retrain_threshold_pct:.1%})"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking retrain threshold: {e}")
            return False

    def trigger_retrain_async(self) -> None:
        """
        Trigger asynchronous model retraining.

        Note: Currently synchronous. In production, this should use
        background tasks (e.g., FastAPI BackgroundTasks, Celery, etc.)
        """
        logger.info("Retraining triggered (currently synchronous)")

        try:
            # Fetch all texts from Qdrant
            logger.info("Fetching all documents from Qdrant for retraining...")
            all_chunks = self._fetch_all_texts()

            if len(all_chunks) < self.config.n_topics:
                logger.warning(f"Not enough documents for retraining ({len(all_chunks)} < {self.config.n_topics})")
                return

            # Create backup of current model
            if self.current_model:
                backup_path = Path(self.config.model_dir) / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.current_model.save(str(backup_path))
                logger.info(f"Backup saved to {backup_path}.pkl")

            # Train new model
            logger.info(f"Retraining model on {len(all_chunks)} chunks...")
            new_model = NMFTopicModel(self.config)
            new_model.train(all_chunks)

            # Save new model
            new_model.save(str(self.model_path))
            self.current_model = new_model

            # Update metadata
            stats = self.qdrant_storage.collection_manager.collection_info()
            self.last_trained_doc_count = stats.get('points_count', 0)
            self._save_metadata(
                n_documents=self.last_trained_doc_count,
                n_training_samples=len(all_chunks)
            )

            logger.info("Model retrained and saved successfully")

        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)

    def _fetch_all_texts(self) -> List[str]:
        """
        Fetch all chunk texts from Qdrant for retraining.

        Returns:
            List of all chunk texts
        """
        try:
            # Use scroll to fetch all points
            all_texts = []
            offset = None
            batch_size = 100

            while True:
                results = self.qdrant_storage.client.scroll(
                    collection_name=self.qdrant_storage.config.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = results

                if not points:
                    break

                for point in points:
                    text = point.payload.get('text', '')
                    if text:
                        all_texts.append(text)

                if next_offset is None:
                    break

                offset = next_offset

            logger.info(f"Fetched {len(all_texts)} texts from Qdrant")
            return all_texts

        except Exception as e:
            logger.error(f"Failed to fetch texts from Qdrant: {e}")
            return []

    def assign_topics_hybrid(
        self,
        chunks: List,
        embeddings_text: List[str]
    ) -> Tuple[TopicAssignment, List[TopicAssignment]]:
        """
        Assign topics at both document and chunk levels.

        Args:
            chunks: List of chunk metadata objects
            embeddings_text: List of cleaned texts (same as used for embeddings)

        Returns:
            Tuple of (document_topic, chunk_topics)
        """
        # Handle empty or insufficient text
        if not embeddings_text or len(embeddings_text) < 3:
            default_topic = TopicAssignment(
                topic_id=-1,
                topic_label="[INSUFFICIENT_TEXT]",
                confidence=0.0,
                keywords=[],
                distribution=[]
            )
            return default_topic, [default_topic] * len(chunks)

        # Filter out very short texts
        valid_texts = [t for t in embeddings_text if len(t.strip()) > 50]
        if len(valid_texts) < 3:
            default_topic = TopicAssignment(
                topic_id=-1,
                topic_label="[SHORT_DOCUMENT]",
                confidence=0.0,
                keywords=[],
                distribution=[]
            )
            return default_topic, [default_topic] * len(chunks)

        # Get or create model
        model = self.get_or_create_model(valid_texts)

        # Document-level: concatenate all texts
        document_text = " ".join(embeddings_text)
        doc_topic = model.get_dominant_topic(document_text)

        # Chunk-level: assign topics to each chunk
        chunk_topics = model.transform(embeddings_text)

        logger.debug(
            f"Assigned topics - Document: {doc_topic.topic_label} ({doc_topic.confidence:.2f}), "
            f"Chunks: {len(chunk_topics)}"
        )

        return doc_topic, chunk_topics

    def get_topic_stats(self) -> Dict[str, Any]:
        """Get statistics about current topic model"""
        if not self.current_model:
            return {
                'model_exists': False,
                'message': 'No model trained yet'
            }

        return {
            'model_exists': True,
            'model_info': self.current_model.get_model_info(),
            'last_trained_doc_count': self.last_trained_doc_count,
            'model_path': str(self.model_path)
        }


# Convenience function for testing
def test_topic_modeling():
    """Test topic modeling functionality"""
    print("=" * 80)
    print("Topic Modeling Service Test")
    print("=" * 80)

    # Sample texts
    texts = [
        "Machine learning algorithms use neural networks for training deep models",
        "Neural networks consist of layers that process information hierarchically",
        "Data science involves statistics and machine learning techniques",
        "Python programming is popular for data analysis and visualization",
        "Web development uses HTML, CSS, and JavaScript frameworks",
        "React and Vue are modern JavaScript frameworks for building UIs",
        "Database management systems store and retrieve structured data",
        "SQL queries extract information from relational databases"
    ]

    # Create config
    config = TopicConfig(
        enabled=True,
        n_topics=3,
        max_features=100
    )

    # Create and train model
    model = NMFTopicModel(config)
    model.train(texts)

    # Test transformation
    test_texts = [
        "Deep learning neural network architectures",
        "JavaScript frontend framework development"
    ]

    assignments = model.transform(test_texts)

    print("\nTopic Assignments:")
    for i, (text, assignment) in enumerate(zip(test_texts, assignments)):
        print(f"\nText {i + 1}: '{text[:50]}...'")
        print(f"  Topic: {assignment.topic_label}")
        print(f"  Confidence: {assignment.confidence:.2f}")
        print(f"  Keywords: {', '.join(assignment.keywords[:5])}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_topic_modeling()
