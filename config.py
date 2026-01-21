"""
Configuration Loader for KnowVec RAG Pipeline
Loads all settings from .env file
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: Optional[str]
    qdrant_collection: str
    qdrant_timeout: int
    qdrant_retry_count: int

    # OpenSearch Configuration (for Hybrid Search)
    opensearch_enabled: bool
    opensearch_host: str
    opensearch_port: int
    opensearch_username: Optional[str]
    opensearch_password: Optional[str]
    opensearch_use_ssl: bool
    opensearch_verify_certs: bool
    opensearch_ssl_show_warn: bool
    opensearch_index: str
    opensearch_timeout: int
    opensearch_max_retries: int

    # Hybrid Search Configuration
    generate_heading_chunks: bool
    generate_clause_chunks: bool
    generate_metadata_chunks: bool
    generate_summary_chunks: bool
    keyword_max_keywords: int
    keyword_min_word_length: int
    keyword_include_phrases: bool

    # Embedding Configuration
    embedding_model: str
    embedding_dimension: int
    embedding_batch_size: int

    # Processing Configuration
    max_chunk_size: int
    target_chunk_size: int
    enable_overlap: bool
    overlap_size: int
    respect_page_boundaries: bool
    keep_tables_intact: bool
    remove_toc_pages: bool
    protect_headings: bool
    protect_tables: bool
    protect_code_blocks: bool
    detect_multi_column: bool
    deduplicate_chunks: bool
    aggressive_text_cleaning: bool

    # Summary Configuration
    generate_document_summary: bool
    summary_method: str
    summary_max_length: int
    summary_min_doc_length: int

    # Search Configuration
    min_similarity_score: float
    default_search_limit: int
    max_search_limit: int

    # API Configuration
    api_host: str
    api_port: int
    api_debug: bool
    max_upload_size_mb: int
    cors_origins: str

    # Performance Configuration
    num_workers: int
    cache_size: int

    # Logging Configuration
    log_level: str
    log_format: str
    enable_file_logging: bool
    log_dir: str
    show_progress: bool
    enable_timing_logs: bool
    enable_memory_profiling: bool

    # Topic Modeling Configuration
    enable_topic_modeling: bool
    topic_n_topics: int
    topic_max_features: int
    topic_min_df: int
    topic_max_df: float
    topic_model_dir: str
    topic_retrain_threshold_docs: int
    topic_retrain_threshold_pct: float

    # Pipeline version
    version: str


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        # Qdrant
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "documents"),
        qdrant_timeout=get_int("QDRANT_TIMEOUT", 5),
        qdrant_retry_count=get_int("QDRANT_RETRY_COUNT", 3),

        # OpenSearch (for Hybrid Search)
        opensearch_enabled=get_bool("OPENSEARCH_ENABLED", True),
        opensearch_host=os.getenv("OPENSEARCH_HOST", "localhost"),
        opensearch_port=get_int("OPENSEARCH_PORT", 9200),
        opensearch_username=os.getenv("OPENSEARCH_USERNAME", "admin"),
        opensearch_password=os.getenv("OPENSEARCH_PASSWORD", "ArivurAI@123"),
        opensearch_use_ssl=get_bool("OPENSEARCH_USE_SSL", False),
        opensearch_verify_certs=get_bool("OPENSEARCH_VERIFY_CERTS", False),
        opensearch_ssl_show_warn=get_bool("OPENSEARCH_SSL_SHOW_WARN", False),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", "knowvec_keywords"),
        opensearch_timeout=get_int("OPENSEARCH_TIMEOUT", 30),
        opensearch_max_retries=get_int("OPENSEARCH_MAX_RETRIES", 3),

        # Hybrid Search Chunk Generation
        generate_heading_chunks=get_bool("GENERATE_HEADING_CHUNKS", True),
        generate_clause_chunks=get_bool("GENERATE_CLAUSE_CHUNKS", True),
        generate_metadata_chunks=get_bool("GENERATE_METADATA_CHUNKS", True),
        generate_summary_chunks=get_bool("GENERATE_SUMMARY_CHUNKS", True),
        keyword_max_keywords=get_int("KEYWORD_MAX_KEYWORDS", 50),
        keyword_min_word_length=get_int("KEYWORD_MIN_WORD_LENGTH", 3),
        keyword_include_phrases=get_bool("KEYWORD_INCLUDE_PHRASES", True),

        # Embedding
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        embedding_dimension=get_int("EMBEDDING_DIMENSION", 768),
        embedding_batch_size=get_int("EMBEDDING_BATCH_SIZE", 32),

        # Processing
        max_chunk_size=get_int("MAX_CHUNK_SIZE", 1500),
        target_chunk_size=get_int("TARGET_CHUNK_SIZE", 1200),
        enable_overlap=get_bool("ENABLE_OVERLAP", True),
        overlap_size=get_int("OVERLAP_SIZE", 250),
        respect_page_boundaries=get_bool("RESPECT_PAGE_BOUNDARIES", False),
        keep_tables_intact=get_bool("KEEP_TABLES_INTACT", True),
        remove_toc_pages=get_bool("REMOVE_TOC_PAGES", True),
        protect_headings=get_bool("PROTECT_HEADINGS", True),
        protect_tables=get_bool("PROTECT_TABLES", True),
        protect_code_blocks=get_bool("PROTECT_CODE_BLOCKS", True),
        detect_multi_column=get_bool("DETECT_MULTI_COLUMN", True),
        deduplicate_chunks=get_bool("DEDUPLICATE_CHUNKS", True),
        aggressive_text_cleaning=get_bool("AGGRESSIVE_TEXT_CLEANING", False),

        # Summary
        generate_document_summary=get_bool("GENERATE_DOCUMENT_SUMMARY", True),
        summary_method=os.getenv("SUMMARY_METHOD", "hybrid"),
        summary_max_length=get_int("SUMMARY_MAX_LENGTH", 500),
        summary_min_doc_length=get_int("SUMMARY_MIN_DOC_LENGTH", 1000),

        # Search
        min_similarity_score=get_float("MIN_SIMILARITY_SCORE", 0.3),
        default_search_limit=get_int("DEFAULT_SEARCH_LIMIT", 10),
        max_search_limit=get_int("MAX_SEARCH_LIMIT", 100),

        # API
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=get_int("API_PORT", 8007),
        api_debug=get_bool("API_DEBUG", False),
        max_upload_size_mb=get_int("MAX_UPLOAD_SIZE_MB", 50),
        cors_origins=os.getenv("CORS_ORIGINS", "*"),

        # Performance
        num_workers=get_int("NUM_WORKERS", 4),
        cache_size=get_int("CACHE_SIZE", 100),

        # Logging
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "standard"),
        enable_file_logging=get_bool("ENABLE_FILE_LOGGING", True),
        log_dir=os.getenv("LOG_DIR", "logs"),
        show_progress=get_bool("SHOW_PROGRESS", True),
        enable_timing_logs=get_bool("ENABLE_TIMING_LOGS", True),
        enable_memory_profiling=get_bool("ENABLE_MEMORY_PROFILING", False),

        # Topic Modeling
        enable_topic_modeling=get_bool("ENABLE_TOPIC_MODELING", True),
        topic_n_topics=get_int("TOPIC_N_TOPICS", 10),
        topic_max_features=get_int("TOPIC_MAX_FEATURES", 5000),
        topic_min_df=get_int("TOPIC_MIN_DF", 1),
        topic_max_df=get_float("TOPIC_MAX_DF", 0.85),
        topic_model_dir=os.getenv("TOPIC_MODEL_DIR", "models/topics"),
        topic_retrain_threshold_docs=get_int("TOPIC_RETRAIN_THRESHOLD_DOCS", 100),
        topic_retrain_threshold_pct=get_float("TOPIC_RETRAIN_THRESHOLD_PCT", 0.20),

        # Version
        version=os.getenv("VERSION", "1.0"),
    )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


if __name__ == "__main__":
    """Test configuration loading."""
    cfg = get_config()

    print("=" * 80)
    print("KnowVec RAG Pipeline - Configuration")
    print("=" * 80)

    print(f"\n📊 QDRANT DATABASE")
    print(f"  URL: {cfg.qdrant_url}")
    print(f"  Collection: {cfg.qdrant_collection}")
    print(f"  API Key: {'***' if cfg.qdrant_api_key else 'Not set'}")

    print(f"\n🔍 OPENSEARCH (HYBRID SEARCH)")
    print(f"  Enabled: {cfg.opensearch_enabled}")
    print(f"  Host: {cfg.opensearch_host}:{cfg.opensearch_port}")
    print(f"  Index: {cfg.opensearch_index}")
    print(f"  Username: {cfg.opensearch_username}")
    print(f"  Password: {'***' if cfg.opensearch_password else 'Not set'}")
    print(f"  SSL: {cfg.opensearch_use_ssl}")
    print(f"  Verify Certs: {cfg.opensearch_verify_certs}")
    print(f"  Timeout: {cfg.opensearch_timeout}s")
    print(f"  Max Retries: {cfg.opensearch_max_retries}")

    print(f"\n🔧 HYBRID SEARCH CHUNKS")
    print(f"  Heading Chunks: {cfg.generate_heading_chunks}")
    print(f"  Clause Chunks: {cfg.generate_clause_chunks}")
    print(f"  Metadata Chunks: {cfg.generate_metadata_chunks}")
    print(f"  Summary Chunks: {cfg.generate_summary_chunks}")
    print(f"  Max Keywords: {cfg.keyword_max_keywords}")
    print(f"  Min Word Length: {cfg.keyword_min_word_length}")
    print(f"  Include Phrases: {cfg.keyword_include_phrases}")

    print(f"\n🔢 EMBEDDING MODEL")
    print(f"  Model: {cfg.embedding_model}")
    print(f"  Dimension: {cfg.embedding_dimension}")
    print(f"  Batch Size: {cfg.embedding_batch_size}")

    print(f"\n⚙️ PROCESSING")
    print(f"  Max Chunk Size: {cfg.max_chunk_size}")
    print(f"  Target Chunk Size: {cfg.target_chunk_size}")
    print(f"  Overlap: {cfg.enable_overlap} ({cfg.overlap_size} chars)")
    print(f"  Deduplicate: {cfg.deduplicate_chunks}")
    print(f"  Remove TOC Pages: {cfg.remove_toc_pages}")
    print(f"  Protect Headings: {cfg.protect_headings}")
    print(f"  Protect Tables: {cfg.protect_tables}")
    print(f"  Protect Code Blocks: {cfg.protect_code_blocks}")
    print(f"  Detect Multi-Column: {cfg.detect_multi_column}")

    print(f"\n📝 DOCUMENT SUMMARY")
    print(f"  Enabled: {cfg.generate_document_summary}")
    print(f"  Method: {cfg.summary_method}")
    print(f"  Max Length: {cfg.summary_max_length} chars")

    print(f"\n🔎 SEARCH")
    print(f"  Min Similarity Score: {cfg.min_similarity_score}")
    print(f"  Default Limit: {cfg.default_search_limit}")
    print(f"  Max Limit: {cfg.max_search_limit}")

    print(f"\n🌐 API SERVER")
    print(f"  Host: {cfg.api_host}")
    print(f"  Port: {cfg.api_port}")
    print(f"  Debug Mode: {cfg.api_debug}")
    print(f"  Max Upload: {cfg.max_upload_size_mb} MB")
    print(f"  CORS Origins: {cfg.cors_origins}")

    print(f"\n⚡ PERFORMANCE")
    print(f"  Workers: {cfg.num_workers}")
    print(f"  Cache Size: {cfg.cache_size}")

    print(f"\n📋 LOGGING")
    print(f"  Level: {cfg.log_level}")
    print(f"  Format: {cfg.log_format}")
    print(f"  File Logging: {cfg.enable_file_logging}")
    print(f"  Log Directory: {cfg.log_dir}")
    print(f"  Show Progress: {cfg.show_progress}")
    print(f"  Timing Logs: {cfg.enable_timing_logs}")

    print(f"\n🏷️ TOPIC MODELING")
    print(f"  Enabled: {cfg.enable_topic_modeling}")
    print(f"  Number of Topics: {cfg.topic_n_topics}")
    print(f"  Max Features: {cfg.topic_max_features}")

    print(f"\n🏷️ VERSION")
    print(f"  Pipeline Version: {cfg.version}")

    print("=" * 80)