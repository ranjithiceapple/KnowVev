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

    # Pipeline version
    version: str


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        # Qdrant
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "knowvec_documents"),

        # Embedding
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_dimension=get_int("EMBEDDING_DIMENSION", 384),
        embedding_batch_size=get_int("EMBEDDING_BATCH_SIZE", 32),

        # Processing
        max_chunk_size=get_int("MAX_CHUNK_SIZE", 1000),
        target_chunk_size=get_int("TARGET_CHUNK_SIZE", 800),
        enable_overlap=get_bool("ENABLE_OVERLAP", True),
        overlap_size=get_int("OVERLAP_SIZE", 200),
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
        api_port=get_int("API_PORT", 8000),
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

    print(f"\n🔢 EMBEDDING MODEL")
    print(f"  Model: {cfg.embedding_model}")
    print(f"  Dimension: {cfg.embedding_dimension}")
    print(f"  Batch Size: {cfg.embedding_batch_size}")

    print(f"\n⚙️  PROCESSING")
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

    print(f"\n🔍 SEARCH")
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

    print(f"\n📝 LOGGING")
    print(f"  Level: {cfg.log_level}")
    print(f"  Format: {cfg.log_format}")
    print(f"  File Logging: {cfg.enable_file_logging}")
    print(f"  Log Directory: {cfg.log_dir}")
    print(f"  Show Progress: {cfg.show_progress}")
    print(f"  Timing Logs: {cfg.enable_timing_logs}")

    print(f"\n🏷️  VERSION")
    print(f"  Pipeline Version: {cfg.version}")

    print("=" * 80)
