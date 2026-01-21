"""
OpenSearch Connection Test Script

Tests the connection to OpenSearch and verifies the setup for hybrid search.
Now supports loading credentials from config.py

Usage:
    python test_opensearch_connection.py                    # Use config.py settings
    python test_opensearch_connection.py --use-config       # Explicit config.py
    python test_opensearch_connection.py --host localhost   # Override host
"""

import sys
import argparse

# Check if opensearch-py is installed
try:
    from opensearchpy import OpenSearch
    OPENSEARCH_INSTALLED = True
except ImportError:
    OPENSEARCH_INSTALLED = False


def load_config_settings():
    """Load OpenSearch settings from config.py if available."""
    try:
        from config import get_config
        cfg = get_config()
        return {
            'host': cfg.opensearch_host,
            'port': cfg.opensearch_port,
            'username': cfg.opensearch_username,
            'password': cfg.opensearch_password,
            'use_ssl': cfg.opensearch_use_ssl,
            'enabled': cfg.opensearch_enabled,
            'index': cfg.opensearch_index
        }
    except ImportError:
        print("WARNING: config.py not found, using default settings")
        return None
    except Exception as e:
        print(f"WARNING: Error loading config: {e}")
        return None


def test_basic_connection(host: str, port: int, username: str, password: str, use_ssl: bool, no_auth: bool = False):
    """Test basic OpenSearch connection."""
    print(f"\n{'='*60}")
    print("OPENSEARCH CONNECTION TEST")
    print(f"{'='*60}")

    # Check if library is installed
    print("\n[1] Checking opensearch-py installation...")
    if not OPENSEARCH_INSTALLED:
        print("   ❌ FAILED: opensearch-py not installed")
        print("   Fix: pip install opensearch-py")
        print("   Or: pip3 install opensearch-py")
        return False
    print("   ✅ OK: opensearch-py is installed")

    # Test connection
    print(f"\n[2] Connecting to OpenSearch at {host}:{port}...")
    print(f"    SSL: {use_ssl}, Auth: {'disabled' if no_auth else username}")

    try:
        client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=(username, password) if (username and not no_auth) else None,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=10
        )

        # Test cluster health
        info = client.info()
        print(f"   ✅ OK: Connected to OpenSearch")
        print(f"       Cluster: {info.get('cluster_name', 'N/A')}")
        print(f"       Version: {info.get('version', {}).get('number', 'N/A')}")

    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print("\n   Troubleshooting:")
        print("   - Check if OpenSearch container is running:")
        print("     docker ps | grep opensearch")
        print("   - Check if port 9200 is accessible:")
        print("     netstat -tuln | grep 9200")
        print("   - Try different host:")
        print("     localhost, 127.0.0.1, or iceapple-server1")
        print("   - Verify credentials in .env file:")
        print("     OPENSEARCH_USERNAME=admin")
        print("     OPENSEARCH_PASSWORD=ArivurAI@123")
        print("   - Check OpenSearch logs:")
        print("     docker logs <opensearch-container-name>")
        return False

    # Test cluster health
    print("\n[3] Checking cluster health...")
    try:
        health = client.cluster.health()
        status = health.get('status', 'unknown')
        print(f"   ✅ OK: Cluster status is '{status}'")
        if status == 'red':
            print("   ⚠️  WARNING: Cluster health is RED - some shards are not allocated")
        elif status == 'yellow':
            print("   ⚠️  INFO: Cluster health is YELLOW - no replica shards (normal for single-node)")
    except Exception as e:
        print(f"   ⚠️  WARNING: Could not get cluster health: {e}")

    # Test index operations
    print("\n[4] Testing index operations...")
    test_index = "connection_test_index"
    try:
        # Create test index
        if client.indices.exists(index=test_index):
            client.indices.delete(index=test_index)

        client.indices.create(index=test_index, body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0}
        })
        print(f"   ✅ OK: Created test index '{test_index}'")

        # Index a document
        client.index(index=test_index, id="1", body={"test": "document"}, refresh=True)
        print("   ✅ OK: Indexed test document")

        # Search
        result = client.search(index=test_index, body={"query": {"match_all": {}}})
        hits = result.get('hits', {}).get('total', {}).get('value', 0)
        print(f"   ✅ OK: Search returned {hits} hit(s)")

        # Cleanup
        client.indices.delete(index=test_index)
        print(f"   ✅ OK: Cleaned up test index")

    except Exception as e:
        print(f"   ❌ FAILED: Index operations failed: {e}")
        return False

    print(f"\n{'='*60}")
    print("✅ ALL TESTS PASSED - OpenSearch is ready!")
    print(f"{'='*60}")
    return True


def test_keyword_store(host: str, port: int, username: str, password: str, use_ssl: bool):
    """Test the OpenSearchKeywordStore class."""
    print(f"\n{'='*60}")
    print("KEYWORD STORE TEST")
    print(f"{'='*60}")

    try:
        from opensearch_keyword_store import OpenSearchKeywordStore
        print("\n[1] ✅ OpenSearchKeywordStore imported successfully")
    except ImportError as e:
        print(f"\n[1] ❌ FAILED to import OpenSearchKeywordStore: {e}")
        return False

    try:
        print("\n[2] Initializing OpenSearchKeywordStore...")
        store = OpenSearchKeywordStore(
            host=host,
            port=port,
            username=username,
            password=password,
            use_ssl=use_ssl,
            verify_certs=False
        )
        print("   ✅ OK: Store initialized")

        print("\n[3] Creating test index 'keyword_store_test'...")
        store.create_index("keyword_store_test", delete_if_exists=True)
        print("   ✅ OK: Index created")

        print("\n[4] Testing keyword extraction...")
        from opensearch_keyword_store import KeywordExtractor
        extractor = KeywordExtractor()
        keywords = extractor.extract_keywords("Machine learning and artificial intelligence are transforming technology.")
        print(f"   ✅ OK: Extracted {len(keywords)} keywords: {keywords[:5]}...")

        print("\n[5] Cleaning up test index...")
        store.delete_index("keyword_store_test")
        print("   ✅ OK: Test index deleted")

        print(f"\n{'='*60}")
        print("✅ KEYWORD STORE TESTS PASSED!")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\n   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_integration():
    """Test integration with config.py."""
    print(f"\n{'='*60}")
    print("CONFIG.PY INTEGRATION TEST")
    print(f"{'='*60}")

    try:
        from config import get_config
        print("\n[1] ✅ config.py imported successfully")

        cfg = get_config()

        print(f"\n[2] Configuration loaded:")
        print(f"    OpenSearch Enabled: {cfg.opensearch_enabled}")
        print(f"    Host: {cfg.opensearch_host}:{cfg.opensearch_port}")
        print(f"    Username: {cfg.opensearch_username}")
        print(f"    Password: {'***' if cfg.opensearch_password else 'Not set'}")
        print(f"    SSL: {cfg.opensearch_use_ssl}")
        print(f"    Index: {cfg.opensearch_index}")
        print(f"    Timeout: {cfg.opensearch_timeout}s")
        print(f"    Max Retries: {cfg.opensearch_max_retries}")

        print(f"\n[3] Hybrid chunk settings:")
        print(f"    Heading chunks: {cfg.generate_heading_chunks}")
        print(f"    Clause chunks: {cfg.generate_clause_chunks}")
        print(f"    Metadata chunks: {cfg.generate_metadata_chunks}")
        print(f"    Summary chunks: {cfg.generate_summary_chunks}")

        if not cfg.opensearch_enabled:
            print("\n   ⚠️  WARNING: OpenSearch is disabled in config!")
            print("   Set OPENSEARCH_ENABLED=true in .env to enable")

        print(f"\n{'='*60}")
        print("✅ CONFIG INTEGRATION READY!")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\n   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test OpenSearch connection")
    parser.add_argument("--host", help="OpenSearch host (overrides config)")
    parser.add_argument("--port", type=int, help="OpenSearch port (overrides config)")
    parser.add_argument("--username", help="Username (overrides config)")
    parser.add_argument("--password", help="Password (overrides config)")
    parser.add_argument("--no-ssl", action="store_true", help="Disable SSL (overrides config)")
    parser.add_argument("--no-auth", action="store_true", help="Disable authentication")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--use-config", action="store_true", help="Use config.py settings (default)")

    args = parser.parse_args()

    # Load config settings
    config_settings = load_config_settings()

    # Determine settings to use (command line args override config)
    if config_settings:
        host = args.host if args.host else config_settings['host']
        port = args.port if args.port else config_settings['port']
        username = args.username if args.username else config_settings['username']
        password = args.password if args.password else config_settings['password']
        use_ssl = (not args.no_ssl) if args.no_ssl else config_settings['use_ssl']
        
        print("\n" + "="*60)
        print("OPENSEARCH CONNECTION DIAGNOSTIC")
        print("="*60)
        print(f"Configuration source: config.py (from .env)")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Username: {'(disabled)' if args.no_auth else username}")
        print(f"SSL: {use_ssl}")
        print(f"OpenSearch Enabled: {config_settings['enabled']}")
        print(f"Index Name: {config_settings['index']}")
        
        if not config_settings['enabled']:
            print("\n⚠️  WARNING: OpenSearch is disabled in config!")
            print("Set OPENSEARCH_ENABLED=true in .env to enable")
            print("Continuing with connection test anyway...")
    else:
        # Fallback to defaults
        host = args.host if args.host else "localhost"
        port = args.port if args.port else 9200
        username = args.username if args.username else "admin"
        password = args.password if args.password else "ArivurAI@123"
        use_ssl = not args.no_ssl
        
        print("\n" + "="*60)
        print("OPENSEARCH CONNECTION DIAGNOSTIC")
        print("="*60)
        print(f"Configuration source: Command line / defaults")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Username: {'(disabled)' if args.no_auth else username}")
        print(f"SSL: {use_ssl}")

    # Run tests
    success = True

    # Test config integration first
    if config_settings:
        if not test_config_integration():
            print("\n⚠️  Config integration test failed, but continuing...")

    # Basic connection test
    if not test_basic_connection(host, port, username, password, use_ssl, args.no_auth):
        success = False
        print("\n❌ Basic connection test FAILED. Fix connection issues before proceeding.")
        sys.exit(1)

    # Full test suite
    if args.full:
        if not test_keyword_store(host, port, username, password, use_ssl):
            success = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if success:
        print("\n✅ All tests PASSED!")
        print("\nYou can now use OpenSearch hybrid search with:")
        print(f"""
from config import get_config
from opensearch_keyword_store import OpenSearchKeywordStore

# Load config
cfg = get_config()

# Create OpenSearch store
store = OpenSearchKeywordStore(
    host=cfg.opensearch_host,
    port=cfg.opensearch_port,
    username=cfg.opensearch_username,
    password=cfg.opensearch_password,
    use_ssl=cfg.opensearch_use_ssl,
    verify_certs=cfg.opensearch_verify_certs
)

# Create index
store.create_index(cfg.opensearch_index)

# Index documents with hybrid chunks
from enterprise_chunking_pipeline import EnterpriseChunkingPipeline, ChunkingConfig

chunking_config = ChunkingConfig(
    max_chunk_size=cfg.max_chunk_size,
    enable_overlap=cfg.enable_overlap,
    generate_heading_chunks=cfg.generate_heading_chunks,
    generate_clause_chunks=cfg.generate_clause_chunks,
    generate_metadata_chunks=cfg.generate_metadata_chunks,
    generate_summary_chunks=cfg.generate_summary_chunks
)

# Process and index documents...
""")
    else:
        print("\n❌ Some tests FAILED. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()