"""
OpenSearch Connection Test Script

Tests the connection to OpenSearch and verifies the setup for hybrid search.
Run this script to diagnose OpenSearch connectivity issues.

Usage:
    python test_opensearch_connection.py
    python test_opensearch_connection.py --host iceapple-server1 --port 9200
"""

import sys
import argparse

# Check if opensearch-py is installed
try:
    from opensearchpy import OpenSearch
    OPENSEARCH_INSTALLED = True
except ImportError:
    OPENSEARCH_INSTALLED = False


def test_basic_connection(host: str, port: int, username: str, password: str, use_ssl: bool, no_auth: bool = False):
    """Test basic OpenSearch connection."""
    print(f"\n{'='*60}")
    print("OPENSEARCH CONNECTION TEST")
    print(f"{'='*60}")

    # Check if library is installed
    print("\n[1] Checking opensearch-py installation...")
    if not OPENSEARCH_INSTALLED:
        print("   FAILED: opensearch-py not installed")
        print("   Fix: pip install opensearch-py")
        return False
    print("   OK: opensearch-py is installed")

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
        print(f"   OK: Connected to OpenSearch")
        print(f"       Cluster: {info.get('cluster_name', 'N/A')}")
        print(f"       Version: {info.get('version', {}).get('number', 'N/A')}")

    except Exception as e:
        print(f"   FAILED: {e}")
        print("\n   Troubleshooting:")
        print("   - Check if OpenSearch container is running: docker ps | grep opensearch")
        print("   - Check if port 9200 is accessible")
        print("   - Try different host (localhost, 127.0.0.1, or server IP)")
        print("   - Verify credentials (default: admin/admin)")
        return False

    # Test cluster health
    print("\n[3] Checking cluster health...")
    try:
        health = client.cluster.health()
        status = health.get('status', 'unknown')
        print(f"   OK: Cluster status is '{status}'")
        if status == 'red':
            print("   WARNING: Cluster health is RED - some shards are not allocated")
    except Exception as e:
        print(f"   WARNING: Could not get cluster health: {e}")

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
        print(f"   OK: Created test index '{test_index}'")

        # Index a document
        client.index(index=test_index, id="1", body={"test": "document"}, refresh=True)
        print("   OK: Indexed test document")

        # Search
        result = client.search(index=test_index, body={"query": {"match_all": {}}})
        hits = result.get('hits', {}).get('total', {}).get('value', 0)
        print(f"   OK: Search returned {hits} hit(s)")

        # Cleanup
        client.indices.delete(index=test_index)
        print(f"   OK: Cleaned up test index")

    except Exception as e:
        print(f"   FAILED: Index operations failed: {e}")
        return False

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED - OpenSearch is ready!")
    print(f"{'='*60}")
    return True


def test_keyword_store(host: str, port: int, username: str, password: str, use_ssl: bool):
    """Test the OpenSearchKeywordStore class."""
    print(f"\n{'='*60}")
    print("KEYWORD STORE TEST")
    print(f"{'='*60}")

    try:
        from opensearch_keyword_store import OpenSearchKeywordStore
        print("\n[1] OpenSearchKeywordStore imported successfully")
    except ImportError as e:
        print(f"\n[1] FAILED to import OpenSearchKeywordStore: {e}")
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
        print("   OK: Store initialized")

        print("\n[3] Creating test index 'keyword_store_test'...")
        store.create_index("keyword_store_test", delete_if_exists=True)
        print("   OK: Index created")

        print("\n[4] Testing keyword extraction...")
        from opensearch_keyword_store import KeywordExtractor
        extractor = KeywordExtractor()
        keywords = extractor.extract_keywords("Machine learning and artificial intelligence are transforming technology.")
        print(f"   OK: Extracted {len(keywords)} keywords: {keywords[:5]}...")

        print("\n[5] Cleaning up test index...")
        store.delete_index("keyword_store_test")
        print("   OK: Test index deleted")

        print(f"\n{'='*60}")
        print("KEYWORD STORE TESTS PASSED!")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\n   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_service_integration(host: str, port: int, username: str, password: str, use_ssl: bool):
    """Test integration with DocumentToVectorService."""
    print(f"\n{'='*60}")
    print("DOCUMENT SERVICE INTEGRATION TEST")
    print(f"{'='*60}")

    try:
        from document_to_vector_service import ServiceConfig
        print("\n[1] ServiceConfig imported successfully")

        config = ServiceConfig(
            opensearch_enabled=True,
            opensearch_host=host,
            opensearch_port=port,
            opensearch_username=username,
            opensearch_password=password,
            opensearch_use_ssl=use_ssl,
            opensearch_verify_certs=False
        )

        print(f"\n[2] ServiceConfig created:")
        print(f"    Host: {config.opensearch_host}:{config.opensearch_port}")
        print(f"    SSL: {config.opensearch_use_ssl}")
        print(f"    Index: {config.opensearch_index}")
        print(f"    Hybrid chunks: heading={config.generate_heading_chunks}, "
              f"clause={config.generate_clause_chunks}, "
              f"metadata={config.generate_metadata_chunks}, "
              f"summary={config.generate_summary_chunks}")

        print(f"\n{'='*60}")
        print("INTEGRATION CONFIG READY!")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\n   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test OpenSearch connection")
    parser.add_argument("--host", default="localhost", help="OpenSearch host (default: localhost)")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port (default: 9200)")
    parser.add_argument("--username", default="admin", help="Username (default: admin)")
    parser.add_argument("--password", default="admin", help="Password (default: admin)")
    parser.add_argument("--no-ssl", action="store_true", help="Disable SSL")
    parser.add_argument("--no-auth", action="store_true", help="Disable authentication (for OpenSearch with security disabled)")
    parser.add_argument("--full", action="store_true", help="Run full test suite including keyword store")

    args = parser.parse_args()

    use_ssl = not args.no_ssl

    print("\n" + "="*60)
    print("OPENSEARCH CONNECTION DIAGNOSTIC")
    print("="*60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Username: {'(disabled)' if args.no_auth else args.username}")
    print(f"SSL: {use_ssl}")

    # Run tests
    success = True

    # Basic connection test
    if not test_basic_connection(args.host, args.port, args.username, args.password, use_ssl, args.no_auth):
        success = False
        print("\nBasic connection test FAILED. Fix connection issues before proceeding.")
        sys.exit(1)

    # Full test suite
    if args.full:
        if not test_keyword_store(args.host, args.port, args.username, args.password, use_ssl):
            success = False

        if not test_document_service_integration(args.host, args.port, args.username, args.password, use_ssl):
            success = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if success:
        print("\nAll tests PASSED!")
        print("\nYou can now use OpenSearch hybrid search with:")
        print(f"""
from document_to_vector_service import DocumentToVectorService, ServiceConfig

config = ServiceConfig(
    opensearch_enabled=True,
    opensearch_host="{args.host}",
    opensearch_port={args.port},
    opensearch_username="{args.username}",
    opensearch_password="{args.password}",
    opensearch_use_ssl={use_ssl},
    opensearch_verify_certs=False
)

service = DocumentToVectorService(config)

# Process a document (indexes to both Qdrant and OpenSearch)
result = service.process_document("your_document.pdf")

# Hybrid search (combines vector + keyword)
results = service.hybrid_search("your query")

# Keyword-only search
results = service.keyword_search("exact terms", strategy="phrase")

# Delete document (from both stores)
service.delete_document("doc_id")
""")
    else:
        print("\nSome tests FAILED. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
