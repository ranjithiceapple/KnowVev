"""
List Documents and Embeddings from Vector Database

This script allows you to:
1. List all documents stored in Qdrant
2. View document metadata
3. View embeddings (vectors) for each document
4. Export to JSON/CSV
"""

import sys
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import ScrollRequest, Filter, FieldCondition, MatchValue
from logger_config import get_logger

logger = get_logger(__name__)


class DocumentLister:
    """List and analyze documents in Qdrant vector database."""

    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "documents_services"):
        """
        Initialize document lister.

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Collection name
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        logger.info(f"Connected to Qdrant at {qdrant_url}, collection: {collection_name}")

    def list_documents(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        List all unique documents in the collection.

        Args:
            limit: Maximum number of points to scan

        Returns:
            List of document metadata
        """
        logger.info(f"Scanning collection '{self.collection_name}' for documents...")

        # Scroll through all points
        documents = {}
        offset = None
        total_points = 0

        while True:
            # Scroll with pagination
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            points, next_offset = scroll_result

            if not points:
                break

            total_points += len(points)

            # Group by document
            for point in points:
                payload = point.payload
                doc_id = payload.get('doc_id')

                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        'doc_id': doc_id,
                        'file_name': payload.get('file_name', 'Unknown'),
                        'total_chunks': 0,
                        'total_pages': payload.get('page_end', 0),
                        'has_summary': False,
                        'created_at': payload.get('created_at', 'Unknown'),
                        'version': payload.get('version', 'Unknown'),
                        'chunks': []
                    }

                # Count chunks
                if doc_id:
                    documents[doc_id]['total_chunks'] += 1

                    # Check for summary chunk
                    if payload.get('section_title') == '[DOCUMENT SUMMARY]':
                        documents[doc_id]['has_summary'] = True

            # Check if we should continue
            offset = next_offset
            if offset is None or total_points >= limit:
                break

        logger.info(f"Scanned {total_points} points, found {len(documents)} unique documents")

        return list(documents.values())

    def get_document_chunks(self, doc_id: str, include_vectors: bool = False) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.

        Args:
            doc_id: Document ID
            include_vectors: Whether to include vector embeddings

        Returns:
            List of chunks with metadata
        """
        logger.info(f"Fetching chunks for document: {doc_id}")

        chunks = []
        offset = None

        while True:
            # Scroll with filter
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=include_vectors
            )

            points, next_offset = scroll_result

            if not points:
                break

            for point in points:
                chunk_data = {
                    'id': point.id,
                    'chunk_id': point.payload.get('chunk_id'),
                    'chunk_index': point.payload.get('chunk_index', 0),
                    'page_start': point.payload.get('page_start', 0),
                    'page_end': point.payload.get('page_end', 0),
                    'section_title': point.payload.get('section_title', 'N/A'),
                    'char_len': point.payload.get('char_len', 0),
                    'word_count': point.payload.get('word_count', 0),
                    'contains_code': point.payload.get('contains_code', False),
                    'contains_tables': point.payload.get('contains_tables', False),
                    'text_preview': point.payload.get('text', '')[:100] + '...',
                }

                if include_vectors and point.vector:
                    chunk_data['vector'] = point.vector
                    chunk_data['vector_dim'] = len(point.vector) if isinstance(point.vector, list) else 'N/A'

                chunks.append(chunk_data)

            offset = next_offset
            if offset is None:
                break

        # Sort by chunk index
        chunks.sort(key=lambda x: x['chunk_index'])

        logger.info(f"Found {len(chunks)} chunks for document {doc_id}")
        return chunks

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        logger.info("Fetching collection statistics...")

        try:
            collection_info = self.client.get_collection(self.collection_name)

            stats = {
                'collection_name': self.collection_name,
                'total_points': collection_info.points_count,
                'total_vectors': collection_info.vectors_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': str(collection_info.config.params.vectors.distance),
                'status': collection_info.status,
                'optimizer_status': collection_info.optimizer_status,
            }

            logger.info(f"Collection stats: {stats['total_points']} points, {stats['total_vectors']} vectors")
            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    def export_documents_to_json(self, output_file: str = "documents_list.json"):
        """Export document list to JSON."""
        logger.info(f"Exporting documents to {output_file}...")

        documents = self.list_documents()

        export_data = {
            'exported_at': datetime.now().isoformat(),
            'collection': self.collection_name,
            'total_documents': len(documents),
            'documents': documents
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Exported {len(documents)} documents to {output_file}")
        return output_file

    def export_documents_to_csv(self, output_file: str = "documents_list.csv"):
        """Export document list to CSV."""
        logger.info(f"Exporting documents to {output_file}...")

        documents = self.list_documents()

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Doc ID', 'File Name', 'Total Chunks', 'Total Pages',
                'Has Summary', 'Created At', 'Version'
            ])

            # Data
            for doc in documents:
                writer.writerow([
                    doc['doc_id'],
                    doc['file_name'],
                    doc['total_chunks'],
                    doc['total_pages'],
                    doc['has_summary'],
                    doc['created_at'],
                    doc['version']
                ])

        logger.info(f"✅ Exported {len(documents)} documents to {output_file}")
        return output_file

    def export_embeddings(self, doc_id: str, output_file: str = None):
        """Export embeddings for a specific document."""
        if output_file is None:
            output_file = f"embeddings_{doc_id[:8]}.json"

        logger.info(f"Exporting embeddings for document {doc_id}...")

        chunks = self.get_document_chunks(doc_id, include_vectors=True)

        export_data = {
            'doc_id': doc_id,
            'total_chunks': len(chunks),
            'exported_at': datetime.now().isoformat(),
            'chunks': chunks
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Exported {len(chunks)} chunks with embeddings to {output_file}")
        return output_file


def display_documents(documents: List[Dict[str, Any]]):
    """Display documents in a formatted table."""
    print("\n" + "=" * 120)
    print(f"{'DOCUMENTS IN VECTOR DATABASE':<120}")
    print("=" * 120)

    if not documents:
        print("\n❌ No documents found in the database.")
        return

    print(f"\nTotal Documents: {len(documents)}\n")

    # Header
    header_format = "{:<10} {:<40} {:<12} {:<12} {:<12} {:<20}"
    print(header_format.format(
        "Doc ID", "File Name", "Chunks", "Pages", "Has Summary", "Created At"
    ))
    print("-" * 120)

    # Documents
    for doc in documents:
        doc_id_short = doc['doc_id'][:8] + "..." if len(doc['doc_id']) > 8 else doc['doc_id']
        file_name_short = doc['file_name'][:38] + ".." if len(doc['file_name']) > 40 else doc['file_name']
        created_at_short = doc['created_at'][:19] if len(doc['created_at']) > 19 else doc['created_at']

        print(header_format.format(
            doc_id_short,
            file_name_short,
            str(doc['total_chunks']),
            str(doc['total_pages']),
            '✓' if doc['has_summary'] else '✗',
            created_at_short
        ))

    print("=" * 120)


def display_document_details(doc_id: str, lister: DocumentLister, show_embeddings: bool = False):
    """Display detailed information about a specific document."""
    logger.info(f"Fetching details for document: {doc_id}")

    chunks = lister.get_document_chunks(doc_id, include_vectors=show_embeddings)

    if not chunks:
        print(f"\n❌ No chunks found for document: {doc_id}")
        return

    print("\n" + "=" * 120)
    print(f"DOCUMENT DETAILS: {doc_id}")
    print("=" * 120)

    # Document summary
    print(f"\nTotal Chunks: {len(chunks)}")

    if chunks:
        print(f"Pages: {chunks[0]['page_start']} - {chunks[-1]['page_end']}")

        # Check for summary chunk
        summary_chunk = next((c for c in chunks if c['section_title'] == '[DOCUMENT SUMMARY]'), None)
        if summary_chunk:
            print(f"\n📄 Document Summary Found (Chunk ID: {summary_chunk['chunk_id']})")
            print(f"   {summary_chunk['text_preview']}")

    # Chunk list
    print(f"\n{'='*120}")
    print("CHUNKS:")
    print(f"{'='*120}\n")

    header_format = "{:<6} {:<25} {:<12} {:<40} {:<10} {:<10}"
    print(header_format.format(
        "Index", "Chunk ID", "Pages", "Section", "Chars", "Words"
    ))
    print("-" * 120)

    for chunk in chunks:
        chunk_id_short = chunk['chunk_id'][:23] + ".." if len(chunk['chunk_id']) > 25 else chunk['chunk_id']
        section_short = chunk['section_title'][:38] + ".." if len(chunk['section_title']) > 40 else chunk['section_title']
        pages = f"{chunk['page_start']}-{chunk['page_end']}"

        print(header_format.format(
            str(chunk['chunk_index']),
            chunk_id_short,
            pages,
            section_short,
            str(chunk['char_len']),
            str(chunk['word_count'])
        ))

    print("=" * 120)

    # Show embedding info if requested
    if show_embeddings and chunks[0].get('vector'):
        print(f"\n{'='*120}")
        print("EMBEDDINGS:")
        print(f"{'='*120}\n")

        print(f"Vector Dimension: {chunks[0]['vector_dim']}")
        print(f"Total Embeddings: {len(chunks)}")

        print(f"\nSample Vector (First Chunk):")
        sample_vector = chunks[0]['vector']
        if isinstance(sample_vector, list):
            print(f"  First 10 dimensions: {sample_vector[:10]}")
            print(f"  Last 10 dimensions: {sample_vector[-10:]}")
            print(f"  Vector norm: {sum(x**2 for x in sample_vector)**0.5:.4f}")


def interactive_mode(lister: DocumentLister):
    """Interactive mode for browsing documents."""
    while True:
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE")
        print("=" * 80)
        print("\nCommands:")
        print("  list        - List all documents")
        print("  view <id>   - View document details (use first 8 chars of doc_id)")
        print("  embed <id>  - View document with embeddings")
        print("  stats       - Show collection statistics")
        print("  export      - Export documents to JSON/CSV")
        print("  quit        - Exit")
        print()

        command = input("Enter command: ").strip().lower()

        if command == 'quit':
            break

        elif command == 'list':
            documents = lister.list_documents()
            display_documents(documents)

        elif command.startswith('view '):
            doc_id_prefix = command.split(' ', 1)[1]
            # Find full doc_id
            documents = lister.list_documents()
            matching_docs = [d for d in documents if d['doc_id'].startswith(doc_id_prefix)]

            if matching_docs:
                display_document_details(matching_docs[0]['doc_id'], lister, show_embeddings=False)
            else:
                print(f"❌ No document found with ID starting with: {doc_id_prefix}")

        elif command.startswith('embed '):
            doc_id_prefix = command.split(' ', 1)[1]
            documents = lister.list_documents()
            matching_docs = [d for d in documents if d['doc_id'].startswith(doc_id_prefix)]

            if matching_docs:
                display_document_details(matching_docs[0]['doc_id'], lister, show_embeddings=True)
            else:
                print(f"❌ No document found with ID starting with: {doc_id_prefix}")

        elif command == 'stats':
            stats = lister.get_collection_stats()
            print("\n" + "=" * 80)
            print("COLLECTION STATISTICS")
            print("=" * 80)
            for key, value in stats.items():
                print(f"{key:.<30} {value}")
            print("=" * 80)

        elif command == 'export':
            print("\nExport format:")
            print("  1. JSON")
            print("  2. CSV")
            choice = input("Choose format (1/2): ").strip()

            if choice == '1':
                output_file = lister.export_documents_to_json()
                print(f"✅ Exported to {output_file}")
            elif choice == '2':
                output_file = lister.export_documents_to_csv()
                print(f"✅ Exported to {output_file}")

        else:
            print("❌ Unknown command")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="List documents and embeddings from Qdrant")
    parser.add_argument(
        '--url',
        default='http://localhost:6333',
        help='Qdrant URL (default: http://localhost:6333)'
    )
    parser.add_argument(
        '--collection',
        default='documents_services',
        help='Collection name (default: documents_services)'
    )
    parser.add_argument(
        '--mode',
        choices=['list', 'view', 'export', 'stats', 'interactive'],
        default='interactive',
        help='Operation mode'
    )
    parser.add_argument(
        '--doc-id',
        help='Document ID (for view mode)'
    )
    parser.add_argument(
        '--embeddings',
        action='store_true',
        help='Include embeddings in output'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Export format (default: json)'
    )
    parser.add_argument(
        '--output',
        help='Output file path'
    )

    args = parser.parse_args()

    # Create lister
    lister = DocumentLister(qdrant_url=args.url, collection_name=args.collection)

    if args.mode == 'interactive':
        interactive_mode(lister)

    elif args.mode == 'list':
        documents = lister.list_documents()
        display_documents(documents)

    elif args.mode == 'view':
        if not args.doc_id:
            print("❌ --doc-id required for view mode")
            sys.exit(1)
        display_document_details(args.doc_id, lister, show_embeddings=args.embeddings)

    elif args.mode == 'stats':
        stats = lister.get_collection_stats()
        print("\n" + "=" * 80)
        print("COLLECTION STATISTICS")
        print("=" * 80)
        print(json.dumps(stats, indent=2))

    elif args.mode == 'export':
        if args.format == 'json':
            output = args.output or 'documents_list.json'
            lister.export_documents_to_json(output)
        else:
            output = args.output or 'documents_list.csv'
            lister.export_documents_to_csv(output)


if __name__ == "__main__":
    main()
