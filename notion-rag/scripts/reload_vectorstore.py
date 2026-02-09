"""
Vector store reload script

Delete existing embeddings and reload documents using NotionRecursiveLoader
with child pages included, then split using SemanticChunker.
"""

import os
import sys

# Project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.loaders import NotionRecursiveLoader
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.postgres_store import PostgresVectorStore


def main():
    # Check environment variables
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    print(f"OpenAI API Key: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"Notion API Key: {'SET' if os.getenv('NOTION_API_KEY') else 'NOT SET'}")
    print(f"PostgreSQL: {os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5433')}")

    database_id = os.getenv("NOTION_DATABASE_ID")
    if not database_id:
        print("\n[ERROR] NOTION_DATABASE_ID is not set.")
        return
    print(f"Notion Database ID: {database_id[:8]}...")

    # Initialize embedding model
    print("\n" + "=" * 60)
    print("Step 0: Initialize Embedding Model")
    print("=" * 60)
    embedding_manager = EmbeddingManager(provider="openai")
    embeddings = embedding_manager.embeddings
    print("[OK] OpenAI Embedding model ready")

    # Delete existing vector store
    print("\n" + "=" * 60)
    print("Step 1: Delete Existing Embeddings")
    print("=" * 60)
    vector_store = PostgresVectorStore(
        embeddings=embeddings,
        collection_name="notion_docs"
    )

    try:
        stats = vector_store.get_collection_stats()
        print(f"Existing collection info: {stats}")
        vector_store.delete_collection()
        print("[OK] Existing collection deleted")
    except Exception as e:
        print(f"Delete error (can be ignored): {e}")

    # Load documents with NotionRecursiveLoader
    print("\n" + "=" * 60)
    print("Step 2: Load Notion Documents (including child pages)")
    print("=" * 60)
    loader = NotionRecursiveLoader(
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    docs = loader.load_database(database_id)
    print(f"\nLoaded documents:")
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Untitled")
        full_title = doc.metadata.get("full_title", title)
        content_len = len(doc.page_content)
        print(f"  {i:2}. {full_title} ({content_len:,} chars)")

    # Split with SemanticChunker
    print("\n" + "=" * 60)
    print("Step 3: Split with SemanticChunker")
    print("=" * 60)
    chunks = loader.split_documents(docs)

    # Sample output
    print(f"\nSplit result sample (first 5):")
    for i, chunk in enumerate(chunks[:5], 1):
        title = chunk.metadata.get("title", "Untitled")
        chunk_idx = chunk.metadata.get("chunk_index", 0)
        content_preview = chunk.page_content[:80].replace("\n", " ")
        # Encode safely for Windows console
        try:
            print(f"  {i}. [{title}][chunk {chunk_idx}] {content_preview}...")
        except UnicodeEncodeError:
            safe_preview = content_preview.encode('ascii', 'replace').decode('ascii')
            print(f"  {i}. [{title}][chunk {chunk_idx}] {safe_preview}...")

    # Save to vector store
    print("\n" + "=" * 60)
    print("Step 4: Save to PostgreSQL Vector Store")
    print("=" * 60)

    # Create new vector store and add documents
    vector_store = PostgresVectorStore(
        embeddings=embeddings,
        collection_name="notion_docs"
    )
    vector_store.from_documents(chunks)

    # Final stats
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
    print(f"\n[OK] {len(docs)} documents loaded")
    print(f"[OK] {len(chunks)} chunks created")
    print(f"[OK] Saved to PostgreSQL")

    # Verify: similarity search test
    print("\n" + "=" * 60)
    print("Verify: Similarity Search Test")
    print("=" * 60)
    test_queries = [
        "Unity HTTP",
        "WebSocket",
        "weekly meeting"
    ]

    for query in test_queries:
        results = vector_store.similarity_search(query, k=2)
        print(f"\nQuery: '{query}'")
        for j, result in enumerate(results, 1):
            title = result.metadata.get("title", "Untitled")
            content_preview = result.page_content[:60].replace("\n", " ")
            print(f"  {j}. [{title}] {content_preview}...")


if __name__ == "__main__":
    main()
