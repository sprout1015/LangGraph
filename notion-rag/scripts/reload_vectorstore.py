"""
Vector store reload script

기존 컬렉션 유무에 따라 동적으로 실행 방식을 결정합니다.

- 기존 컬렉션 있음: JSON 백업 → Notion 로드 → 기존 삭제 → 신규 저장
  (로딩 실패 시 삭제하지 않으므로 기존 데이터 보존)
- 기존 컬렉션 없음: Notion 로드 → 저장 (백업/삭제 불필요)

실행:
    python scripts/reload_vectorstore.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.loaders import NotionRecursiveLoader
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.postgres_store import PostgresVectorStore


def backup_collection(vector_store: PostgresVectorStore, backup_dir: Path) -> Path | None:
    """현재 컬렉션을 JSON으로 백업합니다. 백업 파일 경로를 반환합니다."""
    try:
        from sqlalchemy import create_engine, text
        from urllib.parse import quote_plus

        password = quote_plus(os.getenv("POSTGRES_PASSWORD", ""))
        conn_str = (
            f"postgresql+psycopg://postgres:{password}"
            f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
            f":{os.getenv('POSTGRES_PORT', '5433')}"
            f"/{os.getenv('POSTGRES_DB', 'notion_rag')}"
        )
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT e.id, e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :name
            """), {"name": vector_store.collection_name}).fetchall()

        backup = [
            {"id": str(r.id), "document": r.document, "cmetadata": r.cmetadata}
            for r in rows
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{vector_store.collection_name}_{timestamp}.json"
        backup_dir.mkdir(parents=True, exist_ok=True)

        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(backup, f, ensure_ascii=False, indent=2)

        print(f"[OK] Backup saved: {backup_path} ({len(backup)} docs)")
        return backup_path

    except Exception as e:
        print(f"[WARN] Backup failed: {e}")
        return None


def main():
    # ── 환경 변수 확인 ──
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

    # ── Step 0: 임베딩 모델 초기화 ──
    print("\n" + "=" * 60)
    print("Step 0: Initialize Embedding Model")
    print("=" * 60)
    embeddings = EmbeddingManager(provider="openai").embeddings
    print("[OK] OpenAI Embedding model ready")

    # ── 기존 컬렉션 존재 여부 확인 ──
    vector_store = PostgresVectorStore(embeddings=embeddings, collection_name="notion_docs")
    try:
        stats = vector_store.get_collection_stats()
        existing_count = stats.get("count", 0)
        has_existing = isinstance(existing_count, int) and existing_count > 0
    except Exception:
        has_existing = False

    if has_existing:
        print(f"\n기존 컬렉션 감지: {existing_count}개 문서")
        print("실행 순서: 백업 → Notion 로드 → 기존 삭제 → 신규 저장")
    else:
        print("\n기존 컬렉션 없음")
        print("실행 순서: Notion 로드 → 저장")

    # ── Step 1: Notion 문서 로드 (기존 데이터 건드리기 전에 먼저 실행) ──
    print("\n" + "=" * 60)
    print("Step 1: Load Notion Documents (including child pages)")
    print("=" * 60)
    loader = NotionRecursiveLoader(
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        timeout=60,
    )

    docs = loader.load_database(database_id)
    if not docs:
        print("[ERROR] No documents loaded. Aborting without touching existing data.")
        return

    print(f"\nLoaded documents:")
    for i, doc in enumerate(docs, 1):
        full_title = doc.metadata.get("full_title", doc.metadata.get("title", "Untitled"))
        print(f"  {i:2}. {full_title} ({len(doc.page_content):,} chars)")

    # ── Step 2: SemanticChunker로 분할 ──
    print("\n" + "=" * 60)
    print("Step 2: Split with SemanticChunker (code blocks protected)")
    print("=" * 60)
    chunks = loader.split_documents(docs)

    print(f"\nSplit result sample (first 5):")
    for i, chunk in enumerate(chunks[:5], 1):
        title = chunk.metadata.get("title", "Untitled")
        chunk_idx = chunk.metadata.get("chunk_index", 0)
        content_preview = chunk.page_content[:80].replace("\n", " ")
        try:
            print(f"  {i}. [{title}][chunk {chunk_idx}] {content_preview}...")
        except UnicodeEncodeError:
            safe = content_preview.encode("ascii", "replace").decode("ascii")
            print(f"  {i}. [{title}][chunk {chunk_idx}] {safe}...")

    # ── Step 3: 기존 컬렉션 백업 및 교체 ──
    print("\n" + "=" * 60)
    print("Step 3: Replace PostgreSQL Vector Store")
    print("=" * 60)

    if has_existing:
        # 백업 후 삭제 (로딩 성공이 확인된 이후)
        backup_dir = Path(__file__).parent.parent / "data" / "backups"
        backup_collection(vector_store, backup_dir)
        vector_store.delete_collection()
        print("[OK] Existing collection deleted")

    # 신규 저장
    vector_store = PostgresVectorStore(embeddings=embeddings, collection_name="notion_docs")
    vector_store.from_documents(chunks)

    # ── 완료 ──
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
    print(f"\n[OK] {len(docs)} documents loaded")
    print(f"[OK] {len(chunks)} chunks created")
    print(f"[OK] Saved to PostgreSQL")

    # ── 검증: 유사도 검색 테스트 ──
    print("\n" + "=" * 60)
    print("Verify: Similarity Search Test")
    print("=" * 60)
    test_queries = ["Unity HTTP", "WebSocket", "weekly meeting"]

    for query in test_queries:
        results = vector_store.similarity_search(query, k=2)
        print(f"\nQuery: '{query}'")
        for j, result in enumerate(results, 1):
            title = result.metadata.get("title", "Untitled")
            content_preview = result.page_content[:60].replace("\n", " ")
            try:
                print(f"  {j}. [{title}] {content_preview}...")
            except UnicodeEncodeError:
                safe = content_preview.encode("ascii", "replace").decode("ascii")
                print(f"  {j}. [{title}] {safe}...")


if __name__ == "__main__":
    main()
