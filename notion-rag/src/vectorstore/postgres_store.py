"""
PostgreSQL + pgvector 벡터 스토어 관리 모듈

문서의 벡터 저장 및 유사도 검색을 담당합니다.
"""

import os
from typing import List, Optional, Tuple
from urllib.parse import quote_plus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres.vectorstores import PGVector


class PostgresVectorStore:
    """PostgreSQL + pgvector 기반 벡터 스토어 관리 클래스"""

    def __init__(
        self,
        embeddings: Embeddings,
        connection_string: Optional[str] = None,
        collection_name: str = "notion_docs"
    ):
        """
        Args:
            embeddings: 임베딩 모델 인스턴스
            connection_string: PostgreSQL 연결 문자열 (없으면 환경변수에서 로드)
            collection_name: 컬렉션 이름
        """
        self.embeddings = embeddings
        self.connection_string = connection_string or self._build_connection_string()
        self.collection_name = collection_name
        self._vectorstore: Optional[PGVector] = None

    def _build_connection_string(self) -> str:
        """환경변수에서 연결 문자열 생성"""
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5433")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "")
        database = os.getenv("POSTGRES_DB", "notion_rag")
        # 비밀번호에 특수문자가 있을 수 있으므로 URL 인코딩
        encoded_password = quote_plus(password)
        return f"postgresql+psycopg://{user}:{encoded_password}@{host}:{port}/{database}"

    @property
    def vectorstore(self) -> PGVector:
        """벡터 스토어 인스턴스 반환 (지연 로딩)"""
        if self._vectorstore is None:
            self._vectorstore = self._load_or_create()
        return self._vectorstore

    def _load_or_create(self) -> PGVector:
        """기존 벡터 스토어를 로드하거나 새로 생성"""
        return PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        문서를 벡터 스토어에 추가

        Args:
            documents: 추가할 Document 리스트

        Returns:
            추가된 문서 ID 리스트
        """
        ids = self.vectorstore.add_documents(documents)
        print(f"{len(documents)}개 문서가 추가되었습니다.")
        return ids

    def from_documents(self, documents: List[Document]) -> "PostgresVectorStore":
        """
        문서로부터 새 벡터 스토어 생성

        Args:
            documents: Document 리스트

        Returns:
            PostgresVectorStore 인스턴스
        """
        self._vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True
        )
        print(f"{len(documents)}개 문서로 벡터 스토어가 생성되었습니다.")
        return self

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        유사도 검색

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            가장 유사한 Document 리스트
        """
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        유사도 점수와 함께 검색

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            (Document, 점수) 튜플 리스트
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def as_retriever(self, search_kwargs: Optional[dict] = None, score_threshold: Optional[float] = None):
        """
        Retriever 인터페이스로 변환

        Args:
            search_kwargs: 검색 파라미터 (예: {"k": 3})
            score_threshold: 유사도 점수 임계값 (설정 시 임계값 미만 문서 제외)

        Returns:
            Retriever 인스턴스
        """
        search_kwargs = search_kwargs or {"k": 4}
        if score_threshold is not None:
            return self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={**search_kwargs, "score_threshold": score_threshold},
            )
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    def get_collection_stats(self) -> dict:
        """컬렉션 통계 정보 반환"""
        # PGVector에서 문서 수 조회
        try:
            # 직접 SQL로 카운트 조회
            from sqlalchemy import create_engine, text
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :name)"),
                    {"name": self.collection_name}
                )
                count = result.scalar() or 0
        except Exception:
            count = "알 수 없음"

        return {
            "name": self.collection_name,
            "count": count,
            "connection": self.connection_string.split("@")[-1]  # 비밀번호 제외
        }

    def delete_collection(self):
        """컬렉션 삭제"""
        self.vectorstore.delete_collection()
        self._vectorstore = None
        print(f"컬렉션 '{self.collection_name}'이 삭제되었습니다.")

    def clear(self):
        """모든 문서 삭제 (컬렉션 유지)"""
        self.delete_collection()
        self._vectorstore = self._load_or_create()
        print("모든 문서가 삭제되었습니다.")
