"""
ChromaDB 벡터 스토어 관리 모듈

문서의 벡터 저장 및 유사도 검색을 담당합니다.
"""

import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma


class ChromaVectorStore:
    """ChromaDB 기반 벡터 스토어 관리 클래스"""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: Optional[str] = None,
        collection_name: str = "notion_docs"
    ):
        """
        Args:
            embeddings: 임베딩 모델 인스턴스
            persist_directory: 벡터 DB 저장 경로 (없으면 환경변수에서 로드)
            collection_name: 컬렉션 이름
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", "./data/chroma"
        )
        self.collection_name = collection_name
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """벡터 스토어 인스턴스 반환 (지연 로딩)"""
        if self._vectorstore is None:
            self._vectorstore = self._load_or_create()
        return self._vectorstore

    def _load_or_create(self) -> Chroma:
        """기존 벡터 스토어를 로드하거나 새로 생성"""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
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

    def from_documents(self, documents: List[Document]) -> "ChromaVectorStore":
        """
        문서로부터 새 벡터 스토어 생성

        Args:
            documents: Document 리스트

        Returns:
            ChromaVectorStore 인스턴스
        """
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
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

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Retriever 인터페이스로 변환

        Args:
            search_kwargs: 검색 파라미터 (예: {"k": 3})

        Returns:
            Retriever 인스턴스
        """
        search_kwargs = search_kwargs or {"k": 4}
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    def get_collection_stats(self) -> dict:
        """컬렉션 통계 정보 반환"""
        collection = self.vectorstore._collection
        return {
            "name": self.collection_name,
            "count": collection.count(),
            "persist_directory": self.persist_directory
        }

    def delete_collection(self):
        """컬렉션 삭제"""
        self.vectorstore.delete_collection()
        self._vectorstore = None
        print(f"컬렉션 '{self.collection_name}'이 삭제되었습니다.")

    def clear(self):
        """모든 문서 삭제 (컬렉션 유지)"""
        # ChromaDB에서는 컬렉션을 삭제하고 다시 생성
        self.delete_collection()
        self._vectorstore = self._load_or_create()
        print("모든 문서가 삭제되었습니다.")
