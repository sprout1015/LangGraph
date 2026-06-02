"""
임베딩 관리 모듈

다양한 임베딩 모델을 지원하는 어댑터 패턴 구현
"""

import os
from typing import List, Literal, Optional
from langchain_core.embeddings import Embeddings


EmbeddingProvider = Literal["openai", "huggingface"]


class EmbeddingManager:
    """다양한 임베딩 모델을 관리하는 클래스"""

    def __init__(
        self,
        provider: EmbeddingProvider = "openai",
        model_name: Optional[str] = None
    ):
        """
        Args:
            provider: 임베딩 제공자 ("openai" 또는 "huggingface")
            model_name: 사용할 모델명 (없으면 기본값 사용)
        """
        self.provider = provider
        self.model_name = model_name
        self._embeddings: Optional[Embeddings] = None

    @property
    def embeddings(self) -> Embeddings:
        """임베딩 모델 인스턴스를 반환 (지연 로딩)"""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def _create_embeddings(self) -> Embeddings:
        """임베딩 모델 인스턴스 생성"""
        if self.provider == "openai":
            return self._create_openai_embeddings()
        elif self.provider == "huggingface":
            return self._create_huggingface_embeddings()
        else:
            raise ValueError(f"지원하지 않는 provider: {self.provider}")

    def _create_openai_embeddings(self) -> Embeddings:
        """OpenAI 임베딩 모델 생성"""
        from langchain_openai import OpenAIEmbeddings

        model = self.model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        return OpenAIEmbeddings(model=model)

    def _create_huggingface_embeddings(self) -> Embeddings:
        """HuggingFace 임베딩 모델 생성 (무료, 로컬 실행)"""
        from langchain_community.embeddings import HuggingFaceEmbeddings

        model = self.model_name or "sentence-transformers/all-MiniLM-L6-v2"

        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},  # GPU 사용 시 "cuda"
            encode_kwargs={"normalize_embeddings": True}
        )

    def embed_query(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩"""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 임베딩"""
        return self.embeddings.embed_documents(texts)


def get_default_embeddings() -> Embeddings:
    """기본 임베딩 모델 반환 (OpenAI)"""
    manager = EmbeddingManager(provider="openai")
    return manager.embeddings
