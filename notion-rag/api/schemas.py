"""
API 요청/응답 스키마 정의

FastAPI 엔드포인트의 Pydantic 모델을 정의합니다.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """RAG 질의 요청"""
    question: str = Field(..., description="질문 텍스트", min_length=1)
    category: str = Field(default="", description="카테고리 필터 (빈 문자열이면 전체 검색)")


class SourceDoc(BaseModel):
    """참조 문서 정보"""
    title: str = Field(description="문서 제목")
    content: str = Field(description="문서 내용 미리보기 (200자)")
    category: str = Field(default="", description="문서 카테고리")
    url: str = Field(default="", description="Notion 원문 링크")


class QueryResponse(BaseModel):
    """RAG 질의 응답"""
    answer: str = Field(description="LLM이 생성한 답변")
    sources: List[SourceDoc] = Field(default_factory=list, description="참조 문서 목록")


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = Field(description="서버 상태")
    rag_chain: bool = Field(description="RAG 체인 초기화 여부")
    vector_store: Optional[str] = Field(default=None, description="벡터 스토어 상태")
