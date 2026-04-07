"""
Notion RAG 챗봇 - Gradio UI

스트리밍 응답과 참조 소스 표시를 지원하는 웹 기반 챗봇 인터페이스
"""

import os
import gradio as gr
from dotenv import load_dotenv

from src.llm import LLMAdapter
from src.embeddings import EmbeddingManager
from src.vectorstore import PostgresVectorStore
from src.chains import RAGChain

# 환경 변수 로드
load_dotenv()


def initialize_rag_chain() -> RAGChain:
    """RAG 체인 초기화 (환경변수로 LLM/임베딩 제공자 선택)"""
    # LLM 설정 — LLM_PROVIDER 환경변수로 선택 (기본: anthropic)
    # 선택지: anthropic | ollama | openai
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    llm = LLMAdapter(provider=llm_provider, temperature=0).llm
    print(f"LLM 제공자: {llm_provider}")

    # 임베딩 설정 — OPENAI_API_KEY 있으면 OpenAI, 없으면 HuggingFace 로컬
    embedding_provider = "openai" if os.getenv("OPENAI_API_KEY") else "huggingface"
    embeddings = EmbeddingManager(provider=embedding_provider).embeddings
    print(f"임베딩 제공자: {embedding_provider}")

    # 벡터 스토어 설정 (PostgreSQL + pgvector)
    vector_store = PostgresVectorStore(embeddings, collection_name="notion_docs")
    retriever = vector_store.as_retriever(search_kwargs={"k": 4}, score_threshold=0.3)

    # RAG 체인 생성
    return RAGChain(llm, retriever)


# RAG 체인 초기화
rag_chain = initialize_rag_chain()


def respond_with_streaming(message: str, history: list):
    """
    스트리밍 응답 생성 및 소스 표시

    Args:
        message: 사용자 질문
        history: Gradio 대화 히스토리

    Yields:
        스트리밍 응답 청크
    """
    # 1. 스트리밍 응답 생성
    partial = ""
    for chunk in rag_chain.stream(message):
        partial += chunk
        yield partial

    # 2. 소스 정보 추가 조회
    result = rag_chain.invoke_with_sources(message)
    sources = result.get("sources", [])

    if sources:
        source_text = "\n\n---\n**참조 문서:**\n"
        for i, src in enumerate(sources, 1):
            title = src.get("title", "Unknown")
            category = src.get("category", "")
            url = src.get("url", "")
            label = f"[{category}] {title}" if category else title
            if url:
                source_text += f"{i}. [{label}]({url})\n"
            else:
                source_text += f"{i}. {label}\n"
        yield partial + source_text


# Gradio 인터페이스 구성
demo = gr.ChatInterface(
    fn=respond_with_streaming,
    title="Notion RAG 챗봇",
    description="Notion 문서 기반 질의응답 시스템",
    examples=[
        "Spring Boot에서 JPA 설정은 어떻게 하나요?",
        "REST API URL은 어떻게 설계해야 하나요?",
        "Spring Security에서 인증을 어떻게 구현하나요?",
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
