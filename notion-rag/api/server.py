"""
FastAPI RAG API 서버

RAG 체인을 HTTP API로 래핑하여 외부 클라이언트에서 접근 가능하게 합니다.

실행:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import os
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from api.schemas import QueryRequest, QueryResponse, SourceDoc, HealthResponse
from api.metrics import (
    rag_query_total,
    rag_query_duration_seconds,
    rag_llm_duration_seconds,
    rag_docs_retrieved,
    rag_similarity_score,
)
from src.llm import LLMAdapter
from src.embeddings import EmbeddingManager
from src.vectorstore import PostgresVectorStore
from src.chains import RAGChain

# 환경 변수 로드
load_dotenv()

# 모듈 레벨 RAG 체인 (lifespan에서 초기화)
rag_chain: RAGChain | None = None
vector_store: PostgresVectorStore | None = None


def initialize_rag_chain() -> tuple[RAGChain, PostgresVectorStore]:
    """RAG 체인 초기화 (app.py와 동일한 패턴)"""
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    fallback_provider = os.getenv("FALLBACK_LLM_PROVIDER")

    # Primary LLM
    primary_llm = LLMAdapter(provider=llm_provider, temperature=0).llm

    # Fallback LLM (Phase 3: Qwen → Claude 자동 전환)
    if fallback_provider:
        fallback_llm = LLMAdapter(provider=fallback_provider, temperature=0).llm
        llm = primary_llm.with_fallbacks([fallback_llm])
    else:
        llm = primary_llm

    # 임베딩 설정
    embeddings = EmbeddingManager(provider="openai").embeddings

    # 벡터 스토어 설정
    vs = PostgresVectorStore(embeddings, collection_name="notion_docs")
    retriever = vs.as_retriever(search_kwargs={"k": 4}, score_threshold=0.3)

    return RAGChain(llm, retriever), vs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 RAG 체인 초기화/정리"""
    global rag_chain, vector_store
    rag_chain, vector_store = initialize_rag_chain()
    print("RAG 체인 초기화 완료")
    yield
    print("서버 종료")


app = FastAPI(
    title="Notion RAG API",
    description="Notion 문서 기반 RAG 질의응답 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정 (로컬 개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    vs_status = None
    if vector_store:
        try:
            stats = vector_store.get_collection_stats()
            vs_status = f"{stats['name']}: {stats['count']} docs"
        except Exception:
            vs_status = "error"

    return HealthResponse(
        status="ok",
        rag_chain=rag_chain is not None,
        vector_store=vs_status,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus 메트릭 엔드포인트"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """RAG 질의 엔드포인트 (동기 응답)"""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG 체인이 초기화되지 않았습니다")

    t_start = time.time()
    category_filter = {"카테고리": {"$eq": request.category}} if request.category else None

    try:
        # 문서 검색 (메트릭 수집용 직접 호출)
        docs_with_scores = await asyncio.to_thread(
            vector_store.similarity_search_with_score, request.question, 10, category_filter
        )
        # 유사도 점수 기록
        for _, score in docs_with_scores:
            rag_similarity_score.observe(score)

        # 답변 생성 (LLM 레이턴시 측정)
        t_llm = time.time()
        result = await asyncio.to_thread(
            rag_chain.invoke_with_sources, request.question, category_filter
        )
        rag_llm_duration_seconds.observe(time.time() - t_llm)

        # 검색 문서 수 기록
        rag_docs_retrieved.observe(len(result.get("sources", [])))

        rag_query_total.labels(status="success", endpoint="/api/query").inc()
        rag_query_duration_seconds.labels(endpoint="/api/query").observe(time.time() - t_start)

        return QueryResponse(
            answer=result["answer"],
            sources=[
                SourceDoc(
                    title=s["title"],
                    content=s["content"],
                    category=s.get("category", ""),
                    url=s.get("url", ""),
                )
                for s in result.get("sources", [])
            ],
        )
    except Exception as e:
        rag_query_total.labels(status="error", endpoint="/api/query").inc()
        rag_query_duration_seconds.labels(endpoint="/api/query").observe(time.time() - t_start)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """RAG 질의 스트리밍 엔드포인트 (SSE)"""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG 체인이 초기화되지 않았습니다")

    category_filter = {"카테고리": {"$eq": request.category}} if request.category else None

    async def event_generator():
        t_start = time.time()
        try:
            # 동기 스트림을 비동기로 래핑
            def _stream():
                return list(rag_chain.stream(request.question, filter=category_filter))

            chunks = await asyncio.to_thread(_stream)
            for chunk in chunks:
                yield {"event": "token", "data": chunk}

            # 소스 정보 전송
            result = await asyncio.to_thread(
                rag_chain.invoke_with_sources, request.question, category_filter
            )
            sources = result.get("sources", [])
            if sources:
                import json
                yield {"event": "sources", "data": json.dumps(sources, ensure_ascii=False)}

            rag_query_total.labels(status="success", endpoint="/api/query/stream").inc()
            rag_query_duration_seconds.labels(endpoint="/api/query/stream").observe(time.time() - t_start)
            yield {"event": "done", "data": ""}
        except Exception as e:
            rag_query_total.labels(status="error", endpoint="/api/query/stream").inc()
            rag_query_duration_seconds.labels(endpoint="/api/query/stream").observe(time.time() - t_start)
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())
