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
from src.chains import RAGChain, DecomposedRAGChain, QueryDecomposer

# 환경 변수 로드
load_dotenv()

# 모듈 레벨 RAG 체인 (lifespan에서 초기화)
rag_chain: DecomposedRAGChain | None = None
vector_store: PostgresVectorStore | None = None


def initialize_rag_chain() -> tuple[DecomposedRAGChain, PostgresVectorStore]:
    """RAG 체인 초기화 (쿼리 분해 포함)"""
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

    base_chain = RAGChain(llm, retriever)
    decomposer = QueryDecomposer(llm)
    return DecomposedRAGChain(base_chain, decomposer), vs


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
            sub_queries=result.get("sub_queries", []),
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
        import json
        t_start = time.time()
        try:
            # 답변 + 소스 + 서브쿼리 일괄 조회
            result = await asyncio.to_thread(
                rag_chain.invoke_with_sources, request.question, category_filter
            )

            # 복합 질문이면 분해된 서브쿼리 먼저 전송
            sub_queries = result.get("sub_queries", [])
            if sub_queries:
                yield {"event": "decompose", "data": json.dumps(sub_queries, ensure_ascii=False)}

            # 답변을 토큰 단위로 스트리밍 (청크 분할)
            answer = result.get("answer", "")
            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                yield {"event": "token", "data": answer[i:i + chunk_size]}

            # 소스 정보 전송
            sources = result.get("sources", [])
            if sources:
                yield {"event": "sources", "data": json.dumps(sources, ensure_ascii=False)}

            rag_query_total.labels(status="success", endpoint="/api/query/stream").inc()
            rag_query_duration_seconds.labels(endpoint="/api/query/stream").observe(time.time() - t_start)
            yield {"event": "done", "data": ""}
        except Exception as e:
            rag_query_total.labels(status="error", endpoint="/api/query/stream").inc()
            rag_query_duration_seconds.labels(endpoint="/api/query/stream").observe(time.time() - t_start)
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())
