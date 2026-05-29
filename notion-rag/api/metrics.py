"""
Prometheus 메트릭 정의

RAG API 서버의 주요 지표를 Prometheus 형식으로 노출합니다.
"""

from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# ─────────────────────────────────────────────
# 메트릭 정의
# ─────────────────────────────────────────────

# 질의 횟수 (label: status=success|error)
rag_query_total = Counter(
    "rag_query_total",
    "RAG 질의 총 횟수",
    ["status", "endpoint"],
)

# 전체 질의-응답 레이턴시 (seconds)
rag_query_duration_seconds = Histogram(
    "rag_query_duration_seconds",
    "RAG 질의-응답 전체 레이턴시 (초)",
    ["endpoint"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

# LLM 호출 레이턴시만 (seconds)
rag_llm_duration_seconds = Histogram(
    "rag_llm_duration_seconds",
    "LLM 호출 레이턴시 (초)",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

# 질의당 검색된 문서 수
rag_docs_retrieved = Histogram(
    "rag_docs_retrieved",
    "질의당 검색된 문서 수",
    buckets=[0, 1, 2, 3, 4, 5, 6, 8, 10],
)

# 검색 문서의 유사도 점수 분포
rag_similarity_score = Histogram(
    "rag_similarity_score",
    "검색된 문서의 유사도 점수",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
