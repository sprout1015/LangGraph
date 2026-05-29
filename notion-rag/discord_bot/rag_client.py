"""
RAG API 비동기 클라이언트

FastAPI 서버의 /api/query 엔드포인트를 호출합니다.
"""

from dataclasses import dataclass, field

import httpx


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    sub_queries: list[str] = field(default_factory=list)


class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def query(self, question: str) -> RAGResponse:
        """RAG 질의를 보내고 응답을 반환합니다."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/query",
                json={"question": question},
            )
            resp.raise_for_status()
            data = resp.json()
            return RAGResponse(
                answer=data["answer"],
                sources=data.get("sources", []),
                sub_queries=data.get("sub_queries", []),
            )

    async def health_check(self) -> bool:
        """RAG API 서버 상태를 확인합니다."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/api/health")
            resp.raise_for_status()
            return resp.json().get("status") == "ok"
