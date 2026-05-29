"""
쿼리 분해 모듈

복합 질문을 독립적인 서브쿼리로 분해합니다.
LLM이 질문 구조를 분석하여 단일/복합 여부를 판단하고,
복합인 경우 서브쿼리 리스트를 반환합니다.
"""

import json
import re
from langchain_core.language_models import BaseChatModel, BaseLLM


DECOMPOSE_PROMPT = """사용자의 질문이 독립적인 여러 하위 질문을 포함하는지 판단하세요.

판단 기준 (복합 질문):
- "또한", "그리고", "같은", "아니면", 물음표 여러 개 등의 접속사/구분자
- 서로 다른 도메인(예: 백엔드 구조 + 인프라 + 다른 언어 비교)을 한 번에 묻는 경우
- 주제가 명확히 다른 두 가지 이상의 질문이 하나로 합쳐진 경우

판단 기준 (단일 질문):
- 한 주제를 여러 측면에서 묻는 경우 (예: "Spring Security 인증 구조와 동작 방식은?")
- 단순히 긴 질문이거나 배경 설명이 포함된 경우

질문: {question}

JSON으로만 응답 (다른 텍스트 없이):
{{"is_multi": true/false, "sub_queries": ["q1", "q2", ...]}}

단일 질문이면 is_multi=false, sub_queries에 원본 질문 그대로 1개만 반환."""


def _extract_json(text: str) -> str:
    """LLM 응답에서 JSON 부분만 추출합니다."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text


class QueryDecomposer:
    """LLM을 사용해 복합 질문을 서브쿼리로 분해합니다."""

    def __init__(self, llm: BaseChatModel | BaseLLM):
        self.llm = llm

    def decompose(self, question: str) -> list[str]:
        """
        질문을 분석하여 서브쿼리 리스트를 반환합니다.

        Args:
            question: 사용자 질문

        Returns:
            단일 질문이면 [question], 복합 질문이면 [q1, q2, ...]
            LLM 호출 실패 시 원본 질문으로 폴백합니다.
        """
        try:
            result = self.llm.invoke(DECOMPOSE_PROMPT.format(question=question))
            content = result.content if hasattr(result, "content") else str(result)
            parsed = json.loads(_extract_json(content))
            if not parsed.get("is_multi", False):
                return [question]
            sub_queries = parsed.get("sub_queries", [])
            return sub_queries if sub_queries else [question]
        except Exception:
            return [question]
