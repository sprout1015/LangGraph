"""
Discord 메시지 포맷터

2000자 제한 처리 및 소스 포맷팅을 담당합니다.
"""

DISCORD_MAX_LENGTH = 2000


def format_response(
    answer: str,
    sources: list[dict],
    sub_queries: list[str] | None = None,
) -> list[str]:
    """RAG 응답을 Discord 메시지 리스트로 변환합니다.

    각 메시지는 2000자 이내이며, 소스는 마지막 청크에 첨부됩니다.
    복합 질문의 경우 상단에 분석된 하위 질문 섹션을 추가합니다.
    """
    header = _format_sub_queries(sub_queries)
    source_text = _format_sources(sources)
    body = f"{header}{answer}" if header else answer
    full_text = f"{body}\n\n{source_text}" if source_text else body

    if len(full_text) <= DISCORD_MAX_LENGTH:
        return [full_text]

    # 단락 기준으로 분할
    chunks = _split_by_paragraphs(body, source_text)
    total = len(chunks)
    if total == 1:
        return chunks

    return [f"[{i + 1}/{total}] {chunk}" for i, chunk in enumerate(chunks)]


def format_error(error_type: str) -> str:
    """에러 타입에 따른 한글 메시지를 반환합니다."""
    messages = {
        "timeout": "요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
        "connection": "RAG API 서버에 연결할 수 없습니다. 서버 상태를 확인해주세요.",
        "server": "서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        "empty": "질문을 입력해주세요.",
        "unknown": "알 수 없는 오류가 발생했습니다.",
    }
    return messages.get(error_type, messages["unknown"])


def _format_sub_queries(sub_queries: list[str] | None) -> str:
    """복합 질문의 하위 질문 목록을 포맷팅합니다."""
    if not sub_queries:
        return ""
    lines = ["🔍 **분석된 하위 질문:**"]
    for i, q in enumerate(sub_queries, 1):
        lines.append(f"  {i}. {q}")
    lines.append("")  # 본문과 구분 개행
    return "\n".join(lines) + "\n\n"


def _format_sources(sources: list[dict]) -> str:
    """소스 문서를 포맷팅합니다."""
    if not sources:
        return ""

    lines = ["**참고 문서:**"]
    for i, src in enumerate(sources, 1):
        title = src.get("title", "제목 없음")
        category = src.get("category", "")
        url = src.get("url", "")
        label = f"[{category}] {title}" if category else title
        if url:
            lines.append(f"> {i}. [{label}]({url})")
        else:
            lines.append(f"> {i}. {label}")
    return "\n".join(lines)


def _split_by_paragraphs(answer: str, source_text: str) -> list[str]:
    """답변을 단락 기준으로 분할하고, 소스는 마지막 청크에 첨부합니다."""
    paragraphs = answer.split("\n\n")
    chunks: list[str] = []
    current = ""

    # [n/total] prefix + 여유 공간
    overhead = 10
    limit = DISCORD_MAX_LENGTH - overhead

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # 단일 단락이 limit 초과하면 강제 분할
            if len(para) > limit:
                for start in range(0, len(para), limit):
                    chunks.append(para[start : start + limit])
                current = ""
            else:
                current = para

    # 마지막 청크에 소스 첨부
    if source_text:
        last_with_source = f"{current}\n\n{source_text}" if current else source_text
        if len(last_with_source) <= limit:
            current = last_with_source
        else:
            if current:
                chunks.append(current)
            current = source_text

    if current:
        chunks.append(current)

    return chunks
