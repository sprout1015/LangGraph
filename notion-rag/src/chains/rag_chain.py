"""
RAG 체인 모듈

Retriever와 LLM을 연결하여 문서 기반 질의응답을 수행합니다.
"""

from typing import Optional, List, Dict, Any
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.retrievers import BaseRetriever


# 기본 RAG 프롬프트 (한국어)
DEFAULT_RAG_PROMPT = """다음 컨텍스트를 기반으로 질문에 답변하세요.
컨텍스트에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
답변은 명확하고 간결하게 작성하세요.

컨텍스트:
{context}

질문: {question}

답변:"""


# 대화 히스토리를 포함한 프롬프트
CONVERSATIONAL_RAG_PROMPT = """다음 대화 기록과 컨텍스트를 참고하여 질문에 답변하세요.
컨텍스트에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.

대화 기록:
{chat_history}

컨텍스트:
{context}

현재 질문: {question}

답변:"""


def _build_notion_url(page_id: str) -> str:
    """Notion 페이지 ID에서 URL을 생성합니다."""
    if not page_id:
        return ""
    return f"https://notion.so/{page_id.replace('-', '')}"


def _to_str(value) -> str:
    """메타데이터 값을 문자열로 변환합니다 (리스트/None 대응)."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def format_docs(docs: List[Document]) -> str:
    """Document 리스트를 문자열로 포맷팅"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        # 메타데이터에서 제목 추출
        title = doc.metadata.get("title", f"문서 {i}")
        content = doc.page_content.strip()
        formatted.append(f"[{title}]\n{content}")

    return "\n\n---\n\n".join(formatted)


class RAGChain:
    """RAG 파이프라인을 구성하고 실행하는 클래스"""

    def __init__(
        self,
        llm: BaseChatModel | BaseLLM,
        retriever: BaseRetriever,
        prompt_template: Optional[str] = None
    ):
        """
        Args:
            llm: 언어 모델 인스턴스
            retriever: 문서 검색기
            prompt_template: 커스텀 프롬프트 (없으면 기본값 사용)
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template or DEFAULT_RAG_PROMPT
        self._chain = None

    @property
    def chain(self):
        """RAG 체인 반환 (지연 생성)"""
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain

    def _build_chain(self):
        """LCEL로 RAG 체인 구성"""
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        chain = (
            RunnableParallel(
                context=self.retriever | format_docs,
                question=RunnablePassthrough()
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def invoke(self, question: str) -> str:
        """
        질문에 대한 답변 생성

        Args:
            question: 사용자 질문

        Returns:
            LLM이 생성한 답변
        """
        return self.chain.invoke(question)

    def invoke_with_sources(self, question: str) -> Dict[str, Any]:
        """
        답변과 함께 참조 문서도 반환 (retriever 1회 호출)

        Args:
            question: 사용자 질문

        Returns:
            {"answer": 답변, "sources": 참조 문서 리스트}
        """
        # 1회 검색으로 문서 조회 + 답변 생성
        docs = self.retriever.invoke(question)
        context = format_docs(docs)

        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        answer = (prompt | self.llm | StrOutputParser()).invoke(
            {"context": context, "question": question}
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "content": doc.page_content[:200] + "...",
                    "category": _to_str(doc.metadata.get("카테고리")),
                    "url": _build_notion_url(doc.metadata.get("id", "")),
                }
                for doc in docs
            ]
        }

    def stream(self, question: str):
        """
        스트리밍 방식으로 답변 생성

        Args:
            question: 사용자 질문

        Yields:
            답변 토큰
        """
        for chunk in self.chain.stream(question):
            yield chunk


class ConversationalRAGChain:
    """대화 히스토리를 유지하는 RAG 체인"""

    def __init__(
        self,
        llm: BaseChatModel | BaseLLM,
        retriever: BaseRetriever,
        max_history: int = 5
    ):
        """
        Args:
            llm: 언어 모델 인스턴스
            retriever: 문서 검색기
            max_history: 유지할 최대 대화 수
        """
        self.llm = llm
        self.retriever = retriever
        self.max_history = max_history
        self.chat_history: List[Dict[str, str]] = []
        self._chain = None

    @property
    def chain(self):
        """대화형 RAG 체인 반환"""
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain

    def _build_chain(self):
        """대화 히스토리를 포함한 체인 구성"""
        prompt = ChatPromptTemplate.from_template(CONVERSATIONAL_RAG_PROMPT)

        chain = (
            RunnableParallel(
                context=self.retriever | format_docs,
                question=RunnablePassthrough(),
                chat_history=lambda _: self._format_history()
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _format_history(self) -> str:
        """대화 히스토리를 문자열로 포맷팅"""
        if not self.chat_history:
            return "(이전 대화 없음)"

        formatted = []
        for entry in self.chat_history[-self.max_history:]:
            formatted.append(f"사용자: {entry['question']}")
            formatted.append(f"AI: {entry['answer']}")

        return "\n".join(formatted)

    def invoke(self, question: str) -> str:
        """
        질문에 대한 답변 생성 (히스토리 유지)

        Args:
            question: 사용자 질문

        Returns:
            LLM이 생성한 답변
        """
        answer = self.chain.invoke(question)

        # 히스토리에 추가
        self.chat_history.append({
            "question": question,
            "answer": answer
        })

        # 최대 히스토리 수 유지
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

        return answer

    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []
        print("대화 히스토리가 초기화되었습니다.")
