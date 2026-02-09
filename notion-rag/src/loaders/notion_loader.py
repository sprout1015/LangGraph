"""
노션 문서 로더 모듈

노션 API를 사용하여 데이터베이스 및 페이지를 로드합니다.
하위 페이지 재귀 로드 및 SemanticChunker를 통한 의미 단위 분할을 지원합니다.
"""

import os
from typing import List, Optional, Set
from langchain_core.documents import Document
from langchain_community.document_loaders import NotionDBLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from notion_client import Client


class NotionDocumentLoader:
    """노션 문서를 로드하고 전처리하는 클래스"""

    def __init__(
        self,
        integration_token: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Args:
            integration_token: 노션 Integration 토큰 (없으면 환경변수에서 로드)
            chunk_size: 청크당 최대 문자 수
            chunk_overlap: 청크 간 중복 문자 수
        """
        self.integration_token = integration_token or os.getenv("NOTION_API_KEY")
        if not self.integration_token:
            raise ValueError(
                "Notion API 키가 필요합니다. "
                "NOTION_API_KEY 환경변수를 설정하거나 integration_token을 전달하세요."
            )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )

    def load_database(self, database_id: str) -> List[Document]:
        """
        노션 데이터베이스의 모든 페이지를 로드합니다.

        Args:
            database_id: 노션 데이터베이스 ID

        Returns:
            로드된 Document 리스트
        """
        loader = NotionDBLoader(
            integration_token=self.integration_token,
            database_id=database_id
        )

        docs = loader.load()
        print(f"로드된 문서 수: {len(docs)}")

        return docs

    def load_and_split(self, database_id: str) -> List[Document]:
        """
        노션 데이터베이스를 로드하고 청크로 분할합니다.

        Args:
            database_id: 노션 데이터베이스 ID

        Returns:
            분할된 Document 리스트
        """
        docs = self.load_database(database_id)
        splits = self.text_splitter.split_documents(docs)

        print(f"분할된 청크 수: {len(splits)}")

        return splits

    def load_with_metadata(
        self,
        database_id: str,
        include_metadata: bool = True
    ) -> List[Document]:
        """
        메타데이터를 보존하며 문서를 로드합니다.

        Args:
            database_id: 노션 데이터베이스 ID
            include_metadata: 메타데이터 포함 여부

        Returns:
            Document 리스트 (메타데이터 포함)
        """
        docs = self.load_database(database_id)

        # 메타데이터 정리 및 보강
        for doc in docs:
            if include_metadata and doc.metadata:
                # 소스 정보 추가
                doc.metadata["source"] = "notion"
                doc.metadata["database_id"] = database_id

        return docs


class NotionRecursiveLoader:
    """
    하위 페이지를 재귀적으로 로드하고 SemanticChunker로 의미 단위 분할하는 클래스

    NotionDBLoader로 기본 문서를 로드한 후, 각 페이지의 하위 페이지를 재귀적으로 탐색합니다.
    """

    def __init__(
        self,
        integration_token: Optional[str] = None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95
    ):
        """
        Args:
            integration_token: 노션 Integration 토큰 (없으면 환경변수에서 로드)
            breakpoint_threshold_type: SemanticChunker 분할 기준 타입
            breakpoint_threshold_amount: 분할 임계값 (percentile의 경우 상위 N% 유사도 급락점)
        """
        self.integration_token = integration_token or os.getenv("NOTION_API_KEY")
        if not self.integration_token:
            raise ValueError(
                "Notion API 키가 필요합니다. "
                "NOTION_API_KEY 환경변수를 설정하거나 integration_token을 전달하세요."
            )

        self.client = Client(auth=self.integration_token)
        self.chunker = SemanticChunker(
            embeddings=OpenAIEmbeddings(),
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
        self._loaded_page_ids: Set[str] = set()

    def _format_id(self, notion_id: str) -> str:
        """Notion ID를 UUID 형식으로 변환합니다 (하이픈 없으면 추가)."""
        if "-" in notion_id:
            return notion_id
        if len(notion_id) == 32:
            return f"{notion_id[:8]}-{notion_id[8:12]}-{notion_id[12:16]}-{notion_id[16:20]}-{notion_id[20:]}"
        return notion_id

    def load_database(self, database_id: str) -> List[Document]:
        """
        노션 데이터베이스의 모든 페이지를 하위 페이지 포함하여 재귀적으로 로드합니다.

        Args:
            database_id: 노션 데이터베이스 ID

        Returns:
            로드된 Document 리스트
        """
        self._loaded_page_ids.clear()

        # NotionDBLoader로 기본 문서 로드
        base_loader = NotionDBLoader(
            integration_token=self.integration_token,
            database_id=database_id
        )
        base_docs = base_loader.load()
        print(f"NotionDBLoader: {len(base_docs)} documents loaded")

        # 각 문서의 page_id를 추출하여 하위 페이지 탐색
        all_docs = []
        for doc in base_docs:
            page_id = doc.metadata.get("id")
            if page_id:
                self._loaded_page_ids.add(page_id)
                # 메타데이터에서 제목 추출 (title 또는 노션 DB 속성명 순서로 검색)
                title = self._extract_title_from_metadata(doc.metadata)
                doc.metadata["title"] = title
                doc.metadata["full_title"] = title

            all_docs.append(doc)

            # 하위 페이지 탐색
            if page_id:
                child_docs = self._load_child_pages(page_id, title)
                all_docs.extend(child_docs)

        print(f"Total documents loaded (including child pages): {len(all_docs)}")
        return all_docs

    def _load_child_pages(
        self,
        page_id: str,
        parent_title: str
    ) -> List[Document]:
        """페이지의 하위 페이지를 재귀적으로 로드합니다."""
        child_docs = []
        child_page_ids = self._find_child_pages(page_id)

        for child_id in child_page_ids:
            if child_id in self._loaded_page_ids:
                continue
            self._loaded_page_ids.add(child_id)

            docs = self._load_page_recursive(child_id, parent_title)
            child_docs.extend(docs)

        return child_docs

    def _load_page_recursive(
        self,
        page_id: str,
        parent_title: str
    ) -> List[Document]:
        """단일 페이지와 그 하위 페이지를 로드합니다."""
        docs = []

        try:
            page_data = self.client.pages.retrieve(page_id=page_id)
        except Exception as e:
            print(f"Page {page_id} retrieval failed: {e}")
            return []

        # 페이지 제목 추출
        title = self._extract_title(page_data)
        full_title = f"{parent_title} > {title}"

        # 페이지 블록 콘텐츠 로드
        content = self._load_page_content(page_id)

        if content.strip():
            metadata = {
                "source": "notion",
                "id": page_id,
                "title": title,
                "full_title": full_title,
            }

            # 페이지 속성에서 추가 메타데이터 추출
            if "properties" in page_data:
                for prop_name, prop_value in page_data["properties"].items():
                    extracted = self._extract_property_value(prop_value)
                    if extracted:
                        metadata[prop_name] = extracted

            docs.append(Document(page_content=content, metadata=metadata))
            print(f"  + Child page: {full_title}")

        # 하위 페이지 재귀 탐색
        child_docs = self._load_child_pages(page_id, full_title)
        docs.extend(child_docs)

        return docs

    def _extract_title(self, page_data: dict) -> str:
        """페이지 데이터에서 제목을 추출합니다."""
        properties = page_data.get("properties", {})

        # title 타입 속성 찾기
        for prop_name, prop_value in properties.items():
            if prop_value.get("type") == "title":
                title_array = prop_value.get("title", [])
                if title_array:
                    return "".join(t.get("plain_text", "") for t in title_array)

        return "Untitled"

    def _extract_title_from_metadata(self, metadata: dict) -> str:
        """
        NotionDBLoader 메타데이터에서 제목을 추출합니다.

        노션 DB 속성명이 메타데이터 키가 되므로, 일반적인 제목 속성명을 순서대로 검색합니다.
        """
        # 일반적인 제목 속성명 패턴 (우선순위 순)
        title_keys = [
            "title", "Title", "TITLE",
            "name", "Name", "NAME",
            "이름", "제목", "작업 이름", "문서 이름", "페이지 이름",
            "작업명", "문서명", "항목", "주제"
        ]

        # 지정된 키로 먼저 검색
        for key in title_keys:
            if key in metadata and metadata[key]:
                return str(metadata[key])

        # 못 찾으면 메타데이터 키 중 문자열 값이 있는 첫 번째 항목 사용
        # (id, chunk_index, source 등 시스템 키 제외)
        system_keys = {"id", "source", "database_id", "full_title", "chunk_index"}
        for key, value in metadata.items():
            if key not in system_keys and value and isinstance(value, str):
                return value

        return "Untitled"

    def _extract_property_value(self, prop_value: dict) -> Optional[str]:
        """노션 속성 값을 문자열로 추출합니다."""
        prop_type = prop_value.get("type")

        if prop_type == "title":
            return None  # 제목은 별도로 처리

        elif prop_type == "rich_text":
            texts = prop_value.get("rich_text", [])
            return "".join(t.get("plain_text", "") for t in texts) if texts else None

        elif prop_type == "select":
            select = prop_value.get("select")
            return select.get("name") if select else None

        elif prop_type == "multi_select":
            options = prop_value.get("multi_select", [])
            return ", ".join(o.get("name", "") for o in options) if options else None

        elif prop_type == "date":
            date = prop_value.get("date")
            return date.get("start") if date else None

        elif prop_type == "checkbox":
            return str(prop_value.get("checkbox", False))

        elif prop_type == "number":
            num = prop_value.get("number")
            return str(num) if num is not None else None

        elif prop_type == "url":
            return prop_value.get("url")

        return None

    def _load_page_content(self, page_id: str) -> str:
        """페이지의 블록 콘텐츠를 로드합니다."""
        blocks = self._get_blocks_recursive(page_id)
        return self._blocks_to_text(blocks)

    def _get_blocks_recursive(self, block_id: str) -> List[dict]:
        """블록과 하위 블록을 재귀적으로 가져옵니다."""
        blocks = []
        has_more = True
        next_cursor = None

        try:
            while has_more:
                response = self.client.blocks.children.list(
                    block_id=block_id,
                    start_cursor=next_cursor
                )
                for block in response["results"]:
                    blocks.append(block)
                    # 하위 블록이 있으면 재귀 로드 (child_page 제외)
                    if block.get("has_children") and block.get("type") != "child_page":
                        child_blocks = self._get_blocks_recursive(block["id"])
                        blocks.extend(child_blocks)

                has_more = response.get("has_more", False)
                next_cursor = response.get("next_cursor")
        except Exception as e:
            print(f"블록 로드 실패 (block_id: {block_id}): {e}")

        return blocks

    def _blocks_to_text(self, blocks: List[dict]) -> str:
        """블록 리스트를 텍스트로 변환합니다."""
        text_parts = []

        for block in blocks:
            block_type = block.get("type")
            block_data = block.get(block_type, {})

            text = self._extract_block_text(block_type, block_data)
            if text:
                text_parts.append(text)

        return "\n".join(text_parts)

    def _extract_block_text(self, block_type: str, block_data: dict) -> str:
        """블록에서 텍스트를 추출합니다."""
        # rich_text를 포함하는 블록 타입들
        rich_text_blocks = [
            "paragraph", "heading_1", "heading_2", "heading_3",
            "bulleted_list_item", "numbered_list_item", "toggle",
            "quote", "callout"
        ]

        if block_type in rich_text_blocks:
            rich_text = block_data.get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in rich_text)

            if block_type == "heading_1":
                return f"# {text}"
            elif block_type == "heading_2":
                return f"## {text}"
            elif block_type == "heading_3":
                return f"### {text}"
            elif block_type in ["bulleted_list_item", "numbered_list_item"]:
                return f"- {text}"
            elif block_type == "quote":
                return f"> {text}"
            else:
                return text

        elif block_type == "code":
            rich_text = block_data.get("rich_text", [])
            code = "".join(t.get("plain_text", "") for t in rich_text)
            language = block_data.get("language", "")
            return f"```{language}\n{code}\n```"

        elif block_type == "divider":
            return "---"

        elif block_type == "table_row":
            cells = block_data.get("cells", [])
            row_text = " | ".join(
                "".join(t.get("plain_text", "") for t in cell)
                for cell in cells
            )
            return f"| {row_text} |"

        return ""

    def _find_child_pages(self, page_id: str) -> List[str]:
        """페이지 내의 child_page 블록을 찾습니다."""
        child_page_ids = []

        try:
            has_more = True
            next_cursor = None

            while has_more:
                response = self.client.blocks.children.list(
                    block_id=page_id,
                    start_cursor=next_cursor
                )

                for block in response["results"]:
                    if block.get("type") == "child_page":
                        child_page_ids.append(block["id"])

                has_more = response.get("has_more", False)
                next_cursor = response.get("next_cursor")

        except Exception as e:
            print(f"하위 페이지 탐색 실패 (page_id: {page_id}): {e}")

        return child_page_ids

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        SemanticChunker를 사용하여 문서를 의미 단위로 분할합니다.

        Args:
            docs: 분할할 Document 리스트

        Returns:
            분할된 Document 리스트
        """
        result = []

        for doc in docs:
            if not doc.page_content.strip():
                continue

            try:
                chunks = self.chunker.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        result.append(Document(
                            page_content=chunk,
                            metadata={**doc.metadata, "chunk_index": i}
                        ))
            except Exception as e:
                # 분할 실패 시 원본 문서 유지
                print(f"문서 분할 실패 ({doc.metadata.get('title', 'Unknown')}): {e}")
                result.append(doc)

        print(f"SemanticChunker로 분할된 청크 수: {len(result)}")
        return result

    def load_and_split(self, database_id: str) -> List[Document]:
        """
        데이터베이스를 로드하고 의미 단위로 분할합니다.

        Args:
            database_id: 노션 데이터베이스 ID

        Returns:
            분할된 Document 리스트
        """
        docs = self.load_database(database_id)
        return self.split_documents(docs)


def create_sample_documents() -> List[Document]:
    """
    테스트용 샘플 문서를 생성합니다.
    노션 연동 전 RAG 파이프라인 테스트에 사용합니다.
    """
    sample_docs = [
        Document(
            page_content="""# Spring Boot JPA 설정 가이드

## 1. 의존성 추가
build.gradle에 다음 의존성을 추가합니다:
```
implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
implementation 'com.h2database:h2'
```

## 2. application.yml 설정
```yaml
spring:
  datasource:
    url: jdbc:h2:mem:testdb
    driver-class-name: org.h2.Driver
  jpa:
    hibernate:
      ddl-auto: create-drop
    show-sql: true
```

## 3. 엔티티 생성
@Entity 어노테이션을 사용하여 JPA 엔티티를 정의합니다.""",
            metadata={
                "title": "Spring Boot JPA 설정",
                "source": "notion",
                "tags": ["spring", "jpa", "database"]
            }
        ),
        Document(
            page_content="""# Spring Security 인증 구현

## 1. 기본 설정
Spring Security 의존성을 추가하면 기본적으로 모든 엔드포인트가 보호됩니다.

## 2. SecurityConfig 클래스
```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/public/**").permitAll()
                .anyRequest().authenticated()
            )
            .formLogin(withDefaults());
        return http.build();
    }
}
```

## 3. 사용자 인증
UserDetailsService를 구현하여 사용자 정보를 로드합니다.""",
            metadata={
                "title": "Spring Security 인증",
                "source": "notion",
                "tags": ["spring", "security", "authentication"]
            }
        ),
        Document(
            page_content="""# REST API 설계 원칙

## 1. 리소스 중심 URL
- GET /users - 사용자 목록 조회
- GET /users/{id} - 특정 사용자 조회
- POST /users - 사용자 생성
- PUT /users/{id} - 사용자 수정
- DELETE /users/{id} - 사용자 삭제

## 2. HTTP 상태 코드
- 200 OK: 성공
- 201 Created: 리소스 생성 성공
- 400 Bad Request: 잘못된 요청
- 404 Not Found: 리소스 없음
- 500 Internal Server Error: 서버 오류

## 3. 응답 형식
JSON 형식으로 일관된 응답 구조를 유지합니다.""",
            metadata={
                "title": "REST API 설계",
                "source": "notion",
                "tags": ["api", "rest", "design"]
            }
        )
    ]

    return sample_docs
