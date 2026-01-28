"""
노션 문서 로더 모듈

노션 API를 사용하여 데이터베이스 및 페이지를 로드합니다.
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import NotionDBLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
