# Notion RAG 시스템

Notion 문서를 활용한 RAG (Retrieval-Augmented Generation) 시스템 학습 프로젝트

## 1. 개요

### 프로젝트 목표
Notion에 저장된 문서를 기반으로 질의응답이 가능한 RAG 시스템 구축

### 개발 단계
| 단계 | 설명 | 상태 |
|------|------|------|
| 1 | Notion API 연동 및 문서 로드 | 완료 |
| 2 | PostgreSQL + pgvector 벡터 저장소 구축 | 완료 |
| 3 | Anthropic Claude API로 RAG 테스트 | 완료 |
| 4 | Gradio 챗봇 UI 구현 | 완료 |
| 5 | FastAPI RAG API 서버 | 완료 |
| 6 | Discord 봇 연동 | 완료 |
| 7 | 로컬 Qwen LLM으로 전환 | 예정 |
| 8 | LoRA 파인튜닝 (선택) | 예정 |

### 현재 단계
**6단계 완료**: Discord 봇 연동
- FastAPI RAG API 서버 (`/api/query`, `/api/query/stream`, `/api/health`)
- Discord 봇 (멘션/지정채널 응답, 2000자 분할)
- 유사도 점수 기반 문서 필터링
- 소스에 카테고리·Notion 원문 링크 포함

---

## 2. 시스템 아키텍처

### 전체 구조
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Notion    │───▶│   Loader    │───▶│  Splitter   │
│   페이지    │    │   (API)     │    │  (청크)     │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Claude    │◀───│  RAG Chain  │◀───│  Vector DB  │
│   (LLM)     │    │  (질의)     │    │ PostgreSQL  │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                    ┌─────┴─────┐
                    ▼           ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Gradio   │ │ FastAPI  │ │ Discord  │
              │ :7860    │ │ :8000   │ │ Bot      │
              └──────────┘ └──────────┘ └──────────┘
```

### 벡터 DB 옵션

| 옵션 | 장점 | 단점 |
|------|------|------|
| **ChromaDB** | 간단한 설정, 파일 기반 | 확장성 제한 |
| **PostgreSQL + pgvector** | 운영 환경 적합, SQL 쿼리 지원 | DB 설정 필요 |

### API 역할 분리

> **중요**: Anthropic은 임베딩 API를 제공하지 않습니다. 각 단계별로 다른 API를 사용합니다.

| 단계 | 목적 | 사용 API | 비고 |
|------|------|----------|------|
| 임베딩 | 문서 → 벡터 | OpenAI / HuggingFace | Anthropic은 임베딩 API 없음 |
| LLM 생성 | 답변 생성 | **Anthropic Claude** | 추후 Qwen으로 전환 예정 |

---

## 3. RAG 워크플로우

### 3.1 인덱싱 단계 (1회 또는 주기적 실행)

```
Notion 문서 → 로드 → 청크 분할 → 임베딩 → 벡터 DB 저장
```

1. **로드**: Notion API로 문서 가져오기
2. **청크 분할**: 500자 단위로 분할 (100자 오버랩)
3. **임베딩**: 청크를 벡터로 변환 (OpenAI 또는 HuggingFace)
4. **저장**: ChromaDB 또는 PostgreSQL (pgvector)에 벡터 저장

> 인덱싱 완료 후에는 다음 업데이트 전까지 Notion API를 사용하지 않습니다.

### 3.2 질의 단계 (질문마다 실행)

```
질문 → 임베딩 → 벡터 DB 검색 → Top-K 문서 추출 → LLM → 답변
```

1. **질문 임베딩**: 사용자 질문을 벡터로 변환
2. **검색**: 벡터 DB에서 유사 문서 검색 (Top-K)
3. **생성**: 검색된 문서 + 질문을 Claude에 전달
4. **답변**: LLM이 컨텍스트 기반 응답 생성

### 3.3 문서 업데이트

- **수동**: 인덱싱 스크립트 재실행
- **자동**: 스케줄러 구현 (선택사항)

---

## 4. 사전 요구사항

### 4.1 Notion 통합 설정

1. https://www.notion.so/my-integrations 접속
2. **"새 API 통합"** 클릭
3. 통합 이름 입력 후 생성
4. **Internal Integration Token** 복사

### 4.2 페이지 권한 부여

1. 연결할 Notion 페이지/데이터베이스 열기
2. 우측 상단 **"..."** 클릭
3. **"연결"** → 통합 추가

### 4.3 데이터베이스 ID 확인

Notion 페이지 URL에서 확인:
```
https://www.notion.so/{workspace}/{database_id}?v=...
```

32자 ID를 UUID 형식으로 변환:
```
abc123def456... → abc123de-f456-7890-abcd-ef1234567890
```

### 4.4 API 키 발급

| 서비스 | URL | 필수 여부 |
|--------|-----|-----------|
| Anthropic | https://console.anthropic.com | **필수** (LLM) |
| OpenAI | https://platform.openai.com | 선택 (임베딩) |
| Notion | https://www.notion.so/my-integrations | **필수** |

> OpenAI API 키가 없으면 HuggingFace의 무료 `sentence-transformers/all-MiniLM-L6-v2` 모델을 임베딩에 사용합니다.

---

## 5. 환경 설정

### .env 파일 복사 및 설정
```bash
cp .env.example .env
```

### .env 필수 변수
```bash
# 필수: LLM (답변 생성)
ANTHROPIC_API_KEY=sk-ant-...

# 필수: Notion API
NOTION_API_KEY=ntn_...

# 선택: 임베딩 (미설정 시 HuggingFace 사용)
OPENAI_API_KEY=sk-...

# 벡터 DB (택일)
VECTOR_DB_TYPE=postgres  # 또는 "chroma"

# PostgreSQL + pgvector 사용 시
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
POSTGRES_DB=notion_rag

# Discord 봇 (선택)
DISCORD_BOT_TOKEN=your-token
DISCORD_CHANNEL_IDS=채널ID1,채널ID2
RAG_API_URL=http://localhost:8000
```

---

## 6. 빠른 시작

### 의존성 설치
```bash
cd notion-rag
pip install -r requirements.txt
```

### 벡터 스토어 초기화 (Notion 문서 임베딩)
```bash
python scripts/reload_vectorstore.py
```

### 챗봇 실행 (Gradio UI)
```bash
python app.py
```
브라우저에서 `http://localhost:7860` 접속

### FastAPI API 서버
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Discord 봇
```bash
python -m discord_bot.bot
```

### 노트북 실행
```bash
jupyter notebook
```

### 권장 노트북 순서
1. `01_notion_loader.ipynb` - Notion API 연결 테스트
2. `02_notion_rag.ipynb` - 전체 RAG 파이프라인
3. `03_local_llm.ipynb` - 로컬 LLM 연동 (5단계)
4. `04_lora_tuning.ipynb` - 파인튜닝 (6단계)

---

## 7. 프로젝트 구조

```
notion-rag/
├── api/
│   ├── server.py             # FastAPI RAG API 서버
│   └── schemas.py            # API 요청/응답 스키마
├── discord_bot/
│   ├── bot.py                # Discord 봇 진입점
│   ├── rag_client.py         # RAG API 비동기 클라이언트
│   └── formatter.py          # Discord 메시지 포맷터
├── src/
│   ├── loaders/          # Notion 문서 로더 (NotionRecursiveLoader 포함)
│   ├── embeddings/       # 임베딩 모델 관리 (OpenAI / HuggingFace)
│   ├── vectorstore/      # 벡터 DB (PostgreSQL + pgvector)
│   ├── llm/              # LLM 어댑터 (Claude, Qwen)
│   └── chains/           # RAG 체인 구성
├── infra/
│   └── scripts/
│       └── user_data.sh      # EC2 초기화 스크립트
├── scripts/
│   └── reload_vectorstore.py  # 벡터 스토어 재구축 스크립트
├── notebooks/
│   ├── 01_notion_loader.ipynb
│   ├── 02_notion_rag.ipynb
│   ├── 03_local_llm.ipynb
│   └── 04_lora_tuning.ipynb
├── app.py                # Gradio 챗봇 UI
├── .env.example
├── requirements.txt
└── README.md
```

---

## 8. 문제 해결

### Notion API 오류
- Integration Token이 올바른지 확인
- 페이지 권한 확인 (Integration이 페이지에 추가되었는지)
- 데이터베이스 ID 형식 확인 (대시 포함 UUID)

### 임베딩 오류
- OpenAI 할당량 초과 시 `OPENAI_API_KEY`를 삭제하여 HuggingFace 사용
- HuggingFace 모델은 첫 사용 시 다운로드됨 (~100MB)

### LLM 오류
- Anthropic API 키 유효성 확인
- 모델명 확인: `claude-sonnet-4-20250514`

### PostgreSQL + pgvector 오류
- pgvector 확장 설치 확인: `CREATE EXTENSION IF NOT EXISTS vector;`
- 연결 확인: `psql -h localhost -p 5433 -U postgres -d notion_rag`
- 데이터베이스 존재 확인: `CREATE DATABASE notion_rag;`

---

## 라이선스

이 프로젝트는 학습 목적으로 만들어졌습니다.
