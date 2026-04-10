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
| 7 | 로컬 Qwen LLM으로 전환 (Ollama) | 완료 |
| 8 | LoRA 파인튜닝 (GPU 환경) | 완료 |
| + | RAG 평가 지표 (score_threshold 비교) | 완료 |
| + | Prometheus + Grafana 모니터링 | 완료 |

---

## 2. 시스템 아키텍처

### 전체 데이터 흐름

```
┌──────────────────────────────────────────────────────────────┐
│                        인덱싱 파이프라인                        │
│                                                              │
│  Notion API                                                  │
│  (Database + 하위 페이지 재귀)                                 │
│       │                                                      │
│       ▼                                                      │
│  NotionRecursiveLoader                                       │
│  (블록 단위 파싱 → plain_text)                                 │
│       │                                                      │
│       ▼                                                      │
│  SemanticChunker                                             │
│  (의미 단위 분할 — percentile 95)                              │
│       │                                                      │
│       ▼                                                      │
│  OpenAI Embeddings                    PostgreSQL + pgvector  │
│  (text-embedding-3-small)  ────────▶  (notion_docs 컬렉션)   │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                        질의 파이프라인                          │
│                                                              │
│  사용자 질문                                                   │
│       │                                                      │
│       ▼                                                      │
│  질문 임베딩 (OpenAI)                                          │
│       │                                                      │
│       ▼                                                      │
│  pgvector 유사도 검색                                          │
│  (score_threshold 필터링, Top-K)                              │
│       │                                                      │
│       ▼                                                      │
│  RAG Chain (LCEL)                                            │
│  컨텍스트 + 질문 → LLM                                         │
│       │                                                      │
│       ├─── Anthropic Claude (API)                            │
│       └─── Qwen2.5 via Ollama (로컬)                         │
│                                                              │
│                    답변                                       │
│                     │                                        │
│       ┌─────────────┼─────────────┐                         │
│       ▼             ▼             ▼                          │
│  Gradio UI     FastAPI       Discord Bot                     │
│  (:7860)       (:8000)       (WebSocket)                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                          모니터링                              │
│                                                              │
│  FastAPI /metrics ──▶ Prometheus ──▶ Grafana (:3000)        │
│  (요청수, 응답시간, 검색 문서수, 오류율 등 9개 패널)              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### API 역할 분리

> **중요**: Anthropic은 임베딩 API를 제공하지 않습니다. 임베딩과 LLM에 각각 다른 API를 사용합니다.

| 단계 | 목적 | 사용 API |
|------|------|----------|
| 임베딩 | 문서/질문 → 벡터 | OpenAI `text-embedding-3-small` (또는 HuggingFace) |
| LLM 생성 | 답변 생성 | Anthropic Claude 또는 Qwen2.5 via Ollama |

---

## 3. 청크 전략

### 왜 SemanticChunker인가

고정 크기 분할(`RecursiveCharacterTextSplitter`)은 문장 중간에서 잘려 맥락이 끊기는 문제가 있습니다.  
`SemanticChunker`는 인접 문장 간 임베딩 유사도를 계산하여 **의미가 급격히 달라지는 지점**에서만 분할하므로,  
하나의 청크 내 문장들이 같은 주제를 다룹니다.

### 설정값

```python
SemanticChunker(
    embeddings=OpenAIEmbeddings(),        # 유사도 계산용
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95        # 상위 5% 유사도 급락 지점에서만 분할
)
```

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| `breakpoint_threshold_type` | `"percentile"` | 전체 문장 쌍 유사도 분포를 기준으로 임계값 결정 |
| `breakpoint_threshold_amount` | `95` | 유사도 하위 5%에 해당하는 급락 지점에서만 분할 (청크 수 최소화) |

### 문서 로딩 전략

`NotionRecursiveLoader`는 데이터베이스의 최상위 페이지뿐 아니라 **하위 페이지를 재귀적으로 탐색**합니다.  
각 블록(`heading`, `paragraph`, `bulleted_list`, `code`, `table_row` 등)을 plain text로 파싱한 뒤 SemanticChunker에 전달합니다.

```
Notion Database
├── 페이지 A                ← NotionDBLoader로 로드
│   ├── 하위 페이지 A-1    ← 재귀 탐색으로 추가 로드
│   └── 하위 페이지 A-2
└── 페이지 B
    └── 하위 페이지 B-1
```

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
| Anthropic | https://console.anthropic.com | LLM 사용 시 필수 |
| OpenAI | https://platform.openai.com | 임베딩 필수 (없으면 HuggingFace 사용) |
| Notion | https://www.notion.so/my-integrations | **필수** |

---

## 5. 환경 설정

### .env 파일 복사 및 설정
```bash
cp .env.example .env
```

### .env 필수 변수
```bash
# 필수: Notion API
NOTION_API_KEY=ntn_...
NOTION_DATABASE_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# LLM 선택 (anthropic | ollama | openai)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# 로컬 LLM 사용 시 (LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
# Ollama 실패 시 Claude로 자동 전환 (선택)
FALLBACK_LLM_PROVIDER=anthropic

# 임베딩 (미설정 시 HuggingFace 무료 모델 사용)
OPENAI_API_KEY=sk-...

# PostgreSQL + pgvector
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

### 모니터링 (Prometheus + Grafana)
```bash
cd infra/monitoring
docker compose -f docker-compose.monitoring.yml up -d
```
Grafana: `http://localhost:3000` (admin / admin)

### 로컬 LLM 전환 (Ollama)
```bash
# 1. Ollama 설치 후 모델 다운로드
ollama pull qwen2.5:3b

# 2. .env에서 LLM_PROVIDER 변경
LLM_PROVIDER=ollama

# 3. 서버 재시작
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## 7. 스크립트

| 스크립트 | 설명 |
|----------|------|
| `scripts/reload_vectorstore.py` | Notion 문서를 다시 로드하여 pgvector에 재임베딩. 문서 업데이트 시 실행 |
| `scripts/evaluate_rag.py` | score_threshold 별 검색 품질·Faithfulness 비교 리포트 생성 |
| `scripts/setup_ollama.sh` | Ollama 설치 및 Qwen 모델 다운로드 자동화 (Linux/Mac) |

```bash
# RAG 평가 실행 예시
python scripts/evaluate_rag.py
python scripts/evaluate_rag.py --thresholds 0.0 0.3 0.5 0.7
python scripts/evaluate_rag.py --top-k 6 --output data/my_report.json
```

---

## 8. 프로젝트 구조

```
notion-rag/
├── api/
│   ├── server.py             # FastAPI RAG API 서버 (/api/query, /api/query/stream, /metrics)
│   ├── schemas.py            # API 요청/응답 스키마
│   └── metrics.py            # Prometheus 메트릭 정의
├── discord_bot/
│   ├── bot.py                # Discord 봇 진입점 (멘션 / 지정 채널 응답)
│   ├── rag_client.py         # FastAPI 비동기 클라이언트
│   └── formatter.py          # 2000자 분할 메시지 포맷터
├── src/
│   ├── loaders/
│   │   └── notion_loader.py  # NotionRecursiveLoader (하위 페이지 재귀 + SemanticChunker)
│   ├── embeddings/
│   │   └── embedding_manager.py  # OpenAI / HuggingFace 임베딩 선택
│   ├── vectorstore/
│   │   ├── postgres_store.py     # PostgreSQL + pgvector 벡터 스토어
│   │   └── chroma_store.py       # ChromaDB (로컬 테스트용)
│   ├── llm/
│   │   └── model_adapter.py      # LLMAdapter (anthropic / ollama / openai 동적 선택)
│   └── chains/
│       └── rag_chain.py          # RAGChain, ConversationalRAGChain (LCEL)
├── infra/
│   └── monitoring/
│       ├── docker-compose.monitoring.yml
│       ├── prometheus.yml
│       └── grafana/              # 대시보드 자동 프로비저닝 (9개 패널)
├── scripts/
│   ├── reload_vectorstore.py     # 벡터 스토어 재구축
│   ├── evaluate_rag.py           # RAG 평가 (score_threshold 비교)
│   └── setup_ollama.sh           # Ollama 환경 자동 설치
├── notebooks/                    # 학습용 노트북 (참고 자료)
│   ├── 01_langchain_basics.ipynb
│   ├── 02_notion_rag.ipynb
│   ├── 03_local_llm.ipynb        # Claude vs Qwen 응답 품질 비교
│   └── 04_lora_tuning.ipynb      # QLoRA 파인튜닝 파이프라인 (GPU 필수)
├── data/
│   └── eval_dataset.json         # RAG 평가 데이터셋
├── app.py                        # Gradio 챗봇 UI
├── .env.example
└── requirements.txt
```

---

## 9. 문제 해결

### Notion API 오류
- Integration Token이 올바른지 확인
- 페이지 권한 확인 (Integration이 페이지에 추가되었는지)
- 데이터베이스 ID 형식 확인 (대시 포함 UUID)

### 임베딩 오류
- OpenAI 할당량 초과 시 `OPENAI_API_KEY`를 삭제하여 HuggingFace 사용
- HuggingFace 모델은 첫 사용 시 자동 다운로드됨 (~100MB)

### LLM 오류
- Anthropic: API 키 유효성 및 모델명 확인 (`claude-sonnet-4-20250514`)
- Ollama: `ollama serve` 실행 후 `ollama list`로 모델 설치 확인

### PostgreSQL + pgvector 오류
- pgvector 확장 설치 확인: `CREATE EXTENSION IF NOT EXISTS vector;`
- 연결 확인: `psql -h localhost -p 5433 -U postgres -d notion_rag`

---

## 라이선스

이 프로젝트는 학습 목적으로 만들어졌습니다.
