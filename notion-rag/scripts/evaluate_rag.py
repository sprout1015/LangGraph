"""
RAG 평가 스크립트

score_threshold 설정에 따른 검색 품질과 답변 충실도(Faithfulness)를 비교합니다.

실행:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --thresholds 0.0 0.3 0.5 0.7
    python scripts/evaluate_rag.py --top-k 6 --output data/my_report.json

결과:
    - 콘솔: 임계값별 비교 테이블
    - 파일: data/eval_report.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.chains.rag_chain import format_docs
from src.embeddings import EmbeddingManager
from src.llm import LLMAdapter
from src.vectorstore import PostgresVectorStore


# ─────────────────────────────────────────────
# LLM-as-Judge 프롬프트
# ─────────────────────────────────────────────
FAITHFULNESS_PROMPT = """당신은 RAG 시스템의 답변 품질을 평가하는 전문 평가자입니다.
아래 질문, 컨텍스트, 답변을 보고 답변이 컨텍스트에 얼마나 충실한지 평가하세요.

[평가 기준]
5점: 답변이 컨텍스트에 완전히 근거하며 정확함
4점: 대부분 컨텍스트에 근거하나 일부 추론 포함
3점: 컨텍스트를 부분적으로 활용하거나 일부 부정확
2점: 컨텍스트와 관련이 적거나 많은 추론/오류 포함
1점: 컨텍스트와 무관하거나 완전히 틀린 답변

[참고] 컨텍스트에 정보가 없을 때 "찾을 수 없습니다"라고 답하면 5점입니다.

질문: {question}

컨텍스트:
{context}

답변: {answer}

위 기준에 따라 1~5 사이의 정수 점수만 답하세요. 다른 설명 없이 숫자만 출력하세요.
점수:"""


RAG_ANSWER_PROMPT = """다음 컨텍스트를 기반으로 질문에 답변하세요.
컨텍스트에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
답변은 명확하고 간결하게 작성하세요.

컨텍스트:
{context}

질문: {question}

답변:"""


# ─────────────────────────────────────────────
# 핵심 평가 로직
# ─────────────────────────────────────────────

def generate_answer(llm, question: str, docs) -> str:
    """검색된 문서로 답변 생성"""
    if not docs:
        return "제공된 문서에서 해당 정보를 찾을 수 없습니다."

    context = format_docs(docs)
    prompt = ChatPromptTemplate.from_template(RAG_ANSWER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def judge_faithfulness(judge_llm, question: str, docs, answer: str) -> int:
    """LLM-as-Judge로 faithfulness 점수 반환 (1~5)"""
    if not docs:
        # 검색된 문서가 없을 때 "찾을 수 없습니다" 응답이면 5점
        if "찾을 수 없습니다" in answer:
            return 5
        return 1

    context = format_docs(docs)
    prompt = ChatPromptTemplate.from_template(FAITHFULNESS_PROMPT)
    chain = prompt | judge_llm | StrOutputParser()

    try:
        response = chain.invoke({
            "question": question,
            "context": context[:3000],  # 컨텍스트 길이 제한
            "answer": answer
        })
        score = int(response.strip())
        return max(1, min(5, score))  # 1~5 범위 보정
    except (ValueError, Exception):
        return 3  # 파싱 실패 시 중간값


def evaluate_question(
    question_item: dict,
    raw_results: list,
    llm,
    judge_llm,
    thresholds: list[float]
) -> dict:
    """단일 질문에 대해 각 임계값별 평가 수행"""
    question = question_item["question"]
    results_by_threshold = {}

    for threshold in thresholds:
        # 임계값 기준 필터링
        filtered = [
            (doc, score)
            for doc, score in raw_results
            if score >= threshold
        ]
        docs = [doc for doc, _ in filtered]
        scores = [score for _, score in filtered]

        # 답변 생성
        t0 = time.time()
        answer = generate_answer(llm, question, docs)
        latency = round(time.time() - t0, 2)

        # Faithfulness 채점
        faithfulness = judge_faithfulness(judge_llm, question, docs, answer)

        results_by_threshold[str(threshold)] = {
            "docs_retrieved": len(docs),
            "avg_similarity": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min_similarity": round(min(scores), 4) if scores else 0.0,
            "faithfulness": faithfulness,
            "latency_sec": latency,
            "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
        }

    return {
        "id": question_item["id"],
        "category": question_item.get("category", "general"),
        "question": question,
        "results": results_by_threshold,
    }


# ─────────────────────────────────────────────
# 집계 및 출력
# ─────────────────────────────────────────────

def aggregate_metrics(eval_results: list, thresholds: list[float]) -> dict:
    """임계값별 집계 메트릭 계산"""
    summary = {}
    for threshold in thresholds:
        key = str(threshold)
        valid = [
            r["results"][key]
            for r in eval_results
            if key in r["results"]
        ]
        if not valid:
            continue

        summary[key] = {
            "avg_docs_retrieved": round(
                sum(v["docs_retrieved"] for v in valid) / len(valid), 2
            ),
            "avg_similarity": round(
                sum(v["avg_similarity"] for v in valid) / len(valid), 4
            ),
            "avg_faithfulness": round(
                sum(v["faithfulness"] for v in valid) / len(valid), 2
            ),
            "avg_latency_sec": round(
                sum(v["latency_sec"] for v in valid) / len(valid), 2
            ),
            "zero_retrieval_count": sum(
                1 for v in valid if v["docs_retrieved"] == 0
            ),
            "total_questions": len(valid),
        }
    return summary


def print_comparison_table(summary: dict, thresholds: list[float]):
    """임계값별 비교 테이블 출력"""
    print("\n" + "=" * 72)
    print(" RAG 평가 결과 — score_threshold 비교")
    print("=" * 72)

    header = f"{'항목':<26}"
    for t in thresholds:
        header += f"{'threshold=' + str(t):>14}"
    print(header)
    print("-" * 72)

    metrics_labels = {
        "avg_docs_retrieved": "평균 검색 문서 수",
        "avg_similarity": "평균 유사도 점수",
        "avg_faithfulness": "평균 Faithfulness (1~5)",
        "avg_latency_sec": "평균 응답 시간 (초)",
        "zero_retrieval_count": "검색 결과 0건 질문 수",
    }

    for key, label in metrics_labels.items():
        row = f"{label:<26}"
        for t in thresholds:
            val = summary.get(str(t), {}).get(key, "-")
            row += f"{str(val):>14}"
        print(row)

    print("=" * 72)


def print_question_detail(eval_results: list, thresholds: list[float]):
    """질문별 상세 결과 출력"""
    print("\n[질문별 상세 결과]")
    for r in eval_results:
        print(f"\n  [{r['id']}] ({r['category']}) {r['question'][:60]}...")
        for t in thresholds:
            key = str(t)
            v = r["results"].get(key, {})
            faith = v.get("faithfulness", "-")
            docs = v.get("docs_retrieved", 0)
            sim = v.get("avg_similarity", 0.0)
            print(f"    threshold={t}: 문서 {docs}개 | 유사도 {sim:.3f} | Faithfulness {faith}/5")


def save_report(
    eval_results: list,
    summary: dict,
    thresholds: list[float],
    output_path: str
):
    """평가 결과를 JSON 파일로 저장"""
    report = {
        "metadata": {
            "evaluated_at": datetime.now().isoformat(),
            "thresholds": thresholds,
            "total_questions": len(eval_results),
        },
        "summary": summary,
        "details": eval_results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 평가 리포트 저장: {output_path}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 평가 스크립트")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.5],
        help="비교할 score_threshold 값 목록 (기본: 0.0 0.3 0.5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="벡터 검색 최대 문서 수 (기본: 10, 이후 임계값 필터 적용)",
    )
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.json",
        help="평가 데이터셋 경로",
    )
    parser.add_argument(
        "--output",
        default="data/eval_report.json",
        help="결과 저장 경로",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Faithfulness 채점 건너뛰기 (LLM 비용 절감)",
    )
    args = parser.parse_args()

    # ── 데이터셋 로드 ──
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ 데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        print("   data/eval_dataset.json의 question 항목을 실제 내용으로 수정 후 실행하세요.")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    print(f"📋 평가 데이터셋: {len(questions)}개 질문 로드")
    print(f"🎯 비교 임계값: {args.thresholds}")

    # ── 모델 초기화 ──
    print("\n⏳ 모델 초기화 중...")

    embedding_provider = "openai" if os.getenv("OPENAI_API_KEY") else "huggingface"
    embeddings = EmbeddingManager(provider=embedding_provider).embeddings
    print(f"   임베딩: {embedding_provider}")

    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    llm = LLMAdapter(provider=llm_provider, temperature=0).llm
    print(f"   LLM: {llm_provider}")

    judge_llm = llm  # 동일 LLM을 judge로 사용 (비용 절감)

    vs = PostgresVectorStore(embeddings, collection_name="notion_docs")
    print("   벡터 스토어: PostgreSQL + pgvector")

    # ── 평가 실행 ──
    print(f"\n🔍 평가 시작...\n")
    eval_results = []

    for i, q_item in enumerate(questions, 1):
        qid = q_item["id"]
        question = q_item["question"]
        print(f"  [{i}/{len(questions)}] {qid}: {question[:50]}...")

        try:
            # 벡터 검색 (1회만 실행, 임계값은 Python에서 필터링)
            raw_results = vs.similarity_search_with_score(question, k=args.top_k)

            result = evaluate_question(
                q_item,
                raw_results,
                llm,
                judge_llm if not args.skip_judge else None,
                args.thresholds,
            )
            eval_results.append(result)

        except Exception as e:
            print(f"    ⚠️  오류 발생: {e}")
            eval_results.append({
                "id": qid,
                "category": q_item.get("category", "general"),
                "question": question,
                "results": {},
                "error": str(e),
            })

    # ── 집계 및 출력 ──
    summary = aggregate_metrics(eval_results, args.thresholds)
    print_comparison_table(summary, args.thresholds)
    print_question_detail(eval_results, args.thresholds)

    # 최적 임계값 추천
    best_threshold = max(
        summary.keys(),
        key=lambda t: summary[t]["avg_faithfulness"],
        default=None
    )
    if best_threshold:
        print(f"\n💡 권장 score_threshold: {best_threshold}")
        print(f"   → 평균 Faithfulness {summary[best_threshold]['avg_faithfulness']}/5.0")

    # ── 리포트 저장 ──
    save_report(eval_results, summary, args.thresholds, args.output)


if __name__ == "__main__":
    main()
