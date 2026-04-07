#!/usr/bin/env bash
# ─────────────────────────────────────────────
# Ollama 설치 및 Qwen2.5 모델 다운로드 스크립트
# ─────────────────────────────────────────────
# 실행:
#   chmod +x scripts/setup_ollama.sh
#   ./scripts/setup_ollama.sh
#
# 지원 플랫폼: Linux / macOS
# Windows: https://ollama.com/download 에서 직접 설치
# ─────────────────────────────────────────────

set -e

echo "=========================================="
echo "  Ollama 설치 및 Qwen 모델 설정"
echo "=========================================="

# ── 1. Ollama 설치 (미설치 시) ──
if command -v ollama &>/dev/null; then
    echo "✅ Ollama 이미 설치됨: $(ollama --version)"
else
    echo "⏳ Ollama 설치 중..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama 설치 완료"
fi

# ── 2. Ollama 서버 시작 (백그라운드) ──
if ! pgrep -x "ollama" > /dev/null; then
    echo "⏳ Ollama 서버 시작 중..."
    ollama serve &
    sleep 3
    echo "✅ Ollama 서버 시작됨 (http://localhost:11434)"
else
    echo "✅ Ollama 서버 이미 실행 중"
fi

# ── 3. Qwen2.5 모델 다운로드 ──
MODELS=("qwen2.5:3b" "qwen2.5:7b")
DEFAULT_MODEL="qwen2.5:3b"

echo ""
echo "다운로드할 모델을 선택하세요:"
echo "  1) qwen2.5:3b  — 경량 (~2GB, CPU에서도 동작)"
echo "  2) qwen2.5:7b  — 중형 (~5GB, GPU 권장)"
echo "  3) 둘 다"
echo ""
read -rp "선택 [1/2/3, 기본=1]: " choice

case "${choice:-1}" in
    2)
        echo "⏳ qwen2.5:7b 다운로드 중..."
        ollama pull qwen2.5:7b
        ;;
    3)
        echo "⏳ qwen2.5:3b 다운로드 중..."
        ollama pull qwen2.5:3b
        echo "⏳ qwen2.5:7b 다운로드 중..."
        ollama pull qwen2.5:7b
        ;;
    *)
        echo "⏳ qwen2.5:3b 다운로드 중..."
        ollama pull qwen2.5:3b
        ;;
esac

echo ""
echo "=========================================="
echo "✅ 설정 완료!"
echo ""
echo "사용 방법:"
echo "  # .env 파일에서 LLM_PROVIDER 설정"
echo "  LLM_PROVIDER=ollama"
echo "  OLLAMA_BASE_URL=http://localhost:11434"
echo "  OLLAMA_MODEL=qwen2.5:3b"
echo ""
echo "  # RAG API 서버 실행"
echo "  uvicorn api.server:app --host 0.0.0.0 --port 8000"
echo ""
echo "  # 또는 Gradio UI 실행"
echo "  python app.py"
echo "=========================================="

# ── 4. 동작 테스트 ──
echo ""
echo "⏳ 모델 동작 테스트 중..."
ollama run qwen2.5:3b "안녕하세요. 한 문장으로 자기소개 해주세요." 2>/dev/null || true
echo "✅ 테스트 완료"
