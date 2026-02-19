"""
LLM 어댑터 모듈

API 기반 LLM과 로컬 LLM을 동일한 인터페이스로 사용할 수 있게 합니다.
"""

import os
from typing import Literal, Optional
from langchain_core.language_models import BaseChatModel, BaseLLM


LLMProvider = Literal["openai", "anthropic", "ollama", "huggingface"]


class LLMAdapter:
    """다양한 LLM 제공자를 위한 어댑터 클래스"""

    def __init__(
        self,
        provider: LLMProvider = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ):
        """
        Args:
            provider: LLM 제공자
            model_name: 모델명 (없으면 기본값 사용)
            temperature: 응답의 창의성 (0=일관적, 1=창의적)
            **kwargs: 추가 모델 파라미터
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
        self._llm: Optional[BaseChatModel | BaseLLM] = None

    @property
    def llm(self) -> BaseChatModel | BaseLLM:
        """LLM 인스턴스 반환 (지연 로딩)"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> BaseChatModel | BaseLLM:
        """LLM 인스턴스 생성"""
        if self.provider == "openai":
            return self._create_openai()
        elif self.provider == "anthropic":
            return self._create_anthropic()
        elif self.provider == "ollama":
            return self._create_ollama()
        elif self.provider == "huggingface":
            return self._create_huggingface()
        else:
            raise ValueError(f"지원하지 않는 provider: {self.provider}")

    def _create_openai(self) -> BaseChatModel:
        """OpenAI ChatGPT 모델 생성"""
        from langchain_openai import ChatOpenAI

        model = self.model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")

        return ChatOpenAI(
            model=model,
            temperature=self.temperature,
            **self.kwargs
        )

    def _create_anthropic(self) -> BaseChatModel:
        """Anthropic Claude 모델 생성"""
        from langchain_anthropic import ChatAnthropic

        model = self.model_name or "claude-3-haiku-20240307"

        return ChatAnthropic(
            model=model,
            temperature=self.temperature,
            **self.kwargs
        )

    def _create_ollama(self) -> BaseChatModel:
        """Ollama 로컬/원격 모델 생성 (Phase 3: base_url로 터널 지원)"""
        from langchain_community.chat_models import ChatOllama

        model = self.model_name or "qwen2.5:3b"
        base_url = self.kwargs.pop(
            "base_url",
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

        return ChatOllama(
            model=model,
            temperature=self.temperature,
            base_url=base_url,
            **self.kwargs
        )

    def _create_huggingface(self) -> BaseLLM:
        """HuggingFace Transformers 모델 생성 (Phase 4용)"""
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        model_name = self.model_name or "Qwen/Qwen2.5-3B-Instruct"

        # 4bit 양자화 설정 (VRAM 절약)
        quantization_config = None
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        except ImportError:
            print("bitsandbytes 미설치. 양자화 없이 로드합니다.")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=self.temperature,
            do_sample=self.temperature > 0
        )

        return HuggingFacePipeline(pipeline=pipe)

    def invoke(self, prompt: str) -> str:
        """프롬프트에 대한 응답 생성"""
        response = self.llm.invoke(prompt)

        # ChatModel과 LLM의 응답 형식이 다름
        if hasattr(response, "content"):
            return response.content
        return str(response)


def get_default_llm(provider: LLMProvider = "openai") -> BaseChatModel | BaseLLM:
    """기본 LLM 반환"""
    adapter = LLMAdapter(provider=provider)
    return adapter.llm
