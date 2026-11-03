"""Unit tests for Google Gemini client and LLM factory."""

import json
from typing import Any
import httpx
import pytest
from pydantic import BaseModel, Field

from news_agent.llm.exceptions import GeminiAPIError, GeminiError
from news_agent.llm.factory import create_llm_client
from news_agent.llm.gemini_client import GeminiClient


class SampleSchema(BaseModel):
    title: str = Field(..., min_length=2)
    score: float = Field(..., ge=0.0)


class TestGeminiClient:
    def test_gemini_generate_text_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "gemini-2.5-flash:generateContent" in str(request.url)
            assert "key=fake-key" in str(request.url)
            body = json.loads(request.content.decode("utf-8"))
            assert body["contents"][0]["parts"][0]["text"] == "Hello Gemini"
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Hello human, I am Gemini."}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ]
                },
            )

        client = GeminiClient(
            api_key="fake-key",
            model_name="gemini-2.5-flash",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        output = client.generate("Hello Gemini")
        assert output == "Hello human, I am Gemini."
        assert client.provider_name == "gemini"
        assert client.model_name == "gemini-2.5-flash"

    def test_gemini_generate_structured_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            assert body["generationConfig"]["responseMimeType"] == "application/json"
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": json.dumps({"title": "Verified Claim", "score": 0.95})}
                                ],
                                "role": "model",
                            }
                        }
                    ]
                },
            )

        client = GeminiClient(
            api_key="fake-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        result = client.generate_structured("Extract claim", SampleSchema)
        assert isinstance(result, SampleSchema)
        assert result.title == "Verified Claim"
        assert result.score == 0.95

    def test_gemini_health_check_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "models/gemini-2.5-flash" in str(request.url)
            return httpx.Response(200, json={"name": "models/gemini-2.5-flash"})

        client = GeminiClient(
            api_key="fake-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert client.health_check() is True

    def test_gemini_missing_api_key_raises(self):
        client = GeminiClient(api_key="")
        with pytest.raises(GeminiAPIError) as exc_info:
            client.generate("Test prompt")
        assert exc_info.value.status_code == 401

    def test_gemini_api_error_handling(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"message": "API key not valid. Please pass a valid API key."}},
            )

        client = GeminiClient(
            api_key="bad-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(GeminiAPIError) as exc_info:
            client.generate("Hello")
        assert exc_info.value.status_code == 400
        assert "API key not valid" in str(exc_info.value)


class TestLLMFactory:
    def test_factory_creates_gemini_when_key_passed(self):
        client = create_llm_client(api_key="test-gemini-key")
        assert client.provider_name == "gemini"

    def test_factory_creates_ollama_when_no_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        client = create_llm_client(provider="ollama")
        assert client.provider_name == "ollama"

    def test_factory_explicit_provider_selection(self):
        client_gemini = create_llm_client(provider="gemini", api_key="test-key")
        assert client_gemini.provider_name == "gemini"

        client_ollama = create_llm_client(provider="ollama")
        assert client_ollama.provider_name == "ollama"

