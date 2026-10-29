"""Unit tests for the Ollama client wrapper, error handling, and structured validation."""

import json
from typing import Any
import httpx
import pytest
from pydantic import BaseModel, Field

from news_agent.llm.client import OllamaClient
from news_agent.llm.exceptions import (
    OllamaAPIError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseValidationError,
    OllamaTimeoutError,
)


class DummyStructuredResponse(BaseModel):
    claim_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_verifiable: bool


def make_mock_client(handler) -> OllamaClient:
    """Helper to instantiate OllamaClient with an httpx.MockTransport."""
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://mock-ollama:11434")
    return OllamaClient(
        base_url="http://mock-ollama:11434",
        model_name="llama3.2:3b",
        timeout_seconds=5.0,
        http_client=client,
    )


class TestOllamaClientHealthAndDiscovery:
    def test_health_check_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/version"
            return httpx.Response(200, json={"version": "0.4.1"})

        client = make_mock_client(handler)
        assert client.health_check() is True

    def test_health_check_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        client = make_mock_client(handler)
        assert client.health_check() is False

    def test_list_local_models(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/tags"
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "llama3.2:3b", "size": 2000000000},
                        {"name": "nomic-embed-text:latest", "size": 300000000},
                    ]
                },
            )

        client = make_mock_client(handler)
        models = client.list_local_models()
        assert models == ["llama3.2:3b", "nomic-embed-text:latest"]

    def test_verify_model_installed_found(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})

        client = make_mock_client(handler)
        assert client.verify_model_installed() is True

    def test_verify_model_installed_not_found_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "mistral:7b"}]})

        client = make_mock_client(handler)
        with pytest.raises(OllamaModelNotFoundError) as exc_info:
            client.verify_model_installed()

        # Ensures explicit guidance and no auto download
        assert "Automatic downloads are strictly disabled" in str(exc_info.value)
        assert "ollama pull llama3.2:3b" in str(exc_info.value)


class TestOllamaClientInference:
    def test_generate_text(self):
        recorded_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/generate"
            body = json.loads(request.content.decode("utf-8"))
            recorded_requests.append(body)
            assert body["model"] == "llama3.2:3b"
            assert body["prompt"] == "Summarize the news."
            assert body["stream"] is False
            return httpx.Response(200, json={"response": "Springfield council met today."})

        client = make_mock_client(handler)
        result = client.generate("Summarize the news.", system="You are an analyst.")
        assert result == "Springfield council met today."
        assert recorded_requests[0]["system"] == "You are an analyst."

    def test_chat(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/chat"
            body = json.loads(request.content.decode("utf-8"))
            assert len(body["messages"]) == 1
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "Chat reply from llama."}},
            )

        client = make_mock_client(handler)
        messages = [{"role": "user", "content": "Hello"}]
        reply = client.chat(messages)
        assert reply == "Chat reply from llama."


class TestOllamaStructuredValidation:
    def test_generate_structured_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/generate"
            body = json.loads(request.content.decode("utf-8"))
            assert body["format"] == "json"
            model_json = json.dumps(
                {
                    "claim_summary": "7-2 solar vote",
                    "confidence": 0.95,
                    "is_verifiable": True,
                }
            )
            return httpx.Response(200, json={"response": model_json})

        client = make_mock_client(handler)
        validated = client.generate_structured("Analyze claim", DummyStructuredResponse)
        assert isinstance(validated, DummyStructuredResponse)
        assert validated.claim_summary == "7-2 solar vote"
        assert validated.confidence == 0.95
        assert validated.is_verifiable is True

    def test_generate_structured_with_markdown_fencing(self):
        def handler(request: httpx.Request) -> httpx.Response:
            fenced_json = "```json\n{\"claim_summary\": \"Vote passed\", \"confidence\": 0.8, \"is_verifiable\": true}\n```"
            return httpx.Response(200, json={"response": fenced_json})

        client = make_mock_client(handler)
        validated = client.generate_structured("Analyze claim", DummyStructuredResponse)
        assert validated.claim_summary == "Vote passed"

    def test_generate_structured_invalid_json_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "Not valid JSON at all"})

        client = make_mock_client(handler)
        with pytest.raises(OllamaResponseValidationError) as exc_info:
            client.generate_structured("Analyze claim", DummyStructuredResponse)
        assert "not valid JSON" in str(exc_info.value)
        assert exc_info.value.raw_output == "Not valid JSON at all"

    def test_generate_structured_schema_violation_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            # Missing required is_verifiable and confidence > 1.0
            invalid_payload = json.dumps({"claim_summary": "Test", "confidence": 1.5})
            return httpx.Response(200, json={"response": invalid_payload})

        client = make_mock_client(handler)
        with pytest.raises(OllamaResponseValidationError) as exc_info:
            client.generate_structured("Analyze claim", DummyStructuredResponse)
        assert "schema validation" in str(exc_info.value).lower()


class TestOllamaErrorHandling:
    def test_connection_error_raises_ollama_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused by host", request=request)

        client = make_mock_client(handler)
        with pytest.raises(OllamaConnectionError):
            client.generate("Test prompt")

    def test_timeout_error_raises_ollama_timeout_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Request timed out", request=request)

        client = make_mock_client(handler)
        with pytest.raises(OllamaTimeoutError):
            client.generate("Test prompt")

    def test_model_404_raises_ollama_model_not_found(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model 'llama3.2:3b' not found"})

        client = make_mock_client(handler)
        with pytest.raises(OllamaModelNotFoundError):
            client.generate("Test prompt")

    def test_api_500_raises_ollama_api_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal server error in Ollama engine")

        client = make_mock_client(handler)
        with pytest.raises(OllamaAPIError) as exc_info:
            client.generate("Test prompt")
        assert exc_info.value.status_code == 500
