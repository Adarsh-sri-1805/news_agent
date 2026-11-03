"""Unit tests for atomic claim extraction and LLM failure resilience."""

from datetime import datetime, timezone
import json
import httpx
import pytest

from news_agent.llm.client import OllamaClient
from news_agent.models.article import NormalizedArticle
from news_agent.verification.extractor import ClaimExtractor


def make_sample_article(
    content: str = "The city council voted 7-2 to approve solar panels on municipal buildings.",
) -> NormalizedArticle:
    return NormalizedArticle(
        id="art-council-1",
        title="Council Approves Solar Arrays",
        content=content,
        source_url="https://localnews.com/council-solar",  # type: ignore[arg-type]
        publisher="Local Gazette",
        published_at=datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc),
    )


class TestClaimExtractor:
    def test_extract_multiple_atomic_claims_successful(self):
        llm_json_response = {
            "claims": [
                {
                    "claim_text": "City council voted 7-2 to approve solar panels on municipal rooftops.",
                    "context": "Paragraph 1 describing the Tuesday morning session.",
                    "confidence": 0.95,
                },
                {
                    "claim_text": "The contract value for the solar installation is $2.5 million.",
                    "context": "Paragraph 2 detailing contract figures.",
                    "confidence": 0.90,
                },
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": json.dumps(llm_json_response)})

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        extractor = ClaimExtractor(llm_client=client)

        article = make_sample_article()
        claims = extractor.extract_claims(article)

        assert len(claims) == 2
        assert claims[0].article_id == "art-council-1"
        assert claims[0].claim_text == "City council voted 7-2 to approve solar panels on municipal rooftops."
        assert claims[0].extraction_confidence == 0.95
        assert claims[0].metadata["source_url"] == "https://localnews.com/council-solar"
        assert claims[0].metadata["publisher"] == "Local Gazette"

        assert claims[1].article_id == "art-council-1"
        assert claims[1].claim_text == "The contract value for the solar installation is $2.5 million."

    def test_inadequate_content_skips_llm_and_returns_empty(self):
        # Very short content below min_content_length (30 chars)
        article = make_sample_article(content="Too brief.")

        # Ensure no HTTP request is made
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("LLM should not be called for inadequate content!")

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        extractor = ClaimExtractor(llm_client=client, min_content_length=30)

        claims = extractor.extract_claims(article)
        assert claims == []

    def test_deduplicate_claims_within_same_article(self):
        llm_json_response = {
            "claims": [
                {
                    "claim_text": "City council voted 7-2 to approve the solar contract.",
                    "context": "P1",
                    "confidence": 0.95,
                },
                # Near duplicate of the first claim
                {
                    "claim_text": "City council voted 7-2 to approve the solar contract!",
                    "context": "P2",
                    "confidence": 0.90,
                },
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": json.dumps(llm_json_response)})

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        extractor = ClaimExtractor(llm_client=client)

        claims = extractor.extract_claims(make_sample_article())
        assert len(claims) == 1

    def test_malformed_json_from_llm_handled_gracefully(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "Not valid JSON at all!"})

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        extractor = ClaimExtractor(llm_client=client)

        claims = extractor.extract_claims(make_sample_article())
        assert claims == []

    def test_ollama_timeout_and_connection_error_handled_gracefully(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Ollama timed out", request=request)

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        extractor = ClaimExtractor(llm_client=client)

        claims = extractor.extract_claims(make_sample_article())
        assert claims == []
