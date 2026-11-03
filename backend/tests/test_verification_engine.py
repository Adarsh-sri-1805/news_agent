"""Unit tests for end-to-end VerificationEngine workflow and provenance tracing."""

from datetime import datetime, timezone
import json
import httpx
import pytest

from news_agent.llm.client import OllamaClient
from news_agent.models.article import NormalizedArticle
from news_agent.models.verification import VerificationStatus
from news_agent.verification.engine import VerificationEngine


def make_article(
    id_val: str,
    title: str,
    content: str,
    publisher: str,
    url: str,
) -> NormalizedArticle:
    return NormalizedArticle(
        id=id_val,
        title=title,
        content=content,
        publisher=publisher,
        source_url=url,  # type: ignore[arg-type]
        published_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
    )


class TestVerificationEngineEndToEnd:
    def test_end_to_end_verification_pipeline(self):
        art1 = make_article(
            id_val="art-origin",
            title="Springfield Council Approves Solar Rooftops",
            content="The city council voted 7-2 on Tuesday morning to approve solar panel installations across municipal buildings.",
            publisher="Springfield Gazette",
            url="https://gazette.com/solar-approved",
        )
        art2 = make_article(
            id_val="art-corrob",
            title="Municipal Records: Council Passes Solar Accord",
            content="Official city hall voting records confirm that council resolution #44 passed by a 7 to 2 vote for municipal solar panels.",
            publisher="Civic Records Observer",
            url="https://observer.org/records/solar-44",
        )

        # Mock Ollama behavior:
        # First call: extract claims from art1
        # Second call: extract claims from art2 (no claims needed or 1 claim)
        # Subsequent calls: verify claims
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            prompt = body.get("prompt", "")

            if "Extract up to" in prompt:
                if "art-origin" in prompt or "Springfield Gazette" in prompt:
                    payload = {
                        "claims": [
                            {
                                "claim_text": "City council voted 7-2 to approve solar panel installations.",
                                "context": "Tuesday morning session.",
                                "confidence": 0.95,
                            }
                        ]
                    }
                else:
                    payload = {"claims": []}
                return httpx.Response(200, json={"response": json.dumps(payload)})

            elif "Evaluate whether the evidence" in prompt:
                # Verifier call
                payload = {
                    "status": "supported",
                    "confidence_score": 0.94,
                    "reasoning": "Civic Records Observer confirms resolution #44 passed by 7-2 vote.",
                    "cited_passage_ids": ["art-corrob_p0"],
                }
                return httpx.Response(200, json={"response": json.dumps(payload)})

            return httpx.Response(500, text="Unrecognized mock prompt")

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        engine = VerificationEngine(llm_client=client)

        summary = engine.verify_articles([art1, art2], top_k_evidence=2, exclude_own_article=True)

        assert summary.total_claims == 1
        assert summary.supported_count == 1
        assert summary.contradicted_count == 0
        assert summary.insufficient_evidence_count == 0

        # Provenance verification
        claim = summary.claims[0]
        assert claim.article_id == "art-origin"
        assert claim.metadata["source_url"] == "https://gazette.com/solar-approved"

        result = summary.results[0]
        assert result.claim_id == claim.id
        assert result.status == VerificationStatus.SUPPORTED
        assert len(result.evidence_passages) == 1
        evidence = result.evidence_passages[0]
        # Evidence must come from art-corrob (origin was excluded)
        assert evidence.metadata["article_id"] == "art-corrob"
        assert evidence.publisher == "Civic Records Observer"
