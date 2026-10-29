"""Unit tests for ClaimVerifier covering verdicts, citation filtering, and failure safeguards."""

from datetime import datetime, timezone
import json
import httpx
import pytest

from news_agent.llm.client import OllamaClient
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.verification import VerificationStatus
from news_agent.verification.verifier import ClaimVerifier


def make_test_claim(text: str = "Council voted 7-2 to approve solar panels.") -> FactualClaim:
    return FactualClaim(
        id="claim-test-1",
        article_id="art-test-1",
        claim_text=text,
        extracted_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
    )


def make_test_evidence(id_val: str, text: str, publisher: str = "Gazette") -> EvidencePassage:
    return EvidencePassage(
        id=id_val,
        passage_text=text,
        publisher=publisher,
        source_url=f"https://example.com/{id_val}",  # type: ignore[arg-type]
        retrieved_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
        relevance_score=0.92,
    )


class TestClaimVerifier:
    def test_verify_supported_verdict(self):
        llm_response = {
            "status": "supported",
            "confidence_score": 0.95,
            "reasoning": "Passage p1 directly confirms the 7-2 vote passed by the council.",
            "cited_passage_ids": ["p1"],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": json.dumps(llm_response)})

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        verifier = ClaimVerifier(llm_client=client)

        claim = make_test_claim()
        ev1 = make_test_evidence("p1", "The Springfield city council voted 7-2 in favor of the solar project.")

        result = verifier.verify_claim(claim, [ev1])
        assert result.claim_id == claim.id
        assert result.status == VerificationStatus.SUPPORTED
        assert result.confidence_score == 0.95
        assert "directly confirms" in result.reasoning
        assert len(result.evidence_passages) == 1
        assert result.evidence_passages[0].id == "p1"

    def test_verify_contradicted_verdict(self):
        llm_response = {
            "status": "contradicted",
            "confidence_score": 0.90,
            "reasoning": "Official records show the resolution was rejected 2-7, directly conflicting with the claim.",
            "cited_passage_ids": ["p1"],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": json.dumps(llm_response)})

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        verifier = ClaimVerifier(llm_client=client)

        claim = make_test_claim("Council approved the solar panel contract.")
        ev1 = make_test_evidence("p1", "Resolution 44 to install solar panels was defeated by a 2-7 vote.")

        result = verifier.verify_claim(claim, [ev1])
        assert result.status == VerificationStatus.CONTRADICTED
        assert result.confidence_score == 0.90

    def test_zero_evidence_yields_insufficient_evidence_without_llm(self):
        """CRITICAL: Deterministic zero-evidence guard must never call the LLM."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("LLM must not be called when zero evidence is retrieved!")

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        verifier = ClaimVerifier(llm_client=client)

        claim = make_test_claim()
        result = verifier.verify_claim(claim, evidence_passages=[])

        assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert result.confidence_score == 0.0
        assert "No independent evidence passages were retrieved" in result.reasoning

    def test_unsupplied_passage_citations_are_discarded(self):
        """CRITICAL: A model must not cite an evidence passage that was not actually supplied to it."""
        llm_response = {
            "status": "supported",
            "confidence_score": 0.85,
            "reasoning": "Confirmed by records.",
            # Model cites a hallucinated passage 'p999' not in supplied list ['p1']
            "cited_passage_ids": ["p1", "p999_fabricated"],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": json.dumps(llm_response)})

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        verifier = ClaimVerifier(llm_client=client)

        claim = make_test_claim()
        ev1 = make_test_evidence("p1", "Official council minutes confirm the resolution passed 7-2.")

        result = verifier.verify_claim(claim, [ev1])
        # Only the legitimate passage p1 should be retained
        assert len(result.evidence_passages) == 1
        assert result.evidence_passages[0].id == "p1"

    def test_model_failure_never_converts_to_supported(self):
        """CRITICAL: If the model returns invalid output, do not silently convert the failure into supported."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Ollama Crash")

        mock_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")
        client = OllamaClient(http_client=mock_http)
        verifier = ClaimVerifier(llm_client=client)

        claim = make_test_claim()
        ev1 = make_test_evidence("p1", "Some evidence text.")

        result = verifier.verify_claim(claim, [ev1])
        assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert result.confidence_score == 0.0
        assert "could not be completed safely" in result.reasoning
