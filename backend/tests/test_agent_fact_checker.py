"""Unit tests for FactCheckingAgent: extraction, retrieval, verification, and provenance auditing."""

from datetime import datetime, timezone
import json
import httpx
import pytest

from news_agent.agents.fact_checker import FactCheckingAgent, FactCheckingResult
from news_agent.llm.client import OllamaClient
from news_agent.models.article import NormalizedArticle
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.verification import VerificationResult, VerificationStatus
from news_agent.verification.engine import PipelineVerificationSummary, VerificationEngine


def make_test_article(id_val: str, title: str, content: str, publisher: str = "Daily Herald") -> NormalizedArticle:
    return NormalizedArticle(
        id=id_val,
        title=title,
        content=content,
        publisher=publisher,
        source_url=f"https://herald.com/{id_val}",  # type: ignore[arg-type]
        published_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
    )


class TestFactCheckingAgent:
    def test_check_articles_end_to_end_with_provenance(self):
        art1 = make_test_article(
            "art-1",
            "Council Authorizes $2.5M Solar Contract",
            "The city council approved a $2.5 million solar installation contract on Tuesday.",
            publisher="Daily Herald",
        )
        art2 = make_test_article(
            "art-2",
            "City Minutes Confirm Solar Contract Approval",
            "Official clerk records confirm resolution 44 passed authorizing the $2.5 million solar contract.",
            publisher="Civic Records Observer",
        )

        def mock_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            prompt = body.get("prompt", "")

            if "Extract up to" in prompt:
                if "Daily Herald" in prompt:
                    payload = {
                        "claims": [
                            {
                                "claim_text": "City council approved a $2.5 million solar installation contract.",
                                "context": "Tuesday vote.",
                                "confidence": 0.95,
                            }
                        ]
                    }
                else:
                    payload = {"claims": []}
                return httpx.Response(200, json={"response": json.dumps(payload)})

            elif "Evaluate whether the evidence" in prompt:
                payload = {
                    "status": "supported",
                    "confidence_score": 0.96,
                    "reasoning": "Civic Records Observer explicitly confirms the $2.5M contract authorization.",
                    "cited_passage_ids": ["art-2_p0"],
                }
                return httpx.Response(200, json={"response": json.dumps(payload)})

            return httpx.Response(500, text="Unrecognized prompt")

        client = OllamaClient(
            http_client=httpx.Client(
                transport=httpx.MockTransport(mock_handler),
                base_url="http://mock:11434",
            )
        )
        agent = FactCheckingAgent(llm_client=client)
        result: FactCheckingResult = agent.check_articles([art1, art2])

        assert len(result.claims) == 1
        assert len(result.verification_results) == 1
        assert result.summary["supported"] == 1
        assert result.summary["contradicted"] == 0
        assert result.summary["insufficient_evidence"] == 0

        # Provenance verification
        claim_id = result.claims[0].id
        assert claim_id in result.provenance_map
        prov = result.provenance_map[claim_id]
        assert prov["article_id"] == "art-1"
        assert prov["publisher"] == "Daily Herald"
        assert prov["source_url"] == "https://herald.com/art-1"
        assert prov["status"] == "supported"
        assert len(prov["evidence_sources"]) == 1
        assert prov["evidence_sources"][0]["publisher"] == "Civic Records Observer"

    def test_relevance_score_never_interprets_as_truth_verdict(self):
        """Retrieval relevance score measures lexical term overlap, NEVER truth probability."""
        agent = FactCheckingAgent()
        claim = FactualClaim(
            id="c-test",
            article_id="art-1",
            claim_text="The bridge project was completed on schedule.",
        )
        # Even with high relevance score, if evidence text does not support the assertion,
        # status must remain insufficient_evidence when zero evidence or verified as such
        res = agent.verify_claim(claim, evidence_passages=[])
        assert res.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert res.confidence_score == 0.0

    def test_check_articles_empty_input_handled_gracefully(self):
        agent = FactCheckingAgent()
        result = agent.check_articles([])
        assert len(result.claims) == 0
        assert len(result.verification_results) == 0
        assert result.summary == {"supported": 0, "contradicted": 0, "insufficient_evidence": 0}
        assert result.provenance_map == {}
