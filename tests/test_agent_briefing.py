"""Unit tests for NewsBriefingAgent: grounded brief synthesis, citation preservation, and status distinction."""

from datetime import datetime, timezone
import json
import httpx
import pytest

from news_agent.agents.briefing import NewsBriefingAgent
from news_agent.llm.client import OllamaClient
from news_agent.models.article import NormalizedArticle
from news_agent.models.brief import NewsBrief
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.verification import VerificationResult, VerificationStatus


def make_article(id_val: str, title: str, publisher: str = "Springfield Gazette") -> NormalizedArticle:
    return NormalizedArticle(
        id=id_val,
        title=title,
        content="Springfield municipal council approved the $2.5M clean solar initiative.",
        publisher=publisher,
        source_url=f"https://example.com/{id_val}",  # type: ignore[arg-type]
        published_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
    )


class TestNewsBriefingAgent:
    def test_deterministic_brief_generation_preserves_status_distinction(self):
        art1 = make_article("art-1", "Council Approves $2.5M Solar Contract", "Springfield Gazette")
        art2 = make_article("art-2", "Resolution 44 Passes", "Civic Records Observer")

        claim_supported = FactualClaim(
            id="c1",
            article_id="art-1",
            claim_text="Council approved the solar panel contract.",
        )
        claim_unverified = FactualClaim(
            id="c2",
            article_id="art-1",
            claim_text="Annual utility savings are projected at $400,000.",
        )
        claim_contradicted = FactualClaim(
            id="c3",
            article_id="art-1",
            claim_text="Voting against the measure were three councilors.",
        )

        ev_passage = EvidencePassage(
            id="p-corrob",
            passage_text="Council resolution 44 passed approving solar arrays.",
            source_url="https://example.com/art-2",  # type: ignore[arg-type]
            publisher="Civic Records Observer",
            relevance_score=0.92,
        )

        res_supported = VerificationResult(
            claim_id="c1",
            status=VerificationStatus.SUPPORTED,
            confidence_score=0.95,
            reasoning="Civic Records Observer confirms resolution 44 passed.",
            evidence_passages=[ev_passage],
            model_name="test_model",
        )
        res_unverified = VerificationResult(
            claim_id="c2",
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            confidence_score=0.0,
            reasoning="No independent external sources found regarding the $400,000 figure.",
            evidence_passages=[],
            model_name="test_model",
        )
        res_contradicted = VerificationResult(
            claim_id="c3",
            status=VerificationStatus.CONTRADICTED,
            confidence_score=0.98,
            reasoning="Official records confirm only 2 councilors voted against, not 3.",
            evidence_passages=[ev_passage],
            model_name="test_model",
        )

        agent = NewsBriefingAgent()
        brief: NewsBrief = agent.generate_brief(
            articles=[art1, art2],
            claims=[claim_supported, claim_unverified, claim_contradicted],
            verification_results=[res_supported, res_unverified, res_contradicted],
        )

        assert brief.title.startswith("Intelligence Brief:")
        assert len(brief.source_article_ids) == 2
        assert brief.verification_summary == {
            "supported": 1,
            "contradicted": 1,
            "insufficient_evidence": 1,
        }

        # Check that takeaways label supported, reported, and contradicted distinctly
        takeaway_text = " ".join(brief.headline_takeaways)
        assert "[Supported" in takeaway_text
        assert "[Reported by" in takeaway_text or "Pending external corroboration" in takeaway_text
        assert "Disputed:" in takeaway_text or "[Contradicted" in takeaway_text

        # Synthesized body must group corroborated developments and uncorroborated claims
        assert "Corroborated Developments" in brief.synthesized_body
        assert "Single-Source Reported Developments" in brief.synthesized_body
        assert "Contradicted or Conflicted Assertions" in brief.synthesized_body

    def test_brief_generation_empty_articles_handled(self):
        agent = NewsBriefingAgent()
        brief = agent.generate_brief(articles=[], claims=[], verification_results=[])
        assert brief.title == "No News Available"
        assert len(brief.source_article_ids) == 0
        assert brief.verification_summary == {"supported": 0, "contradicted": 0, "insufficient_evidence": 0}

    def test_brief_generation_llm_synthesis_success(self):
        art = make_article("art-1", "Clean Energy Accord Reached")
        claim = FactualClaim(id="c1", article_id="art-1", claim_text="City reached accord.")
        res = VerificationResult(
            claim_id="c1",
            status=VerificationStatus.SUPPORTED,
            confidence_score=0.9,
            reasoning="Supported by records.",
            evidence_passages=[],
            model_name="test_model",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            payload = {
                "title": "Executive Brief: Clean Energy Accord",
                "headline_takeaways": ["City council approves accord [Springfield Gazette] [Supported]"],
                "synthesized_body": "Council reached clean energy agreement with Midwest Solar. [Source: Springfield Gazette]",
            }
            return httpx.Response(200, json={"response": json.dumps(payload)})

        client = OllamaClient(
            http_client=httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="http://mock:11434",
            )
        )
        agent = NewsBriefingAgent(llm_client=client)
        brief = agent.generate_brief([art], [claim], [res])

        assert brief.title == "Executive Brief: Clean Energy Accord"
        assert len(brief.headline_takeaways) == 1
        assert "Springfield Gazette" in brief.synthesized_body
        assert brief.metadata["generation_mode"] == "llm_grounded"
