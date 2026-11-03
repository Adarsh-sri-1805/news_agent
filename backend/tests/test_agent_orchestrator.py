"""Unit tests for NewsIntelligenceOrchestrator: multi-agent data flow, error resilience, and provenance integrity."""

from datetime import datetime, timezone
import json
import httpx
import pytest

from news_agent.agents.briefing import NewsBriefingAgent
from news_agent.agents.fact_checker import FactCheckingAgent
from news_agent.agents.orchestrator import NewsIntelligenceOrchestrator, OrchestratorResult
from news_agent.agents.researcher import NewsResearcherAgent
from news_agent.agents.social import SocialContentAgent
from news_agent.ingestion.adapters.base import BaseNewsSourceAdapter
from news_agent.ingestion.pipeline import IngestionPipeline
from news_agent.llm.client import OllamaClient
from news_agent.models.article import NormalizedArticle
from news_agent.models.verification import VerificationStatus


class MockStaticAdapter(BaseNewsSourceAdapter):
    def __init__(self, name: str, articles: list[NormalizedArticle]) -> None:
        self._name = name
        self._articles = articles

    @property
    def source_name(self) -> str:
        return self._name

    def fetch_articles(self) -> list[NormalizedArticle]:
        return self._articles


class MockFailingAdapter(BaseNewsSourceAdapter):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def source_name(self) -> str:
        return self._name

    def fetch_articles(self) -> list[NormalizedArticle]:
        raise ConnectionResetError("Remote server closed connection unexpectedly")


from news_agent.models.base import utc_now


def make_article(
    id_val: str,
    title: str,
    content: str,
    publisher: str,
    url: str,
    published_at: datetime | None = None,
) -> NormalizedArticle:
    return NormalizedArticle(
        id=id_val,
        title=title,
        content=content,
        publisher=publisher,
        source_url=url,  # type: ignore[arg-type]
        published_at=published_at or utc_now(),
    )


class TestNewsIntelligenceOrchestrator:
    def test_full_orchestration_workflow_with_provenance(self):
        art1 = make_article(
            "art-1",
            "Council Approves $2.5M Solar Panel Installation",
            "The Springfield city council voted 7-2 on Tuesday morning to approve a $2.5 million solar installation.",
            publisher="Springfield Gazette",
            url="https://gazette.com/solar",
        )
        art2 = make_article(
            "art-2",
            "Resolution 44 Passes with 7-2 Vote",
            "Official city clerk records confirm that council resolution #44 approved the solar panel installation.",
            publisher="Civic Records Observer",
            url="https://observer.org/res44",
        )

        def mock_llm_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            prompt = body.get("prompt", "")

            if "Extract up to" in prompt:
                if "Springfield Gazette" in prompt:
                    payload = {
                        "claims": [
                            {
                                "claim_text": "Springfield city council approved the solar installation by a 7-2 vote.",
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
                    "confidence_score": 0.94,
                    "reasoning": "Civic Records Observer voting ledgers confirm 7 ayes and 2 nays.",
                    "cited_passage_ids": ["art-2_p0"],
                }
                return httpx.Response(200, json={"response": json.dumps(payload)})

            return httpx.Response(500, text="Unrecognized prompt")

        client = OllamaClient(
            http_client=httpx.Client(
                transport=httpx.MockTransport(mock_llm_handler),
                base_url="http://mock:11434",
            )
        )

        pipeline = IngestionPipeline(adapters=[MockStaticAdapter("StaticNews", [art1, art2])])
        orchestrator = NewsIntelligenceOrchestrator(
            researcher=NewsResearcherAgent(pipeline=pipeline),
            fact_checker=FactCheckingAgent(llm_client=client),
            briefing_agent=NewsBriefingAgent(llm_client=client),
            social_agent=SocialContentAgent(llm_client=client),
            llm_client=client,
        )

        result: OrchestratorResult = orchestrator.run(lookback_hours=48)

        # 1. Ingestion metadata
        assert result.ingestion_metadata["retained_count"] == 2
        assert len(result.articles) == 2

        # 2. Fact-checking results
        assert len(result.claims) == 1
        assert len(result.verification_results) == 1
        assert result.verification_results[0].status == VerificationStatus.SUPPORTED

        # 3. News Brief
        assert result.brief is not None
        assert "Solar" in result.brief.title
        assert len(result.brief.source_article_ids) == 2
        assert result.brief.verification_summary["supported"] == 1

        # 4. Social drafts
        assert len(result.social_drafts) == 2
        for draft in result.social_drafts:
            assert draft.character_count <= (280 if draft.platform == "x_twitter" else 3000)

        # 5. Serialization check
        res_dict = result.to_dict()
        assert res_dict["articles_count"] == 2
        assert res_dict["claims_count"] == 1
        assert res_dict["verification_results_count"] == 1

    def test_empty_ingestion_results_handled_gracefully(self):
        pipeline = IngestionPipeline(adapters=[MockStaticAdapter("EmptySource", [])])
        mock_client = OllamaClient(
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"response": "{}"})),
                base_url="http://mock:11434",
            )
        )
        orchestrator = NewsIntelligenceOrchestrator(
            researcher=NewsResearcherAgent(pipeline=pipeline),
            llm_client=mock_client,
        )

        result = orchestrator.run(lookback_hours=24)

        assert len(result.articles) == 0
        assert len(result.claims) == 0
        assert len(result.verification_results) == 0
        assert result.brief is not None
        assert result.brief.title == "No News Available"
        assert len(result.social_drafts) == 0

    def test_partial_source_failure_preserves_successful_articles_and_logs_error(self):
        art1 = make_article("art-1", "Local Fair Opens", "The annual county fair opened today.", "County News", "https://fair.com")
        pipeline = IngestionPipeline(
            adapters=[
                MockStaticAdapter("HealthySource", [art1]),
                MockFailingAdapter("FaultySource"),
            ]
        )
        mock_client = OllamaClient(
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"response": "{}"})),
                base_url="http://mock:11434",
            )
        )
        orchestrator = NewsIntelligenceOrchestrator(
            researcher=NewsResearcherAgent(pipeline=pipeline),
            llm_client=mock_client,
        )

        result = orchestrator.run()

        # Healthy source ingested
        assert len(result.articles) == 1
        assert result.articles[0].id == "art-1"
        # Error from faulty source recorded
        assert len(result.errors) >= 1
        assert any(e.get("source") == "FaultySource" for e in result.errors)

    def test_run_with_articles_direct_mode(self):
        art = make_article("art-direct", "Transit Overhaul", "City announces bus overhaul.", "Transit Gazette", "https://transit.com")
        mock_client = OllamaClient(
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"response": "{}"})),
                base_url="http://mock:11434",
            )
        )
        orchestrator = NewsIntelligenceOrchestrator(llm_client=mock_client)
        result = orchestrator.run_with_articles([art])

        assert len(result.articles) == 1
        assert result.articles[0].id == "art-direct"
        assert result.brief is not None
        assert len(result.social_drafts) >= 1
