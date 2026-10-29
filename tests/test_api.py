"""API test suite for Local News Intelligence Agent (Phase 5).

Covers:
1. Health endpoint success.
2. Readiness endpoint when dependencies are healthy.
3. Readiness endpoint when Ollama is unavailable.
4. Readiness endpoint when Ollama is up but model is missing.
5. Successful news workflow using mocked dependencies.
6. Safe mock/demo mode workflow.
7. Invalid request validation (bounds and platform constraints).
8. Empty news results handling.
9. Orchestrator failure and sanitized error response.
10. Preservation of verification statuses, scores, and provenance.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock
from fastapi import status
from fastapi.testclient import TestClient
import pytest

from news_agent.agents.orchestrator import NewsIntelligenceOrchestrator, OrchestratorResult
from news_agent.api.app import create_app
from news_agent.api.dependencies import get_ollama_client, get_orchestrator
from news_agent.llm.client import OllamaClient
from news_agent.llm.exceptions import OllamaConnectionError
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now
from news_agent.models.brief import NewsBrief
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.social import SocialMediaDraft
from news_agent.models.verification import VerificationResult, VerificationStatus


@pytest.fixture
def app():
    """Create a fresh FastAPI test application."""
    return create_app()


@pytest.fixture
def client(app):
    """TestClient instance for HTTP assertions."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_article() -> NormalizedArticle:
    return NormalizedArticle(
        id="art-test-1",
        title="City Council Enacts Green Energy Ordinance",
        content="The municipal city council voted 8-1 to allocate funding for local solar arrays.",
        source_url="https://civicgazette.org/news/solar-ordinance",  # type: ignore[arg-type]
        publisher="Civic Gazette",
        published_at=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_claim() -> FactualClaim:
    return FactualClaim(
        id="claim-test-1",
        article_id="art-test-1",
        claim_text="The municipal city council voted 8-1 to allocate funding for local solar arrays.",
        context="Paragraph 1 vote count details.",
        extraction_confidence=0.97,
    )


@pytest.fixture
def sample_verification(sample_claim) -> VerificationResult:
    return VerificationResult(
        id="ver-test-1",
        claim_id=sample_claim.id,
        status=VerificationStatus.SUPPORTED,
        confidence_score=0.96,
        reasoning="Corroborated by municipal records showing 8 ayes and 1 nay.",
        evidence_passages=[
            EvidencePassage(
                id="evi-1",
                passage_text="Official voting tally confirms 8 affirmative votes and 1 dissenting vote.",
                source_url="https://cityhall.gov/voting-records",  # type: ignore[arg-type]
                publisher="City Clerk Office",
                relevance_score=0.95,
            )
        ],
        model_name="llama3.2:3b",
    )


@pytest.fixture
def sample_brief(sample_article) -> NewsBrief:
    return NewsBrief(
        id="brief-test-1",
        title="Civic Energy Transition Approved",
        headline_takeaways=["Council passed solar funding 8-1."],
        synthesized_body="The municipal council overwhelmingly approved new solar infrastructure investments.",
        source_article_ids=[sample_article.id],
        verification_summary={"supported": 1, "contradicted": 0, "insufficient_evidence": 0},
    )


@pytest.fixture
def sample_draft(sample_brief, sample_article) -> SocialMediaDraft:
    return SocialMediaDraft(
        id="draft-test-1",
        platform="x_twitter",
        content="City Council voted 8-1 to allocate municipal solar funding. #CleanEnergy #CivicNews",
        brief_id=sample_brief.id,
        source_article_ids=[sample_article.id],
        hashtags=["CleanEnergy", "CivicNews"],
    )


class TestHealthEndpoint:
    def test_health_success(self, client):
        """GET /health must return 200 without executing LLM or scraping."""
        resp = client.get("/health")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["app"] == "local-news-intelligence-agent"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data


class TestReadinessEndpoint:
    def test_readiness_healthy_when_ollama_and_model_available(self, app, client):
        """GET /ready returns 200 when Ollama is running and configured model is present."""
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.base_url = "http://localhost:11434"
        mock_ollama.model_name = "llama3.2:3b"
        mock_ollama.health_check.return_value = True
        mock_ollama.list_local_models.return_value = ["llama3.2:3b", "other:latest"]

        app.dependency_overrides[get_ollama_client] = lambda: mock_ollama

        resp = client.get("/ready")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["status"] == "ready"
        assert data["ready"] is True
        assert data["dependencies"]["ollama"]["status"] == "ready"
        assert data["dependencies"]["ollama"]["details"]["model_installed"] is True
        assert data["dependencies"]["ingestion_sources"]["status"] == "ready"

    def test_readiness_fails_gracefully_when_ollama_unreachable(self, app, client):
        """GET /ready returns 503 when Ollama is down without crashing process."""
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.base_url = "http://localhost:11434"
        mock_ollama.model_name = "llama3.2:3b"
        mock_ollama.health_check.return_value = False

        app.dependency_overrides[get_ollama_client] = lambda: mock_ollama

        resp = client.get("/ready")
        assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["ready"] is False
        assert data["dependencies"]["ollama"]["status"] == "unavailable"

    def test_readiness_fails_when_ollama_raises_connection_exception(self, app, client):
        """GET /ready catches Ollama connection errors without raising 500 or crashing."""
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.base_url = "http://localhost:11434"
        mock_ollama.health_check.side_effect = OllamaConnectionError("Connection refused")

        app.dependency_overrides[get_ollama_client] = lambda: mock_ollama

        resp = client.get("/ready")
        assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = resp.json()
        assert data["ready"] is False
        assert data["dependencies"]["ollama"]["status"] == "unavailable"

    def test_readiness_fails_when_model_is_not_installed(self, app, client):
        """GET /ready returns 503 when Ollama is up but the configured model is absent."""
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.base_url = "http://localhost:11434"
        mock_ollama.model_name = "llama3.2:3b"
        mock_ollama.health_check.return_value = True
        mock_ollama.list_local_models.return_value = ["mistral:latest", "gemma:2b"]

        app.dependency_overrides[get_ollama_client] = lambda: mock_ollama

        resp = client.get("/ready")
        assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = resp.json()
        assert data["ready"] is False
        assert data["dependencies"]["ollama"]["status"] == "not_ready"
        assert data["dependencies"]["ollama"]["details"]["model_installed"] is False


class TestNewsRunEndpoint:
    def test_news_run_success_with_mocked_dependencies(
        self,
        app,
        client,
        sample_article,
        sample_claim,
        sample_verification,
        sample_brief,
        sample_draft,
    ):
        """POST /api/v1/news/run successfully coordinates workflow and returns all artifacts."""
        mock_orch = MagicMock(spec=NewsIntelligenceOrchestrator)
        mock_result = OrchestratorResult(
            ingestion_metadata={"retained_count": 1, "raw_collected_count": 2},
            articles=[sample_article],
            claims=[sample_claim],
            verification_results=[sample_verification],
            brief=sample_brief,
            social_drafts=[sample_draft],
            errors=[],
            completed_at=utc_now(),
        )
        mock_orch.run.return_value = mock_result
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch

        payload = {"lookback_hours": 12, "platforms": ["x_twitter"]}
        resp = client.post("/api/v1/news/run", json=payload)

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()

        # Ingestion telemetry
        assert data["ingestion_metadata"]["retained_count"] == 1

        # Articles
        assert len(data["articles"]) == 1
        assert data["articles"][0]["id"] == sample_article.id
        assert data["articles"][0]["publisher"] == "Civic Gazette"

        # Claims
        assert len(data["claims"]) == 1
        assert data["claims"][0]["id"] == sample_claim.id

        # Verification
        assert len(data["verification_results"]) == 1
        assert data["verification_results"][0]["status"] == "supported"
        assert data["verification_results"][0]["confidence_score"] == 0.96
        assert len(data["verification_results"][0]["evidence_passages"]) == 1

        # Brief
        assert data["brief"] is not None
        assert data["brief"]["title"] == "Civic Energy Transition Approved"
        assert data["brief"]["verification_summary"]["supported"] == 1

        # Social Drafts
        assert len(data["social_drafts"]) == 1
        assert data["social_drafts"][0]["platform"] == "x_twitter"
        assert data["social_drafts"][0]["character_count"] > 0

        # Orchestrator invoked with correct arguments
        mock_orch.run.assert_called_once_with(lookback_hours=12, platforms=["x_twitter"])

    def test_news_run_safe_mock_mode(self, client):
        """POST /api/v1/news/run in mock_mode runs full pipeline offline deterministically."""
        payload = {"mock_mode": True, "lookback_hours": 24, "platforms": ["x_twitter", "linkedin"]}
        resp = client.post("/api/v1/news/run", json=payload)

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()

        assert len(data["articles"]) >= 1
        assert len(data["claims"]) >= 1
        assert len(data["verification_results"]) >= 1
        assert data["brief"] is not None
        assert len(data["social_drafts"]) >= 1
        verdict_statuses = {v["status"] for v in data["verification_results"]}
        assert "supported" in verdict_statuses

    def test_news_run_validation_invalid_lookback_hours(self, client):
        """POST /api/v1/news/run rejects lookback_hours <= 0 or > 168 with 422."""
        # Non-positive lookback
        resp = client.post("/api/v1/news/run", json={"lookback_hours": 0})
        assert resp.status_code == 422

        # Excessively large lookback
        resp2 = client.post("/api/v1/news/run", json={"lookback_hours": 200})
        assert resp2.status_code == 422

    def test_news_run_validation_invalid_platform(self, client):
        """POST /api/v1/news/run rejects unsupported social platforms with 422."""
        resp = client.post("/api/v1/news/run", json={"platforms": ["myspace", "x_twitter"]})
        assert resp.status_code == 422
        assert "Unsupported platform 'myspace'" in resp.text

    def test_news_run_empty_news_results_handled_gracefully(self, app, client):
        """POST /api/v1/news/run preserves empty result states without failing."""
        mock_orch = MagicMock(spec=NewsIntelligenceOrchestrator)
        mock_result = OrchestratorResult(
            ingestion_metadata={"retained_count": 0, "raw_collected_count": 0},
            articles=[],
            claims=[],
            verification_results=[],
            brief=NewsBrief(
                title="No News Available",
                headline_takeaways=["No news articles were ingested within the requested window."],
                synthesized_body="The news ingestion pipeline returned zero articles for analysis.",
            ),
            social_drafts=[],
            errors=[],
            completed_at=utc_now(),
        )
        mock_orch.run.return_value = mock_result
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch

        resp = client.post("/api/v1/news/run", json={"lookback_hours": 24})
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert len(data["articles"]) == 0
        assert len(data["claims"]) == 0
        assert len(data["verification_results"]) == 0
        assert data["brief"]["title"] == "No News Available"
        assert len(data["social_drafts"]) == 0

    def test_news_run_orchestrator_failure_returns_sanitized_500(self, app, client):
        """POST /api/v1/news/run catches unhandled orchestrator crashes and returns 500 without trace leaks."""
        mock_orch = MagicMock(spec=NewsIntelligenceOrchestrator)
        mock_orch.run.side_effect = RuntimeError("Fatal pipeline failure: database connection timeout")
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch

        resp = client.post("/api/v1/news/run", json={"lookback_hours": 24})
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = resp.json()
        assert "detail" in data
        assert "Fatal pipeline failure" in data["detail"]
        # Ensure raw Python stack traces are not leaked in detail string
        assert "Traceback" not in data["detail"]

    def test_news_run_preserves_provenance_and_verification_verdicts(
        self,
        app,
        client,
        sample_article,
        sample_claim,
        sample_verification,
        sample_brief,
        sample_draft,
    ):
        """Verification statuses (supported, contradicted, insufficient_evidence) and URLs are strictly preserved."""
        # Test preservation across supported and contradicted
        contradicted_claim = FactualClaim(
            id="claim-contra",
            article_id="art-test-1",
            claim_text="The vote was unanimously approved.",
            context="Vote context.",
            extraction_confidence=0.85,
        )
        contradicted_ver = VerificationResult(
            id="ver-contra",
            claim_id=contradicted_claim.id,
            status=VerificationStatus.CONTRADICTED,
            confidence_score=0.91,
            reasoning="Official records document 1 nay vote.",
            model_name="llama3.2:3b",
        )

        mock_orch = MagicMock(spec=NewsIntelligenceOrchestrator)
        mock_orch.run.return_value = OrchestratorResult(
            ingestion_metadata={"retained_count": 1},
            articles=[sample_article],
            claims=[sample_claim, contradicted_claim],
            verification_results=[sample_verification, contradicted_ver],
            brief=sample_brief,
            social_drafts=[sample_draft],
            errors=[],
            completed_at=utc_now(),
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch

        resp = client.post("/api/v1/news/run", json={"lookback_hours": 24})
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()

        verdicts = {v["claim_id"]: v["status"] for v in data["verification_results"]}
        assert verdicts[sample_claim.id] == "supported"
        assert verdicts[contradicted_claim.id] == "contradicted"

        # Check evidence passage provenance
        supported_ver = next(v for v in data["verification_results"] if v["claim_id"] == sample_claim.id)
        assert len(supported_ver["evidence_passages"]) == 1
        passage = supported_ver["evidence_passages"][0]
        assert str(passage["source_url"]) == "https://cityhall.gov/voting-records"
        assert passage["publisher"] == "City Clerk Office"
        assert passage["relevance_score"] == 0.95

    def test_news_run_empty_articles_list_invokes_live_orchestrator(self, app, client, sample_article):
        """When articles is explicitly [] and mock_mode is false, invoke live orchestrator.run()."""
        mock_orch = MagicMock(spec=NewsIntelligenceOrchestrator)
        mock_orch.run.return_value = OrchestratorResult(
            ingestion_metadata={"raw_collected_count": 5, "retained_count": 3},
            articles=[sample_article],
            claims=[],
            verification_results=[],
            brief=None,
            social_drafts=[],
            errors=[],
            completed_at=utc_now(),
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch

        payload = {"articles": [], "lookback_hours": 36, "mock_mode": False}
        resp = client.post("/api/v1/news/run", json=payload)

        assert resp.status_code == status.HTTP_200_OK
        mock_orch.run.assert_called_once_with(lookback_hours=36, platforms=None)
        mock_orch.run_with_articles.assert_not_called()
        data = resp.json()
        assert data["ingestion_metadata"]["raw_collected_count"] == 5

    def test_news_run_omitted_articles_invokes_live_orchestrator(self, app, client, sample_article):
        """When articles is omitted, invoke live orchestrator.run()."""
        mock_orch = MagicMock(spec=NewsIntelligenceOrchestrator)
        mock_orch.run.return_value = OrchestratorResult(
            ingestion_metadata={"raw_collected_count": 2, "retained_count": 1},
            articles=[sample_article],
            claims=[],
            verification_results=[],
            brief=None,
            social_drafts=[],
            errors=[],
            completed_at=utc_now(),
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch

        payload = {"lookback_hours": 48}
        resp = client.post("/api/v1/news/run", json=payload)

        assert resp.status_code == status.HTTP_200_OK
        mock_orch.run.assert_called_once_with(lookback_hours=48, platforms=None)
        mock_orch.run_with_articles.assert_not_called()

    def test_news_run_supplied_articles_invokes_run_with_articles(self, app, client, sample_article):
        """When articles are explicitly provided with content, invoke orchestrator.run_with_articles()."""
        mock_orch = MagicMock(spec=NewsIntelligenceOrchestrator)
        mock_orch.run_with_articles.return_value = OrchestratorResult(
            ingestion_metadata={"mode": "supplied_articles", "retained_count": 1},
            articles=[sample_article],
            claims=[],
            verification_results=[],
            brief=None,
            social_drafts=[],
            errors=[],
            completed_at=utc_now(),
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch

        payload = {
            "articles": [sample_article.model_dump(mode="json")],
            "platforms": ["x_twitter"],
            "mock_mode": False,
        }
        resp = client.post("/api/v1/news/run", json=payload)

        assert resp.status_code == status.HTTP_200_OK
        mock_orch.run_with_articles.assert_called_once()
        mock_orch.run.assert_not_called()
        data = resp.json()
        assert data["ingestion_metadata"]["mode"] == "supplied_articles"

    def test_news_run_mock_mode_with_empty_articles_uses_mock_workflow(self, client):
        """When mock_mode is True, even with articles: [], use mock workflow."""
        payload = {"mock_mode": True, "articles": [], "platforms": ["x_twitter", "linkedin"]}
        resp = client.post("/api/v1/news/run", json=payload)

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert len(data["articles"]) >= 1
        assert data["brief"] is not None

    def test_social_drafts_truth_wording_when_articles_empty(self):
        """Drafts generated with zero articles must never claim verified intelligence."""
        from news_agent.agents.social import SocialContentAgent

        agent = SocialContentAgent()
        brief = NewsBrief(
            title="No News Available",
            headline_takeaways=["No articles were ingested."],
            synthesized_body="Zero articles analyzed.",
        )
        drafts = agent.generate_drafts(
            brief=brief,
            articles=[],
            verification_results=[],
            platforms=["x_twitter", "linkedin"],
        )

        assert len(drafts) == 2
        for d in drafts:
            assert "Key verified intelligence" not in d.content
            assert "Source: News ()" not in d.content
            assert "Notice: No News Available" in d.content
