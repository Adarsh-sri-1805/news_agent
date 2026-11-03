"""Safe mock/demo runner enabling full request-response testing without Ollama or live feeds."""

from datetime import datetime, timezone
import json
import httpx

from news_agent.agents.briefing import NewsBriefingAgent
from news_agent.agents.fact_checker import FactCheckingAgent
from news_agent.agents.orchestrator import NewsIntelligenceOrchestrator, OrchestratorResult
from news_agent.agents.researcher import NewsResearcherAgent
from news_agent.agents.social import SocialContentAgent
from news_agent.ingestion.adapters.base import BaseNewsSourceAdapter
from news_agent.ingestion.pipeline import IngestionPipeline
from news_agent.llm.client import OllamaClient
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now


class StaticMockAdapter(BaseNewsSourceAdapter):
    """Feeds pre-defined sample articles for offline mock runs."""

    def __init__(self, name: str, articles: list[NormalizedArticle]) -> None:
        self._name = name
        self._articles = articles

    @property
    def source_name(self) -> str:
        return self._name

    def fetch_articles(self) -> list[NormalizedArticle]:
        return self._articles


def get_demo_articles() -> list[NormalizedArticle]:
    """Return realistic demo articles for civic solar contract news."""
    now = utc_now()
    art1 = NormalizedArticle(
        id="art-gazette-solar-01",
        title="Springfield City Council Approves $2.5M Solar Panel Installation",
        content=(
            "The Springfield city council voted 7-2 on Tuesday morning to approve a $2.5 million "
            "solar panel installation contract across municipal buildings. Councilor Jenkins sponsored "
            "the bill, projecting $400,000 in annual utility cost reductions. Voting against were Councilors "
            "Vance and Miller, who cited concerns over short-term municipal debt issuance."
        ),
        publisher="Springfield Gazette",
        source_url="https://springfieldgazette.com/articles/solar-vote",  # type: ignore[arg-type]
        published_at=now,
    )

    art2 = NormalizedArticle(
        id="art-observer-records-02",
        title="Municipal Voting Records: Resolution #44 Clean Energy Accord Passes",
        content=(
            "Official voting ledgers from the Springfield city clerk confirm that resolution #44 passed "
            "with a 7-2 vote. The city council approved the $2.5 million solar panel installation contract, "
            "authorizing Midwest Solar Dynamics to install photovoltaic arrays on city hall starting next spring."
        ),
        publisher="Civic Records Observer",
        source_url="https://civicobserver.org/records/resolution-44",  # type: ignore[arg-type]
        published_at=now,
    )

    return [art1, art2]


def create_mock_orchestrator(custom_articles: list[NormalizedArticle] | None = None) -> NewsIntelligenceOrchestrator:
    """Create a fully wired orchestrator using deterministic in-memory mock LLM transport."""
    articles = custom_articles if custom_articles is not None else get_demo_articles()

    def mock_llm_handler(request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content.decode("utf-8"))
        except Exception:
            body = {}
        prompt = body.get("prompt", "")

        # Claim extraction prompt handling
        if "Extract up to" in prompt:
            if "Springfield Gazette" in prompt or "Council Approves" in prompt:
                payload = {
                    "claims": [
                        {
                            "claim_text": "The Springfield city council voted 7-2 to approve a $2.5 million solar installation.",
                            "context": "Paragraph 1 detailing voting count and project valuation.",
                            "confidence": 0.96,
                        }
                    ]
                }
            elif "Civic Records" in prompt or "Resolution #44" in prompt:
                payload = {
                    "claims": [
                        {
                            "claim_text": "Resolution #44 passed with 7 ayes and 2 nays.",
                            "context": "Official voting ledgers verification.",
                            "confidence": 0.98,
                        }
                    ]
                }
            else:
                payload = {"claims": []}
            return httpx.Response(200, json={"response": json.dumps(payload)})

        # Verification prompt handling
        elif "Evaluate whether the evidence" in prompt:
            payload = {
                "status": "supported",
                "confidence_score": 0.95,
                "reasoning": "The evidence passage from the Civic Records Observer directly confirms the 7-2 vote margin.",
                "cited_passage_ids": ["art-observer-records-02_p0"],
            }
            return httpx.Response(200, json={"response": json.dumps(payload)})

        # Briefing prompt handling
        elif "Synthesize a structured news brief" in prompt:
            payload = {
                "title": "Springfield Municipal Solar Installation Approved",
                "headline_takeaways": [
                    "Springfield city council passed a $2.5M solar contract by a 7-2 vote.",
                    "Midwest Solar Dynamics will perform rooftop installations starting in the spring.",
                    "The initiative is projected to reduce municipal utility expenses by $400,000 annually.",
                ],
                "synthesized_body": (
                    "In a 7-2 decision, the Springfield city council ratified resolution #44 authorizing "
                    "a $2.5 million municipal solar deployment. Voting records from the city clerk corroborate "
                    "the vote outcome, confirming Midwest Solar Dynamics as the contractor."
                ),
            }
            return httpx.Response(200, json={"response": json.dumps(payload)})

        # Social copy prompt handling
        elif "Draft promotional social copy" in prompt or "social media" in prompt.lower():
            payload = {
                "content": (
                    "Springfield City Council has approved a $2.5M solar installation across municipal buildings "
                    "in a 7-2 vote. The project is projected to save $400k in annual utility costs."
                ),
                "hashtags": ["CleanEnergy", "Springfield", "CivicNews"],
            }
            return httpx.Response(200, json={"response": json.dumps(payload)})

        # Fallback safe empty JSON object
        return httpx.Response(200, json={"response": "{}"})

    mock_client = OllamaClient(
        http_client=httpx.Client(
            transport=httpx.MockTransport(mock_llm_handler),
            base_url="http://mock:11434",
        )
    )

    pipeline = IngestionPipeline(adapters=[StaticMockAdapter("DemoMockSource", articles)])
    return NewsIntelligenceOrchestrator(
        researcher=NewsResearcherAgent(pipeline=pipeline),
        fact_checker=FactCheckingAgent(llm_client=mock_client),
        briefing_agent=NewsBriefingAgent(llm_client=mock_client),
        social_agent=SocialContentAgent(llm_client=mock_client),
        llm_client=mock_client,
    )


def run_mock_workflow(
    lookback_hours: int | None = None,
    platforms: list[str] | None = None,
    articles: list[NormalizedArticle] | None = None,
) -> OrchestratorResult:
    """Execute the multi-agent workflow in offline mock mode."""
    orchestrator = create_mock_orchestrator(custom_articles=articles)
    if articles is not None:
        return orchestrator.run_with_articles(articles=articles, platforms=platforms)
    return orchestrator.run(lookback_hours=lookback_hours or 48, platforms=platforms)
