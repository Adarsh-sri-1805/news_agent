"""Unit tests for MCP server: initialization, tool registration, input validation, and execution."""

import asyncio
from datetime import datetime, timezone
import json
import httpx
import pytest

from news_agent.agents.briefing import NewsBriefingAgent
from news_agent.agents.fact_checker import FactCheckingAgent
from news_agent.agents.orchestrator import NewsIntelligenceOrchestrator
from news_agent.agents.researcher import NewsResearcherAgent
from news_agent.agents.social import SocialContentAgent
from news_agent.ingestion.adapters.base import BaseNewsSourceAdapter
from news_agent.ingestion.pipeline import IngestionPipeline
from news_agent.llm.client import OllamaClient
from news_agent.mcp.server import create_mcp_server
from news_agent.models.article import NormalizedArticle


class MockStaticAdapter(BaseNewsSourceAdapter):
    def __init__(self, name: str, articles: list[NormalizedArticle]) -> None:
        self._name = name
        self._articles = articles

    @property
    def source_name(self) -> str:
        return self._name

    def fetch_articles(self) -> list[NormalizedArticle]:
        return self._articles


def make_test_article(id_val: str, title: str, content: str, publisher: str = "Gazette") -> NormalizedArticle:
    return NormalizedArticle(
        id=id_val,
        title=title,
        content=content,
        publisher=publisher,
        source_url=f"https://example.com/{id_val}",  # type: ignore[arg-type]
        published_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def test_orchestrator():
    art1 = make_test_article(
        "art-1",
        "Springfield Approves $2.5M Solar Panel Installation",
        "The city council voted 7-2 on Tuesday morning to approve a $2.5 million solar installation across municipal buildings.",
        publisher="Springfield Gazette",
    )
    art2 = make_test_article(
        "art-2",
        "Resolution 44 Passes with 7-2 Vote",
        "Official city clerk voting records confirm that council approved the solar installation by a 7-2 vote.",
        publisher="Civic Records Observer",
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        prompt = body.get("prompt", "")

        if "Extract up to" in prompt:
            payload = {
                "claims": [
                    {
                        "claim_text": "The Springfield city council approved the solar installation by a 7-2 vote.",
                        "context": "Tuesday morning session.",
                        "confidence": 0.95,
                    }
                ]
            }
            return httpx.Response(200, json={"response": json.dumps(payload)})
        elif "Evaluate whether the evidence" in prompt:
            payload = {
                "status": "supported",
                "confidence_score": 0.94,
                "reasoning": "Civic Records Observer confirms resolution 44 passed.",
                "cited_passage_ids": ["art-2_p0"],
            }
            return httpx.Response(200, json={"response": json.dumps(payload)})
        elif "Synthesize a concise news" in prompt:
            payload = {
                "title": "Intelligence Brief: Solar Panel Installation Approved",
                "headline_takeaways": ["Council approves solar project [Supported]"],
                "synthesized_body": "Springfield council authorized installation with 7-2 vote.",
            }
            return httpx.Response(200, json={"response": json.dumps(payload)})

        return httpx.Response(200, json={"response": "{}"})

    client = OllamaClient(
        http_client=httpx.Client(
            transport=httpx.MockTransport(mock_handler),
            base_url="http://mock:11434",
        )
    )

    pipeline = IngestionPipeline(adapters=[MockStaticAdapter("StaticNews", [art1, art2])])
    return NewsIntelligenceOrchestrator(
        researcher=NewsResearcherAgent(pipeline=pipeline),
        fact_checker=FactCheckingAgent(llm_client=client),
        briefing_agent=NewsBriefingAgent(llm_client=client),
        social_agent=SocialContentAgent(llm_client=client),
        llm_client=client,
    )


class TestMCPServer:
    def test_server_initialization_and_tool_registration(self, test_orchestrator):
        async def _run():
            server = create_mcp_server(orchestrator=test_orchestrator)
            tools = await server.list_tools()
            tool_names = [t.name for t in tools]

            expected_tools = [
                "search_news",
                "get_article",
                "extract_claims",
                "retrieve_evidence",
                "verify_claim",
                "generate_news_brief",
                "generate_social_drafts",
                "run_news_intelligence",
            ]
            for exp in expected_tools:
                assert exp in tool_names, f"Expected tool '{exp}' not registered on MCP server"

        asyncio.run(_run())

    def test_search_news_and_get_article_tools(self, test_orchestrator):
        async def _run():
            server = create_mcp_server(orchestrator=test_orchestrator)

            # 1. Execute search_news
            search_res = await server.call_tool("search_news", {"lookback_hours": 48})
            assert search_res.is_error is False
            assert len(search_res.content) == 1
            data = json.loads(search_res.content[0].text)
            assert data["count"] == 2
            assert len(data["articles"]) == 2

            # 2. Retrieve existing article by ID
            get_res = await server.call_tool("get_article", {"article_id": "art-1"})
            assert get_res.is_error is False
            art_data = json.loads(get_res.content[0].text)
            assert "article" in art_data
            assert art_data["article"]["id"] == "art-1"

            # 3. Retrieve non-existent article by ID
            missing_res = await server.call_tool("get_article", {"article_id": "art-nonexistent"})
            assert missing_res.is_error is False
            missing_data = json.loads(missing_res.content[0].text)
            assert "error" in missing_data
            assert "not found" in missing_data["error"]

        asyncio.run(_run())

    def test_extract_claims_validation(self, test_orchestrator):
        async def _run():
            server = create_mcp_server(orchestrator=test_orchestrator)

            # Error if neither article_id nor article_text provided
            res = await server.call_tool("extract_claims", {})
            data = json.loads(res.content[0].text)
            assert "error" in data

            # Error if article_text too short
            short_res = await server.call_tool("extract_claims", {"article_text": "too short"})
            short_data = json.loads(short_res.content[0].text)
            assert "error" in short_data

        asyncio.run(_run())

    def test_retrieve_evidence_and_verify_claim_tools(self, test_orchestrator):
        async def _run():
            server = create_mcp_server(orchestrator=test_orchestrator)

            # First populate articles via search_news
            await server.call_tool("search_news", {"lookback_hours": 48})

            # Retrieve evidence
            ev_res = await server.call_tool(
                "retrieve_evidence",
                {
                    "claim_text": "The Springfield city council approved the solar installation by a 7-2 vote.",
                    "top_k": 2,
                },
            )
            assert ev_res.is_error is False
            ev_data = json.loads(ev_res.content[0].text)
            assert ev_data["count"] >= 1
            assert "evidence_passages" in ev_data

            # Verify claim with supplied passages
            v_res = await server.call_tool(
                "verify_claim",
                {
                    "claim_text": "The council voted 7-2 for solar panels.",
                    "evidence_passage_texts": [
                        "Official voting ledgers verify that resolution #44 passed with 7 ayes and 2 nays."
                    ],
                },
            )
            assert v_res.is_error is False
            v_data = json.loads(v_res.content[0].text)
            assert "verification_result" in v_data
            assert "status" in v_data["verification_result"]

        asyncio.run(_run())

    def test_generate_news_brief_and_social_drafts_tools(self, test_orchestrator):
        async def _run():
            server = create_mcp_server(orchestrator=test_orchestrator)

            # Index articles
            await server.call_tool("search_news", {"lookback_hours": 48})

            # Generate brief
            brief_res = await server.call_tool("generate_news_brief", {})
            assert brief_res.is_error is False
            brief_data = json.loads(brief_res.content[0].text)
            assert "brief" in brief_data
            assert "title" in brief_data["brief"]

            # Generate social drafts
            draft_res = await server.call_tool(
                "generate_social_drafts",
                {"platforms": ["x_twitter", "linkedin"]},
            )
            assert draft_res.is_error is False
            draft_data = json.loads(draft_res.content[0].text)
            assert draft_data["count"] == 2
            for d in draft_data["drafts"]:
                assert d["character_count"] <= (280 if d["platform"] == "x_twitter" else 3000)

        asyncio.run(_run())

    def test_run_news_intelligence_full_pipeline_mcp(self, test_orchestrator):
        async def _run():
            server = create_mcp_server(orchestrator=test_orchestrator)

            res = await server.call_tool(
                "run_news_intelligence",
                {"lookback_hours": 48, "platforms": ["x_twitter", "linkedin"]},
            )
            assert res.is_error is False
            data = json.loads(res.content[0].text)
            assert data["articles_count"] == 2
            assert "brief" in data
            assert len(data["social_drafts"]) == 2
            assert "ingestion_metadata" in data

        asyncio.run(_run())
