"""Unit tests for NewsResearcherAgent: ingestion, deduplication, filtering, and source failure reporting."""

from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest

from news_agent.agents.researcher import NewsResearcherAgent, ResearcherResult
from news_agent.ingestion.adapters.base import BaseNewsSourceAdapter
from news_agent.ingestion.dedup import DeduplicationReport
from news_agent.ingestion.pipeline import IngestionPipeline, IngestionResult
from news_agent.models.article import NormalizedArticle


def make_test_article(id_val: str, title: str, publisher: str = "Test Times") -> NormalizedArticle:
    return NormalizedArticle(
        id=id_val,
        title=title,
        content="Sample news content for research testing purposes with sufficient length.",
        publisher=publisher,
        source_url=f"https://example.com/{id_val}",  # type: ignore[arg-type]
        published_at=datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc),
    )


class TestNewsResearcherAgent:
    def test_research_news_successful_collection(self):
        art1 = make_test_article("art-1", "Council Approves Budget Plan")
        art2 = make_test_article("art-2", "New Downtown Transit Route Opened")

        mock_pipeline = MagicMock(spec=IngestionPipeline)
        mock_pipeline.lookback_hours = 48
        mock_pipeline.run.return_value = IngestionResult(
            articles=[art1, art2],
            raw_collected_count=4,
            after_filter_count=3,
            deduplication_report=DeduplicationReport(
                exact_url_duplicates_removed=1,
                near_duplicate_titles_removed=0,
                retained_count=2,
            ),
            source_errors=[],
        )

        agent = NewsResearcherAgent(pipeline=mock_pipeline)
        result: ResearcherResult = agent.research_news(lookback_hours=24)

        assert len(result.articles) == 2
        assert result.raw_collected_count == 4
        assert result.after_filter_count == 3
        assert result.retained_count == 2
        assert result.has_errors is False
        assert len(result.source_errors) == 0

    def test_research_news_partial_source_failures_reported(self):
        art1 = make_test_article("art-1", "Solar Project Finalized")

        mock_pipeline = MagicMock(spec=IngestionPipeline)
        mock_pipeline.lookback_hours = 24
        mock_pipeline.run.return_value = IngestionResult(
            articles=[art1],
            raw_collected_count=1,
            after_filter_count=1,
            deduplication_report=DeduplicationReport(retained_count=1),
            source_errors=[
                {
                    "source": "Faulty RSS Feed",
                    "error": "HTTP 503 Service Unavailable",
                    "timestamp": "2026-10-27T12:00:00Z",
                }
            ],
        )

        agent = NewsResearcherAgent(pipeline=mock_pipeline)
        result: ResearcherResult = agent.research_news()

        # Partial failure: successfully returns retained articles AND reports source failure
        assert len(result.articles) == 1
        assert result.has_errors is True
        assert len(result.source_errors) == 1
        assert result.source_errors[0]["source"] == "Faulty RSS Feed"

    def test_research_news_empty_ingestion_results(self):
        mock_pipeline = MagicMock(spec=IngestionPipeline)
        mock_pipeline.lookback_hours = 24
        mock_pipeline.run.return_value = IngestionResult(
            articles=[],
            raw_collected_count=0,
            after_filter_count=0,
            deduplication_report=DeduplicationReport(),
            source_errors=[],
        )

        agent = NewsResearcherAgent(pipeline=mock_pipeline)
        result = agent.research_news()

        assert len(result.articles) == 0
        assert result.raw_collected_count == 0
        assert result.has_errors is False

    def test_research_news_catastrophic_pipeline_exception_handled(self):
        mock_pipeline = MagicMock(spec=IngestionPipeline)
        mock_pipeline.lookback_hours = 24
        mock_pipeline.run.side_effect = RuntimeError("Fatal socket error in pipeline")

        agent = NewsResearcherAgent(pipeline=mock_pipeline)
        result = agent.research_news()

        # Does not crash; reports the pipeline fatal error
        assert len(result.articles) == 0
        assert result.has_errors is True
        assert "Fatal socket error in pipeline" in result.source_errors[0]["error"]
