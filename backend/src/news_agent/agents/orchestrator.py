"""News Intelligence Orchestrator: coordinates specialist agents across data flow, error handling, and truth gating."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
from typing import Any

from news_agent.agents.briefing import NewsBriefingAgent
from news_agent.agents.fact_checker import FactCheckingAgent, FactCheckingResult
from news_agent.agents.researcher import NewsResearcherAgent, ResearcherResult
from news_agent.agents.social import SocialContentAgent
from news_agent.llm import BaseLLMClient, create_llm_client
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now
from news_agent.models.brief import NewsBrief
from news_agent.models.claim import FactualClaim
from news_agent.models.social import SocialMediaDraft
from news_agent.models.verification import VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Consolidated, structured output from the multi-agent news intelligence pipeline."""

    ingestion_metadata: dict[str, Any]
    articles: list[NormalizedArticle]
    claims: list[FactualClaim]
    verification_results: list[VerificationResult]
    brief: NewsBrief | None
    social_drafts: list[SocialMediaDraft]
    errors: list[dict[str, Any]] = field(default_factory=list)
    completed_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to machine-readable dictionary format."""
        return {
            "ingestion_metadata": self.ingestion_metadata,
            "articles_count": len(self.articles),
            "claims_count": len(self.claims),
            "verification_results_count": len(self.verification_results),
            "brief": self.brief.model_dump(mode="json") if self.brief else None,
            "social_drafts": [d.model_dump(mode="json") for d in self.social_drafts],
            "errors": self.errors,
            "completed_at": self.completed_at.isoformat(),
        }


class NewsIntelligenceOrchestrator:
    """Coordinates specialist agents (Researcher, Fact-Checker, Briefing, Social) with robust control flow."""

    def __init__(
        self,
        researcher: NewsResearcherAgent | None = None,
        fact_checker: FactCheckingAgent | None = None,
        briefing_agent: NewsBriefingAgent | None = None,
        social_agent: SocialContentAgent | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.llm_client = llm_client or create_llm_client()
        self.researcher = researcher or NewsResearcherAgent()
        self.fact_checker = fact_checker or FactCheckingAgent(llm_client=self.llm_client)
        self.briefing_agent = briefing_agent or NewsBriefingAgent(llm_client=self.llm_client)
        self.social_agent = social_agent or SocialContentAgent(llm_client=self.llm_client)

    def run(
        self,
        lookback_hours: int | None = None,
        platforms: list[str] | None = None,
        reference_time: datetime | None = None,
        max_articles: int | None = None,
    ) -> OrchestratorResult:
        """Execute the complete end-to-end multi-agent news intelligence workflow.

        Steps:
        1. News Researcher collects, filters, and deduplicates news articles.
        2. Fact-Checking Agent extracts atomic claims, searches evidence, and assigns verdicts.
        3. News Briefing Agent synthesizes an executive brief with provenance citations.
        4. Social Content Agent generates platform-validated promotional drafts.
        5. Returns structured OrchestratorResult with all artifacts and recorded error logs.
        """
        all_errors: list[dict[str, Any]] = []
        t_start = time.perf_counter()

        # 1. News Research Phase
        t_res_start = time.perf_counter()
        research_out: ResearcherResult = self.researcher.research_news(
            lookback_hours=lookback_hours,
            reference_time=reference_time,
        )
        res_ms = round((time.perf_counter() - t_res_start) * 1000, 2)
        logger.info("News research stage completed in %.2f ms (retained=%d)", res_ms, research_out.retained_count)

        if research_out.source_errors:
            all_errors.extend(research_out.source_errors)

        ingestion_metadata = {
            "raw_collected_count": research_out.raw_collected_count,
            "after_filter_count": research_out.after_filter_count,
            "retained_count": research_out.retained_count,
            "exact_url_duplicates_removed": research_out.deduplication_report.exact_url_duplicates_removed,
            "near_duplicate_titles_removed": research_out.deduplication_report.near_duplicate_titles_removed,
            "source_errors_count": len(research_out.source_errors),
        }

        # Handle empty articles scenario gracefully without failing downstream
        if not research_out.articles:
            logger.warning("No articles were retained after ingestion and filtering.")
            brief = self.briefing_agent.generate_brief(
                articles=[],
                claims=[],
                verification_results=[],
            )
            ingestion_metadata["stage_timings_ms"] = {
                "research": res_ms,
                "fact_checking": 0.0,
                "briefing": 0.0,
                "social_drafting": 0.0,
                "total": round((time.perf_counter() - t_start) * 1000, 2),
            }
            return OrchestratorResult(
                ingestion_metadata=ingestion_metadata,
                articles=[],
                claims=[],
                verification_results=[],
                brief=brief,
                social_drafts=[],
                errors=all_errors,
            )

        # 2. Execute downstream stages with collected articles (optionally capped for responsive execution)
        articles_to_process = research_out.articles
        if max_articles is not None and max_articles > 0:
            articles_to_process = articles_to_process[:max_articles]

        return self._execute_downstream(
            articles=articles_to_process,
            ingestion_metadata=ingestion_metadata,
            errors=all_errors,
            platforms=platforms,
            t_start=t_start,
            research_ms=res_ms,
            evidence_corpus=research_out.articles,
        )

    def run_with_articles(
        self,
        articles: list[NormalizedArticle],
        platforms: list[str] | None = None,
    ) -> OrchestratorResult:
        """Execute the multi-agent workflow starting from pre-supplied articles.

        Useful for offline tests, manual briefings, and direct MCP tool requests.
        """
        t_start = time.perf_counter()
        ingestion_metadata = {
            "raw_collected_count": len(articles),
            "after_filter_count": len(articles),
            "retained_count": len(articles),
            "exact_url_duplicates_removed": 0,
            "near_duplicate_titles_removed": 0,
            "source_errors_count": 0,
            "mode": "supplied_articles",
        }
        return self._execute_downstream(
            articles=articles,
            ingestion_metadata=ingestion_metadata,
            errors=[],
            platforms=platforms,
            t_start=t_start,
            research_ms=0.0,
        )

    def _execute_downstream(
        self,
        articles: list[NormalizedArticle],
        ingestion_metadata: dict[str, Any],
        errors: list[dict[str, Any]],
        platforms: list[str] | None = None,
        t_start: float | None = None,
        research_ms: float = 0.0,
        evidence_corpus: list[NormalizedArticle] | None = None,
    ) -> OrchestratorResult:
        """Execute Fact-Checking, Briefing, and Social Content generation with error resilience and latency logging."""
        claims: list[FactualClaim] = []
        verification_results: list[VerificationResult] = []
        brief: NewsBrief | None = None
        social_drafts: list[SocialMediaDraft] = []
        overall_start = t_start if t_start is not None else time.perf_counter()

        # 2. Fact-Checking Phase
        t_fc_start = time.perf_counter()
        try:
            fc_result: FactCheckingResult = self.fact_checker.check_articles(
                articles=articles,
                evidence_corpus=evidence_corpus,
            )
            claims = fc_result.claims
            verification_results = fc_result.verification_results
        except Exception as err:
            logger.exception("Fact-checking stage failed: %s", err)
            errors.append({"stage": "fact_checking", "error": str(err)})
        fc_ms = round((time.perf_counter() - t_fc_start) * 1000, 2)
        logger.info("Fact-checking stage completed in %.2f ms (claims=%d, verifications=%d)", fc_ms, len(claims), len(verification_results))

        # 3. News Briefing Phase
        t_br_start = time.perf_counter()
        try:
            brief = self.briefing_agent.generate_brief(
                articles=articles,
                claims=claims,
                verification_results=verification_results,
            )
        except Exception as err:
            logger.exception("News briefing stage failed: %s", err)
            errors.append({"stage": "news_briefing", "error": str(err)})
        br_ms = round((time.perf_counter() - t_br_start) * 1000, 2)
        logger.info("News briefing stage completed in %.2f ms", br_ms)

        # 4. Social Content Phase
        t_so_start = time.perf_counter()
        if brief is not None:
            try:
                social_drafts = self.social_agent.generate_drafts(
                    brief=brief,
                    articles=articles,
                    verification_results=verification_results,
                    platforms=platforms,
                )
            except Exception as err:
                logger.exception("Social drafting stage failed: %s", err)
                errors.append({"stage": "social_drafting", "error": str(err)})
        so_ms = round((time.perf_counter() - t_so_start) * 1000, 2)
        logger.info("Social drafting stage completed in %.2f ms (drafts=%d)", so_ms, len(social_drafts))

        total_ms = round((time.perf_counter() - overall_start) * 1000, 2)
        ingestion_metadata["stage_timings_ms"] = {
            "research": research_ms,
            "fact_checking": fc_ms,
            "briefing": br_ms,
            "social_drafting": so_ms,
            "total": total_ms,
        }

        return OrchestratorResult(
            ingestion_metadata=ingestion_metadata,
            articles=articles,
            claims=claims,
            verification_results=verification_results,
            brief=brief,
            social_drafts=social_drafts,
            errors=errors,
        )
