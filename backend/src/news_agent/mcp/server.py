"""Model Context Protocol (MCP) server for Local News Intelligence Agent.

Exposes typed, machine-readable tools using the official Python MCP SDK over stdio transport.
All diagnostics and logs are strictly redirected to stderr so stdout remains clean for protocol frames.
"""

from datetime import datetime, timezone
import logging
import sys
from typing import Any
from uuid import uuid4

from mcp.server.mcpserver import MCPServer

from news_agent.agents.orchestrator import NewsIntelligenceOrchestrator, OrchestratorResult
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now
from news_agent.models.brief import NewsBrief
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.social import SocialMediaDraft
from news_agent.models.verification import VerificationResult

# Ensure logging outputs to stderr exclusively
logger = logging.getLogger("news_agent.mcp")


class MCPNewsState:
    """In-memory cache preserving articles, claims, and briefs across MCP tool invocations."""

    def __init__(self) -> None:
        self.articles: dict[str, NormalizedArticle] = {}
        self.claims: dict[str, FactualClaim] = {}
        self.briefs: dict[str, NewsBrief] = {}
        self.verification_results: dict[str, VerificationResult] = {}
        self.latest_result: OrchestratorResult | None = None

    def store_articles(self, articles: list[NormalizedArticle]) -> None:
        for a in articles:
            self.articles[a.id] = a

    def store_claims(self, claims: list[FactualClaim]) -> None:
        for c in claims:
            self.claims[c.id] = c

    def store_results(self, results: list[VerificationResult]) -> None:
        for r in results:
            self.verification_results[r.claim_id] = r

    def store_brief(self, brief: NewsBrief) -> None:
        self.briefs[brief.id] = brief


def create_mcp_server(
    orchestrator: NewsIntelligenceOrchestrator | None = None,
    server_name: str = "news-intelligence-server",
) -> MCPServer:
    """Factory creating and configuring the official MCP server with news intelligence tools.

    All tool implementations reuse the core multi-agent orchestrator and specialist agents.
    """
    orch = orchestrator or NewsIntelligenceOrchestrator()
    state = MCPNewsState()
    server = MCPServer(server_name)

    # 1. Tool: search_news
    @server.tool()
    def search_news(lookback_hours: int = 24) -> dict[str, Any]:
        """Collect and deduplicate recent news articles from configured sources.

        Args:
            lookback_hours: Time-window in hours for filtering articles (default: 24).

        Returns:
            Structured dictionary with collected articles, counts, and source error logs.
        """
        try:
            res = orch.researcher.research_news(lookback_hours=lookback_hours)
            state.store_articles(res.articles)
            return {
                "count": len(res.articles),
                "articles": [a.model_dump(mode="json") for a in res.articles],
                "raw_collected_count": res.raw_collected_count,
                "after_filter_count": res.after_filter_count,
                "exact_duplicates_removed": res.deduplication_report.exact_url_duplicates_removed,
                "near_duplicates_removed": res.deduplication_report.near_duplicate_titles_removed,
                "source_errors": res.source_errors,
            }
        except Exception as err:
            logger.exception("Error executing search_news tool: %s", err)
            return {"error": f"Failed to search news: {str(err)}", "count": 0, "articles": []}

    # 2. Tool: get_article
    @server.tool()
    def get_article(article_id: str) -> dict[str, Any]:
        """Retrieve a collected article by its unique identifier.

        Args:
            article_id: The ID of the article to retrieve.

        Returns:
            Normalized article data or an error description.
        """
        if not article_id or not article_id.strip():
            return {"error": "Invalid article_id provided."}

        article = state.articles.get(article_id.strip())
        if not article:
            return {
                "error": f"Article '{article_id}' not found in active corpus.",
                "available_article_ids": list(state.articles.keys())[:10],
            }
        return {"article": article.model_dump(mode="json")}

    # 3. Tool: extract_claims
    @server.tool()
    def extract_claims(
        article_id: str | None = None,
        article_text: str | None = None,
        title: str = "External Article",
        publisher: str = "External Source",
    ) -> dict[str, Any]:
        """Extract atomic, verifiable factual claims from an article.

        Args:
            article_id: ID of an article in the active corpus (optional).
            article_text: Raw text of an article if not in corpus (optional).
            title: Headline of the article if supplying raw text.
            publisher: Publisher name if supplying raw text.

        Returns:
            Extracted atomic claims with confidence estimates.
        """
        target_article: NormalizedArticle | None = None
        if article_id:
            target_article = state.articles.get(article_id.strip())
            if not target_article:
                return {"error": f"Article ID '{article_id}' not found in cache."}
        elif article_text:
            clean_text = article_text.strip()
            if len(clean_text) < 15:
                return {"error": "article_text must contain at least 15 characters of content."}
            target_article = NormalizedArticle(
                id=f"art-temp-{uuid4().hex[:8]}",
                title=title,
                content=clean_text,
                publisher=publisher,
                source_url="https://local.mcp/article",  # type: ignore[arg-type]
                published_at=utc_now(),
            )
        else:
            return {"error": "Must provide either article_id or article_text."}

        try:
            claims = orch.fact_checker.extract_claims(target_article)
            state.store_claims(claims)
            return {
                "article_id": target_article.id,
                "count": len(claims),
                "claims": [c.model_dump(mode="json") for c in claims],
            }
        except Exception as err:
            logger.exception("Error extracting claims: %s", err)
            return {"error": f"Claim extraction failed: {str(err)}", "claims": []}

    # 4. Tool: retrieve_evidence
    @server.tool()
    def retrieve_evidence(
        claim_text: str,
        top_k: int = 3,
        originating_article_id: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve ranked evidence passages for a factual claim from indexed articles.

        Args:
            claim_text: The factual assertion to search corroboration/refutation for.
            top_k: Maximum number of passages to return (default: 3).
            originating_article_id: Article ID to isolate and exclude from corroboration.

        Returns:
            Ranked evidence passages with lexical relevance scores.
        """
        if not claim_text or not claim_text.strip():
            return {"error": "claim_text cannot be empty."}

        claim_obj = FactualClaim(
            id=f"c-mcp-{uuid4().hex[:8]}",
            article_id=originating_article_id or "external",
            claim_text=claim_text.strip(),
        )

        articles_list = list(state.articles.values())
        if not articles_list:
            return {
                "claim_text": claim_text,
                "evidence_passages": [],
                "count": 0,
                "message": "Corpus is empty. Please run 'search_news' first to index articles.",
            }

        evidence = orch.fact_checker.retrieve_evidence(
            claim=claim_obj,
            articles=articles_list,
            top_k=top_k,
            exclude_own_article=bool(originating_article_id),
        )

        return {
            "claim_text": claim_text,
            "count": len(evidence),
            "evidence_passages": [p.model_dump(mode="json") for p in evidence],
        }

    # 5. Tool: verify_claim
    @server.tool()
    def verify_claim(
        claim_text: str,
        evidence_passage_texts: list[str] | None = None,
        claim_id: str | None = None,
    ) -> dict[str, Any]:
        """Verify a factual claim against provided evidence passages.

        Preserves discrete verdicts: 'supported', 'contradicted', or 'insufficient_evidence'.

        Args:
            claim_text: The assertion to verify.
            evidence_passage_texts: List of candidate evidence texts to evaluate against.
            claim_id: Optional claim identifier for tracking provenance.

        Returns:
            VerificationResult domain model with verdict, confidence, and reasoning.
        """
        if not claim_text or not claim_text.strip():
            return {"error": "claim_text cannot be empty."}

        claim_obj = state.claims.get(claim_id) if claim_id else None
        if not claim_obj:
            claim_obj = FactualClaim(
                id=claim_id or f"c-mcp-{uuid4().hex[:8]}",
                article_id="mcp_direct",
                claim_text=claim_text.strip(),
            )

        # Convert raw texts into EvidencePassage contracts if provided
        passages: list[EvidencePassage] = []
        if evidence_passage_texts:
            for idx, text in enumerate(evidence_passage_texts):
                passages.append(
                    EvidencePassage(
                        id=f"ev-mcp-{idx}",
                        passage_text=text.strip(),
                        source_url="https://local.mcp/evidence",  # type: ignore[arg-type]
                        publisher="Supplied Evidence",
                        relevance_score=1.0,
                    )
                )
        else:
            # Auto-retrieve from cached corpus if available
            articles_list = list(state.articles.values())
            if articles_list:
                passages = orch.fact_checker.retrieve_evidence(
                    claim=claim_obj,
                    articles=articles_list,
                    top_k=3,
                    exclude_own_article=False,
                )

        result = orch.fact_checker.verify_claim(claim=claim_obj, evidence_passages=passages)
        state.store_results([result])
        return {"verification_result": result.model_dump(mode="json")}

    # 6. Tool: generate_news_brief
    @server.tool()
    def generate_news_brief(article_ids: list[str] | None = None) -> dict[str, Any]:
        """Synthesize an evidence-cited executive news brief from active corpus articles.

        Args:
            article_ids: Optional list of article IDs to include (defaults to all cached articles).

        Returns:
            Structured NewsBrief with takeaways, synthesized body, and verification summary.
        """
        if article_ids:
            target_articles = [state.articles[aid] for aid in article_ids if aid in state.articles]
        else:
            target_articles = list(state.articles.values())

        if not target_articles:
            return {
                "error": "No articles available to synthesize brief. Run 'search_news' first or provide valid article_ids."
            }

        # Run fact checking for articles
        fc_res = orch.fact_checker.check_articles(target_articles)
        state.store_claims(fc_res.claims)
        state.store_results(fc_res.verification_results)

        brief = orch.briefing_agent.generate_brief(
            articles=target_articles,
            claims=fc_res.claims,
            verification_results=fc_res.verification_results,
        )
        state.store_brief(brief)
        return {"brief": brief.model_dump(mode="json")}

    # 7. Tool: generate_social_drafts
    @server.tool()
    def generate_social_drafts(
        brief_id: str | None = None,
        platforms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate platform-specific promotional drafts with deterministic character limit enforcement.

        Args:
            brief_id: Optional ID of an existing brief in cache.
            platforms: List of target platforms (e.g., ['x_twitter', 'linkedin']).

        Returns:
            List of platform drafts with validated character counts and truth gating.
        """
        target_brief: NewsBrief | None = None
        if brief_id:
            target_brief = state.briefs.get(brief_id)
            if not target_brief:
                return {"error": f"Brief ID '{brief_id}' not found in cache."}
        elif state.briefs:
            # Pick latest brief
            target_brief = list(state.briefs.values())[-1]
        else:
            # Automatically synthesize brief from active articles if present
            articles_list = list(state.articles.values())
            if not articles_list:
                return {"error": "No brief or articles found. Run 'run_news_intelligence' or 'search_news' first."}
            brief_resp = generate_news_brief()
            if "brief" in brief_resp:
                target_brief = NewsBrief.model_validate(brief_resp["brief"])
            else:
                return {"error": "Could not generate parent brief for drafts."}

        target_articles = [state.articles[aid] for aid in target_brief.source_article_ids if aid in state.articles]
        if not target_articles:
            target_articles = list(state.articles.values())

        results_list = list(state.verification_results.values())
        drafts = orch.social_agent.generate_drafts(
            brief=target_brief,
            articles=target_articles,
            verification_results=results_list,
            platforms=platforms or ["x_twitter", "linkedin"],
        )

        return {
            "count": len(drafts),
            "drafts": [d.model_dump(mode="json") for d in drafts],
        }

    # 8. Tool: run_news_intelligence
    @server.tool()
    def run_news_intelligence(
        lookback_hours: int = 24,
        platforms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute the complete multi-agent news intelligence pipeline end-to-end.

        Performs research, fact-checking, executive briefing, and social draft generation.

        Args:
            lookback_hours: Ingestion lookback filter window (default: 24).
            platforms: Target social channels (default: ['x_twitter', 'linkedin']).

        Returns:
            Full structured OrchestratorResult containing all artifacts and execution audit.
        """
        try:
            result = orch.run(
                lookback_hours=lookback_hours,
                platforms=platforms or ["x_twitter", "linkedin"],
            )
            state.store_articles(result.articles)
            state.store_claims(result.claims)
            state.store_results(result.verification_results)
            if result.brief:
                state.store_brief(result.brief)
            state.latest_result = result
            return result.to_dict()
        except Exception as err:
            logger.exception("Orchestration workflow failed: %s", err)
            return {"error": f"Orchestrator execution error: {str(err)}"}

    return server


def run_mcp_server(orchestrator: NewsIntelligenceOrchestrator | None = None) -> None:
    """Entry point to launch the MCP server over standard stdio transport."""
    # Configure logging strictly to stderr to keep stdout clean for JSON-RPC MCP messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Initializing Local News Intelligence MCP Server (stdio transport)...")
    server = create_mcp_server(orchestrator=orchestrator)
    logger.info("Starting MCP stdio event loop...")
    server.run(transport="stdio")


def main() -> None:
    """CLI script entrypoint for running the MCP server."""
    run_mcp_server()


if __name__ == "__main__":
    main()
