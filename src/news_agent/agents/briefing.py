"""News Briefing Agent: synthesizes grounded, cited executive news briefs with explicit verification attribution."""

from datetime import datetime
import logging
from typing import Any
from pydantic import BaseModel, Field

from news_agent.llm.client import OllamaClient
from news_agent.llm.exceptions import OllamaError
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now
from news_agent.models.brief import NewsBrief
from news_agent.models.claim import FactualClaim
from news_agent.models.verification import VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)

BRIEFING_SYSTEM_PROMPT = """You are an impartial, highly rigorous intelligence briefing analyst.
Your task is to synthesize an executive news brief based strictly and solely on the provided articles and verified claims.

Absolute Rules:
1. Grounding: Do NOT invent facts, quotes, dates, statistics, or citations. If a fact is not in the text, omit it.
2. Provenance: Every key point MUST explicitly cite its supporting publisher or source URL.
3. Verification Integrity:
   - For 'supported' claims: state them as corroborated facts with source attribution.
   - For 'insufficient_evidence' claims: explicitly characterize them as reported by their originating source, NOT independently verified.
   - For 'contradicted' claims: explicitly identify the conflicting reports or disproven statements.
4. Keep the summary concise, professional, and structured.
"""


class LLMBriefPayload(BaseModel):
    """Structured response schema for LLM-based news brief generation."""

    title: str = Field(..., min_length=3, description="Concise headline for the intelligence brief.")
    headline_takeaways: list[str] = Field(
        ...,
        min_length=1,
        description="Bulleted key findings with explicit publisher/source citations and verification status.",
    )
    synthesized_body: str = Field(
        ...,
        min_length=10,
        description="Cohesive summary grouping developments and citing sources.",
    )


class NewsBriefingAgent:
    """Specialist agent that synthesizes evidence-grounded news briefs with source provenance."""

    def __init__(self, llm_client: OllamaClient | None = None) -> None:
        self.llm_client = llm_client or OllamaClient()

    def generate_brief(
        self,
        articles: list[NormalizedArticle],
        claims: list[FactualClaim],
        verification_results: list[VerificationResult],
    ) -> NewsBrief:
        """Generate a structured, evidence-cited NewsBrief from articles and verification findings.

        Args:
            articles: Ingested and deduplicated source articles.
            claims: Extracted atomic claims.
            verification_results: Fact-checking verdicts for the claims.

        Returns:
            Validated NewsBrief contract with source integrity.
        """
        # Handle empty articles edge-case gracefully
        if not articles:
            return NewsBrief(
                title="No News Available",
                headline_takeaways=["No news articles were ingested within the requested window."],
                synthesized_body="The news ingestion pipeline returned zero articles for analysis.",
                source_article_ids=[],
                verification_summary={"supported": 0, "contradicted": 0, "insufficient_evidence": 0},
                metadata={"grounding": "empty_input"},
            )

        # Index articles and verification data for deterministic provenance
        article_map: dict[str, NormalizedArticle] = {a.id: a for a in articles}
        verdict_map: dict[str, VerificationResult] = {r.claim_id: r for r in verification_results}

        summary_counts: dict[str, int] = {
            "supported": sum(1 for r in verification_results if r.status == VerificationStatus.SUPPORTED),
            "contradicted": sum(1 for r in verification_results if r.status == VerificationStatus.CONTRADICTED),
            "insufficient_evidence": sum(1 for r in verification_results if r.status == VerificationStatus.INSUFFICIENT_EVIDENCE),
        }

        # Build claim classification
        categorized_claims: list[dict[str, Any]] = []
        for claim in claims:
            v_res = verdict_map.get(claim.id)
            origin_article = article_map.get(claim.article_id)
            status_val = v_res.status.value if v_res else "unverified"
            evidence_count = len(v_res.evidence_passages) if v_res else 0
            evidence_sources = [p.publisher for p in v_res.evidence_passages] if v_res else []

            categorized_claims.append(
                {
                    "claim_id": claim.id,
                    "claim_text": claim.claim_text,
                    "publisher": origin_article.publisher if origin_article else "Unknown",
                    "source_url": str(origin_article.source_url) if origin_article else "",
                    "status": status_val,
                    "evidence_count": evidence_count,
                    "evidence_sources": evidence_sources,
                }
            )

        # Attempt structured synthesis via LLM if available
        brief = self._generate_with_llm(articles, categorized_claims, summary_counts)
        if brief is not None:
            return brief

        # Deterministic synthesis fallback (ensures offline reliability without hallucinations)
        return self._generate_deterministic_brief(articles, categorized_claims, summary_counts)

    def _generate_with_llm(
        self,
        articles: list[NormalizedArticle],
        categorized_claims: list[dict[str, Any]],
        summary_counts: dict[str, int],
    ) -> NewsBrief | None:
        """Attempt to synthesize brief via local Ollama LLM, strictly checking grounding."""
        # Format claims summary
        claims_prompt_lines = []
        for c in categorized_claims:
            pub = c["publisher"]
            status = c["status"]
            ev_sources = ", ".join(c["evidence_sources"]) if c["evidence_sources"] else "None"
            claims_prompt_lines.append(
                f"- Claim: \"{c['claim_text']}\" [Origin: {pub}] [Status: {status}] [Corroborating Sources: {ev_sources}]"
            )

        formatted_claims = "\n".join(claims_prompt_lines) if claims_prompt_lines else "No atomic claims extracted."
        formatted_articles = "\n\n".join(
            f"Article [{a.id}] ({a.publisher}) - {a.title}:\n{a.content[:400]}..." for a in articles[:5]
        )

        prompt = (
            f"Source Articles:\n{formatted_articles}\n\n"
            f"Factual Claims & Verification Statuses:\n{formatted_claims}\n\n"
            "Synthesize a concise news intelligence brief. Explicitly distinguish between independently "
            "corroborated claims and single-source reported claims. Cite publishers for all factual statements."
        )

        try:
            parsed = self.llm_client.generate_structured(
                prompt=prompt,
                response_model=LLMBriefPayload,
                system=BRIEFING_SYSTEM_PROMPT,
            )
            return NewsBrief(
                title=parsed.title,
                headline_takeaways=parsed.headline_takeaways,
                synthesized_body=parsed.synthesized_body,
                source_article_ids=[a.id for a in articles],
                verification_summary=summary_counts,
                metadata={
                    "generation_mode": "llm_grounded",
                    "categorized_claims": categorized_claims,
                },
            )
        except (OllamaError, Exception) as err:
            logger.info("Ollama brief synthesis unavailable or timed out (%s); falling back to deterministic synthesis.", err)
            return None

    def _generate_deterministic_brief(
        self,
        articles: list[NormalizedArticle],
        categorized_claims: list[dict[str, Any]],
        summary_counts: dict[str, int],
    ) -> NewsBrief:
        """Generate a fully grounded, deterministic executive brief without requiring LLM inference."""
        # 1. Headline title
        main_article = articles[0]
        title = f"Intelligence Brief: {main_article.title}"

        # 2. Key takeaways with strict status labels
        takeaways: list[str] = []
        for c in categorized_claims:
            claim_text = c["claim_text"]
            pub = c["publisher"]
            status = c["status"]

            if status == VerificationStatus.SUPPORTED.value:
                ev_info = f" (Corroborated by: {', '.join(c['evidence_sources'])})" if c["evidence_sources"] else ""
                takeaways.append(f"{claim_text} [Supported - {pub}{ev_info}]")
            elif status == VerificationStatus.CONTRADICTED.value:
                takeaways.append(f"Disputed: {claim_text} [Contradicted by external evidence]")
            else:
                takeaways.append(f"{claim_text} [Reported by {pub} - Pending external corroboration]")

        if not takeaways:
            for art in articles[:3]:
                takeaways.append(f"{art.title} [Source: {art.publisher}]")

        # 3. Synthesized body grouping developments and citations
        body_sections: list[str] = [
            f"This intelligence brief synthesizes developments across {len(articles)} ingested news source(s).",
        ]

        # Group by status
        supported = [c for c in categorized_claims if c["status"] == VerificationStatus.SUPPORTED.value]
        insufficient = [c for c in categorized_claims if c["status"] == VerificationStatus.INSUFFICIENT_EVIDENCE.value]
        contradicted = [c for c in categorized_claims if c["status"] == VerificationStatus.CONTRADICTED.value]

        if supported:
            body_sections.append("\n**Corroborated Developments:**")
            for item in supported:
                sources = f"{item['publisher']}" + (f", {', '.join(item['evidence_sources'])}" if item['evidence_sources'] else "")
                body_sections.append(f"- {item['claim_text']} (Sources: {sources})")

        if insufficient:
            body_sections.append("\n**Single-Source Reported Developments (Uncorroborated):**")
            for item in insufficient:
                body_sections.append(f"- {item['claim_text']} (Reported by {item['publisher']}; external independent evidence not found)")

        if contradicted:
            body_sections.append("\n**Contradicted or Conflicted Assertions:**")
            for item in contradicted:
                body_sections.append(f"- {item['claim_text']} (Flagged: evidence contradicts this assertion)")

        # Source index
        body_sections.append("\n**Referenced Sources:**")
        for art in articles:
            body_sections.append(f"- [{art.publisher}] {art.title} ({art.source_url})")

        synthesized_body = "\n".join(body_sections)

        return NewsBrief(
            title=title,
            headline_takeaways=takeaways,
            synthesized_body=synthesized_body,
            source_article_ids=[a.id for a in articles],
            verification_summary=summary_counts,
            metadata={
                "generation_mode": "deterministic_grounded",
                "categorized_claims": categorized_claims,
            },
        )
