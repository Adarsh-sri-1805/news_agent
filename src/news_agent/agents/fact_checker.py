"""Fact-Checking Agent: responsible for claim extraction, evidence retrieval, and discrete fact verification."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any

from news_agent.llm.client import OllamaClient
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.verification import VerificationResult, VerificationStatus
from news_agent.verification.engine import PipelineVerificationSummary, VerificationEngine
from news_agent.verification.extractor import ClaimExtractor
from news_agent.verification.retriever import EvidenceRetriever
from news_agent.verification.verifier import ClaimVerifier

logger = logging.getLogger(__name__)


@dataclass
class FactCheckingResult:
    """Structured output returned by the Fact-Checking Agent."""

    claims: list[FactualClaim]
    verification_results: list[VerificationResult]
    provenance_map: dict[str, dict[str, Any]]
    summary: dict[str, int]
    completed_at: datetime = field(default_factory=utc_now)


class FactCheckingAgent:
    """Specialist agent that extracts factual assertions and verifies them against retrieved evidence."""

    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        verification_engine: VerificationEngine | None = None,
        extractor: ClaimExtractor | None = None,
        verifier: ClaimVerifier | None = None,
    ) -> None:
        self.llm_client = llm_client or OllamaClient()
        self.extractor = extractor or ClaimExtractor(llm_client=self.llm_client)
        self.verifier = verifier or ClaimVerifier(llm_client=self.llm_client)
        self.verification_engine = verification_engine or VerificationEngine(
            llm_client=self.llm_client,
            extractor=self.extractor,
            verifier=self.verifier,
        )

    def extract_claims(self, article: NormalizedArticle) -> list[FactualClaim]:
        """Extract atomic, verifiable claims from a single normalized article."""
        return self.extractor.extract_claims(article)

    def retrieve_evidence(
        self,
        claim: FactualClaim,
        articles: list[NormalizedArticle] | None = None,
        top_k: int = 3,
        exclude_own_article: bool = True,
    ) -> list[EvidencePassage]:
        """Retrieve candidate evidence passages from a corpus for a specific claim.

        Note: High relevance scores reflect lexical term overlap, NOT truth probabilities.
        A claim must be evaluated by the verifier before any verification status is assigned.
        """
        if not articles:
            return []
        retriever = EvidenceRetriever.from_articles(articles)
        return retriever.retrieve_evidence(
            claim=claim,
            top_k=top_k,
            exclude_own_article=exclude_own_article,
        )

    def verify_claim(
        self,
        claim: FactualClaim,
        evidence_passages: list[EvidencePassage],
    ) -> VerificationResult:
        """Verify a single claim against supplied evidence passages.

        Preserves discrete verdicts: supported, contradicted, or insufficient_evidence.
        Never upgrades a claim to supported merely because evidence was retrieved.
        """
        return self.verifier.verify_claim(claim=claim, evidence_passages=evidence_passages)

    def check_articles(
        self,
        articles: list[NormalizedArticle],
        top_k_evidence: int = 3,
        exclude_own_article: bool = True,
    ) -> FactCheckingResult:
        """Run the end-to-end fact-checking pipeline across all supplied articles.

        Builds complete provenance linking claims to originating articles,
        source URLs, and retrieved evidence passages.
        """
        if not articles:
            return FactCheckingResult(
                claims=[],
                verification_results=[],
                provenance_map={},
                summary={
                    "supported": 0,
                    "contradicted": 0,
                    "insufficient_evidence": 0,
                },
            )

        # Run VerificationEngine
        pipeline_summary: PipelineVerificationSummary = self.verification_engine.verify_articles(
            articles=articles,
            top_k_evidence=top_k_evidence,
            exclude_own_article=exclude_own_article,
        )

        # Map article metadata for provenance lookups
        article_lookup: dict[str, NormalizedArticle] = {a.id: a for a in articles}

        # Build provenance map
        provenance_map: dict[str, dict[str, Any]] = {}
        for result in pipeline_summary.results:
            # Find the corresponding claim
            matched_claim = next((c for c in pipeline_summary.claims if c.id == result.claim_id), None)
            origin_article = article_lookup.get(matched_claim.article_id) if matched_claim else None

            provenance_map[result.claim_id] = {
                "claim_id": result.claim_id,
                "claim_text": matched_claim.claim_text if matched_claim else "",
                "article_id": matched_claim.article_id if matched_claim else "",
                "publisher": origin_article.publisher if origin_article else "Unknown",
                "source_url": str(origin_article.source_url) if origin_article else "",
                "status": result.status.value,
                "confidence_score": result.confidence_score,
                "reasoning": result.reasoning,
                "evidence_passage_ids": [p.id for p in result.evidence_passages],
                "evidence_sources": [
                    {
                        "passage_id": p.id,
                        "source_url": str(p.source_url),
                        "publisher": p.publisher,
                        "relevance_score": p.relevance_score,
                    }
                    for p in result.evidence_passages
                ],
            }

        return FactCheckingResult(
            claims=pipeline_summary.claims,
            verification_results=pipeline_summary.results,
            provenance_map=provenance_map,
            summary={
                "supported": pipeline_summary.supported_count,
                "contradicted": pipeline_summary.contradicted_count,
                "insufficient_evidence": pipeline_summary.insufficient_evidence_count,
            },
            completed_at=pipeline_summary.completed_at,
        )
