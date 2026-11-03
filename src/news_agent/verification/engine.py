"""End-to-end verification engine orchestrating claim extraction, retrieval, and verification."""

from dataclasses import dataclass, field
from datetime import datetime
import logging

from news_agent.llm import BaseLLMClient, create_llm_client
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now
from news_agent.models.claim import FactualClaim
from news_agent.models.verification import VerificationResult, VerificationStatus
from news_agent.verification.extractor import ClaimExtractor
from news_agent.verification.retriever import EvidenceRetriever
from news_agent.verification.verifier import ClaimVerifier

logger = logging.getLogger(__name__)


@dataclass
class PipelineVerificationSummary:
    """Summary of verification outcomes across a collection of claims."""

    total_claims: int = 0
    supported_count: int = 0
    contradicted_count: int = 0
    insufficient_evidence_count: int = 0
    claims: list[FactualClaim] = field(default_factory=list)
    results: list[VerificationResult] = field(default_factory=list)
    completed_at: datetime = field(default_factory=utc_now)


class VerificationEngine:
    """Orchestrates end-to-end verification workflow across articles and claims."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        extractor: ClaimExtractor | None = None,
        verifier: ClaimVerifier | None = None,
    ) -> None:
        self.llm_client = llm_client or create_llm_client()
        self.extractor = extractor or ClaimExtractor(llm_client=self.llm_client)
        self.verifier = verifier or ClaimVerifier(llm_client=self.llm_client)

    def verify_articles(
        self,
        articles: list[NormalizedArticle],
        top_k_evidence: int = 3,
        exclude_own_article: bool = True,
        evidence_corpus: list[NormalizedArticle] | None = None,
    ) -> PipelineVerificationSummary:
        """Run the end-to-end verification engine over a set of ingested articles.

        Steps:
        1. Segment and index all articles in the corpus for lexical retrieval.
        2. Extract atomic verifiable claims from each article.
        3. Retrieve ranked, deduplicated evidence passages for each claim.
        4. Interpret evidence and generate discrete verification results.
        5. Compile and return a PipelineVerificationSummary.
        """
        summary = PipelineVerificationSummary()
        if not articles:
            return summary

        # 1. Build retrieval index across all available articles (use evidence_corpus if supplied)
        corpus = evidence_corpus if evidence_corpus is not None else articles
        retriever = EvidenceRetriever.from_articles(corpus)

        # 2. Extract claims from all articles
        all_claims: list[FactualClaim] = []
        for article in articles:
            claims = self.extractor.extract_claims(article)
            all_claims.extend(claims)

        summary.total_claims = len(all_claims)
        summary.claims = all_claims

        # 3. Retrieve evidence and verify each claim
        for claim in all_claims:
            evidence = retriever.retrieve_evidence(
                claim=claim,
                top_k=top_k_evidence,
                exclude_own_article=exclude_own_article,
            )

            result = self.verifier.verify_claim(claim=claim, evidence_passages=evidence)
            summary.results.append(result)

            if result.status == VerificationStatus.SUPPORTED:
                summary.supported_count += 1
            elif result.status == VerificationStatus.CONTRADICTED:
                summary.contradicted_count += 1
            else:
                summary.insufficient_evidence_count += 1

        return summary
