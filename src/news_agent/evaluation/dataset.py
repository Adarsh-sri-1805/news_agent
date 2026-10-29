"""Deterministic ground-truth evaluation dataset for Phase 6 benchmarking."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from news_agent.models.article import NormalizedArticle
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.verification import VerificationStatus


@dataclass
class VerificationEvalCase:
    """Evaluation test case for claim verification."""

    case_id: str
    claim_text: str
    passages: list[EvidencePassage]
    expected_status: VerificationStatus
    description: str
    category: str  # "supported", "contradicted", "insufficient_evidence", "distractor", "hallucinated_citation"
    simulated_model_output: dict[str, Any] | None = None


@dataclass
class RetrievalEvalQuery:
    """Evaluation query for evidence retrieval."""

    query_id: str
    claim_text: str
    gold_passage_ids: list[str]
    gold_article_ids: list[str]
    description: str


@dataclass
class ExtractionEvalCandidate:
    """Test candidate for claim extraction quality and fragment rejection."""

    candidate_id: str
    raw_text: str
    is_valid_assertion: bool
    fragment_type: str | None = None  # "standalone_entity", "single_number", "date_only", "predicate_missing"
    description: str = ""


@dataclass
class EvaluationDataset:
    """Complete container for Phase 6 deterministic evaluation dataset."""

    verification_cases: list[VerificationEvalCase]
    retrieval_corpus: list[NormalizedArticle]
    retrieval_queries: list[RetrievalEvalQuery]
    extraction_candidates: list[ExtractionEvalCandidate]


def _build_dataset() -> EvaluationDataset:
    # 1. Corpus of Articles for Retrieval & Verification
    base_time = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)

    art1 = NormalizedArticle(
        id="art-bond-001",
        title="Springfield Council Approves $15M Road Resurfacing Bond",
        content=(
            "The Springfield City Council voted 7-2 on Tuesday evening to approve a $15 million municipal bond. "
            "The funding will finance roadway resurfacing across twenty downtown transit corridors over the next two fiscal years. "
            "Council president Maria Alvarez praised the bipartisan vote, noting that critical thoroughfares had gone unpaved for six years."
        ),
        publisher="Springfield Gazette",
        source_url="https://springfieldgazette.com/council-road-bond-15m",  # type: ignore[arg-type]
        published_at=base_time,
    )

    art2 = NormalizedArticle(
        id="art-bond-002-syndicated",
        title="Springfield Council Approves $15M Road Resurfacing Bond",
        content=(
            "The Springfield City Council voted 7-2 on Tuesday evening to approve a $15 million municipal bond. "
            "The funding will finance roadway resurfacing across twenty downtown transit corridors over the next two fiscal years."
        ),
        publisher="Valley News Wire",
        source_url="https://valleynews.org/wire/springfield-road-bond",  # type: ignore[arg-type]
        published_at=base_time,
    )

    art3 = NormalizedArticle(
        id="art-transit-003",
        title="Regional Transit Board Rejects Downtown Light Rail Proposal",
        content=(
            "Following six hours of contentious public testimony, the regional transit authority board voted 5-4 "
            "to reject the $450 million downtown light rail expansion proposal. Board members cited soaring projected construction costs "
            "and insufficient federal matching grants as the primary reasons for turning down the project."
        ),
        publisher="Metro Transit Chronicle",
        source_url="https://metrotransit.org/news/light-rail-rejected-vote",  # type: ignore[arg-type]
        published_at=base_time,
    )

    art4 = NormalizedArticle(
        id="art-health-004",
        title="Health Officials Confirm Three Cases of Avian Flu in County",
        content=(
            "The State Department of Agriculture and Metro Health officially confirmed three cases of avian influenza "
            "in commercial poultry farms across the county this morning. Quarantine perimeters have been established around all affected facilities."
        ),
        publisher="State Agriculture Bulletin",
        source_url="https://agri.state.gov/bulletin/avian-flu-oct-2026",  # type: ignore[arg-type]
        published_at=base_time,
    )

    art5 = NormalizedArticle(
        id="art-distractor-005",
        title="Governor Attends Clean Harbor Conference in Springfield",
        content=(
            "Governor Thomas delivered a keynote address at the Clean Harbor environmental summit in Springfield on Thursday. "
            "The governor discussed broad regional strategies to mitigate urban river runoff, but no legislative bills or accords "
            "were signed during the academic summit."
        ),
        publisher="Capital Daily",
        source_url="https://capitaldaily.com/governor-summit-springfield",  # type: ignore[arg-type]
        published_at=base_time,
    )

    art6 = NormalizedArticle(
        id="art-library-006",
        title="County Auditor Releases Final Costs for Central Library Renovation",
        content=(
            "A comprehensive report by the County Auditor revealed that final expenditures for the new Central Public Library "
            "totaled $8.7 million. Original budget estimates had placed the estimated project cost at $3.2 million."
        ),
        publisher="Civic Ledger",
        source_url="https://civicledger.com/library-audit-8m",  # type: ignore[arg-type]
        published_at=base_time,
    )

    retrieval_corpus = [art1, art2, art3, art4, art5, art6]

    # Passages extracted from corpus
    pass_bond_1 = EvidencePassage(
        id="pass-bond-1",
        source_url=art1.source_url,
        publisher=art1.publisher,
        passage_text="The Springfield City Council voted 7-2 on Tuesday evening to approve a $15 million municipal bond.",
        relevance_score=0.92,
        metadata={"article_id": art1.id, "passage_index": 0},
    )
    pass_bond_synd = EvidencePassage(
        id="pass-bond-2",
        source_url=art2.source_url,
        publisher=art2.publisher,
        passage_text="The Springfield City Council voted 7-2 on Tuesday evening to approve a $15 million municipal bond.",
        relevance_score=0.91,
        metadata={"article_id": art2.id, "passage_index": 0},
    )
    pass_transit_1 = EvidencePassage(
        id="pass-transit-1",
        source_url=art3.source_url,
        publisher=art3.publisher,
        passage_text="Following six hours of contentious public testimony, the regional transit authority board voted 5-4 to reject the $450 million downtown light rail expansion proposal.",
        relevance_score=0.94,
        metadata={"article_id": art3.id, "passage_index": 0},
    )
    pass_health_1 = EvidencePassage(
        id="pass-health-1",
        source_url=art4.source_url,
        publisher=art4.publisher,
        passage_text="The State Department of Agriculture and Metro Health officially confirmed three cases of avian influenza in commercial poultry farms across the county this morning.",
        relevance_score=0.88,
        metadata={"article_id": art4.id, "passage_index": 0},
    )
    pass_distractor_1 = EvidencePassage(
        id="pass-distract-1",
        source_url=art5.source_url,
        publisher=art5.publisher,
        passage_text="Governor Thomas delivered a keynote address at the Clean Harbor environmental summit in Springfield on Thursday.",
        relevance_score=0.65,
        metadata={"article_id": art5.id, "passage_index": 0},
    )
    pass_library_1 = EvidencePassage(
        id="pass-lib-1",
        source_url=art6.source_url,
        publisher=art6.publisher,
        passage_text="A comprehensive report by the County Auditor revealed that final expenditures for the new Central Public Library totaled $8.7 million.",
        relevance_score=0.89,
        metadata={"article_id": art6.id, "passage_index": 0},
    )

    # 2. Verification Test Cases
    verification_cases = [
        # Case 1: Supported directly by municipal reporting
        VerificationEvalCase(
            case_id="ver-sup-01",
            claim_text="Springfield City Council voted 7-2 to approve a $15 million municipal road bond.",
            passages=[pass_bond_1, pass_bond_synd],
            expected_status=VerificationStatus.SUPPORTED,
            description="Council vote directly confirmed by independent and syndicated reporting.",
            category="supported",
            simulated_model_output={
                "status": "supported",
                "confidence_score": 0.95,
                "reasoning": "Both passages explicitly confirm the 7-2 council vote and $15 million municipal bond.",
                "cited_passage_ids": ["pass-bond-1"],
            },
        ),
        # Case 2: Contradicted directly by transit board vote
        VerificationEvalCase(
            case_id="ver-con-01",
            claim_text="The regional transit board unanimously approved the downtown light rail expansion.",
            passages=[pass_transit_1],
            expected_status=VerificationStatus.CONTRADICTED,
            description="Passage directly disproves unanimous approval, confirming a 5-4 rejection vote.",
            category="contradicted",
            simulated_model_output={
                "status": "contradicted",
                "confidence_score": 0.92,
                "reasoning": "The evidence states the board voted 5-4 to reject the proposal, directly contradicting unanimous approval.",
                "cited_passage_ids": ["pass-transit-1"],
            },
        ),
        # Case 3: Contradicted numerical value
        VerificationEvalCase(
            case_id="ver-con-02",
            claim_text="The new Central Public Library construction was completed for under $4 million.",
            passages=[pass_library_1],
            expected_status=VerificationStatus.CONTRADICTED,
            description="Auditor figures prove total cost was $8.7 million, contradicting the under $4 million claim.",
            category="contradicted",
            simulated_model_output={
                "status": "contradicted",
                "confidence_score": 0.90,
                "reasoning": "Auditor report states total expenditures were $8.7 million, directly contradicting 'under $4 million'.",
                "cited_passage_ids": ["pass-lib-1"],
            },
        ),
        # Case 4: Insufficient evidence - Missing evidence entirely
        VerificationEvalCase(
            case_id="ver-ins-01",
            claim_text="The mayor issued a midnight emergency curfew across all municipal districts.",
            passages=[],
            expected_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            description="Empty passage list must deterministically yield insufficient_evidence.",
            category="insufficient_evidence",
            simulated_model_output=None,
        ),
        # Case 5: Insufficient evidence - Distractor with misleading lexical overlap
        VerificationEvalCase(
            case_id="ver-ins-02",
            claim_text="The Governor signed the Clean Harbor Accord into law on Thursday.",
            passages=[pass_distractor_1],
            expected_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            description="High lexical overlap (Governor, Clean Harbor, Thursday), but text notes no bills were signed.",
            category="distractor",
            simulated_model_output={
                "status": "insufficient_evidence",
                "confidence_score": 0.85,
                "reasoning": "Passage confirms attendance at the summit but does not confirm signing an accord into law.",
                "cited_passage_ids": ["pass-distract-1"],
            },
        ),
        # Case 6: Supported - Health agency confirmation
        VerificationEvalCase(
            case_id="ver-sup-02",
            claim_text="Health officials confirmed three cases of avian flu in county commercial poultry.",
            passages=[pass_health_1],
            expected_status=VerificationStatus.SUPPORTED,
            description="State agricultural release explicitly confirms three avian influenza cases.",
            category="supported",
            simulated_model_output={
                "status": "supported",
                "confidence_score": 0.94,
                "reasoning": "Passage explicitly confirms three avian flu cases in commercial poultry farms.",
                "cited_passage_ids": ["pass-health-1"],
            },
        ),
        # Case 7: Hardening guard - Model claims supported but only cites hallucinated unsupplied passage
        VerificationEvalCase(
            case_id="ver-sec-01",
            claim_text="The city launched a zero-emission bus fleet across seven suburban routes.",
            passages=[pass_transit_1],
            expected_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            description="Model claims supported but cites fabricated passage 'pass-ghost-999'. Must demote to insufficient_evidence.",
            category="hallucinated_citation",
            simulated_model_output={
                "status": "supported",
                "confidence_score": 0.88,
                "reasoning": "Supported based on external knowledge.",
                "cited_passage_ids": ["pass-ghost-999"],
            },
        ),
    ]

    # 3. Retrieval Evaluation Queries
    retrieval_queries = [
        RetrievalEvalQuery(
            query_id="ret-q-01",
            claim_text="Springfield council approved a $15 million road bond.",
            gold_passage_ids=["art-bond-001_p0", "art-bond-002-syndicated_p0"],
            gold_article_ids=["art-bond-001", "art-bond-002-syndicated"],
            description="Road bond retrieval test across municipal reporting and syndicated wire.",
        ),
        RetrievalEvalQuery(
            query_id="ret-q-02",
            claim_text="Regional transit board voted to reject the light rail project.",
            gold_passage_ids=["art-transit-003_p0"],
            gold_article_ids=["art-transit-003"],
            description="Light rail vote retrieval test.",
        ),
        RetrievalEvalQuery(
            query_id="ret-q-03",
            claim_text="Confirmed avian flu cases in county poultry farms.",
            gold_passage_ids=["art-health-004_p0"],
            gold_article_ids=["art-health-004"],
            description="Avian influenza health announcement retrieval.",
        ),
        RetrievalEvalQuery(
            query_id="ret-q-04",
            claim_text="Central library renovation costs totaled $8.7 million in county audit.",
            gold_passage_ids=["art-library-006_p0"],
            gold_article_ids=["art-library-006"],
            description="Library expenditures audit retrieval.",
        ),
    ]

    # 4. Extraction Quality & Fragment Filtering Candidates
    extraction_candidates = [
        # Valid assertions
        ExtractionEvalCandidate(
            candidate_id="ext-val-01",
            raw_text="The Springfield City Council voted 7-2 to approve a $15 million municipal bond.",
            is_valid_assertion=True,
            description="Complete atomic factual claim with subject, verb, and numerical predicate.",
        ),
        ExtractionEvalCandidate(
            candidate_id="ext-val-02",
            raw_text="Regional transit authority board rejected the light rail expansion proposal.",
            is_valid_assertion=True,
            description="Complete atomic factual claim.",
        ),
        ExtractionEvalCandidate(
            candidate_id="ext-val-03",
            raw_text="Three cases of avian influenza were confirmed in commercial poultry farms.",
            is_valid_assertion=True,
            description="Complete passive factual claim with quantifiable subject.",
        ),
        ExtractionEvalCandidate(
            candidate_id="ext-val-04",
            raw_text="Final expenditures for the central library renovation reached $8.7 million.",
            is_valid_assertion=True,
            description="Complete financial audit claim.",
        ),
        ExtractionEvalCandidate(
            candidate_id="ext-val-05",
            raw_text="Governor Thomas delivered a keynote address at the Clean Harbor summit.",
            is_valid_assertion=True,
            description="Complete event attendance assertion.",
        ),
        # Poorly formed fragments (must be rejected)
        ExtractionEvalCandidate(
            candidate_id="ext-frag-01",
            raw_text="Fernando Seco",
            is_valid_assertion=False,
            fragment_type="standalone_entity",
            description="Standalone person name without assertion.",
        ),
        ExtractionEvalCandidate(
            candidate_id="ext-frag-02",
            raw_text="Toledo",
            is_valid_assertion=False,
            fragment_type="standalone_entity",
            description="Standalone single-word location name.",
        ),
        ExtractionEvalCandidate(
            candidate_id="ext-frag-03",
            raw_text="68",
            is_valid_assertion=False,
            fragment_type="single_number",
            description="Standalone digit string.",
        ),
        ExtractionEvalCandidate(
            candidate_id="ext-frag-04",
            raw_text="October 28, 2026",
            is_valid_assertion=False,
            fragment_type="date_only",
            description="Standalone date fragment.",
        ),
        ExtractionEvalCandidate(
            candidate_id="ext-frag-05",
            raw_text="train crash",
            is_valid_assertion=False,
            fragment_type="predicate_missing",
            description="Two-word noun phrase lacking predicate.",
        ),
    ]

    return EvaluationDataset(
        verification_cases=verification_cases,
        retrieval_corpus=retrieval_corpus,
        retrieval_queries=retrieval_queries,
        extraction_candidates=extraction_candidates,
    )


EVALUATION_DATASET = _build_dataset()
