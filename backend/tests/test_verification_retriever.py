"""Unit tests for lexical BM25 evidence retrieval, originating article exclusion, and passage deduplication."""

from datetime import datetime, timezone
import pytest

from news_agent.models.article import NormalizedArticle
from news_agent.models.claim import FactualClaim
from news_agent.verification.retriever import EvidenceRetriever
from news_agent.verification.segmenter import PassageSegment


def make_test_segment(
    passage_id: str,
    article_id: str,
    text: str,
    publisher: str = "Test Gazette",
) -> PassageSegment:
    return PassageSegment(
        passage_id=passage_id,
        article_id=article_id,
        text=text,
        start_char=0,
        end_char=len(text),
        source_url=f"https://example.com/articles/{article_id}",
        publisher=publisher,
        published_at=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
    )


class TestEvidenceRetriever:
    def test_retrieve_ranked_relevant_passages(self):
        seg1 = make_test_segment(
            "p1", "art-1", "City council voted 7-2 on Tuesday morning to approve municipal solar panels."
        )
        seg2 = make_test_segment(
            "p2", "art-2", "The city budget allocates funds for downtown park renovations."
        )
        seg3 = make_test_segment(
            "p3", "art-3", "Official council minutes confirm resolution 44 passed 7 to 2 regarding solar panels."
        )

        retriever = EvidenceRetriever(passages=[seg1, seg2, seg3])

        claim = FactualClaim(
            id="c1",
            article_id="art-other",
            claim_text="The city council approved solar panels with a 7-2 vote.",
        )

        results = retriever.retrieve_evidence(claim, top_k=2)
        assert len(results) == 2
        # The solar passages (p1 and p3) should be returned, not the park renovation (p2)
        retrieved_ids = [p.id for p in results]
        assert "p2" not in retrieved_ids
        assert all(0.0 <= p.relevance_score <= 1.0 for p in results)

    def test_originating_article_excluded_by_default(self):
        """CRITICAL: Exclude the claim's own source article when searching for corroborating sources by default."""
        seg_own = make_test_segment(
            "p_own", "art-origin", "The council voted 7-2 to install solar panels on municipal rooftops."
        )
        seg_corroborating = make_test_segment(
            "p_corrob", "art-external", "Council meeting minutes verify the 7-2 vote on the solar project."
        )

        retriever = EvidenceRetriever(passages=[seg_own, seg_corroborating])

        claim = FactualClaim(
            id="c1",
            article_id="art-origin",  # Originating article
            claim_text="Council voted 7-2 for solar panels.",
        )

        # Default: exclude_own_article=True
        results = retriever.retrieve_evidence(claim, exclude_own_article=True)
        assert len(results) == 1
        assert results[0].id == "p_corrob"
        assert results[0].metadata["article_id"] == "art-external"

    def test_originating_article_included_when_flag_disabled(self):
        seg_own = make_test_segment(
            "p_own", "art-origin", "The council voted 7-2 to install solar panels on municipal rooftops."
        )
        retriever = EvidenceRetriever(passages=[seg_own])

        claim = FactualClaim(
            id="c1",
            article_id="art-origin",
            claim_text="Council voted 7-2 for solar panels.",
        )

        results = retriever.retrieve_evidence(claim, exclude_own_article=False)
        assert len(results) == 1
        assert results[0].id == "p_own"

    def test_identical_syndicated_passages_deduplicated(self):
        """CRITICAL: Do not count repeated copies of the same passage as independent evidence."""
        verbatim_text = "Midwest Solar Dynamics won the $2.5 million municipal bidding process."
        seg_pub1 = make_test_segment("p_wire_1", "art-1", verbatim_text, publisher="Publisher A")
        # Exact copy syndicated to Publisher B
        seg_pub2 = make_test_segment("p_wire_2", "art-2", verbatim_text, publisher="Publisher B")

        retriever = EvidenceRetriever(passages=[seg_pub1, seg_pub2])

        claim = FactualClaim(
            id="c1",
            article_id="art-other",
            claim_text="Midwest Solar Dynamics won the $2.5 million contract.",
        )

        results = retriever.retrieve_evidence(claim, top_k=5)
        # Should deduplicate repeated passage copies
        assert len(results) == 1

    def test_empty_corpus_and_no_match(self):
        retriever = EvidenceRetriever(passages=[])
        claim = FactualClaim(id="c1", article_id="art-1", claim_text="Some claim text.")
        assert retriever.retrieve_evidence(claim) == []

        # Irrelevant query
        seg = make_test_segment("p1", "art-1", "Cooking recipes for apple pie.")
        retriever2 = EvidenceRetriever(passages=[seg], min_relevance_threshold=0.1)
        claim2 = FactualClaim(id="c2", article_id="art-2", claim_text="Quantum mechanics gravitational waves.")
        assert retriever2.retrieve_evidence(claim2) == []

    def test_unrelated_passage_with_low_query_coverage_rejected_regression(self):
        """REGRESSION: Passages matching only incidental words (e.g. 'contract') but missing

        critical claim assertions (e.g. '$2.5 million valuation') must not be returned.
        """
        observer_text = (
            "Official voting ledgers from the Springfield city clerk verify that resolution #44 passed "
            "with 7 ayes and 2 nays. The approved contract authorizes Midwest Solar Dynamics to install "
            "photovoltaic arrays on city hall and the central library rooftop starting next spring."
        )
        seg = make_test_segment("p_observer", "art-observer", observer_text, publisher="Civic Records Observer")
        retriever = EvidenceRetriever(passages=[seg])

        # Claim 1: Mentions vote and approval -> sufficient overlap with observer passage
        claim_vote = FactualClaim(
            id="c_vote",
            article_id="art-other",
            claim_text="The Springfield city council approved the solar installation by a 7-2 vote.",
        )
        assert len(retriever.retrieve_evidence(claim_vote)) == 1

        # Claim 2: Valuation claim ($2.5 million) -> observer passage has no valuation info
        claim_value = FactualClaim(
            id="c_val",
            article_id="art-other",
            claim_text="The municipal solar panel contract is valued at $2.5 million.",
        )
        # Must NOT return the observer passage for this valuation claim
        assert retriever.retrieve_evidence(claim_value) == []

