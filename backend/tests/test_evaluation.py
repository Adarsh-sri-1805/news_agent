"""Automated unit and regression tests for Phase 6 evaluation suite."""

import pytest
from news_agent.evaluation.dataset import EVALUATION_DATASET
from news_agent.evaluation.metrics import (
    compute_classification_metrics,
    compute_extraction_quality_metrics,
    compute_retrieval_metrics,
)
from news_agent.evaluation.runner import EvaluationRunner
from news_agent.verification.extractor import is_meaningful_claim


class TestEvaluationMetricsComputation:
    def test_classification_metrics_perfect_predictions(self):
        gt = ["supported", "contradicted", "insufficient_evidence"]
        pred = ["supported", "contradicted", "insufficient_evidence"]
        res = compute_classification_metrics(gt, pred)

        assert res.accuracy == 1.0
        assert res.macro_f1 == 1.0
        assert res.macro_precision == 1.0
        assert res.macro_recall == 1.0
        assert res.class_metrics["supported"].f1_score == 1.0

    def test_classification_metrics_with_misclassifications(self):
        gt = ["supported", "supported", "contradicted", "insufficient_evidence"]
        pred = ["supported", "insufficient_evidence", "contradicted", "insufficient_evidence"]
        res = compute_classification_metrics(gt, pred)

        assert res.total_samples == 4
        assert res.correct_predictions == 3
        assert res.accuracy == 0.75
        assert res.class_metrics["supported"].precision == 1.0
        assert res.class_metrics["supported"].recall == 0.5

    def test_retrieval_metrics_computation(self):
        query_results = [
            {
                "gold_passage_ids": ["p1", "p2"],
                "gold_article_ids": ["art1"],
                "retrieved_passage_ids": ["p1", "p3", "p4"],
                "retrieved_article_ids": ["art1", "art2"],
            },
            {
                "gold_passage_ids": ["p5"],
                "gold_article_ids": ["art3"],
                "retrieved_passage_ids": ["p9", "p10", "p5"],
                "retrieved_article_ids": ["art4", "art5", "art3"],
            },
        ]
        res = compute_retrieval_metrics(query_results)
        assert res.recall_at_1 == 0.5
        assert res.recall_at_3 == 1.0
        assert res.gold_page_hit_rate_at_3 == 1.0

    def test_extraction_quality_metrics_computation(self):
        candidates = [
            {"is_valid_assertion": True, "retained": True},
            {"is_valid_assertion": True, "retained": True},
            {"is_valid_assertion": False, "retained": False},
            {"is_valid_assertion": False, "retained": False},
        ]
        res = compute_extraction_quality_metrics(candidates)
        assert res.claim_quality_rate == 1.0
        assert res.fragment_rejection_rate == 1.0


class TestEvaluationDatasetIntegrity:
    def test_evaluation_dataset_structure(self):
        assert len(EVALUATION_DATASET.verification_cases) >= 6
        assert len(EVALUATION_DATASET.retrieval_corpus) >= 5
        assert len(EVALUATION_DATASET.retrieval_queries) >= 4
        assert len(EVALUATION_DATASET.extraction_candidates) >= 8

    def test_fragment_rejection_rules(self):
        # Fragments must be rejected
        assert is_meaningful_claim("Fernando Seco") is False
        assert is_meaningful_claim("Toledo") is False
        assert is_meaningful_claim("68") is False
        assert is_meaningful_claim("October 28, 2026") is False
        assert is_meaningful_claim("train crash") is False
        assert is_meaningful_claim("$12,000,000") is False

        # Complete propositions must be accepted
        assert is_meaningful_claim("City council approved the municipal solar installation.") is True
        assert is_meaningful_claim("The regional transit board voted 5-4 to reject the proposal.") is True
        assert is_meaningful_claim("Three cases of avian flu were confirmed in local poultry farms.") is True


class TestEvaluationRunnerEndToEnd:
    def test_runner_all_benchmarks(self):
        runner = EvaluationRunner(use_live_llm=False)
        metrics = runner.run_all(measure_latency=False)

        # Verification asserts
        assert metrics.verification.accuracy >= 0.90
        assert metrics.verification.macro_f1 >= 0.85

        # Retrieval asserts
        assert metrics.retrieval.recall_at_3 >= 0.90
        assert metrics.retrieval.gold_page_hit_rate_at_3 >= 0.90

        # Extraction asserts
        assert metrics.extraction.claim_quality_rate == 1.0
        assert metrics.extraction.fragment_rejection_rate == 1.0

        # Citations audit
        assert metrics.citations.total_citations >= 5
        assert metrics.citations.valid_citations >= 4
