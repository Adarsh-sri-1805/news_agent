"""Evaluation and benchmarking module for the Local News Intelligence Agent."""

from news_agent.evaluation.dataset import EVALUATION_DATASET, EvaluationDataset
from news_agent.evaluation.metrics import (
    ClassificationMetrics,
    EvaluationMetrics,
    compute_classification_metrics,
    compute_retrieval_metrics,
    compute_extraction_quality_metrics,
)
from news_agent.evaluation.runner import EvaluationRunner

__all__ = [
    "EVALUATION_DATASET",
    "EvaluationDataset",
    "ClassificationMetrics",
    "EvaluationMetrics",
    "compute_classification_metrics",
    "compute_retrieval_metrics",
    "compute_extraction_quality_metrics",
    "EvaluationRunner",
]
