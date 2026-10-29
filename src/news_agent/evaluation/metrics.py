"""Evaluation metrics calculators for Phase 6 benchmarking.

All formulas are explicitly defined, mathematically documented, and reproducible.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ClassificationMetrics:
    """Precision, Recall, and F1 metrics for a single classification class."""

    class_name: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float


@dataclass
class VerificationEvaluationResult:
    """Summary of claim verification accuracy, precision, recall, and F1."""

    total_samples: int
    correct_predictions: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    class_metrics: dict[str, ClassificationMetrics]


@dataclass
class RetrievalEvaluationResult:
    """Evidence retrieval performance across Recall@k and Gold Page Hit Rate."""

    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    gold_page_hit_rate_at_3: float
    gold_page_hit_rate_at_5: float


@dataclass
class CitationIntegrityResult:
    """Integrity of model citations against supplied evidence passages."""

    total_citations: int
    valid_citations: int
    invalid_citations: int
    invalid_citation_rate: float


@dataclass
class ExtractionQualityResult:
    """Quality of claim extraction and fragment rejection."""

    total_candidates: int
    valid_assertions_total: int
    valid_assertions_retained: int
    fragments_total: int
    fragments_rejected: int
    fragment_rejection_rate: float
    claim_quality_rate: float


@dataclass
class EvaluationMetrics:
    """Complete aggregated evaluation report for the News Intelligence Agent."""

    verification: VerificationEvaluationResult
    retrieval: RetrievalEvaluationResult
    citations: CitationIntegrityResult
    extraction: ExtractionQualityResult
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to machine-readable dictionary."""
        return asdict(self)


def compute_classification_metrics(
    ground_truth: list[str],
    predictions: list[str],
    classes: list[str] | None = None,
) -> VerificationEvaluationResult:
    """Compute overall accuracy, per-class precision, recall, F1, and macro averages.

    Formulas:
        Accuracy = Sum(TP) / Total
        Precision_c = TP_c / (TP_c + FP_c)
        Recall_c = TP_c / (TP_c + FN_c)
        F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
        Macro F1 = Mean(F1_c for all classes)
    """
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have identical lengths.")

    total = len(ground_truth)
    if total == 0:
        return VerificationEvaluationResult(
            total_samples=0,
            correct_predictions=0,
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            class_metrics={},
        )

    all_classes = classes or sorted(list(set(ground_truth) | set(predictions)))
    correct = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
    accuracy = round(correct / total, 4)

    class_metrics: dict[str, ClassificationMetrics] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for c in all_classes:
        tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == c and pred == c)
        fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt != c and pred == c)
        fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == c and pred != c)
        tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt != c and pred != c)

        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        class_metrics[c] = ClassificationMetrics(
            class_name=c,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=prec,
            recall=rec,
            f1_score=f1,
        )

    macro_precision = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
    macro_recall = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
    macro_f1 = round(sum(f1s) / len(f1s), 4) if f1s else 0.0

    return VerificationEvaluationResult(
        total_samples=total,
        correct_predictions=correct,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        class_metrics=class_metrics,
    )


def compute_retrieval_metrics(
    query_results: list[dict[str, Any]],
) -> RetrievalEvaluationResult:
    """Compute Recall@k and Gold Page Hit Rate for evidence retrieval.

    Args:
        query_results: List of dicts containing:
            - 'gold_passage_ids': list[str]
            - 'gold_article_ids': list[str]
            - 'retrieved_passage_ids': list[str]
            - 'retrieved_article_ids': list[str]
    """
    total = len(query_results)
    if total == 0:
        return RetrievalEvaluationResult(
            total_queries=0,
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            gold_page_hit_rate_at_3=0.0,
            gold_page_hit_rate_at_5=0.0,
        )

    r_at_1_hits = 0
    r_at_3_hits = 0
    r_at_5_hits = 0
    gold_page_3_hits = 0
    gold_page_5_hits = 0

    for item in query_results:
        gold_passages = set(item.get("gold_passage_ids", []))
        gold_articles = set(item.get("gold_article_ids", []))
        ret_passages = item.get("retrieved_passage_ids", [])
        ret_articles = item.get("retrieved_article_ids", [])

        # Recall@k (at least one gold passage retrieved in top-k)
        if any(pid in gold_passages for pid in ret_passages[:1]):
            r_at_1_hits += 1
        if any(pid in gold_passages for pid in ret_passages[:3]):
            r_at_3_hits += 1
        if any(pid in gold_passages for pid in ret_passages[:5]):
            r_at_5_hits += 1

        # Gold Page Hit Rate (gold article in top-k)
        if any(aid in gold_articles for aid in ret_articles[:3]):
            gold_page_3_hits += 1
        if any(aid in gold_articles for aid in ret_articles[:5]):
            gold_page_5_hits += 1

    return RetrievalEvaluationResult(
        total_queries=total,
        recall_at_1=round(r_at_1_hits / total, 4),
        recall_at_3=round(r_at_3_hits / total, 4),
        recall_at_5=round(r_at_5_hits / total, 4),
        gold_page_hit_rate_at_3=round(gold_page_3_hits / total, 4),
        gold_page_hit_rate_at_5=round(gold_page_5_hits / total, 4),
    )


def compute_extraction_quality_metrics(
    candidate_results: list[dict[str, Any]],
) -> ExtractionQualityResult:
    """Compute claim extraction quality and fragment rejection rate.

    Args:
        candidate_results: List of dicts with:
            - 'is_valid_assertion': bool
            - 'retained': bool
    """
    total = len(candidate_results)
    valid_total = sum(1 for c in candidate_results if c["is_valid_assertion"])
    fragments_total = sum(1 for c in candidate_results if not c["is_valid_assertion"])

    valid_retained = sum(1 for c in candidate_results if c["is_valid_assertion"] and c["retained"])
    fragments_rejected = sum(1 for c in candidate_results if not c["is_valid_assertion"] and not c["retained"])

    quality_rate = round(valid_retained / valid_total, 4) if valid_total > 0 else 0.0
    fragment_rejection_rate = round(fragments_rejected / fragments_total, 4) if fragments_total > 0 else 1.0

    return ExtractionQualityResult(
        total_candidates=total,
        valid_assertions_total=valid_total,
        valid_assertions_retained=valid_retained,
        fragments_total=fragments_total,
        fragments_rejected=fragments_rejected,
        fragment_rejection_rate=fragment_rejection_rate,
        claim_quality_rate=quality_rate,
    )
