"""Deterministic lexical evidence retrieval using BM25 with source isolation."""

import logging
import math
import re
from typing import Any
from rank_bm25 import BM25Plus

from news_agent.models.article import NormalizedArticle
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.verification.segmenter import PassageSegment, segment_article

logger = logging.getLogger(__name__)

# Basic English stopwords for deterministic lexical tokenization
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


def tokenize_text(text: str) -> list[str]:
    """Tokenize and normalize text into clean, unaccented, lowercase word tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class EvidenceRetriever:
    """Lexical passage retrieval engine indexing articles for factual evidence search."""

    def __init__(
        self,
        passages: list[PassageSegment] | None = None,
        min_relevance_threshold: float = 0.05,
        min_query_coverage: float = 0.35,
    ) -> None:
        self.min_relevance_threshold = min_relevance_threshold
        self.min_query_coverage = min_query_coverage
        self.passages: list[PassageSegment] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Plus | None = None

        if passages:
            self.index_passages(passages)

    @classmethod
    def from_articles(
        cls,
        articles: list[NormalizedArticle],
        target_chars: int = 450,
        min_relevance_threshold: float = 0.05,
        min_query_coverage: float = 0.35,
    ) -> "EvidenceRetriever":
        """Construct an EvidenceRetriever by segmenting a list of NormalizedArticles."""
        all_passages: list[PassageSegment] = []
        for article in articles:
            segments = segment_article(article, target_chars=target_chars)
            all_passages.extend(segments)
        return cls(
            passages=all_passages,
            min_relevance_threshold=min_relevance_threshold,
            min_query_coverage=min_query_coverage,
        )

    def index_passages(self, passages: list[PassageSegment]) -> None:
        """Index a set of passages for BM25 ranking."""
        self.passages = passages
        self._tokenized_corpus = [tokenize_text(p.text) for p in passages]
        # Only fit BM25 if there are non-empty token lists
        if any(self._tokenized_corpus):
            self._bm25 = BM25Plus(self._tokenized_corpus)
        else:
            self._bm25 = None

    def retrieve_evidence(
        self,
        claim: FactualClaim,
        top_k: int = 5,
        exclude_own_article: bool = True,
    ) -> list[EvidencePassage]:
        """Retrieve and rank evidence passages relevant to the claim.

        Args:
            claim: The FactualClaim to find corroborating/refuting evidence for.
            top_k: Maximum number of evidence passages to return.
            exclude_own_article: Whether to exclude passages originating from
                the claim's own parent article (default: True for corroboration).

        Returns:
            List of ranked EvidencePassage models with normalized relevance scores.
        """
        if not self._bm25 or not self.passages:
            return []

        query_tokens = tokenize_text(claim.claim_text)
        if not query_tokens:
            return []

        query_set = set(query_tokens)
        raw_scores = self._bm25.get_scores(query_tokens)

        # Gate raw scores: require minimum token coverage of the query
        gated_scores: list[float] = []
        for idx, score in enumerate(raw_scores):
            matches = len(query_set & set(self._tokenized_corpus[idx]))
            coverage = matches / len(query_set) if query_set else 0.0
            gated_scores.append(float(score) if matches > 0 and coverage >= self.min_query_coverage else 0.0)

        max_score = max(gated_scores) if gated_scores else 0.0
        if max_score <= 0.0:
            return []

        scored_results: list[tuple[PassageSegment, float]] = []
        seen_passage_fingerprints: set[str] = set()

        for idx, score in enumerate(gated_scores):
            if score <= 0.0:
                continue

            segment = self.passages[idx]

            # 1. Enforce originating article exclusion for independent corroboration
            if exclude_own_article and segment.article_id == claim.article_id:
                continue

            # 2. Deduplicate identical or syndicated passages across articles
            norm_fingerprint = re.sub(r"\s+", " ", segment.text.lower().strip())
            if norm_fingerprint in seen_passage_fingerprints:
                continue

            # 3. Normalize score to [0.0, 1.0] range
            normalized_score = round(min(1.0, score / max_score), 4)

            if normalized_score >= self.min_relevance_threshold:
                seen_passage_fingerprints.add(norm_fingerprint)
                scored_results.append((segment, normalized_score))

        # Sort descending by relevance score
        scored_results.sort(key=lambda item: item[1], reverse=True)

        ranked_passages: list[EvidencePassage] = []
        for segment, relevance in scored_results[:top_k]:
            ranked_passages.append(segment.to_evidence_passage(relevance_score=relevance))

        return ranked_passages
