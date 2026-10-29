"""Passage segmentation for news article content with character offset tracking."""

from dataclasses import dataclass
from datetime import datetime
import re
from uuid import uuid4

from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now
from news_agent.models.evidence import EvidencePassage


@dataclass
class PassageSegment:
    """An exact textual passage extracted from an article with provenance metadata."""

    passage_id: str
    article_id: str
    text: str
    start_char: int
    end_char: int
    source_url: str
    publisher: str
    published_at: datetime | None

    def to_evidence_passage(self, relevance_score: float | None = None) -> EvidencePassage:
        """Convert segment into a validated EvidencePassage domain model."""
        return EvidencePassage(
            id=self.passage_id,
            passage_text=self.text,
            source_url=self.source_url,  # type: ignore[arg-type]
            publisher=self.publisher,
            retrieved_at=utc_now(),
            relevance_score=relevance_score,
            metadata={
                "article_id": self.article_id,
                "start_char": self.start_char,
                "end_char": self.end_char,
            },
        )


def segment_article(
    article: NormalizedArticle,
    target_chars: int = 450,
    min_chars: int = 50,
) -> list[PassageSegment]:
    """Segment an article's content into manageable verbatim passages with character offsets.

    Splits along paragraph or sentence boundaries to maintain semantic completeness,
    preserving exact source character positions.

    Args:
        article: The NormalizedArticle to segment.
        target_chars: Target character length per passage (default: 450).
        min_chars: Minimum character length to qualify as a passage (default: 50).

    Returns:
        List of PassageSegment objects with verified character offsets.
    """
    content = article.content.strip()
    if not content:
        return []

    # If content is short enough, return single passage
    if len(content) <= target_chars:
        return [
            PassageSegment(
                passage_id=f"{article.id}_p0",
                article_id=article.id,
                text=content,
                start_char=0,
                end_char=len(content),
                source_url=str(article.source_url),
                publisher=article.publisher,
                published_at=article.published_at,
            )
        ]

    # Split on paragraph boundaries first
    paragraphs: list[tuple[str, int, int]] = []
    # Find all paragraph matches preserving start and end offsets in original content
    for match in re.finditer(r"\S[^\n\r]*", content):
        p_text = match.group().strip()
        if p_text:
            paragraphs.append((p_text, match.start(), match.end()))

    segments: list[PassageSegment] = []
    current_text_parts: list[str] = []
    current_start = 0
    current_end = 0
    passage_idx = 0

    for p_text, p_start, p_end in paragraphs:
        if not current_text_parts:
            current_start = p_start
            current_text_parts.append(p_text)
            current_end = p_end
        else:
            combined_len = (p_end - current_start)
            if combined_len <= target_chars:
                current_text_parts.append(p_text)
                current_end = p_end
            else:
                # Flush current accumulated passage
                passage_text = content[current_start:current_end].strip()
                if len(passage_text) >= min_chars:
                    segments.append(
                        PassageSegment(
                            passage_id=f"{article.id}_p{passage_idx}",
                            article_id=article.id,
                            text=passage_text,
                            start_char=current_start,
                            end_char=current_end,
                            source_url=str(article.source_url),
                            publisher=article.publisher,
                            published_at=article.published_at,
                        )
                    )
                    passage_idx += 1
                # Start new chunk
                current_start = p_start
                current_text_parts = [p_text]
                current_end = p_end

    # Flush any remaining text
    if current_text_parts:
        passage_text = content[current_start:current_end].strip()
        if len(passage_text) >= min_chars or not segments:
            segments.append(
                PassageSegment(
                    passage_id=f"{article.id}_p{passage_idx}",
                    article_id=article.id,
                    text=passage_text,
                    start_char=current_start,
                    end_char=current_end,
                    source_url=str(article.source_url),
                    publisher=article.publisher,
                    published_at=article.published_at,
                )
            )

    return segments
