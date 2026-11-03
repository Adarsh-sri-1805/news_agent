"""Atomic factual claim extraction using local Ollama LLM inference."""

import logging
import re
from typing import Any
from pydantic import AliasChoices, BaseModel, Field, model_validator

from news_agent.ingestion.dedup import calculate_title_similarity
from news_agent.llm import BaseLLMClient, LLMError, OllamaError, create_llm_client
from news_agent.models.article import NormalizedArticle
from news_agent.models.base import utc_now
from news_agent.models.claim import FactualClaim

logger = logging.getLogger(__name__)


def is_meaningful_claim(text: str) -> bool:
    """Validate that an extracted claim is a complete, meaningful, assessable assertion rather than a fragment.

    Rejects:
    - Text shorter than 10 characters.
    - Under 3 whitespace-separated tokens (e.g. 'Fernando Seco', 'Toledo', '12').
    - Pure numbers, currency amounts, or dates without predicates.
    - Entities/locations lacking an asserting action or statement.
    """
    cleaned = text.strip()
    if len(cleaned) < 10:
        return False

    tokens = cleaned.split()
    if len(tokens) < 3:
        return False

    # Check if string has sufficient alphabetic content (not just symbols/digits)
    alpha_chars = [c for c in cleaned if c.isalpha()]
    if len(alpha_chars) < 6:
        return False

    # Reject date-only fragments (e.g. "October 28, 2026", "2024-05-12")
    date_patterns = [
        r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(?:, \d{4})?$",
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",
    ]
    for pat in date_patterns:
        if re.match(pat, cleaned, re.IGNORECASE):
            return False

    return True

EXTRACTION_SYSTEM_PROMPT = """You are a rigorous investigative news analyst.
Extract atomic, verifiable factual claims from the provided article text.

Strict Rules:
1. Extract only objective factual assertions (actions, numbers, votes, dates, policies, quotes).
2. Completely EXCLUDE personal opinions, subjective editorial commentary, speculation, and marketing hyperbole.
3. Keep each claim atomic and self-contained (understandable without full context).
4. Do NOT rewrite or alter the factual meaning of the text.
5. If the text does not contain any concrete factual claims, return an empty claims list.
"""


class ExtractedClaimItem(BaseModel):
    """Schema for individual claim extracted by LLM."""

    claim_text: str = Field(
        ...,
        min_length=3,
        validation_alias=AliasChoices("claim_text", "text", "claim", "assertion", "statement", "value", "description"),
        description="Atomic, verifiable factual assertion.",
    )
    context: str | None = Field(
        default=None,
        validation_alias=AliasChoices("context", "snippet", "source_context", "quote"),
        description="Direct supporting context snippet from article.",
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("confidence", "score", "confidence_score"),
        description="Extraction confidence score.",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_raw_input(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"claim_text": data.strip()}
        if isinstance(data, dict):
            cleaned = {}
            for k, v in data.items():
                if isinstance(k, str):
                    cleaned[k.strip().lower()] = v
                else:
                    cleaned[k] = v
            # If claim_text isn't found under standard aliases, check for description or value
            if "claim_text" not in cleaned:
                for alt in ("claim_text", "text", "claim", "assertion", "statement", "description", "value"):
                    val = cleaned.get(alt)
                    if isinstance(val, str) and len(val.strip()) >= 3:
                        cleaned["claim_text"] = val.strip()
                        break
            return cleaned
        return data


class ClaimExtractionResponse(BaseModel):
    """Schema for structured LLM extraction output."""

    claims: list[ExtractedClaimItem] = Field(
        default_factory=list,
        description="List of extracted verifiable claims.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_response(cls, data: Any) -> Any:
        if isinstance(data, list):
            data = {"claims": data}
        if isinstance(data, dict):
            raw_claims = data.get("claims")
            if isinstance(raw_claims, list):
                valid_items = []
                for item in raw_claims:
                    if isinstance(item, str) and len(item.strip()) >= 3:
                        valid_items.append(item.strip())
                    elif isinstance(item, dict):
                        # Verify it has at least one valid string field with length >= 3
                        has_text = False
                        for alt in ("claim_text", "text", "claim", "assertion", "statement", "description", "value"):
                            val = item.get(alt)
                            if isinstance(val, str) and len(val.strip()) >= 3:
                                has_text = True
                                break
                        if has_text:
                            valid_items.append(item)
                data["claims"] = valid_items
        return data


class ClaimExtractor:
    """Extracts atomic factual claims from NormalizedArticles using local Ollama inference."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        max_claims_per_article: int = 5,
        min_content_length: int = 30,
        system_prompt: str = EXTRACTION_SYSTEM_PROMPT,
    ) -> None:
        self.llm_client = llm_client or create_llm_client()
        self.max_claims_per_article = max_claims_per_article
        self.min_content_length = min_content_length
        self.system_prompt = system_prompt

    def extract_claims(self, article: NormalizedArticle) -> list[FactualClaim]:
        """Extract atomic factual claims from an article with robust validation.

        Args:
            article: Normalized news article.

        Returns:
            List of validated FactualClaim domain models.
        """
        content = (article.content or "").strip()
        if len(content) < self.min_content_length:
            logger.info(
                "Article '%s' (id=%s) has inadequate content length (%d chars); skipping claim extraction.",
                article.title,
                article.id,
                len(content),
            )
            return []

        prompt = (
            f"Headline: {article.title}\n"
            f"Publisher: {article.publisher}\n"
            f"Article Content:\n{content[:2500]}\n\n"
            f"Extract up to {self.max_claims_per_article} atomic factual claims in the requested JSON format."
        )

        try:
            response = self.llm_client.generate_structured(
                prompt=prompt,
                response_model=ClaimExtractionResponse,
                system=self.system_prompt,
            )
        except OllamaError as err:
            logger.warning(
                "Ollama extraction failed for article '%s' (id=%s): %s",
                article.title,
                article.id,
                err,
            )
            return []

        raw_items = response.claims[: self.max_claims_per_article]
        valid_claims: list[FactualClaim] = []

        for item in raw_items:
            clean_text = item.claim_text.strip()
            # Reject empty, too-short, or poorly formed entity/number fragments
            if not is_meaningful_claim(clean_text):
                logger.debug("Rejected non-meaningful claim fragment: '%s'", clean_text)
                continue

            # Deduplicate against already extracted claims in this article
            if self._is_duplicate_claim(clean_text, valid_claims):
                continue

            claim = FactualClaim(
                article_id=article.id,
                claim_text=clean_text,
                context=item.context.strip() if item.context else None,
                extracted_at=utc_now(),
                extraction_confidence=item.confidence,
                metadata={
                    "source_url": str(article.source_url),
                    "publisher": article.publisher,
                },
            )
            valid_claims.append(claim)

        return valid_claims

    @staticmethod
    def _is_duplicate_claim(new_claim: str, existing_claims: list[FactualClaim]) -> bool:
        """Check if claim is a near duplicate of any already extracted claim."""
        for existing in existing_claims:
            if calculate_title_similarity(new_claim, existing.claim_text) >= 0.85:
                return True
        return False
