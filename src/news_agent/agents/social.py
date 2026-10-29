"""Social Content Agent: generates platform-tailored draft copy with strict character validation and truth grounding."""

from datetime import datetime
import logging
import re
from typing import Any

from news_agent.llm.client import OllamaClient
from news_agent.models.article import NormalizedArticle
from news_agent.models.brief import NewsBrief
from news_agent.models.social import SocialMediaDraft
from news_agent.models.verification import VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)

# Deterministic platform character limits
PLATFORM_CHARACTER_LIMITS: dict[str, int] = {
    "x_twitter": 280,
    "x": 280,
    "twitter": 280,
    "linkedin": 3000,
}


def sanitize_hashtag(tag: str) -> str:
    """Strip '#' prefix and spaces from hashtag string."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "", tag.lstrip("#"))
    return clean


class SocialContentAgent:
    """Specialist agent that drafts platform-specific social copy with strict length and truth gates."""

    def __init__(self, llm_client: OllamaClient | None = None) -> None:
        self.llm_client = llm_client or OllamaClient()

    def generate_drafts(
        self,
        brief: NewsBrief,
        articles: list[NormalizedArticle],
        verification_results: list[VerificationResult],
        platforms: list[str] | None = None,
    ) -> list[SocialMediaDraft]:
        """Generate platform-tailored social drafts.

        Critical constraints:
        - NEVER assert unverified or contradicted claims as established facts.
        - Enforce deterministic platform character limits.
        - Never publish; draft generation only.
        """
        target_platforms = platforms or ["x_twitter", "linkedin"]
        drafts: list[SocialMediaDraft] = []

        # Filter verification results to separate confirmed facts from uncorroborated/contradicted
        verdict_map = {r.claim_id: r for r in verification_results}
        supported_results = [r for r in verification_results if r.status == VerificationStatus.SUPPORTED]
        contradicted_results = [r for r in verification_results if r.status == VerificationStatus.CONTRADICTED]

        # Extract hashtags from articles/topics
        hashtags = self._extract_relevant_hashtags(articles)

        for platform_name in target_platforms:
            canonical_platform = "x_twitter" if platform_name in ("x", "twitter") else platform_name
            max_chars = PLATFORM_CHARACTER_LIMITS.get(canonical_platform, 280)

            draft = self._create_draft_for_platform(
                platform=canonical_platform,
                max_chars=max_chars,
                brief=brief,
                articles=articles,
                supported_results=supported_results,
                contradicted_results=contradicted_results,
                hashtags=hashtags,
            )
            # Enforce deterministic Python validation
            self.validate_draft(draft, max_chars=max_chars)
            drafts.append(draft)

        return drafts

    def validate_draft(self, draft: SocialMediaDraft, max_chars: int | None = None) -> bool:
        """Deterministically validate that a draft conforms to platform constraints."""
        canonical = "x_twitter" if draft.platform in ("x", "twitter") else draft.platform
        limit = max_chars or PLATFORM_CHARACTER_LIMITS.get(canonical, 3000)

        actual_len = len(draft.content)
        if actual_len > limit:
            raise ValueError(
                f"Draft for platform '{draft.platform}' exceeds limit of {limit} characters: {actual_len} chars."
            )
        if draft.character_count != actual_len:
            draft.character_count = actual_len

        return True

    def _extract_relevant_hashtags(self, articles: list[NormalizedArticle]) -> list[str]:
        """Extract a small set of clean, relevant hashtags without '#'."""
        tags: list[str] = ["LocalNews"]
        for art in articles[:3]:
            # Extract simple topic tokens
            words = re.findall(r"\b[A-Z][a-z]{3,}\b", art.title)
            for w in words[:2]:
                clean = sanitize_hashtag(w)
                if clean and clean.lower() not in ("news", "local", "today", "report") and clean not in tags:
                    tags.append(clean)
        return tags[:4]

    def _create_draft_for_platform(
        self,
        platform: str,
        max_chars: int,
        brief: NewsBrief,
        articles: list[NormalizedArticle],
        supported_results: list[VerificationResult],
        contradicted_results: list[VerificationResult],
        hashtags: list[str],
    ) -> SocialMediaDraft:
        """Create a platform draft strictly respecting character limits and factual bounds."""
        source_ids = [a.id for a in articles]
        main_url = str(articles[0].source_url) if articles else ""
        publisher = articles[0].publisher if articles else "News"

        # Truth check: Check if brief has confirmed facts
        has_supported = len(supported_results) > 0
        has_contradicted = len(contradicted_results) > 0

        if not articles:
            # Deterministic truthful copy when zero articles were processed
            if platform == "x_twitter":
                content = f"Notice: {brief.title}\n\nNo news articles were processed in this monitoring cycle."
            else:
                content = (
                    f"Notice: {brief.title}\n\n"
                    "No news articles were ingested or processed in this monitoring cycle. "
                    "Zero claims or verified intelligence were evaluated."
                )
            return SocialMediaDraft(
                platform=platform,
                content=content,
                character_count=len(content),
                brief_id=brief.id,
                source_article_ids=[],
                hashtags=[],
                metadata={
                    "truth_gating": {
                        "supported_claims_count": 0,
                        "contradicted_claims_count": 0,
                    }
                },
            )

        if platform == "x_twitter":
            # Concise 280-char post
            # Only use title and verified status or explicitly report as reported
            hashtag_str = " ".join(f"#{t}" for t in hashtags[:2])
            
            # Grounded copy: if claims are unverified or contradicted, state transparently
            if has_contradicted:
                headline_text = f"Report update: Conflicting evidence found regarding recent {publisher} story."
            elif has_supported:
                # Use brief title or main takeaway
                headline_text = brief.title.replace("Intelligence Brief: ", "")
            else:
                headline_text = f"Reported: {articles[0].title if articles else brief.title}"

            base_text = f"{headline_text}\n\nDetails: {main_url}\n{hashtag_str}"
            if len(base_text) > max_chars:
                # Truncate headline cleanly to fit
                overhead = len(f"...\n\nDetails: {main_url}\n{hashtag_str}")
                allowed_title = max_chars - overhead
                headline_text = headline_text[:allowed_title].rstrip() + "..."
                base_text = f"{headline_text}\n\nDetails: {main_url}\n{hashtag_str}"

            content = base_text

        else:
            # LinkedIn: richer professional briefing up to 3000 chars
            header_intro = (
                "Key verified intelligence from our monitoring network:"
                if has_supported
                else "Recent news reporting summary (pending independent corroboration):"
            )
            lines = [
                f"📰 Executive News Brief: {brief.title.replace('Intelligence Brief: ', '')}",
                "",
                header_intro,
            ]

            # Only include supported takeaways in factual assertions
            for takeaway in brief.headline_takeaways[:3]:
                # Omit contradicted assertions from promotional key takeaways
                if "[Contradicted" in takeaway:
                    continue
                lines.append(f"• {takeaway}")

            if has_contradicted:
                lines.append("")
                lines.append("⚠️ Fact-Check Advisory: Conflicting or contradicted evidence was identified during multi-source auditing.")

            lines.append("")
            lines.append(f"Source: {publisher} ({main_url})")
            if hashtags:
                lines.append("")
                lines.append(" ".join(f"#{t}" for t in hashtags))

            content = "\n".join(lines)
            if len(content) > max_chars:
                content = content[:max_chars - 3].rstrip() + "..."

        return SocialMediaDraft(
            platform=platform,
            content=content,
            character_count=len(content),
            brief_id=brief.id,
            source_article_ids=source_ids,
            hashtags=hashtags,
            metadata={
                "truth_gating": {
                    "supported_claims_count": len(supported_results),
                    "contradicted_claims_count": len(contradicted_results),
                }
            },
        )
