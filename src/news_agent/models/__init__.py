"""Export domain models and contracts."""

from news_agent.models.article import NormalizedArticle
from news_agent.models.base import NewsAgentBaseModel, ensure_utc, utc_now
from news_agent.models.brief import NewsBrief
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.social import SocialMediaDraft
from news_agent.models.verification import VerificationResult, VerificationStatus

__all__ = [
    "NewsAgentBaseModel",
    "NormalizedArticle",
    "FactualClaim",
    "EvidencePassage",
    "VerificationStatus",
    "VerificationResult",
    "NewsBrief",
    "SocialMediaDraft",
    "utc_now",
    "ensure_utc",
]
