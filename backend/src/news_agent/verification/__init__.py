"""Evidence and Verification Engine."""

from news_agent.verification.corroboration import (
    CorroborationAnalysis,
    analyze_corroboration,
)
from news_agent.verification.engine import (
    PipelineVerificationSummary,
    VerificationEngine,
)
from news_agent.verification.extractor import (
    ClaimExtractionResponse,
    ClaimExtractor,
    ExtractedClaimItem,
)
from news_agent.verification.retriever import EvidenceRetriever, tokenize_text
from news_agent.verification.segmenter import PassageSegment, segment_article
from news_agent.verification.verifier import ClaimVerifier, LLMVerificationPayload

__all__ = [
    "PassageSegment",
    "segment_article",
    "ClaimExtractor",
    "ExtractedClaimItem",
    "ClaimExtractionResponse",
    "EvidenceRetriever",
    "tokenize_text",
    "CorroborationAnalysis",
    "analyze_corroboration",
    "ClaimVerifier",
    "LLMVerificationPayload",
    "VerificationEngine",
    "PipelineVerificationSummary",
]
