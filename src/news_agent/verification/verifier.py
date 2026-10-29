"""Evidence interpretation and claim verification using local Ollama inference."""

import logging
from typing import Any
from pydantic import AliasChoices, BaseModel, Field, model_validator

from news_agent.llm.client import OllamaClient
from news_agent.llm.exceptions import OllamaError
from news_agent.models.base import utc_now
from news_agent.models.claim import FactualClaim
from news_agent.models.evidence import EvidencePassage
from news_agent.models.verification import VerificationResult, VerificationStatus
from news_agent.verification.corroboration import analyze_corroboration

logger = logging.getLogger(__name__)

VERIFICATION_SYSTEM_PROMPT = """You are an impartial, highly rigorous fact-checking judge.
Your task is to determine whether the provided evidence passages support, contradict, or provide insufficient evidence for the factual claim.

Verdict Definitions:
- 'supported': The available evidence directly and unambiguously confirms the factual claim.
- 'contradicted': The available evidence directly and unambiguously disproves or conflicts with the factual claim.
- 'insufficient_evidence': The evidence is incomplete, weak, tangential, missing, or ambiguous. When in doubt, you MUST choose 'insufficient_evidence'.

Rules:
1. Base your verdict strictly and solely on the provided evidence passages. Do NOT assume facts not stated.
2. Confidence score must be between 0.0 and 1.0, reflecting how clearly the text justifies the verdict.
3. You must only cite passage IDs (e.g., 'p1', 'p2') that are explicitly listed below.
4. Keep categorical status strictly distinct from confidence.
"""


class LLMVerificationPayload(BaseModel):
    """Schema for structured model response during claim verification."""

    status: VerificationStatus = Field(
        ...,
        validation_alias=AliasChoices("status", "verdict", "decision"),
        description="Categorical verdict: supported, contradicted, or insufficient_evidence.",
    )
    confidence_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("confidence_score", "confidence", "score"),
        description="Numeric estimate in [0.0, 1.0] representing strength of textual justification.",
    )
    reasoning: str = Field(
        default="Evidence evaluated by verification model.",
        min_length=5,
        validation_alias=AliasChoices("reasoning", "explanation", "justification", "evidence"),
        description="Detailed chain of evidence explaining the verdict.",
    )
    cited_passage_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("cited_passage_ids", "cited_passages", "passages"),
        description="List of passage identifiers cited in justification.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_payload_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cleaned = {}
            for k, v in data.items():
                if isinstance(k, str):
                    cleaned[k.strip().lower()] = v
                else:
                    cleaned[k] = v
            if "status" not in cleaned:
                for alt in ("verdict", "decision", "result"):
                    if alt in cleaned:
                        cleaned["status"] = cleaned[alt]
                        break
            if "confidence_score" not in cleaned:
                for alt in ("confidence", "score", "confidence_level"):
                    if alt in cleaned:
                        cleaned["confidence_score"] = cleaned[alt]
                        break
            if "reasoning" not in cleaned:
                for alt in ("explanation", "justification", "evidence", "rationale"):
                    if alt in cleaned:
                        cleaned["reasoning"] = cleaned[alt]
                        break
            return cleaned
        return data


class ClaimVerifier:
    """Interprets evidence passages to evaluate claims against discrete verification verdicts."""

    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        system_prompt: str = VERIFICATION_SYSTEM_PROMPT,
    ) -> None:
        self.llm_client = llm_client or OllamaClient()
        self.system_prompt = system_prompt

    def verify_claim(
        self,
        claim: FactualClaim,
        evidence_passages: list[EvidencePassage],
    ) -> VerificationResult:
        """Evaluate a claim against retrieved evidence passages.

        Args:
            claim: The FactualClaim to evaluate.
            evidence_passages: List of retrieved candidate EvidencePassages.

        Returns:
            Validated VerificationResult domain contract.
        """
        # Audit source diversity and independence
        corroboration_audit = analyze_corroboration(evidence_passages)

        # 1. Deterministic guard: zero evidence immediately yields INSUFFICIENT_EVIDENCE
        if not evidence_passages:
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence_score=0.0,
                reasoning="No independent evidence passages were retrieved to evaluate this claim.",
                evidence_passages=[],
                verified_at=utc_now(),
                model_name=self.llm_client.model_name,
                metadata={
                    "corroboration": corroboration_audit.__dict__,
                    "evaluation_mode": "deterministic_zero_evidence",
                },
            )

        # 2. Format prompt with verbatim passages and explicit IDs
        passages_text_blocks = []
        valid_passage_id_map: dict[str, EvidencePassage] = {}

        for idx, passage in enumerate(evidence_passages, 1):
            valid_passage_id_map[passage.id] = passage
            # Also support 1-based index 'p1', 'p2' alias
            valid_passage_id_map[f"p{idx}"] = passage
            passages_text_blocks.append(
                f"[{passage.id}] (Publisher: {passage.publisher or 'Unknown'}, Relevance: {passage.relevance_score})\n"
                f'"{passage.passage_text}"'
            )

        formatted_passages = "\n\n".join(passages_text_blocks)
        prompt = (
            f"Factual Claim to Evaluate:\n\"{claim.claim_text}\"\n\n"
            f"Retrieved Evidence Passages:\n{formatted_passages}\n\n"
            "Evaluate whether the evidence supports, contradicts, or is insufficient for the claim."
        )

        # 3. Model inference with structured output validation
        try:
            parsed = self.llm_client.generate_structured(
                prompt=prompt,
                response_model=LLMVerificationPayload,
                system=self.system_prompt,
            )
        except OllamaError as err:
            logger.warning(
                "Model verification error for claim '%s' (id=%s): %s",
                claim.claim_text,
                claim.id,
                err,
            )
            # NEVER silently convert failure into supported!
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence_score=0.0,
                reasoning=f"Verification inference could not be completed safely: {err}",
                evidence_passages=evidence_passages,
                verified_at=utc_now(),
                model_name=self.llm_client.model_name,
                metadata={
                    "error": str(err),
                    "corroboration": corroboration_audit.__dict__,
                },
            )

        # 4. Strict deterministic Python validation of cited passage references
        # Ensure model never cites passages that were not supplied
        retained_cited_passages: list[EvidencePassage] = []
        seen_cited_ids: set[str] = set()

        for cited_ref in parsed.cited_passage_ids:
            clean_ref = cited_ref.strip()
            if clean_ref in valid_passage_id_map:
                matched_passage = valid_passage_id_map[clean_ref]
                if matched_passage.id not in seen_cited_ids:
                    seen_cited_ids.add(matched_passage.id)
                    retained_cited_passages.append(matched_passage)
            else:
                logger.warning(
                    "Model cited unsupplied passage reference '%s'; discarded.",
                    clean_ref,
                )

        # If model did not cite specific passages, associate all supplied passages
        final_passages = retained_cited_passages if retained_cited_passages else evidence_passages

        # Critical Hardening Invariant: If model returns SUPPORTED but all cited passages were unsupplied/hallucinated,
        # missing evidence must NEVER result in a supported classification.
        status = parsed.status
        reasoning = parsed.reasoning
        confidence_score = parsed.confidence_score

        if status == VerificationStatus.SUPPORTED and parsed.cited_passage_ids and not retained_cited_passages:
            logger.warning(
                "Claim '%s' (id=%s) received supported verdict but all cited passages %s were invalid/unsupplied. Demoting to INSUFFICIENT_EVIDENCE.",
                claim.claim_text,
                claim.id,
                parsed.cited_passage_ids,
            )
            status = VerificationStatus.INSUFFICIENT_EVIDENCE
            reasoning = f"Model verdict demoted to insufficient_evidence: all cited passage references {parsed.cited_passage_ids} were unsupplied."
            confidence_score = 0.0

        return VerificationResult(
            claim_id=claim.id,
            status=status,
            confidence_score=confidence_score,
            reasoning=reasoning,
            evidence_passages=final_passages,
            verified_at=utc_now(),
            model_name=self.llm_client.model_name,
            metadata={
                "corroboration": corroboration_audit.__dict__,
                "raw_cited_ids": parsed.cited_passage_ids,
            },
        )
