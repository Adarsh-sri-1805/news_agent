"""Google Gemini cloud inference client wrapper with structured output validation."""

import json
import logging
import os
import re
from typing import Any, Type, TypeVar
import httpx
from pydantic import BaseModel, ValidationError

from news_agent.config import settings
from news_agent.llm.base import BaseLLMClient
from news_agent.llm.exceptions import (
    GeminiAPIError,
    GeminiError,
    GeminiResponseValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GeminiClient(BaseLLMClient):
    """Client wrapper for interacting with the Google Gemini API (Free Tier).

    Features:
    - Fast cloud inference (~1-2 seconds per request vs local CPU).
    - Native JSON structured generation with Pydantic validation.
    - Zero cost using Google AI Studio free tier.
    - Full mock transport support for deterministic offline testing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or settings.gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()
        self._model_name = model_name or settings.gemini_model or "gemini-3.5-flash"
        self.timeout_seconds = timeout_seconds or 60.0
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._owned_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def close(self) -> None:
        """Close the underlying HTTP client session if owned."""
        if self._owned_client and not self._client.is_closed:
            self._client.close()

    def health_check(self) -> bool:
        """Check whether the Gemini API key is valid and the model is reachable."""
        if not self.api_key:
            return False
        try:
            url = f"{self.base_url}/models/{self._model_name}?key={self.api_key}"
            response = self._client.get(url)
            return response.status_code == 200
        except Exception as err:
            logger.debug("Gemini health check probe failed: %s", err)
            return False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Generate raw text completion from Gemini."""
        if not self.api_key:
            raise GeminiAPIError(
                401,
                "Gemini API key is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env.",
            )

        url = f"{self.base_url}/models/{self._model_name}:generateContent?key={self.api_key}"
        gen_config: dict[str, Any] = {
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if options:
            gen_config.update(options)

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": gen_config,
        }
        if system:
            payload["systemInstruction"] = {
                "parts": [{"text": system}]
            }

        data = self._post_request(url, payload)
        return self._extract_candidate_text(data)

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> T:
        """Generate structured JSON and validate strictly against a Pydantic contract."""
        if not self.api_key:
            raise GeminiAPIError(
                401,
                "Gemini API key is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env.",
            )

        url = f"{self.base_url}/models/{self._model_name}:generateContent?key={self.api_key}"
        gen_config = {
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if options:
            gen_config.update(options)

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": gen_config,
        }
        if system:
            payload["systemInstruction"] = {
                "parts": [{"text": system}]
            }

        data = self._post_request(url, payload)
        raw_output = self._extract_candidate_text(data)
        return self._validate_structured_payload(raw_output, response_model)

    def _post_request(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute POST request against Gemini REST API with error mapping."""
        try:
            response = self._client.post(url, json=payload)
        except httpx.TimeoutException as err:
            raise GeminiError(f"Gemini API request timed out after {self.timeout_seconds}s: {err}") from err
        except httpx.RequestError as err:
            raise GeminiError(f"Network error connecting to Gemini API: {err}") from err

        if response.status_code != 200:
            error_msg = response.text
            try:
                err_json = response.json()
                error_msg = err_json.get("error", {}).get("message", response.text)
            except Exception:
                pass
            raise GeminiAPIError(response.status_code, error_msg)

        try:
            return response.json()
        except ValueError as err:
            raise GeminiAPIError(
                response.status_code,
                f"Failed to parse Gemini response JSON: {response.text}",
            ) from err

    @staticmethod
    def _extract_candidate_text(data: dict[str, Any]) -> str:
        """Extract response text from Gemini generateContent JSON response."""
        candidates = data.get("candidates", [])
        if not candidates:
            # Check prompt feedback (blocked content etc.)
            feedback = data.get("promptFeedback", {})
            block_reason = feedback.get("blockReason")
            if block_reason:
                raise GeminiError(f"Gemini blocked content with reason: {block_reason}")
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        text_segments = [p.get("text", "") for p in parts if "text" in p]
        return "".join(text_segments).strip()

