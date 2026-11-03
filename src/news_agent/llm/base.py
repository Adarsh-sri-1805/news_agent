"""Base interface and common utilities for LLM clients."""

import abc
import json
import re
from typing import Any, Type, TypeVar
from pydantic import BaseModel, ValidationError

from news_agent.llm.exceptions import LLMError, OllamaResponseValidationError

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(abc.ABC):
    """Abstract base client for LLM providers (Google Gemini, Ollama)."""

    provider_name: str = "base"
    model_name: str = "base"

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Verify service accessibility."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Generate raw text completion."""

    @abc.abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> T:
        """Generate structured data and strictly validate against Pydantic contract."""

    def close(self) -> None:
        """Release underlying client resources."""

    @staticmethod
    def _validate_structured_payload(raw_output: str, response_model: Type[T]) -> T:
        """Parse raw model text as JSON and validate against the provided Pydantic model."""
        cleaned = raw_output.strip()

        # Handle markdown fenced code blocks if returned by the LLM
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed_json = json.loads(cleaned)
        except json.JSONDecodeError as err:
            raise OllamaResponseValidationError(
                message=f"Model output is not valid JSON: {err}",
                raw_output=raw_output,
                validation_error=err,
            ) from err

        try:
            validated = response_model.model_validate(parsed_json)
            return validated
        except ValidationError as err:
            raise OllamaResponseValidationError(
                message=f"Model output failed Pydantic schema validation: {err}",
                raw_output=raw_output,
                validation_error=err,
            ) from err
