"""Ollama local inference client wrapper with structured output validation."""

import json
import re
from typing import Any, Type, TypeVar
import httpx
from pydantic import BaseModel, ValidationError

from news_agent.config import settings
from news_agent.llm.base import BaseLLMClient
from news_agent.llm.exceptions import (
    OllamaAPIError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseValidationError,
    OllamaTimeoutError,
)

T = TypeVar("T", bound=BaseModel)


class OllamaClient(BaseLLMClient):
    """Synchronous client wrapper for interacting with a local Ollama instance.

    Design constraints:
    - Target hardware: 16 GB RAM, CPU-only.
    - Zero automatic model downloads.
    - Strict validation of all model responses.
    - Explicit timeout and connection error handling.
    """

    @property
    def provider_name(self) -> str:
        return "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
        keep_alive: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model_name = model_name or settings.ollama_model
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.ollama_timeout_seconds
        )
        self.keep_alive = keep_alive or settings.ollama_keep_alive
        self._owned_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
        )

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client session if created internally."""
        if self._owned_client and not self._client.is_closed:
            self._client.close()

    def health_check(self) -> bool:
        """Check whether the local Ollama server is running and accessible."""
        try:
            response = self._client.get("/api/version")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    def list_local_models(self) -> list[str]:
        """Fetch list of models currently installed in the local Ollama daemon.

        Does NOT download or pull any models.
        """
        try:
            response = self._client.get("/api/tags")
        except httpx.ConnectError as err:
            raise OllamaConnectionError(
                f"Could not connect to Ollama daemon at {self.base_url}. Ensure Ollama is running."
            ) from err
        except httpx.TimeoutException as err:
            raise OllamaTimeoutError(
                f"Timeout ({self.timeout_seconds}s) connecting to Ollama at {self.base_url}."
            ) from err

        if response.status_code != 200:
            raise OllamaAPIError(response.status_code, response.text)

        payload = response.json()
        models = [m.get("name") for m in payload.get("models", []) if m.get("name")]
        return models

    def verify_model_installed(self) -> bool:
        """Verify if the configured model is installed locally.

        Raises:
            OllamaModelNotFoundError: If model is not found.
        """
        models = self.list_local_models()
        # Ollama tags may have ':latest' appended by default
        normalized_target = self.model_name if ":" in self.model_name else f"{self.model_name}:latest"
        available = {m for m in models} | {m.split(":")[0] for m in models}

        if self.model_name not in available and normalized_target not in available:
            raise OllamaModelNotFoundError(
                f"Model '{self.model_name}' is not installed locally. "
                f"Available models: {models}. "
                f"Please manually execute 'ollama pull {self.model_name}'. "
                "Automatic downloads are strictly disabled."
            )
        return True

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Generate a raw text completion from the local Ollama model."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        data = self._post_request("/api/generate", payload)
        raw_response = data.get("response", "")
        return raw_response

    def chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a multi-turn or formatted chat request to the local Ollama model."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        if options:
            payload["options"] = options

        data = self._post_request("/api/chat", payload)
        message = data.get("message", {})
        return message.get("content", "")

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> T:
        """Generate structured data and strictly validate it against a Pydantic contract.

        Model outputs are treated as untrusted and must pass validation.
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        data = self._post_request("/api/generate", payload)
        raw_output = data.get("response", "")

        return self._validate_structured_payload(raw_output, response_model)

    def _post_request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute a POST request with robust error handling."""
        try:
            response = self._client.post(endpoint, json=payload)
        except httpx.ConnectError as err:
            raise OllamaConnectionError(
                f"Could not connect to Ollama daemon at {self.base_url}."
            ) from err
        except httpx.TimeoutException as err:
            raise OllamaTimeoutError(
                f"Ollama request exceeded timeout of {self.timeout_seconds} seconds."
            ) from err
        except httpx.RequestError as err:
            raise OllamaConnectionError(f"HTTP request error: {err}") from err

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{self.model_name}' not found on Ollama server at {self.base_url}. "
                "Automatic downloading is disabled."
            )
        if response.status_code != 200:
            raise OllamaAPIError(response.status_code, response.text)

        try:
            return response.json()
        except ValueError as err:
            raise OllamaAPIError(
                response.status_code,
                f"Failed to parse Ollama daemon response JSON: {response.text}",
            ) from err

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
