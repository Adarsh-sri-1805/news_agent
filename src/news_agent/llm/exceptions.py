"""Exceptions for the Ollama LLM integration."""


class OllamaError(Exception):
    """Base exception for all Ollama-related errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama service cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request times out."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not installed locally in Ollama."""


class OllamaResponseValidationError(OllamaError):
    """Raised when Ollama output fails structured validation or JSON parsing."""

    def __init__(self, message: str, raw_output: str, validation_error: Exception | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.validation_error = validation_error


class OllamaAPIError(OllamaError):
    """Raised when Ollama returns an HTTP error or error message in response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Ollama API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message
