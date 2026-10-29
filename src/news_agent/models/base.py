"""Base model and timestamp utilities for news agent domain entities."""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator


def utc_now() -> datetime:
    """Return the current UTC-aware datetime."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware and set to UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # If naive datetime provided, treat as UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class NewsAgentBaseModel(BaseModel):
    """Base Pydantic model with strict validation and standard configuration."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )
