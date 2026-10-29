"""Health and readiness check endpoints."""

import logging
from fastapi import APIRouter, Depends, Response, status

from news_agent.api.dependencies import get_ollama_client, get_settings
from news_agent.api.schemas import DependencyDetail, HealthResponse, ReadinessResponse
from news_agent.config import Settings
from news_agent.llm.client import OllamaClient
from news_agent.models.base import utc_now

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health & Readiness"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic service liveness check",
    description="Returns immediate service status. Does not execute model inference or news ingestion.",
)
def get_health() -> HealthResponse:
    """Simple service-health liveness probe."""
    return HealthResponse(
        status="healthy",
        app="local-news-intelligence-agent",
        version="0.1.0",
        timestamp=utc_now(),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        200: {"description": "Service and all required dependencies are ready."},
        503: {"description": "One or more critical dependencies (e.g. Ollama daemon) are unavailable."},
    },
    summary="Comprehensive dependency readiness check",
    description=(
        "Checks availability of required dependencies including local Ollama connectivity "
        "and configured model installation. Never automatically pulls or downloads models."
    ),
)
def get_readiness(
    response: Response,
    client: OllamaClient = Depends(get_ollama_client),
    app_settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """Verify system readiness and external component health without crashing on failure."""
    dependencies: dict[str, DependencyDetail] = {}
    ollama_ready = False

    # 1. Ollama Daemon Connectivity Check
    try:
        is_accessible = client.health_check()
        if is_accessible:
            # Check model presence without triggering downloads
            installed_models = client.list_local_models()
            target_model = client.model_name
            normalized_target = target_model if ":" in target_model else f"{target_model}:latest"
            available_set = {m for m in installed_models} | {m.split(":")[0] for m in installed_models}

            model_installed = target_model in available_set or normalized_target in available_set
            if model_installed:
                ollama_ready = True
                dependencies["ollama"] = DependencyDetail(
                    status="ready",
                    details={
                        "base_url": client.base_url,
                        "model": target_model,
                        "model_installed": True,
                        "available_models": installed_models,
                    },
                )
            else:
                dependencies["ollama"] = DependencyDetail(
                    status="not_ready",
                    details={
                        "base_url": client.base_url,
                        "model": target_model,
                        "model_installed": False,
                        "available_models": installed_models,
                        "instruction": f"Run 'ollama pull {target_model}' to install the configured model.",
                    },
                )
        else:
            dependencies["ollama"] = DependencyDetail(
                status="unavailable",
                details={
                    "base_url": client.base_url,
                    "model": client.model_name,
                    "instruction": "Ensure the Ollama daemon is running locally.",
                },
            )
    except Exception as exc:
        logger.warning("Error inspecting Ollama readiness: %s", exc)
        dependencies["ollama"] = DependencyDetail(
            status="unavailable",
            details={
                "base_url": client.base_url,
                "error": str(exc),
            },
        )

    # 2. Configured News Ingestion Sources Check
    feeds_count = len(app_settings.rss_feeds)
    dependencies["ingestion_sources"] = DependencyDetail(
        status="ready" if feeds_count > 0 else "warning",
        details={
            "configured_rss_feeds": feeds_count,
            "gdelt_enabled": app_settings.gdelt_enabled,
        },
    )

    is_overall_ready = ollama_ready
    if not is_overall_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_overall_ready else "not_ready",
        ready=is_overall_ready,
        dependencies=dependencies,
        timestamp=utc_now(),
    )
