"""News intelligence execution endpoint."""

import logging
import time
from fastapi import APIRouter, Depends, HTTPException, status

from news_agent.agents.orchestrator import NewsIntelligenceOrchestrator, OrchestratorResult
from news_agent.api.dependencies import get_orchestrator
from news_agent.api.mock_runner import run_mock_workflow
from news_agent.api.schemas import ErrorResponse, NewsRunRequest, NewsRunResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/news", tags=["News Intelligence"])


@router.post(
    "/run",
    response_model=NewsRunResponse,
    responses={
        200: {"description": "Workflow completed successfully with full structured results."},
        422: {"description": "Input validation error (e.g. invalid lookback hours or unsupported platform)."},
        500: {"model": ErrorResponse, "description": "Internal error during intelligence pipeline execution."},
    },
    summary="Execute end-to-end news intelligence workflow",
    description=(
        "Invokes the multi-agent orchestrator across News Researcher, Fact-Checker, "
        "Briefing Agent, and Social Content Agent. Supports optional mock_mode for safe offline testing "
        "without requiring active feeds or Ollama daemon."
    ),
)
def run_news_pipeline(
    request: NewsRunRequest,
    orchestrator: NewsIntelligenceOrchestrator = Depends(get_orchestrator),
) -> NewsRunResponse:
    """Execute complete news intelligence workflow and return structured results."""
    req_t0 = time.perf_counter()
    logger.info(
        "Received /api/v1/news/run request (mock_mode=%s, lookback_hours=%s, platforms=%s, custom_articles=%s, max_articles=%s)",
        request.mock_mode,
        request.lookback_hours,
        request.platforms,
        len(request.articles) if request.articles else 0,
        request.max_articles,
    )

    try:
        if request.mock_mode:
            # Safe mock mode for offline demonstration and testing
            result: OrchestratorResult = run_mock_workflow(
                lookback_hours=request.lookback_hours,
                platforms=request.platforms,
                articles=request.articles if request.articles else None,
            )
        elif request.articles:
            # Direct article input mode (invoked only when articles are explicitly supplied and non-empty)
            result = orchestrator.run_with_articles(
                articles=request.articles,
                platforms=request.platforms,
            )
        else:
            # Live researcher/ingestion workflow (invoked when articles is omitted or explicitly empty [])
            run_kwargs = {
                "lookback_hours": request.lookback_hours,
                "platforms": request.platforms,
            }
            if request.max_articles is not None:
                run_kwargs["max_articles"] = request.max_articles
            result = orchestrator.run(**run_kwargs)

        duration_ms = round((time.perf_counter() - req_t0) * 1000, 2)
        logger.info(
            "/api/v1/news/run completed in %.2f ms (retained=%d, claims=%d, verifications=%d, drafts=%d)",
            duration_ms,
            len(result.articles),
            len(result.claims),
            len(result.verification_results),
            len(result.social_drafts),
        )

        return NewsRunResponse.from_orchestrator_result(result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error executing news intelligence pipeline: %s", exc)
        # Avoid exposing raw stack traces, secrets, or internal paths
        sanitized_msg = str(exc).split("\n")[0]
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"News intelligence execution failed: {sanitized_msg}",
        ) from exc
