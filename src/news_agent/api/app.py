"""FastAPI application factory and middleware configuration."""

import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from news_agent.api.routes.health import router as health_router
from news_agent.api.routes.news import router as news_router
from news_agent.models.base import utc_now

logger = logging.getLogger("news_agent.api")


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    application = FastAPI(
        title="Local News Intelligence Agent API",
        description=(
            "REST API for the Local News Intelligence Agent. "
            "Coordinates news research, factual verification, synthesis of executive briefs, "
            "and promotional social copy generation using local Ollama inference."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS configuration for local testing and developer UIs
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    application.include_router(health_router)
    application.include_router(news_router)

    # Service information root endpoint
    @application.get(
        "/",
        summary="Service index",
        description="Index endpoint providing documentation links and service metadata.",
        tags=["Index"],
    )
    def index():
        return {
            "name": "Local News Intelligence Agent API",
            "version": "0.1.0",
            "status": "online",
            "docs_url": "/docs",
            "health_url": "/health",
            "readiness_url": "/ready",
            "timestamp": utc_now().isoformat(),
        }

    # Global unhandled exception handler to protect against sensitive information leaks
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred processing your request.",
                "timestamp": utc_now().isoformat(),
            },
        )

    return application


# Global application instance for uvicorn
app = create_app()
