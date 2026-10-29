"""CLI entrypoint for running the API server directly via `python -m news_agent.api`."""

import argparse
import uvicorn

from news_agent.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Local News Intelligence Agent REST API")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reload (development only)")
    args = parser.parse_args()

    print(f"Starting Local News Intelligence Agent API on http://{args.host}:{args.port}")
    print(f"Swagger interactive documentation: http://{args.host}:{args.port}/docs")
    print(f"Readiness endpoint: http://{args.host}:{args.port}/ready")
    uvicorn.run(
        "news_agent.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
