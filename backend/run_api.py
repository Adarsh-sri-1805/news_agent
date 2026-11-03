"""Convenience CLI entrypoint to launch the FastAPI server directly.

Usage:
    python run_api.py
    python run_api.py --port 8000 --host 127.0.0.1 --reload
"""

import argparse
import os
from pathlib import Path
import sys

# Ensure src/ is on sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn
from news_agent.config import settings


def main() -> None:
    default_port = int(os.environ.get("PORT", 8000))
    default_host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")

    parser = argparse.ArgumentParser(description="Local News Intelligence Agent REST API Runner")
    parser.add_argument("--host", default=default_host, help="Host interface to bind to (default: 0.0.0.0 if PORT set, else 127.0.0.1)")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to listen on (default: {default_port})")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reloading")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  LOCAL NEWS INTELLIGENCE AGENT - REST API SERVER")
    print("=" * 70)
    print(f"  Base URL:     http://{args.host}:{args.port}")
    print(f"  Swagger Docs: http://{args.host}:{args.port}/docs")
    print(f"  Health Check: http://{args.host}:{args.port}/health")
    print(f"  Readiness:    http://{args.host}:{args.port}/ready")
    print("=" * 70 + "\n")

    uvicorn.run(
        "news_agent.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
