"""
Entry point for FastAPI inference server.

Usage:
  python run_fastapi.py [--host HOST] [--port PORT] [--workers N]

Example:
  python run_fastapi.py --host 0.0.0.0 --port 8000 --workers 1
"""

import argparse
import logging
import os
from pathlib import Path

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fastapi")


def main():
    parser = argparse.ArgumentParser(description="FastAPI Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker threads for inference queue"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (best_unet.pt)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Directory to store inference results",
    )
    args = parser.parse_args()

    # Set env vars for app
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.runs_dir:
        os.environ["API_RUNS_DIR"] = args.runs_dir

    # Import app after env vars set
    from fastapi_api.app import create_app

    app = create_app()
    logger.info("Starting FastAPI server on %s:%d with %d inference worker(s)", args.host, args.port, args.workers)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,  # FastAPI workers (not inference workers)
        log_level="info",
    )


if __name__ == "__main__":
    main()