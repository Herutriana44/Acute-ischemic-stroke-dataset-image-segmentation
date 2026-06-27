"""
Ngrok port forwarding for FastAPI inference server.
Reads NGROK_AUTHTOKEN from .env file.

Usage:
  python ngrok_tunnel.py [--port PORT]

Example:
  python ngrok_tunnel.py --port 65234
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import ngrok
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ngrok_tunnel")

ENV_PATH = Path(__file__).resolve().parent / ".env"


def main():
    parser = argparse.ArgumentParser(description="Ngrok tunnel for FastAPI server")
    parser.add_argument("--port", type=int, default=65234, help="Local port to forward")
    args = parser.parse_args()

    # Load .env
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        logger.info("Loaded .env from %s", ENV_PATH)
    else:
        logger.warning(".env not found at %s, trying environment variables", ENV_PATH)

    authtoken = os.getenv("NGROK_AUTHTOKEN")
    if not authtoken:
        logger.error("NGROK_AUTHTOKEN not set. Add it to %s or export as env var.", ENV_PATH)
        sys.exit(1)

    # Authenticate
    listener = ngrok.forward(
        addr=f"localhost:{args.port}",
        authtoken=authtoken,
    )

    public_url = listener.url()
    logger.info("Ngrok tunnel established")
    logger.info("  Public URL: %s", public_url)
    logger.info("  Local port: %d", args.port)
    logger.info("  Docs:       %s/docs", public_url)
    logger.info("  Health:     %s/health", public_url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down ngrok tunnel...")
        listener.close()
        logger.info("Tunnel closed.")


if __name__ == "__main__":
    main()
