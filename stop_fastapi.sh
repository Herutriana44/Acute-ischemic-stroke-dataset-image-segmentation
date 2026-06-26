#!/bin/bash
# Stop FastAPI inference server
# Usage: ./stop_fastapi.sh [--force]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/.fastapi.pid"
FORCE="${1:-}"

if [ ! -f "$PID_FILE" ]; then
    echo "ERROR: FastAPI server is not running (no PID file found)"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: Process $PID is not running"
    rm -f "$PID_FILE"
    exit 1
fi

echo "Stopping FastAPI server (PID $PID)..."

if [ "$FORCE" = "--force" ]; then
    echo "Force stopping..."
    kill -9 "$PID" || true
else
    echo "Sending SIGTERM (graceful shutdown)..."
    kill -15 "$PID" || true

    # Wait up to 10 seconds for graceful shutdown
    for i in {1..20}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "Server stopped gracefully."
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 0.5
    done

    echo "Graceful shutdown timeout, force killing..."
    kill -9 "$PID" || true
fi

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Server stopped."
else
    echo "WARNING: Process may still be running (PID $PID)"
fi

rm -f "$PID_FILE"
echo "PID file removed."