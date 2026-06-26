#!/bin/bash
# Start FastAPI inference server in background (nohup)
# Usage: ./start_fastapi.sh [port] [workers]
# Default: port=8000, workers=1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-65234}"
WORKERS="${2:-1}"
PID_FILE="$SCRIPT_DIR/.fastapi.pid"
LOG_FILE="$SCRIPT_DIR/fastapi.log"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: FastAPI server already running (PID $OLD_PID)"
        echo "  Stop it first: ./stop_fastapi.sh"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

echo "Starting FastAPI inference server..."
echo "  Port:    $PORT"
echo "  Workers: $WORKERS"
echo "  Log:     $LOG_FILE"
echo "  PID:     $PID_FILE"

cd "$SCRIPT_DIR"
nohup python run_fastapi.py --host 0.0.0.0 --port "$PORT" --workers "$WORKERS" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "Server started successfully (PID $PID)"
    echo "  API:      http://0.0.0.0:$PORT"
    echo "  Docs:     http://0.0.0.0:$PORT/docs"
    echo "  Health:   http://0.0.0.0:$PORT/health"
    echo "  Stop:     ./stop_fastapi.sh"
else
    echo "ERROR: Server failed to start. Check $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi