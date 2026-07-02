# ── Hugging Face Spaces Dockerfile ──────────────────────────────────
# FastAPI inference API for acute ischemic stroke segmentation
# HF Spaces default port: 7860

FROM python:3.11-slim

# System deps for OpenCV, patool (archive extraction), and misc
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    p7zip-full \
    unrar-free \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────
COPY . .

# ── Runtime ────────────────────────────────────────────────────────
ENV MODEL_PATH=/app/best_unet.pt
ENV YOLO_MODEL_PATH=/app/yolo_best.pt
ENV API_RUNS_DIR=/app/api_runs
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "fastapi_api.app:app", "--host", "0.0.0.0", "--port", "7860"]
