"""
FastAPI inference API application.

Endpoints:
- POST /api/v1/jobs/submit-image    → submit 2D image
- POST /api/v1/jobs/submit-dicom    → submit single DICOM file
- POST /api/v1/jobs/submit-series   → submit DICOM series (ZIP)
- GET  /api/v1/jobs/{job_id}        → poll job status/result
- GET  /api/v1/jobs                 → list all jobs
- GET  /api/v1/stats                → queue + GPU stats
- GET  /api/v1/runs/{job_id}/{file} → download result files
- GET  /health                      → health check
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from fastapi_api.models import JobType, ModelType
from fastapi_api.queue import JobQueue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fastapi_api")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "best_unet.pt"))
YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", PROJECT_ROOT / "yolo_best.pt"))
RUNS_DIR = Path(os.getenv("API_RUNS_DIR", PROJECT_ROOT / "api_runs"))
UPLOAD_TMP = RUNS_DIR / "_uploads"

# ── Allowed formats ────────────────────────────────────────────────────
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ARCHIVE_SUFFIXES = {".zip", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".tbz", ".xz", ".txz", ".7z"}

# Global queue instance (initialized in lifespan)
job_queue: JobQueue | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: manage job queue lifecycle."""
    global job_queue
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_TMP.mkdir(parents=True, exist_ok=True)

    job_queue = JobQueue(model_path=MODEL_PATH, runs_dir=RUNS_DIR, num_workers=1, yolo_model_path=YOLO_MODEL_PATH)
    job_queue.start()
    logger.info("FastAPI inference API ready. Model: %s | Runs: %s", MODEL_PATH, RUNS_DIR)
    logger.info("GPU available: %s | device: %s", job_queue.gpu_available, job_queue.device)
    yield
    job_queue.stop()
    logger.info("FastAPI inference API shutdown.")


def _get_queue() -> JobQueue:
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    return job_queue


def _save_upload(upload: UploadFile, allowed_suffixes: set[str]) -> Path:
    """Validate and save an uploaded file to a temp location. Returns path."""
    name = (upload.filename or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = "".join(Path(name).suffixes).lower()
    # For archives, check compound suffix; for images, simple suffix
    simple_suffix = Path(name).suffix.lower()
    is_allowed = simple_suffix in allowed_suffixes or any(
        suffix.endswith(s) for s in allowed_suffixes
    )
    if not is_allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{simple_suffix}'. Allowed: {sorted(allowed_suffixes)}",
        )

    tmp_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_TMP / f"{tmp_id}{simple_suffix}"
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
    finally:
        upload.file.close()
    return dest


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stroke Segmentation Inference API",
        description="Async inference API for 2D images, single DICOM, and DICOM series with auto-GPU + queue.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Health ─────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        q = _get_queue()
        return {"status": "ok", "gpu_available": q.gpu_available, "device": q.device}

    # ── Submit endpoints ───────────────────────────────────────────────
    @app.post("/api/v1/jobs/submit-image")
    async def submit_image(
        file: UploadFile = File(...),
        model_type: str = "unet",
    ):
        """Submit a 2D image for inference. Returns job_id immediately.

        Query params:
        - model_type: 'unet' (default) or 'yolo'
        """
        q = _get_queue()
        saved = _save_upload(file, IMAGE_SUFFIXES)
        mt = ModelType.YOLO if model_type.lower() == "yolo" else ModelType.UNET
        job_id = q.submit(JobType.IMAGE, str(saved), model_type=mt)
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "pending", "job_type": "image", "model_type": mt.value},
        )

    @app.post("/api/v1/jobs/submit-dicom")
    async def submit_dicom(
        file: UploadFile = File(...),
        model_type: str = "unet",
    ):
        """Submit a single DICOM (.dcm) file for inference. Returns job_id immediately.

        Query params:
        - model_type: 'unet' (default) or 'yolo'
        """
        q = _get_queue()
        saved = _save_upload(file, {".dcm"})
        mt = ModelType.YOLO if model_type.lower() == "yolo" else ModelType.UNET
        job_id = q.submit(JobType.DICOM, str(saved), model_type=mt)
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "pending", "job_type": "dicom", "model_type": mt.value},
        )

    @app.post("/api/v1/jobs/submit-series")
    async def submit_series(
        file: UploadFile = File(...),
        model_type: str = "unet",
    ):
        """Submit a DICOM series (ZIP/archive) for inference. Returns job_id immediately.

        Query params:
        - model_type: 'unet' (default) or 'yolo'
        """
        q = _get_queue()
        saved = _save_upload(file, ARCHIVE_SUFFIXES)
        mt = ModelType.YOLO if model_type.lower() == "yolo" else ModelType.UNET
        job_id = q.submit(JobType.SERIES, str(saved), model_type=mt)
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "pending", "job_type": "series", "model_type": mt.value},
        )

    # ── Query endpoints ────────────────────────────────────────────────
    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        """Get job status and result (if completed)."""
        q = _get_queue()
        job = q.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job.to_result()

    @app.get("/api/v1/jobs")
    async def list_jobs():
        """List all jobs (newest first)."""
        q = _get_queue()
        jobs = q.list_jobs()
        return {"count": len(jobs), "jobs": [j.to_status() for j in jobs]}

    @app.get("/api/v1/stats")
    async def stats():
        """Queue and GPU statistics."""
        q = _get_queue()
        return q.stats()

    # ── Result file download ───────────────────────────────────────────
    @app.get("/api/v1/runs/{job_id}/{filename:path}")
    async def get_run_file(job_id: str, filename: str):
        """Download a result file from a completed run."""
        # Sanitize job_id
        safe_id = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))
        if safe_id != job_id:
            raise HTTPException(status_code=404, detail="Invalid job_id")

        run_dir = (RUNS_DIR / job_id).resolve()
        if RUNS_DIR.resolve() not in run_dir.parents and run_dir != RUNS_DIR.resolve():
            raise HTTPException(status_code=404, detail="Run not found")

        file_path = (run_dir / filename).resolve()
        # Prevent path traversal
        if run_dir not in file_path.parents and file_path != run_dir:
            raise HTTPException(status_code=404, detail="File not found")
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(str(file_path))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
