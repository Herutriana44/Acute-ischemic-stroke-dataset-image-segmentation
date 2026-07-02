"""
Simple in-memory thread-safe job queue with background workers.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Callable, Optional

from fastapi_api.models import JobStatus, JobType, ModelType

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """A single inference job."""
    job_id: str
    job_type: JobType
    status: JobStatus
    input_path: str
    created_at: datetime
    model_type: ModelType = ModelType.UNET
    completed_at: Optional[datetime] = None
    result: Optional[dict] = None
    error_message: Optional[str] = None

    def to_status(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "job_type": self.job_type.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }

    def to_result(self) -> dict:
        d = self.to_status()
        d["result"] = self.result
        return d


class JobQueue:
    """Thread-safe job queue with background worker pool.

    Workers process inference jobs sequentially (single worker default)
    to avoid GPU memory contention.
    """

    def __init__(self, model_path: Path, runs_dir: Path, num_workers: int = 1, yolo_model_path: Path | None = None):
        self._model_path = model_path
        self._yolo_model_path = yolo_model_path or model_path
        self._runs_dir = runs_dir
        self._num_workers = num_workers
        self._gpu_available: Optional[bool] = None  # determined on first use

        self._queue: Queue[Job] = Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._active: bool = False

        # Detect GPU once at init
        import torch
        self._gpu_available = torch.cuda.is_available()
        logger.info(
            "GPU %s — using %d worker(s)",
            "available" if self._gpu_available else "NOT available",
            self._num_workers,
        )

    @property
    def gpu_available(self) -> bool:
        return bool(self._gpu_available)

    @property
    def device(self) -> str:
        return "cuda" if self.gpu_available else "cpu"

    def _new_job_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def submit(self, job_type: JobType, input_path: str, model_type: ModelType = ModelType.UNET) -> str:
        """Submit a new inference job. Returns job_id immediately."""
        job = Job(
            job_id=self._new_job_id(),
            job_type=job_type,
            status=JobStatus.PENDING,
            input_path=input_path,
            created_at=datetime.utcnow(),
            model_type=model_type,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put(job)
        logger.info("Job %s submitted (%s, %s)", job.job_id, job_type.value, input_path)
        return job.job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            # Return newest first
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def stats(self) -> dict:
        with self._lock:
            pending = sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)
            running = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
            completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
            failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)
        return {
            "pending_count": pending,
            "running_count": running,
            "completed_count": completed,
            "failed_count": failed,
            "worker_count": self._num_workers,
            "gpu_available": self.gpu_available,
        }

    def start(self) -> None:
        """Start worker threads."""
        if self._active:
            return
        self._active = True
        for i in range(self._num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"infer-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        logger.info("JobQueue started with %d worker(s)", self._num_workers)

    def stop(self) -> None:
        """Signal workers to stop after current job finishes."""
        self._active = False
        logger.info("JobQueue stopping...")

    def _worker_loop(self) -> None:
        """Worker pulls jobs from queue and processes them."""
        while self._active:
            try:
                job = self._queue.get(timeout=1.0)
            except:  # queue empty, retry
                continue

            # Mark running
            with self._lock:
                job.status = JobStatus.RUNNING

            logger.info("Worker processing job %s (%s)...", job.job_id, job.job_type.value)
            try:
                self._process_job(job)
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                logger.info("Job %s completed.", job.job_id)
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error_message = str(exc)
                logger.error("Job %s failed: %s", job.job_id, exc)

            self._queue.task_done()

    def _process_job(self, job: Job) -> None:
        """Route job to appropriate inference function based on job_type and model_type."""
        from fastapi_api.inference_adapter import (
            infer_dicom_series,
            infer_dicom_series_yolo,
            infer_single_dicom,
            infer_single_dicom_yolo,
            infer_2d_image,
            infer_2d_image_yolo,
        )

        device = self.device
        input_path = Path(job.input_path)
        run_id = job.job_id

        if job.job_type == JobType.SERIES:
            if job.model_type == ModelType.YOLO:
                result = infer_dicom_series_yolo(
                    archive_path=input_path,
                    run_id=run_id,
                    model_path=self._yolo_model_path,
                    runs_dir=self._runs_dir,
                    device=device,
                )
            else:
                result = infer_dicom_series(
                    archive_path=input_path,
                    run_id=run_id,
                    model_path=self._model_path,
                    runs_dir=self._runs_dir,
                    device=device,
                )
        elif job.job_type == JobType.DICOM:
            if job.model_type == ModelType.YOLO:
                result = infer_single_dicom_yolo(
                    dicom_path=input_path,
                    run_id=run_id,
                    model_path=self._yolo_model_path,
                    runs_dir=self._runs_dir,
                    device=device,
                )
            else:
                result = infer_single_dicom(
                    dicom_path=input_path,
                    run_id=run_id,
                    model_path=self._model_path,
                    runs_dir=self._runs_dir,
                    device=device,
                )
        elif job.job_type == JobType.IMAGE:
            if job.model_type == ModelType.YOLO:
                result = infer_2d_image_yolo(
                    image_path=input_path,
                    run_id=run_id,
                    model_path=self._yolo_model_path,
                    runs_dir=self._runs_dir,
                    device=device,
                )
            else:
                result = infer_2d_image(
                    image_path=input_path,
                    run_id=run_id,
                    model_path=self._model_path,
                    runs_dir=self._runs_dir,
                    device=device,
                )
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")

        job.result = result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
