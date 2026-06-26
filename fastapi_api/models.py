"""
Pydantic models for FastAPI inference API.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class JobStatus(str, enum.Enum):
    """Job lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, enum.Enum):
    """Type of inference job."""
    IMAGE = "image"
    DICOM = "dicom"
    SERIES = "series"


class JobSubmitRequest(BaseModel):
    """Request to submit a new inference job."""
    # No fields for now — file upload handled via FastAPI's UploadFile
    pass


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: JobStatus
    job_type: JobType
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class JobResultResponse(BaseModel):
    """Response for completed job result."""
    job_id: str
    status: JobStatus
    job_type: JobType
    created_at: datetime
    completed_at: datetime
    result: dict  # InferenceResult or ImageInferenceResult dict
    error_message: Optional[str] = None


class JobListResponse(BaseModel):
    """Response for listing all jobs."""
    count: int
    jobs: list[JobStatusResponse]


class JobQueueStats(BaseModel):
    """Queue statistics."""
    pending_count: int
    running_count: int
    completed_count: int
    failed_count: int
    worker_count: int
    gpu_available: bool