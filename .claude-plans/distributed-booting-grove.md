# FastAPI Inference API with Auto-GPU & Queue System

## Context

Currently, the project has a Flask webapp with synchronous inference endpoints for:
- 2D images (PNG/JPG)
- Single DICOM files (.dcm)
- DICOM series (ZIP archives)

The inference logic is battle-tested in `webapp/services/inference_service.py` and `webapp/services/archive_service.py`. However:
- **No async/queue support**: inference blocks the request thread (problematic for GPU inference which can take minutes)
- **No background task tracking**: no way to poll job status or handle long-running ops
- **GPU detection is present but minimal**: only `torch.cuda.is_available()` in one place

Goal: Build a new FastAPI API layer that:
1. Reuses existing inference logic (no reimplementation)
2. Adds a simple in-memory queue for async job submission
3. Auto-detects GPU and routes inference to it
4. Provides endpoints for: submit job, poll status, list jobs, get results

## Architecture

### Queue System
- **Type**: Simple in-memory queue using Python's `queue.Queue` + threading
- **Why**: No external dependency (Redis/RabbitMQ), simple, sufficient for this use case
- **Workers**: Single background thread pool (initially 1 worker to avoid GPU contention)
- **Job states**: `pending` → `running` → `completed` / `failed`
- **Storage**: Jobs stored in `runs/` directory (persistent across restarts for historical queries)

### GPU Auto-Detection
- Detect CUDA at startup via `torch.cuda.is_available()`
- Pass device (`cuda` / `cpu`) to existing inference functions
- Log device used in each job's result.json

### API Endpoints (FastAPI)

#### Submit endpoints (async, immediate response with job_id):
- `POST /api/v1/jobs/submit-image` → upload 2D image → return `{job_id, status}`
- `POST /api/v1/jobs/submit-dicom` → upload single .dcm → return `{job_id, status}`
- `POST /api/v1/jobs/submit-series` → upload DICOM ZIP → return `{job_id, status}`

#### Query endpoints (sync, fast):
- `GET /api/v1/jobs/{job_id}` → return full job result or status if running
- `GET /api/v1/jobs` → list all jobs with summary (run_id, status, created_at)

#### Backward compat (optional):
- Keep Flask endpoints working for UI; FastAPI is API-only for now

### File Structure
```
fastapi_api/
  __init__.py
  app.py                  # FastAPI app factory + routes
  queue.py                # JobQueue class (thread-safe queue + worker pool)
  models.py               # Pydantic models (JobStatus, JobResult, etc.)
  inference_adapter.py    # Wrapper around existing inference_service functions
```

## Implementation Steps

### Phase 1: Create queue system + models (no endpoints yet)
1. **Create `fastapi_api/models.py`**:
   - Pydantic models: `JobStatus` (enum: pending/running/completed/failed)
   - `JobSubmitRequest`, `JobStatusResponse`, `JobResultResponse`
   
2. **Create `fastapi_api/queue.py`**:
   - `Job` dataclass: {job_id, job_type (image/dicom/series), status, input_path, result, error_msg, created_at, completed_at}
   - `JobQueue` class: manages queue.Queue + background worker thread
   - Methods: `submit_job()`, `get_job()`, `list_jobs()`, `_worker_loop()`

3. **Create `fastapi_api/inference_adapter.py`**:
   - Wrapper functions around `webapp.services.inference_service`
   - Handle device selection (auto-GPU) + exception wrapping
   - Signature: `infer_2d_image(image_path, run_id, model_path, runs_dir, device) → dict`

### Phase 2: Create FastAPI app + endpoints
1. **Create `fastapi_api/app.py`**:
   - FastAPI app factory with global JobQueue instance
   - Routes: POST submit-* (queue job, return immediately)
   - Routes: GET /jobs/{job_id} (poll status)
   - Routes: GET /jobs (list all)

2. **Update `requirements.txt`**:
   - Add: `fastapi`, `uvicorn[standard]`, `python-multipart`

3. **Create entrypoint script** (e.g., `run_fastapi.py`):
   ```python
   if __name__ == "__main__":
       import uvicorn
       uvicorn.run("fastapi_api.app:create_app()", host="0.0.0.0", port=8000, reload=False)
   ```

### Phase 3: Integration & Testing
1. Verify existing inference functions still work when called from adapter
2. Test queue submission + job polling
3. Test GPU detection (log to console)
4. Verify results match Flask app (bit-for-bit or close enough)

## Key Reuse Points

| Component | Location | Reuse Strategy |
|-----------|----------|-----------------|
| `run_inference` | `webapp/services/inference_service.py:180` | Import & call directly |
| `run_inference_image` | `webapp/services/inference_service.py:372` | Import & call directly |
| `extract_archive` | `webapp/services/archive_service.py:56` | Import & call directly |
| `find_dicom_series_dir` | `webapp/services/archive_service.py:99` | Import & call directly |
| `InferenceError` | `webapp/services/inference_service.py:31` | Import exception class |
| GPU detection | `infer_dicom_unet.py:118` | Pattern: `torch.cuda.is_available()` |

## Critical Files to Modify

- **Create new**: `fastapi_api/__init__.py`, `fastapi_api/app.py`, `fastapi_api/queue.py`, `fastapi_api/models.py`, `fastapi_api/inference_adapter.py`
- **Update**: `requirements.txt` (add fastapi, uvicorn, python-multipart)
- **Create new**: `run_fastapi.py` (entry point)

## Verification Plan

1. **Functional tests**:
   - Submit 2D image → poll until complete → verify output files exist
   - Submit DICOM series → poll until complete → verify mask volume in result
   - Submit single DICOM → verify quick completion
   - Verify GPU is used if available (check console logs + device field in result.json)

2. **Comparison tests**:
   - Run same image through Flask /api/predict/image vs FastAPI POST /api/v1/jobs/submit-image
   - Ensure result.json fields match or are compatible

3. **Queue tests**:
   - Submit 2 jobs → verify first completes before second starts (single worker)
   - Verify job status transitions: pending → running → completed
   - Verify failed job (bad image) transitions to failed state

## Constraints & Assumptions

- Model checkpoint (`best_unet.pt`) location same as Flask app expects
- CUDA availability is stable (no hot-plugging GPUs)
- Single-worker queue is sufficient (avoid GPU memory contention)
- Inference output format from existing functions is immutable
- Using existing `runs/` directory structure for result storage

## Non-Goals

- Multi-GPU support (single worker, single device)
- Distributed queue (Celery/RabbitMQ) — in-memory only
- Authentication/rate limiting on queue endpoints
- WebSocket progress updates — polling only
- Modifying Flask app (it stays as-is for UI)
