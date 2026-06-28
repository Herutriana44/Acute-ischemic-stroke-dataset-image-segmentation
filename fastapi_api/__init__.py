"""
FastAPI inference API with auto-GPU and queue system.

Provides async endpoints for:
- 2D image inference
- Single DICOM file inference
- DICOM series (ZIP) inference

Reuses existing inference logic from webapp/services.
"""