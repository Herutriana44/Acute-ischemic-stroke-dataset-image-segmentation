"""Streamlit webapp for Stroke Segmentation FastAPI. Configurable API URL."""

from __future__ import annotations

import io
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Segmentation",
    page_icon="🧠",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────────────
JOB_TYPES = {
    "2D Image (PNG/JPG)": "image",
    "Single DICOM (.dcm)": "dicom",
    "DICOM Series (ZIP)": "series",
}

POLL_INTERVAL = 2  # seconds


# ── Helpers ──────────────────────────────────────────────────────────────
def api_health(api_url: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def submit_job(api_url: str, job_type: str, file_bytes: bytes, filename: str) -> dict | None:
    match job_type:
        case "image":
            endpoint = f"{api_url}/api/v1/jobs/submit-image"
        case "dicom":
            endpoint = f"{api_url}/api/v1/jobs/submit-dicom"
        case "series":
            endpoint = f"{api_url}/api/v1/jobs/submit-series"
        case _:
            return None

    try:
        r = requests.post(
            endpoint,
            files={"file": (filename, file_bytes)},
            timeout=30,
        )
        return r.json() if r.ok else {"error": r.status_code, "detail": r.text}
    except Exception as e:
        return {"error": str(e)}


def get_job(api_url: str, job_id: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}/api/v1/jobs/{job_id}", timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


def get_jobs(api_url: str) -> list[dict]:
    try:
        r = requests.get(f"{api_url}/api/v1/jobs", timeout=10)
        return r.json().get("jobs", []) if r.ok else []
    except Exception:
        return []


def get_stats(api_url: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}/api/v1/stats", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def download_result(api_url: str, job_id: str, filename: str) -> bytes | None:
    try:
        r = requests.get(
            f"{api_url}/api/v1/runs/{job_id}/{filename}",
            timeout=15,
        )
        return r.content if r.ok else None
    except Exception:
        return None


def status_badge(status: str) -> str:
    colors = {
        "pending": "🟡",
        "running": "🔵",
        "completed": "🟢",
        "failed": "🔴",
    }
    return colors.get(status, "⚪")


# ── Sidebar: Config ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Config")

    if "api_url" not in st.session_state:
        st.session_state.api_url = "http://localhost:65234"

    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_url,
        placeholder="http://localhost:65234",
        key="api_url_input",
    ).rstrip("/")
    st.session_state.api_url = api_url

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Check Health", use_container_width=True):
            with st.spinner("Checking..."):
                health = api_health(api_url)
                if health:
                    st.session_state.health = health
                else:
                    st.session_state.health = None
                    st.error("Cannot reach API")

    with col2:
        if st.button("📊 Refresh Stats", use_container_width=True):
            stats = get_stats(api_url)
            if stats:
                st.session_state.stats = stats

    if "health" in st.session_state and st.session_state.health:
        h = st.session_state.health
        st.success(f"✅ Connected — GPU: {h.get('gpu_available', 'N/A')} | Device: {h.get('device', 'N/A')}")

    if "stats" in st.session_state and st.session_state.stats:
        s = st.session_state.stats
        st.divider()
        st.subheader("📊 Queue Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pending", s.get("pending_count", 0))
        c2.metric("Running", s.get("running_count", 0))
        c3.metric("Done", s.get("completed_count", 0))
        c4.metric("Failed", s.get("failed_count", 0))

    st.divider()
    st.caption("Stroke Segmentation API Client v1.0")

# ── Main ─────────────────────────────────────────────────────────────────
st.title("🧠 Acute Ischemic Stroke Segmentation")
st.caption("Upload DICOM or 2D images for automatic lesion segmentation via UNet.")

tabs = st.tabs(["📤 Submit Job", "📋 Job History", "🔍 Job Detail"])

# ── Tab 1: Submit ────────────────────────────────────────────────────────
with tabs[0]:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("New Inference Job")
        job_type_label = st.selectbox(
            "Job Type",
            list(JOB_TYPES.keys()),
            help="Image: PNG/JPG/BMP/TIFF  |  DICOM: single .dcm  |  Series: ZIP of DICOM slices",
        )
        job_type = JOB_TYPES[job_type_label]

        accepted = {
            "image": ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            "dicom": ["dcm"],
            "series": ["zip", "rar", "tar", "gz", "tgz", "bz2", "tbz", "xz", "txz", "7z"],
        }
        uploaded = st.file_uploader(
            f"Choose file ({', '.join(accepted[job_type])})",
            type=accepted[job_type],
            key="file_uploader",
        )

        if uploaded:
            if job_type == "image":
                try:
                    img = Image.open(uploaded)
                    st.image(img, caption=f"Preview: {uploaded.name}", use_container_width=True)
                except Exception:
                    st.warning("Cannot preview this image")

            if st.button("🚀 Submit Job", type="primary", use_container_width=True):
                file_bytes = uploaded.read()
                with st.spinner("Submitting..."):
                    result = submit_job(api_url, job_type, file_bytes, uploaded.name)
                if result and "error" in result:
                    st.error(f"Failed: {result}")
                elif result:
                    st.session_state.last_job_id = result["job_id"]
                    st.success(f"Job submitted! ID: `{result['job_id']}`")
                    st.info("Go to **Job Detail** tab to track progress.")
                else:
                    st.error("Submission failed. Check API connection.")

    with col_right:
        st.subheader("Quick Poll")
        if st.button("🔄 List All Jobs", use_container_width=True):
            with st.spinner("Fetching..."):
                jobs = get_jobs(api_url)
                st.session_state.job_list = jobs

        if "job_list" in st.session_state:
            jobs = st.session_state.job_list
            if not jobs:
                st.info("No jobs found.")
            else:
                for j in jobs[:20]:
                    st.write(
                        f"{status_badge(j['status'])} `{j['job_id']}` — "
                        f"{j.get('job_type', '?')} — {j['status']} "
                        f"({j.get('created_at', '')[:19]})"
                    )
                    if j.get("error_message"):
                        st.caption(f"  Error: {j['error_message']}")

# ── Tab 2: Job History ───────────────────────────────────────────────────
with tabs[1]:
    st.subheader("All Jobs")

    if st.button("🔄 Refresh", key="history_refresh"):
        st.session_state.job_list = get_jobs(api_url)

    if "job_list" not in st.session_state:
        st.session_state.job_list = get_jobs(api_url)

    jobs = st.session_state.job_list
    if not jobs:
        st.info("No jobs submitted yet.")
    else:
        for j in jobs:
            with st.expander(
                f"{status_badge(j['status'])} {j['job_id']} — {j.get('job_type', '?')} — {j['status']}"
            ):
                st.json(j)
                if st.button(f"🔍 View Details", key=f"view_{j['job_id']}"):
                    st.session_state.view_job_id = j["job_id"]
                    st.rerun()

# ── Tab 3: Job Detail ────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Job Detail & Results")

    default_id = st.session_state.get("last_job_id", "")
    job_id = st.text_input("Job ID", value=default_id, key="detail_job_id")

    if job_id and st.button("🔍 Fetch Job", type="primary", key="fetch_job"):
        job = get_job(api_url, job_id)
        if job:
            st.session_state.current_job = job
            st.session_state.current_job_id = job_id
        else:
            st.error(f"Job `{job_id}` not found.")

    if "current_job" in st.session_state and "current_job_id" in st.session_state:
        with st.spinner("Refreshing job status..."):
            job = get_job(api_url, st.session_state.get("current_job_id", job_id))
            if job:
                st.session_state.current_job = job
            else:
                job = st.session_state.current_job

        st.write(f"**Status:** {status_badge(job['status'])} {job['status']}")
        st.write(f"**Type:** {job.get('job_type', '?')}")
        st.write(f"**Created:** {job.get('created_at', '')}")

        if job.get("error_message"):
            st.error(f"Error: {job['error_message']}")

        # Auto-poll if pending/running
        if job["status"] in ("pending", "running"):
            placeholder = st.empty()
            while True:
                time.sleep(POLL_INTERVAL)
                updated = get_job(api_url, st.session_state.current_job_id)
                if updated:
                    st.session_state.current_job = updated
                    placeholder.info(
                        f"Status: {status_badge(updated['status'])} {updated['status']}"
                    )
                    if updated["status"] in ("completed", "failed"):
                        break
                else:
                    break
            st.rerun()

        # Show results if completed
        if job["status"] == "completed" and job.get("result"):
            st.success("✅ Inference completed!")
            result = job["result"]
            st.divider()

            cols = st.columns(3)
            run_id = result.get("run_id", result.get("job_id", st.session_state.current_job_id))

            files_to_show = [
                ("Input", result.get("input_name", result.get("original_png", "input.png"))),
                ("Mask", result.get("mask_png", "mask_pred.png")),
                ("Overlay", result.get("overlay_png", "overlay.png")),
            ]

            for idx, (label, fname) in enumerate(files_to_show):
                with cols[idx]:
                    st.caption(label)
                    img_bytes = download_result(api_url, run_id, fname)
                    if img_bytes:
                        try:
                            st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
                        except Exception:
                            st.warning(f"Cannot display {fname}")

            # Metrics
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                lesion = result.get("lesion_pixels", "N/A")
                st.metric("Lesion Pixels", lesion)
            with c2:
                shape = result.get("shape_hw", [])
                st.metric("Shape (H×W)", f"{shape[0]}×{shape[1]}" if shape else "N/A")

            # Raw result JSON
            with st.expander("Raw Result JSON"):
                st.json(result)