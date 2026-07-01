"""Streamlit dashboard for Acute Ischemic Stroke Segmentation API."""

from __future__ import annotations

import io
import time

import requests
import streamlit as st
from PIL import Image

API_BASE_URL = "https://herutriana44-acute-ischemic-stroke-segmentation.hf.space"

st.set_page_config(
    page_title="Stroke Segmentation",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.main-header { font-size:2.2rem; font-weight:bold; color:#E84545; text-align:center; margin-bottom:0.5rem; }
.sub-header  { font-size:1.1rem; color:#555; text-align:center; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

JOB_TYPES = {
    "2D Image (PNG/JPG/BMP/TIFF)": "image",
    "Single DICOM (.dcm)": "dicom",
    "DICOM Series (ZIP/RAR/TAR)": "series",
}

ACCEPTED_EXT = {
    "image":  ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
    "dicom":  ["dcm"],
    "series": ["zip", "rar", "tar", "gz", "tgz", "bz2", "7z"],
}

POLL_INTERVAL = 2


# ── API helpers ────────────────────────────────────────────────────────────

def _get(path: str, timeout: int = 10) -> dict | None:
    try:
        r = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


def _post_file(path: str, file_bytes: bytes, filename: str, timeout: int = 60) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}{path}",
            files={"file": (filename, file_bytes)},
            timeout=timeout,
        )
        return r.json() if r.ok else {"error": r.status_code, "detail": r.text}
    except Exception as e:
        return {"error": str(e)}


def _download(job_id: str, filename: str) -> bytes | None:
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/runs/{job_id}/{filename}", timeout=15)
        return r.content if r.ok else None
    except Exception:
        return None


def submit_job(job_type: str, file_bytes: bytes, filename: str) -> dict | None:
    ep = {
        "image":  "/api/v1/jobs/submit-image",
        "dicom":  "/api/v1/jobs/submit-dicom",
        "series": "/api/v1/jobs/submit-series",
    }
    return _post_file(ep[job_type], file_bytes, filename)


def get_job(job_id: str) -> dict | None:
    return _get(f"/api/v1/jobs/{job_id}")


def get_jobs() -> list[dict]:
    data = _get("/api/v1/jobs")
    return data.get("jobs", []) if isinstance(data, dict) else []


def get_stats() -> dict | None:
    return _get("/api/v1/stats")


def get_health() -> dict | None:
    return _get("/health")


def status_icon(status: str) -> str:
    return {"pending": "🟡", "running": "🔵", "completed": "🟢", "failed": "🔴"}.get(status, "⚪")


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Config")

    if st.button("🔍 Check Health", use_container_width=True):
        with st.spinner("Checking..."):
            h = get_health()
        if h:
            st.session_state.health = h
        else:
            st.session_state.pop("health", None)
            st.error("Tidak bisa terhubung ke API")

    if "health" in st.session_state:
        h = st.session_state.health
        status = h.get("status", "unknown")
        if status == "ok":
            st.success(f"✅ API: {status.upper()}")
        else:
            st.warning(f"⚠️ API: {status.upper()}")
        gpu = h.get("gpu_available", h.get("device", "N/A"))
        st.caption(f"GPU/Device: {gpu}")

    if st.button("📊 Refresh Stats", use_container_width=True):
        s = get_stats()
        if s:
            st.session_state.stats = s

    if "stats" in st.session_state:
        s = st.session_state.stats
        st.divider()
        st.subheader("📊 Queue Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pending",   s.get("pending_count",   s.get("pending",   0)))
        c2.metric("Running",   s.get("running_count",   s.get("running",   0)))
        c3.metric("Done",      s.get("completed_count", s.get("completed", 0)))
        c4.metric("Failed",    s.get("failed_count",    s.get("failed",    0)))

    st.divider()
    st.caption(f"API: `{API_BASE_URL}`")
    st.caption("Stroke Segmentation Client v2.0")


# ── Header ─────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🧠 Acute Ischemic Stroke Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload DICOM atau 2D image untuk segmentasi lesi otomatis via UNet</div>', unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────

tab_submit, tab_history, tab_detail = st.tabs(["📤 Submit Job", "📋 Riwayat Job", "🔍 Detail Job"])


# ── Tab 1: Submit ──────────────────────────────────────────────────────────

with tab_submit:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Kirim Job Inferensi")

        job_type_label = st.selectbox("Tipe Job", list(JOB_TYPES.keys()))
        job_type = JOB_TYPES[job_type_label]

        uploaded = st.file_uploader(
            f"Pilih file ({', '.join(ACCEPTED_EXT[job_type])})",
            type=ACCEPTED_EXT[job_type],
            key="uploader",
        )

        if uploaded:
            if job_type == "image":
                try:
                    st.image(Image.open(uploaded), caption=f"Preview: {uploaded.name}", use_container_width=True)
                    uploaded.seek(0)
                except Exception:
                    st.warning("Tidak bisa preview gambar")

            if st.button("🚀 Submit Job", type="primary", use_container_width=True):
                file_bytes = uploaded.read()
                with st.spinner("Mengirim..."):
                    res = submit_job(job_type, file_bytes, uploaded.name)
                if not res:
                    st.error("Gagal submit. Periksa koneksi API.")
                elif "error" in res:
                    st.error(f"Gagal: {res}")
                else:
                    jid = res.get("job_id")
                    st.session_state.last_job_id = jid
                    st.success(f"Job terkirim! ID: `{jid}`")
                    st.info("Buka tab **Detail Job** untuk pantau progres.")

    with col_r:
        st.subheader("Job Terbaru")
        if st.button("🔄 List Semua Job", use_container_width=True):
            with st.spinner("Mengambil..."):
                st.session_state.job_list = get_jobs()

        if "job_list" in st.session_state:
            jobs = st.session_state.job_list
            if not jobs:
                st.info("Belum ada job.")
            else:
                for j in jobs[:20]:
                    st.write(
                        f"{status_icon(j['status'])} `{j['job_id']}` — "
                        f"{j.get('job_type','?')} — **{j['status']}** "
                        f"({str(j.get('created_at',''))[:19]})"
                    )
                    if j.get("error_message"):
                        st.caption(f"  Error: {j['error_message']}")


# ── Tab 2: History ─────────────────────────────────────────────────────────

with tab_history:
    st.subheader("Semua Job")

    if st.button("🔄 Refresh", key="history_refresh"):
        st.session_state.job_list = get_jobs()

    if "job_list" not in st.session_state:
        st.session_state.job_list = get_jobs()

    jobs = st.session_state.job_list
    if not jobs:
        st.info("Belum ada job.")
    else:
        for j in jobs:
            with st.expander(
                f"{status_icon(j['status'])}  {j['job_id']}  —  "
                f"{j.get('job_type','?')}  —  {j['status']}"
            ):
                st.json(j)
                if st.button("🔍 Lihat Detail", key=f"view_{j['job_id']}"):
                    st.session_state.view_job_id = j["job_id"]
                    st.rerun()


# ── Tab 3: Detail ──────────────────────────────────────────────────────────

with tab_detail:
    st.subheader("Detail & Hasil Job")

    default_id = st.session_state.get("view_job_id", st.session_state.get("last_job_id", ""))
    job_id_input = st.text_input("Job ID", value=default_id, key="detail_job_id_input")

    if job_id_input and st.button("🔍 Fetch Job", type="primary"):
        job = get_job(job_id_input)
        if job:
            st.session_state.current_job = job
            st.session_state.current_job_id = job_id_input
            st.session_state.pop("view_job_id", None)
        else:
            st.error(f"Job `{job_id_input}` tidak ditemukan.")

    if "current_job" in st.session_state and "current_job_id" in st.session_state:
        cur_id = st.session_state.current_job_id

        # Selalu refresh status terkini
        fresh = get_job(cur_id)
        if fresh:
            st.session_state.current_job = fresh
        job = st.session_state.current_job

        st.write(f"**Status:** {status_icon(job['status'])} {job['status']}")
        st.write(f"**Tipe:** {job.get('job_type', '?')}")
        st.write(f"**Dibuat:** {str(job.get('created_at',''))[:19]}")

        if job.get("error_message"):
            st.error(f"Error: {job['error_message']}")

        # Auto-poll jika masih pending/running
        if job["status"] in ("pending", "running"):
            ph = st.empty()
            ph.info(f"⏳ Menunggu... status: **{job['status']}**")
            while True:
                time.sleep(POLL_INTERVAL)
                updated = get_job(cur_id)
                if updated:
                    st.session_state.current_job = updated
                    ph.info(f"⏳ Status: **{updated['status']}**")
                    if updated["status"] in ("completed", "failed"):
                        break
                else:
                    break
            st.rerun()

        # Hasil jika selesai
        if job["status"] == "completed":
            result = job.get("result") or {}
            st.success("✅ Inferensi selesai!")
            st.divider()

            run_id = result.get("run_id", cur_id)

            # Tampilkan gambar hasil
            file_map = [
                ("Input Asli",  result.get("original_png",  result.get("input_name",  "input.png"))),
                ("Mask Prediksi", result.get("mask_png",    "mask_pred.png")),
                ("Overlay",     result.get("overlay_png",   "overlay.png")),
            ]

            img_cols = st.columns(3)
            for idx, (label, fname) in enumerate(file_map):
                with img_cols[idx]:
                    st.caption(label)
                    raw = _download(run_id, fname)
                    if raw:
                        try:
                            st.image(Image.open(io.BytesIO(raw)), use_container_width=True)
                        except Exception:
                            st.warning(f"Tidak bisa tampilkan {fname}")
                    else:
                        st.info("File tidak tersedia")

            # Metrics
            st.divider()
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Lesion Pixels", result.get("lesion_pixels", "N/A"))
            with m2:
                shape = result.get("shape_hw", [])
                st.metric("Shape (H×W)", f"{shape[0]}×{shape[1]}" if len(shape) == 2 else "N/A")
            with m3:
                st.metric("Job Type", job.get("job_type", "N/A"))

            # Tombol download
            st.divider()
            st.subheader("⬇️ Download Hasil")
            dl_cols = st.columns(3)
            for idx, (label, fname) in enumerate(file_map):
                with dl_cols[idx]:
                    raw = _download(run_id, fname)
                    if raw:
                        st.download_button(
                            label=f"⬇️ {label}",
                            data=raw,
                            file_name=fname,
                            mime="image/png",
                            use_container_width=True,
                        )

            # Raw JSON
            with st.expander("🔍 Raw Result JSON"):
                st.json(result)


# ── Footer ─────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    f"<div style='text-align:center;color:#888;padding:1rem'>"
    f"Stroke Segmentation Dashboard | API: <code>{API_BASE_URL}</code>"
    f"</div>",
    unsafe_allow_html=True,
)
