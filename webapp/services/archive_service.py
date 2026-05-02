from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from webapp.services.inference_service import InferenceError

REQUIRED_DICOM_MODALITY = "CT"

SUPPORTED_SUFFIXES = {
    ".zip",
    ".rar",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".tbz",
    ".xz",
    ".txz",
    ".7z",
}


def _safe_unarchive(archive_path: Path, out_dir: Path) -> None:
    suffixes = "".join(archive_path.suffixes).lower()
    try:
        if suffixes.endswith(".zip"):
            shutil.unpack_archive(str(archive_path), str(out_dir), format="zip")
            return
        if suffixes.endswith((".tar.gz", ".tgz")):
            shutil.unpack_archive(str(archive_path), str(out_dir), format="gztar")
            return
        if suffixes.endswith((".tar.bz2", ".tbz")):
            shutil.unpack_archive(str(archive_path), str(out_dir), format="bztar")
            return
        if suffixes.endswith((".tar.xz", ".txz")):
            shutil.unpack_archive(str(archive_path), str(out_dir), format="xztar")
            return
        if suffixes.endswith(".tar"):
            shutil.unpack_archive(str(archive_path), str(out_dir), format="tar")
            return

        import patoolib

        patoolib.extract_archive(str(archive_path), outdir=str(out_dir), verbosity=-1)
    except Exception as exc:
        raise InferenceError(
            "Gagal ekstrak archive. Pastikan format didukung dan file tidak korup."
        ) from exc


def extract_archive(file: FileStorage, upload_root: Path) -> tuple[Path, str]:
    original_name = secure_filename(file.filename or "")
    if not original_name:
        raise InferenceError("Nama file archive tidak valid.")

    suffixes = "".join(Path(original_name).suffixes).lower()
    if not any(suffixes.endswith(sfx) for sfx in SUPPORTED_SUFFIXES):
        raise InferenceError("Format archive belum didukung. Gunakan zip/rar/tar.gz/7z dan sejenisnya.")

    run_id = uuid.uuid4().hex[:12]
    run_dir = upload_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    archive_path = run_dir / original_name
    file.save(str(archive_path))

    extracted_dir = run_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    _safe_unarchive(archive_path, extracted_dir)
    return extracted_dir, run_id


def _peek_dicom_modality(series_dir: Path) -> str | None:
    """Baca tag Modality dari salah satu file .dcm (tanpa memuat pixel data)."""
    import pydicom

    files = sorted([p for p in series_dir.glob("*.dcm") if p.is_file()])
    for fp in files[:5]:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            m = getattr(ds, "Modality", None)
            if m:
                return str(m).strip().upper()
        except Exception:
            continue
    return None


def _path_has_folder_name(path: Path, name: str) -> bool:
    token = name.upper()
    return any(part.upper() == token for part in path.parts)


def find_dicom_series_dir(extracted_dir: Path) -> Path:
    """Pilih folder series DICOM CT.

    Dari semua folder yang berisi .dcm, hanya kandidat dengan Modality CT
    (atau path berisi subfolder `CT`) yang dipertimbangkan; lalu dipilih yang
    paling banyak slice-nya."""
    candidate_dirs: dict[Path, int] = {}
    for dicom_path in extracted_dir.rglob("*.dcm"):
        parent = dicom_path.parent
        candidate_dirs[parent] = candidate_dirs.get(parent, 0) + 1

    if not candidate_dirs:
        raise InferenceError("Tidak ditemukan file .dcm pada archive.")

    ct_from_tag: list[tuple[Path, int]] = []
    for d, n in candidate_dirs.items():
        if _peek_dicom_modality(d) == REQUIRED_DICOM_MODALITY:
            ct_from_tag.append((d, n))

    if ct_from_tag:
        return max(ct_from_tag, key=lambda item: item[1])[0]

    # Ekspor kadang tidak mengisi Modality; andai subfolder `CT` seperti contoh dataset.
    ct_from_path: list[tuple[Path, int]] = []
    for d, n in candidate_dirs.items():
        if _path_has_folder_name(d, REQUIRED_DICOM_MODALITY):
            ct_from_path.append((d, n))

    if ct_from_path:
        return max(ct_from_path, key=lambda item: item[1])[0]

    raise InferenceError(
        "Tidak ditemukan series DICOM CT. Pastikan arsip berisi pemindaian CT "
        "(tag Modality CT), atau struktur folder dengan subfolder 'CT' berisi file .dcm."
    )
