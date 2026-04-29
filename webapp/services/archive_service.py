from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from webapp.services.inference_service import InferenceError

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


def find_dicom_series_dir(extracted_dir: Path) -> Path:
    candidate_dirs: dict[Path, int] = {}
    for dicom_path in extracted_dir.rglob("*.dcm"):
        parent = dicom_path.parent
        candidate_dirs[parent] = candidate_dirs.get(parent, 0) + 1

    if not candidate_dirs:
        raise InferenceError("Tidak ditemukan file .dcm pada archive.")

    best_dir = max(candidate_dirs.items(), key=lambda item: item[1])[0]
    return best_dir
