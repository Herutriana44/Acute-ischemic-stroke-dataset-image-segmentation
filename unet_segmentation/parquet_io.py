"""Utilitas I/O Parquet yang aman dari korupsi.

Inti masalah "file Parquet rusak":
    File Parquet baru dianggap sah jika *footer* (metadata) + magic bytes
    "PAR1" di akhir file berhasil ditulis. Kalau proses mati di tengah
    penulisan — atau kita menimpa (overwrite) file yang sama berulang di
    dalam loop dan ter-interupsi — footer tidak lengkap dan file menjadi
    tidak bisa dibaca (pd.read_parquet / pq.ParquetFile melempar error).

Solusi di modul ini:
    * atomic_write_table(): tulis ke file sementara di folder yang sama,
      fsync, lalu os.replace() (rename atomik di POSIX). Path tujuan SELALU
      berisi file lama yang utuh atau file baru yang utuh — tidak pernah
      versi setengah jadi. Inilah yang mencegah "rusak".
    * validate_parquet(): buka kembali hasil tulis untuk memastikan footer
      benar-benar terbaca (deteksi dini bila ada yang salah).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def atomic_write_table(
    table: pa.Table,
    output_path: str | os.PathLike,
    *,
    compression: str = "snappy",
) -> None:
    """Tulis ``table`` ke ``output_path`` secara atomik (anti-korupsi).

    Langkah:
        1. Tulis ke file sementara ``.<nama>.xxxx.parquet.tmp`` di folder
           yang sama dengan tujuan (wajib se-filesystem agar rename atomik).
        2. flush + os.fsync agar byte benar-benar sampai ke disk.
        3. os.replace(tmp -> tujuan): operasi rename atomik. Pembaca lain
           hanya akan melihat file lama (utuh) atau file baru (utuh).
        4. fsync direktori agar rename itu sendiri durabel.

    Jika terjadi error/interupsi, file sementara dibersihkan dan file tujuan
    yang lama (bila ada) tetap tidak tersentuh.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".parquet.tmp",
        dir=str(output_path.parent),
    )
    os.close(fd)  # pyarrow akan menulis lewat path ini sendiri
    try:
        pq.write_table(table, tmp_name, compression=compression)
        # Pastikan data fisik tertulis sebelum rename.
        with open(tmp_name, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_name, output_path)  # <-- atomik
        _fsync_dir(output_path.parent)
    except BaseException:
        # Jangan tinggalkan file .tmp menggantung bila gagal.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def validate_parquet(
    output_path: str | os.PathLike,
    expected_rows: int | None = None,
) -> int:
    """Buka ulang file untuk memastikan tidak korup. Mengembalikan jumlah baris.

    Membuka ``ParquetFile`` memaksa footer dibaca; bila footer tidak lengkap
    (file rusak) maka pyarrow langsung melempar error di sini.
    """
    output_path = Path(output_path)
    pf = pq.ParquetFile(output_path)
    num_rows = pf.metadata.num_rows
    if expected_rows is not None and num_rows != expected_rows:
        raise RuntimeError(
            f"Validasi Parquet gagal untuk {output_path}: file berisi "
            f"{num_rows} baris, seharusnya {expected_rows}."
        )
    return num_rows


def _fsync_dir(directory: Path) -> None:
    """fsync direktori agar rename durabel. Abaikan jika OS tak mendukung."""
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def relative_name(path: str | os.PathLike, base: str | os.PathLike) -> str:
    """Nama file relatif terhadap ``base`` yang aman (fallback ke basename).

    Menghindari ValueError dari Path.relative_to ketika path absolut/relatif
    tidak sejajar (mis. base relatif, path absolut).
    """
    p = Path(path)
    b = Path(base)
    try:
        return str(p.resolve().relative_to(b.resolve()))
    except Exception:
        return p.name
