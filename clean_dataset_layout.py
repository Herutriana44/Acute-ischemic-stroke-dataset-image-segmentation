#!/usr/bin/env python3
"""
Merapikan dataset: dari dataset/image/<id>/000.png
menjadi clean_dataset/image/<id>_000.png (idem untuk mask).

Secara default membaca ./dataset dan menulis ke ./clean_dataset (salinan file).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def collect_pairs(root: Path, sub: str) -> list[tuple[Path, str, str]]:
    """
    root/sub/<patient_id>/<file> -> daftar (src_path, patient_id, filename).
    """
    base = root / sub
    if not base.is_dir():
        return []
    out: list[tuple[Path, str, str]] = []
    for patient_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        pid = patient_dir.name
        for f in sorted(patient_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                out.append((f, pid, f.name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Flatten dataset/image|mask/<id>/slice.ext -> clean_dataset/.../<id>_slice.ext"
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Folder berisi image/ dan mask/ (default: ./dataset)",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent / "clean_dataset",
        help="Output root (default: ./clean_dataset)",
    )
    ap.add_argument(
        "--symlink",
        action="store_true",
        help="Buat symlink ke file asli (hemat ruang); default: salin file",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya cetak rencana, tidak menulis",
    )
    args = ap.parse_args()
    src_root: Path = args.source.resolve()
    dst_root: Path = args.dest.resolve()

    if not src_root.is_dir():
        print(f"Sumber tidak ada atau bukan folder: {src_root}", file=sys.stderr)
        return 1

    for kind in ("image", "mask"):
        pairs = collect_pairs(src_root, kind)
        if not pairs:
            print(f"[skip] Tidak ada file di {src_root / kind}")
            continue
        out_dir = dst_root / kind
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        for src, pid, fname in pairs:
            new_name = f"{pid}_{fname}"
            dst = out_dir / new_name
            if args.dry_run:
                print(f"{src} -> {dst}")
                continue
            if dst.exists() or dst.is_symlink():
                # Hindari timpa tanpa sadar; bisa diubah ke overwrite jika perlu
                print(f"[skip exists] {dst}", file=sys.stderr)
                continue
            if args.symlink:
                dst.symlink_to(src.resolve(), target_is_directory=False)
            else:
                shutil.copy2(src, dst)

        print(f"[{kind}] {len(pairs)} file -> {out_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # stdout ditutup lebih awal oleh pembaca (mis. `| head`); bukan error skrip
        raise SystemExit(0)
