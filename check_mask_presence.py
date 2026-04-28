#!/usr/bin/env python3
"""
Cek keberadaan masking pada satu direktori berisi file PNG mask.

Dianggap **ada masking** jika ada piksel dengan intensitas di atas ambang
(default: > 0). Mask sepenuhnya hitam (semua 0) dihitung sebagai tanpa masking.

Contoh:
  python3 check_mask_presence.py --mask ./clean_dataset/mask
  python3 check_mask_presence.py ./dataset/mask/0019983
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXT = {".png", ".PNG"}


def max_intensity(path: Path) -> float:
    """Nilai intensitas maksimum (semua kanal). Grayscale/RGB/RGBA didukung."""
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 2:
        return float(arr.max())
    return float(arr.max())


def iter_png_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in IMAGE_EXT:
            files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Deteksi mask PNG yang berisi piksel non-hitam vs mask kosong (full black)."
    )
    ap.add_argument(
        "mask_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Direktori berisi file mask .png (opsional jika memakai --mask)",
    )
    ap.add_argument(
        "--mask",
        type=Path,
        default=None,
        dest="mask_flag",
        help="Alias direktori mask (sama seperti argumen posisi)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Piksel di atas nilai ini dianggap bagian mask (default: 0)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=8,
        help="Jumlah contoh path yang ditampilkan per kategori (default: 8)",
    )
    args = ap.parse_args()

    mask_root = args.mask_flag or args.mask_dir
    if mask_root is None:
        print("Wajib menyertakan direktori mask, contoh:", file=sys.stderr)
        print("  python3 check_mask_presence.py --mask ./clean_dataset/mask", file=sys.stderr)
        return 2

    mask_root = mask_root.resolve()
    if not mask_root.is_dir():
        print(f"Bukan direktori: {mask_root}", file=sys.stderr)
        return 1

    paths = iter_png_files(mask_root)
    total = len(paths)
    if total == 0:
        print(f"Tidak ada file .png di: {mask_root}")
        return 0

    with_mask: list[Path] = []
    empty_black: list[Path] = []
    errors: list[tuple[Path, str]] = []

    thr = args.threshold
    sample_n = max(0, args.sample)

    for p in paths:
        try:
            mx = max_intensity(p)
        except Exception as e:  # noqa: BLE001 — laporkan file rusak/format aneh
            errors.append((p, str(e)))
            continue
        if mx > thr:
            with_mask.append(p)
        else:
            empty_black.append(p)

    n_detected = len(with_mask)
    n_empty = len(empty_black)
    n_err = len(errors)

    print(f"Direktori: {mask_root}")
    print(f"Total file PNG: {total}")
    print(f"Terdeteksi ada masking (max intensitas > {thr}): {n_detected}")
    print(f"Tanpa masking / hitam penuh: {n_empty}")
    if n_err:
        print(f"Gagal dibaca: {n_err}")

    def print_samples(label: str, items: list[Path]) -> None:
        if not items:
            print(f"\n{label}: (tidak ada)")
            return
        show = items[:sample_n]
        print(f"\n{label} (menampilkan {len(show)} dari {len(items)}):")
        for q in show:
            print(f"  {q}")

    print_samples("Sample — ada masking", with_mask)
    print_samples("Sample — hitam penuh / kosong", empty_black)
    if errors:
        print("\nSample — error:")
        for q, msg in errors[:sample_n]:
            print(f"  {q}: {msg}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
