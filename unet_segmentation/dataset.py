"""Dataset slice dari clean_dataset (image/ + mask/ datar, nama sama)."""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from pathlib import Path
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

def patient_id_from_stem(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def list_patients_and_files(clean_root: Path) -> tuple[list[str], list[str]]:
    img_dir = clean_root / "image"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Tidak ada folder image: {img_dir}")
    stems: list[str] = []
    patients: set[str] = set()
    for p in sorted(img_dir.glob("*.png")):
        stem = p.stem
        m = clean_root / "mask" / f"{stem}.png"
        if not m.is_file():
            continue
        stems.append(stem)
        patients.add(patient_id_from_stem(stem))
    return sorted(patients), stems


def split_by_patient(
    stems: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    by_p: dict[str, list[str]] = {}
    for s in stems:
        pid = patient_id_from_stem(s)
        by_p.setdefault(pid, []).append(s)
    pids = list(by_p.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)
    n_val = max(1, int(round(len(pids) * val_ratio)))
    if n_val >= len(pids):
        n_val = max(1, len(pids) - 1)
    val_p = set(pids[:n_val])
    train_stems = [s for s in stems if patient_id_from_stem(s) not in val_p]
    val_stems = [s for s in stems if patient_id_from_stem(s) in val_p]
    return train_stems, val_stems


def build_transforms(train: bool, image_size: int | None = None) -> A.Compose:
    """Augmentasi spasial + kontras pada image; mask ikut flip/resize saja."""
    ops: list = []
    if train:
        ops.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.GaussNoise(var_limit=(5.0, 40.0), p=0.2),
            ]
        )
    if image_size is not None:
        ops.append(A.Resize(image_size, image_size))
    ops.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(ops)


class CleanDataset(Dataset):
    def __init__(
        self,
        clean_root: Path,
        stems: list[str],
        train: bool,
        image_size: int | None = None,
    ) -> None:
        self.root = Path(clean_root)
        self.stems = stems
        self.train = train
        self.tf = build_transforms(train=train, image_size=image_size)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.stems[idx]
        ip = self.root / "image" / f"{stem}.png"
        mp = self.root / "mask" / f"{stem}.png"
        img = np.array(Image.open(ip).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mp).convert("L"), dtype=np.float32)
        mask_u8 = (mask > 127.0).astype(np.uint8)
        aug = self.tf(image=img, mask=mask_u8)
        img_t = aug["image"]
        m = aug["mask"]
        if m.ndim == 2:
            m = m.unsqueeze(0)
        mask_bin = (m.float() > 0.5).float()
        return {"image": img_t, "mask": mask_bin, "stem": stem}

def build_transforms(train: bool, image_size: int | None = None) -> A.Compose:
    """Augmentasi spasial + kontras pada image; mask ikut flip/resize saja."""
    ops: list = []
    if train:
        ops.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.GaussNoise(var_limit=(5.0, 40.0), p=0.2),
            ]
        )
    if image_size is not None:
        ops.append(A.Resize(image_size, image_size))
    ops.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(ops)
    
class YOLODataset(Dataset):
    def __init__(
        self,
        yolo_root: Path,
        split: str,  # 'train', 'val', atau 'test'
        image_size: int | None = None,
    ) -> None:
        """
        Args:
            yolo_root: Path ke root folder dataset YOLO.
            split: String penanda sub-folder ('train', 'val', 'test').
            image_size: Target ukuran resize gambar.
        """
        self.yolo_root = Path(yolo_root)
        self.split = split
        self.image_size = image_size
        self.tf = build_transforms(train=(split == "train"), image_size=image_size)

        # Definisikan path untuk images dan labels sesuai split
        self.img_dir = self.yolo_root / "images" / split
        self.label_dir = self.yolo_root / "labels" / split

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Folder images tidak ditemukan: {self.img_dir}")

        # List semua gambar (mendukung berbagai macam ekstensi populer)
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        self.img_paths = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in valid_extensions]
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_yolo_mask(self, label_path: Path, img_width: int, img_height: int) -> np.ndarray:
        """Membaca file koordinat poligon YOLO .txt dan merendernya menjadi mask biner."""
        # Buat mask kosong sewarna hitam
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if not label_path.is_file():
            # Jika file label tidak ada, asumsikan gambar tersebut adalah background (mask kosong)
            return mask

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:  # YOLO segmentasi minimal butuh class_id + 2 koordinat (x,y) * 2
                    continue
                
                # parts[0] adalah class_id (diabaikan jika ini segmentasi biner/1 kelas)
                # Ambil koordinat poligon (x1, y1, x2, y2, ...)
                coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                
                # Kembalikan koordinat dari skala 0-1 ke ukuran piksel asli gambar
                coords[:, 0] *= img_width
                coords[:, 1] *= img_height
                coords = coords.astype(np.int32)
                
                # Gambar poligon penuh di atas mask kosong dengan warna putih (255)
                cv2.fillPoly(mask, [coords], color=255)
                
        return mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ip = self.img_paths[idx]
        stem = ip.stem
        
        # Cari file label .txt yang namanya sama dengan gambar
        lp = self.label_dir / f"{stem}.txt"

        # 1. Load Image
        img_pil = Image.open(ip).convert("RGB")
        img_w, img_h = img_pil.size
        img = np.array(img_pil, dtype=np.float32)

        # 2. Load & Konversi Label Poligon YOLO menjadi Mask Biner
        mask_u8 = self._load_yolo_mask(lp, img_w, img_h)

        # 3. Terapkan Augmentasi Albumentations
        aug = self.tf(image=img, mask=mask_u8)
        img_t = aug["image"]
        m = aug["mask"]

        # 4. Standarisasi Dimensi Mask ke Tensor (C, H, W)
        if m.ndim == 2:
            m = m.unsqueeze(0)
            
        mask_bin = (m.float() > 0.5).float()

        return {"image": img_t, "mask": mask_bin, "stem": stem}