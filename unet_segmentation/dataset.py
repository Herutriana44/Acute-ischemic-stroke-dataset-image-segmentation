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
