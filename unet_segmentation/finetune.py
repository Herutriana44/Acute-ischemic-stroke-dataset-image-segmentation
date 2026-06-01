"""
Fine-tuning U-Net (PyTorch) dengan checkpoint yang sudah ada.
Dataset: clean_dataset/image + clean_dataset/mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet_segmentation.dataset import CleanDataset, list_patients_and_files, split_by_patient
from unet_segmentation.train import (
    build_model,
    evaluate,
    train_one_epoch,
    SLICE_DICE_TARGET,
)

def is_mask_present(clean_root: Path, stem: str) -> bool:
    from PIL import Image
    import numpy as np
    mask_path = clean_root / "mask" / f"{stem}.png"
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask.max() > 0

def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tuning U-Net")
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path ke file checkpoint (.pt)")
    ap.add_argument(
        "--clean-root",
        type=Path,
        default=root / "clean_dataset",
        help="Root clean_dataset",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5) # LR lebih rendah untuk fine-tuning
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=root / "checkpoints_finetune")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    
    # 1. Load Checkpoint
    if not args.checkpoint.exists():
        print(f"Checkpoint tidak ditemukan: {args.checkpoint}")
        return 1
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder = ckpt.get("encoder", "resnet34")
    
    # 2. Build Model
    model = build_model(encoder, pretrained=True) # Gunakan pretrained=True untuk inisialisasi yang lebih baik
    model.load_state_dict(ckpt["model_state"])
    
    # UNFREEZING: Izinkan encoder dilatih kembali dengan LR lebih kecil
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    model.to(device)
    
    # 3. Setup Dataset/Loader
    clean_root = args.clean_root.resolve()
    _, stems = list_patients_and_files(clean_root)
    train_stems_raw, val_stems = split_by_patient(stems, args.val_ratio, args.seed)
    
    # PENYEIMBANGAN DATASET
    import random
    random.seed(args.seed)
    all_mask_stems = [s for s in train_stems_raw if is_mask_present(clean_root, s)]
    all_empty_stems = [s for s in train_stems_raw if s not in all_mask_stems]
    
    # Ambil 50% dari data kosong agar lebih seimbang
    balanced_empty = random.sample(all_empty_stems, int(len(all_mask_stems) * 0.5))
    train_stems = all_mask_stems + balanced_empty
    random.shuffle(train_stems)
    
    print(f"Balanced train size: {len(train_stems)}")

    train_ds = CleanDataset(clean_root, train_stems, train=True)
    val_ds = CleanDataset(clean_root, val_stems, train=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type=="cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type=="cuda")
    
    # 4. Training loop setup
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
        
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    # Optimizer lebih agresif untuk fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) 
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG: Directory checkpoint: {args.out_dir.resolve()}")
    best_path = args.out_dir / "best_finetuned_unet.pt"
    best_dice = ckpt.get("metrics", {}).get("slice_level_dice", 0.0)
    print(f"Starting fine-tuning from dice: {best_dice:.4f}")
    
    # EARLY STOPPING SETUP: Pantau 3 epoch terakhir
    patience = 3
    counter = 0
    dice_history = []

    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        loss_tr = train_one_epoch(model, train_loader, optimizer, scaler, dice_loss, bce_loss, device, use_amp)
        metrics = evaluate(model, val_loader, device)
        current_dice = metrics['slice_level_dice']
        sched.step(current_dice) 

        print(f"Epoch {epoch}/{args.epochs}  loss={loss_tr:.4f}  dice={current_dice:.4f}")
        
        # Simpan checkpoint terakhir
        last_path = args.out_dir / "last_finetuned_unet.pt"
        torch.save({"model_state": model.state_dict(), "encoder": encoder, "metrics": metrics}, last_path)
        
        # Logika Early Stopping berdasarkan riwayat 3 epoch (DITUTUP SEMENTARA)
        # if len(dice_history) >= patience:
        #     if current_dice <= max(dice_history[-patience:]):
        #         counter += 1
        #         print(f"  Dice tidak meningkat dari riwayat 3 epoch (best_recent: {max(dice_history[-patience:]):.4f}). Counter: {counter}/{patience}")
        #     else:
        #         counter = 0
        
        dice_history.append(current_dice)
        
        # Simpan best secara global
        if current_dice > best_dice:
            print(f"  Dice mencapai rekor baru ({best_dice:.4f} -> {current_dice:.4f}). Menyimpan checkpoint terbaik.")
            best_dice = current_dice
            torch.save({"model_state": model.state_dict(), "encoder": encoder, "metrics": metrics}, best_path)

        # if counter >= patience:
        #     print("Early stopping triggered: tidak ada kenaikan dalam 3 epoch terakhir.")
        #     break

    return 0

if __name__ == "__main__":
    sys.exit(main())
