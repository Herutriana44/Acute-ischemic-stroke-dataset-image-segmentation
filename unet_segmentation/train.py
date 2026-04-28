"""
Training U-Net (PyTorch) dengan encoder ImageNet-pretrained (segmentation_models_pytorch).
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
from unet_segmentation.metrics import (
    dice_per_slice,
    iou_per_slice,
    precision_recall_per_slice,
)


SLICE_DICE_TARGET = 0.76


def build_model(encoder: str, pretrained: bool) -> nn.Module:
    weights = "imagenet" if pretrained else None
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation=None,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_slice_dice: list[torch.Tensor] = []
    all_slice_iou: list[torch.Tensor] = []
    all_prec: list[torch.Tensor] = []
    all_rec: list[torch.Tensor] = []
    inter = torch.zeros(1, device=device)
    pred_sum = torch.zeros(1, device=device)
    gt_sum = torch.zeros(1, device=device)
    tn_acc = torch.zeros(1, device=device)
    fp_acc = torch.zeros(1, device=device)
    pix_correct = 0
    pix_total = 0
    eps = 1e-6

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(imgs)
        prob = torch.sigmoid(logits)

        d = dice_per_slice(prob, masks)
        j = iou_per_slice(prob, masks)
        pr, rc = precision_recall_per_slice(prob, masks)
        all_slice_dice.append(d.cpu())
        all_slice_iou.append(j.cpu())
        all_prec.append(pr.cpu())
        all_rec.append(rc.cpu())

        pred_bin = (prob > 0.5).float()
        gt = (masks > 0.5).float()
        inter += (pred_bin * gt).sum()
        pred_sum += pred_bin.sum()
        gt_sum += gt.sum()
        tn_acc += ((1.0 - pred_bin) * (1.0 - gt)).sum()
        fp_acc += (pred_bin * (1.0 - gt)).sum()
        pix_correct += int((pred_bin == gt).sum().item())
        pix_total += int(gt.numel())

    slice_dice = torch.cat(all_slice_dice).mean().item()
    slice_iou = torch.cat(all_slice_iou).mean().item()
    slice_precision = torch.cat(all_prec).mean().item()
    slice_recall = torch.cat(all_rec).mean().item()
    f1 = (
        2 * slice_precision * slice_recall / (slice_precision + slice_recall + 1e-8)
        if (slice_precision + slice_recall) > 0
        else 0.0
    )

    global_dice = float((2.0 * inter + eps) / (pred_sum + gt_sum + eps))
    global_spec = float((tn_acc + eps) / (tn_acc + fp_acc + eps))
    mean_pix = float((pix_correct + eps) / (pix_total + eps)) if pix_total else 0.0

    return {
        "slice_level_dice": slice_dice,
        "slice_level_iou": slice_iou,
        "slice_level_precision": slice_precision,
        "slice_level_recall": slice_recall,
        "slice_level_f1": float(f1),
        "mean_pixel_accuracy": mean_pix,
        "global_dice_micro": global_dice,
        "global_specificity": global_spec,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    dice_loss: nn.Module,
    bce_loss: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    running = 0.0
    n = 0
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for batch in tqdm(loader, desc="train", leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(imgs)
                loss = dice_loss(logits, masks) + bce_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = dice_loss(logits, masks) + bce_loss(logits, masks)
            loss.backward()
            optimizer.step()
        running += float(loss.detach())
        n += 1
    return running / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="U-Net segmentasi (clean_dataset)")
    root = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--clean-root",
        type=Path,
        default=root / "clean_dataset",
        help="Root clean_dataset berisi image/ dan mask/",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--encoder", type=str, default="resnet34", help="Encoder smp, mis. resnet34, efficientnet-b0")
    ap.add_argument("--no-pretrained", action="store_true", help="Tanpa bobot ImageNet pada encoder")
    ap.add_argument("--image-size", type=int, default=None, help="Resize sisi (default: ukuran asli)")
    ap.add_argument("--out-dir", type=Path, default=root / "checkpoints_unet")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    clean_root = args.clean_root.resolve()
    if not clean_root.is_dir():
        print(f"clean_dataset tidak ditemukan: {clean_root}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler()

    _, stems = list_patients_and_files(clean_root)
    if len(stems) < 10:
        print(f"Terlalu sedikit sampel berpasangan: {len(stems)}", file=sys.stderr)
        return 1

    train_stems, val_stems = split_by_patient(stems, args.val_ratio, args.seed)
    print(f"Sampel train: {len(train_stems)}, val: {len(val_stems)} (split per pasien)")

    train_ds = CleanDataset(clean_root, train_stems, train=True, image_size=args.image_size)
    val_ds = CleanDataset(clean_root, val_stems, train=False, image_size=args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.encoder, pretrained=not args.no_pretrained)
    model.to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    best_path = args.out_dir / "best_unet.pt"

    for epoch in range(1, args.epochs + 1):
        loss_tr = train_one_epoch(
            model, train_loader, optimizer, scaler, dice_loss, bce_loss, device, use_amp
        )
        metrics = evaluate(model, val_loader, device)
        sched.step()

        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={loss_tr:.4f}  "
            f"slice_level_dice={metrics['slice_level_dice']:.4f}  "
            f"slice_level_iou={metrics['slice_level_iou']:.4f}  "
            f"slice_prec={metrics['slice_level_precision']:.4f}  "
            f"slice_rec={metrics['slice_level_recall']:.4f}  "
            f"slice_f1={metrics['slice_level_f1']:.4f}  "
            f"pix_acc={metrics['mean_pixel_accuracy']:.4f}  "
            f"global_dice={metrics['global_dice_micro']:.4f}  "
            f"spec={metrics['global_specificity']:.4f}"
        )

        sd = metrics["slice_level_dice"]
        if sd > SLICE_DICE_TARGET:
            print(
                f">>> slice_level_dice ({sd:.4f}) > target ({SLICE_DICE_TARGET}): "
                "Sudah memenuhi target evaluasi."
            )

        if sd > best_dice:
            best_dice = sd
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "encoder": args.encoder,
                    "metrics": metrics,
                },
                best_path,
            )
            print(f"  (simpan checkpoint terbaik -> {best_path})")

    print(f"Selesai. slice_level_dice terbaik validasi: {best_dice:.4f}")
    if best_dice > SLICE_DICE_TARGET:
        print(
            f"Model terbaik (slice_level_dice={best_dice:.4f}) di atas target {SLICE_DICE_TARGET}."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
