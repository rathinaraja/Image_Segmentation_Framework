"""
eval.py
-------
Standalone evaluation of a trained model on any image/mask dataset.
Loads a .pth checkpoint, runs inference, and records all metrics to CSV.

Usage:
    # Use same dataset paths as in config
    python test.py --config configs/unet.yaml \
                   --checkpoint logs/unet_dataset/20260402_143022/fold_1/checkpoints/best_model.pth

    # Override dataset paths at runtime
    python test.py --config configs/unet.yaml \
                   --checkpoint logs/.../best_model.pth \
                   --images_dir /data/test/images \
                   --masks_dir  /data/test/masks

    # Save predicted mask PNGs alongside metrics
    python test.py --config configs/unet.yaml \
                   --checkpoint logs/.../best_model.pth \
                   --save_preds
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader

from modules           import get_model
from utils.config   import load_config, print_config
from utils.dataset  import SegmentationDataset
from utils.logger   import get_logger
from utils.metrics  import MetricTracker, pixel_accuracy, mean_iou, dice_score
from utils.augmentations import get_val_augmentations


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Segmentation Model Evaluation")
    p.add_argument("--config",      type=str, required=True,
                   help="Path to model YAML config (e.g. configs/unet.yaml)")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--images_dir",  type=str, default=None,
                   help="Override images directory from config")
    p.add_argument("--masks_dir",   type=str, default=None,
                   help="Override masks directory from config")
    p.add_argument("--device",      type=str, default=None,
                   help="Device: cuda, cuda:0, cpu (default: auto-detect)")
    p.add_argument("--batch_size",  type=int, default=None,
                   help="Override batch size from config")
    p.add_argument("--save_preds",  action="store_true",
                   help="Save predicted mask PNGs to output_dir")
    p.add_argument("--output_dir",  type=str, default=None,
                   help="Where to save predictions and results (default: next to checkpoint)")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(cfg, checkpoint_path, device):
    model = get_model(cfg)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    trained_epoch = ckpt.get("epoch", "unknown")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Trained epoch    : {trained_epoch}")
    if "metrics" in ckpt:
        print(f"Saved metrics    : {ckpt['metrics']}")
    return model, trained_epoch


def build_eval_loader(images_dir, masks_dir, img_size, batch_size, num_workers):
    dataset = SegmentationDataset(
        images_dir = images_dir,
        masks_dir  = masks_dir,
        img_size   = tuple(img_size),
        transform  = get_val_augmentations(),   # no augmentation during eval
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print(f"Dataset          : {len(dataset)} images  ({len(loader)} batches)")
    return loader, dataset


def save_predictions(preds_batch, image_paths, output_dir):
    """Save predicted class-index masks as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    for pred, img_path in zip(preds_batch, image_paths):
        pred_np  = pred.numpy().astype(np.uint8) * 255   # binary: 0→0, 1→255
        out_path = os.path.join(output_dir, Path(img_path).stem + "_pred.png")
        Image.fromarray(pred_np).save(out_path)


# ── Core evaluation loop ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, dataset, cfg, device, save_preds, pred_dir):
    n_classes  = cfg["model"]["n_classes"]
    amp_enabled = cfg["training"].get("amp", True)
    criterion  = torch.nn.CrossEntropyLoss()

    tracker    = MetricTracker()
    batch_idx  = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with autocast('cuda', enabled=amp_enabled):
            logits = model(images)
            loss   = criterion(logits, masks)

        preds = logits.argmax(dim=1).cpu()
        m_cpu = masks.cpu()
        n     = images.size(0)

        tracker.update("loss",      loss.item(),                        n=n)
        tracker.update("pixel_acc", pixel_accuracy(preds, m_cpu),       n=n)
        tracker.update("mean_iou",  mean_iou(preds, m_cpu, n_classes),  n=n)
        tracker.update("dice",      dice_score(preds, m_cpu, n_classes),n=n)

        if save_preds:
            start = batch_idx * loader.batch_size
            end   = start + n
            batch_paths = [str(dataset.image_paths[i]) for i in range(start, end)]
            save_predictions(preds, batch_paths, pred_dir)

        batch_idx += 1

    return tracker.summary()


# ── Write results CSV ──────────────────────────────────────────────────────────

def write_results(metrics, checkpoint_path, images_dir, output_dir, trained_epoch):
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"test_results_{ts}.csv")

    row = {
        "timestamp":       ts,
        "checkpoint":      checkpoint_path,
        "trained_epoch":   trained_epoch,
        "images_dir":      images_dir,
        **{k: f"{v:.4f}" for k, v in metrics.items()},
    }

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"\nResults saved to : {csv_path}")
    return csv_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    print_config(cfg)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device           : {device}")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    images_dir = args.images_dir or cfg["dataset"]["images_dir"]
    masks_dir  = args.masks_dir  or cfg["dataset"]["masks_dir"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    img_size   = cfg["dataset"].get("img_size", [512, 512])
    num_workers= cfg["dataset"].get("num_workers", 4)

    # Output dir defaults to checkpoint's parent folder
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    pred_dir   = os.path.join(output_dir, "predictions") if args.save_preds else None

    # ── Load model ────────────────────────────────────────────────────────────
    model, trained_epoch = load_model(cfg, args.checkpoint, device)

    # ── Build loader ──────────────────────────────────────────────────────────
    loader, dataset = build_eval_loader(images_dir, masks_dir, img_size,
                                        batch_size, num_workers)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nRunning evaluation ...")
    metrics = evaluate(model, loader, dataset, cfg, device,
                       save_preds=args.save_preds, pred_dir=pred_dir)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*45)
    print(" Evaluation Results")
    print("="*45)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("="*45)

    # ── Save results CSV ──────────────────────────────────────────────────────
    write_results(metrics, args.checkpoint, images_dir, output_dir, trained_epoch)

    if args.save_preds:
        print(f"Predictions saved: {pred_dir}")


if __name__ == "__main__":
    main()
