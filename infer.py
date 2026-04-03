"""
infer.py
--------
Run inference on a folder of image patches with no masks required.
Saves predicted mask PNGs to output directory.

Usage:
    python infer.py --config configs/unet.yaml \
                    --checkpoint logs/.../best_model.pth \
                    --images_dir /path/to/patches

    # Custom output directory
    python infer.py --config configs/unet.yaml \
                    --checkpoint logs/.../best_model.pth \
                    --images_dir /path/to/patches \
                    --output_dir /path/to/save/predictions
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from modules       import get_model
from utils.config  import load_config, print_config


# ── Dataset (images only, no masks) ───────────────────────────────────────────

class PatchDataset(Dataset):
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, images_dir, img_size=(512, 512)):
        self.img_size    = img_size
        self.image_paths = sorted([
            p for p in Path(images_dir).iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])
        if not self.image_paths:
            raise RuntimeError(f"No images found in: {images_dir}")
        print(f"Found {len(self.image_paths)} patches in {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = TF.resize(image, self.img_size, interpolation=Image.BILINEAR)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        return image, str(self.image_paths[idx])   # return path for saving


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Segmentation Inference (no masks needed)")
    p.add_argument("--config",      type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--images_dir",  type=str, required=True,
                   help="Folder containing input patches")
    p.add_argument("--output_dir",  type=str, default=None,
                   help="Where to save predictions (default: <images_dir>/../predictions)")
    p.add_argument("--device",      type=str, default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    print_config(cfg)

    device     = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    img_size   = tuple(cfg["dataset"].get("img_size", [512, 512]))
    num_workers= cfg["dataset"].get("num_workers", 4)
    output_dir = args.output_dir or os.path.join(
                     os.path.dirname(args.images_dir.rstrip("/")), "predictions")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Device      : {device}")
    print(f"Output dir  : {output_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = get_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Checkpoint  : {args.checkpoint}  (epoch {ckpt.get('epoch','?')})")

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset = PatchDataset(args.images_dir, img_size=img_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    amp     = cfg["training"].get("amp", True)
    saved   = 0

    print("\nRunning inference ...")
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)
            with autocast('cuda', enabled=amp):
                logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()   # [B, H, W]

            for pred, src_path in zip(preds, paths):
                pred_img = (pred * 255).astype(np.uint8)  # 0→0, 1→255
                out_name = Path(src_path).stem + "_pred.png"
                Image.fromarray(pred_img).save(os.path.join(output_dir, out_name))
                saved += 1

            print(f"  Processed {saved}/{len(dataset)}", end="\r")

    print(f"\nDone. {saved} predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()