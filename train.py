"""
train.py
--------
Usage:
    python train.py --config configs/unet.yaml
    python train.py --config configs/unet.yaml --device cuda:1
    python train.py --config configs/unet.yaml --resume logs/.../fold_1/checkpoints/best_model.pth

    # Override any yaml parameter at runtime
    python train.py --config configs/unet.yaml \
                    --set dataset.images_dir=/new/path \
                          training.epochs=50 \
                          training.batch_size=4 \
                          model.n_classes=3
"""

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
import yaml

from utils.config      import load_config, print_config
from utils.dataset     import get_splits
from utils.train_utils import ModelProcess
from modules           import get_model


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Semantic Segmentation Training")
    p.add_argument("--config",  type=str, required=True,
                   help="Path to YAML config (e.g. configs/unet.yaml)")
    p.add_argument("--device",  type=str, default=None,
                   help="Device: cuda, cuda:0, cpu (default: auto-detect)")
    p.add_argument("--resume",  type=str, default=None,
                   help="Checkpoint path to resume from (single-fold only)")
    p.add_argument("--set",     nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="Override config params e.g. --set dataset.images_dir=/path training.epochs=50")
    return p.parse_args()


# ── Config override ────────────────────────────────────────────────────────────

def _apply_overrides(cfg: dict, overrides: list) -> dict:
    """
    Apply CLI overrides to a config dict.
    Supports nested keys with dot notation: dataset.images_dir=/new/path
    Auto-casts to int, float, or bool where possible.

    Examples:
        dataset.images_dir=/data/patches
        training.epochs=100
        training.batch_size=4
        training.amp=false
        model.n_classes=3
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set override must be KEY=VALUE, got: '{item}'")
        key_path, _, value = item.partition("=")
        keys = key_path.strip().split(".")

        # Navigate to parent node
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                raise KeyError(f"Config key '{k}' not found in path '{key_path}'")
            node = node[k]

        # Auto-cast value
        final_key = keys[-1]
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:    value = int(value)
            except ValueError:
                try: value = float(value)
                except ValueError:
                    pass   # keep as string

        node[final_key] = value
        print(f"  Config override: {key_path} = {value}")

    return cfg


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config then apply CLI overrides
    cfg = load_config(args.config)
    if args.set:
        print("\nApplying overrides:")
        cfg = _apply_overrides(cfg, args.set)

    print_config(cfg)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")

    # ── Datetime-stamped run directory ────────────────────────────────────────
    run_dir = os.path.join(cfg["logging"]["log_dir"],
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ── Save config snapshot to run directory ─────────────────────────────────
    # Dumps the LIVE cfg dict (after overrides) — not the original file.
    # This guarantees the saved yaml always reflects exactly what was used.
    saved_cfg_path = os.path.join(run_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        if args.set:
            f.write("# ── CLI overrides were applied to this config ──\n")
            for item in args.set:
                f.write(f"# --set {item}\n")
            f.write("\n")
        yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to: {saved_cfg_path}")

    # ── Dataset splits ────────────────────────────────────────────────────────
    splits    = get_splits(cfg)
    fold_mode = cfg["training"].get("fold_mode", "single")
    print(f"Mode   : {fold_mode} | Folds: {len(splits)} | "
          f"eval_mode: {cfg['training'].get('eval_mode', 'train_val_test')}")

    fold_results = []

    for split in splits:
        fold_num     = split["fold"]
        fold_dir     = os.path.join(run_dir, f"fold_{fold_num}")
        ckpt_dir     = os.path.join(fold_dir, "checkpoints")

        train_loader = split["train_loader"]
        val_loader   = split.get("val_loader")
        test_loader  = split.get("test_loader")

        n_train = len(train_loader.dataset) if train_loader else 0
        n_val   = len(val_loader.dataset)   if val_loader   else 0
        n_test  = len(test_loader.dataset)  if test_loader  else 0
        print(f"\n{'='*55}")
        print(f"Fold {fold_num}/{len(splits)} | train={n_train} val={n_val} test={n_test}")
        print(f"{'='*55}")

        # Fresh model + process per fold
        model   = get_model(cfg)
        process = ModelProcess(model, cfg, device,
                               checkpoint_dir=ckpt_dir,
                               log_dir=fold_dir)

        if args.resume and fold_num == 1:
            process.load_checkpoint(args.resume)

        process.train(train_loader, val_loader=val_loader, test_loader=test_loader)

        # Collect best metrics for summary
        best_ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"),
                               map_location="cpu", weights_only=False)
        fold_results.append({"fold": fold_num, **best_ckpt.get("metrics", {})})

    # ── Cross-fold summary ────────────────────────────────────────────────────
    if fold_mode == "kfold" and len(fold_results) > 1:
        _write_summary(fold_results, os.path.join(run_dir, "summary.csv"))
        print(f"\nSummary written to {run_dir}/summary.csv")

    print("\nAll folds complete.")


def _write_summary(fold_results: list, path: str):
    """Write per-fold metrics + mean/std rows to summary CSV."""
    keys = [k for k in fold_results[0].keys() if k != "fold"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + keys)
        writer.writeheader()
        for row in fold_results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})
        mean_row = {"fold": "mean"}
        std_row  = {"fold": "std"}
        for k in keys:
            vals = [r[k] for r in fold_results if isinstance(r.get(k), float)]
            if vals:
                mean_row[k] = f"{float(np.mean(vals)):.4f}"
                std_row[k]  = f"{float(np.std(vals)):.4f}"
        writer.writerow(mean_row)
        writer.writerow(std_row)


if __name__ == "__main__":
    main()