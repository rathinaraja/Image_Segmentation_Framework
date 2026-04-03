"""
utils/dataset.py
----------------
Supports four dataset split modes driven entirely by config:

  fold_mode: single  + eval_mode: train_val_test  ->  train / val / test
  fold_mode: single  + eval_mode: train_test       ->  train / test  (no val)
  fold_mode: kfold   + eval_mode: train_val_test   ->  K folds on (all - test); each fold has train+val; holdout test
  fold_mode: kfold   + eval_mode: train_test        ->  K folds; each fold is train/test (no holdout)

Public API
----------
  get_splits(cfg)  ->  list of dicts, one per fold:
      {
          "fold":         int,
          "train_loader": DataLoader,
          "val_loader":   DataLoader | None,
          "test_loader":  DataLoader | None,
      }
"""

import numpy as np
import torch
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF


# ── Dataset ────────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, images_dir, masks_dir, img_size=(512, 512), transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.img_size   = img_size
        self.transform  = transform

        self.image_paths = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])
        if not self.image_paths:
            raise RuntimeError(f"No images found in {images_dir}")

        self.mask_paths = []
        for img_path in self.image_paths:
            mask = self._find_mask(img_path.stem)
            if mask is None:
                raise FileNotFoundError(f"No matching mask for '{img_path.name}' in {masks_dir}")
            self.mask_paths.append(mask)

    def _find_mask(self, stem):
        for ext in self.IMG_EXTENSIONS | {".png"}:
            c = self.masks_dir / f"{stem}{ext}"
            if c.exists():
                return c
        return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx]).convert("L")

        image = TF.resize(image, self.img_size, interpolation=Image.BILINEAR)
        mask  = TF.resize(mask,  self.img_size, interpolation=Image.NEAREST)

        if self.transform:
            image, mask = self.transform(image, mask)

        image    = TF.to_tensor(image)
        image    = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask_np  = (np.array(mask, dtype=np.float32) / 255).round().astype(np.int64)
        return image, torch.from_numpy(mask_np)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_loader(dataset, indices, batch_size, num_workers, shuffle):
    if not indices:
        return None
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def _loader_kw(cfg):
    return dict(batch_size=cfg["training"]["batch_size"],
                num_workers=cfg["dataset"].get("num_workers", 4))


def _split_indices(n, fractions, seed=42):
    """Split range(n) into len(fractions) index lists."""
    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(n).tolist()
    splits, cursor = [], 0
    for i, frac in enumerate(fractions):
        size = int(n * frac) if i < len(fractions) - 1 else n - cursor
        splits.append(idx[cursor: cursor + size])
        cursor += size
    return splits


def _build_datasets(cfg):
    from utils.augmentations import get_train_augmentations, get_val_augmentations
    ds_cfg   = cfg["dataset"]
    img_size = tuple(ds_cfg.get("img_size", [512, 512]))
    kw       = dict(images_dir=ds_cfg["images_dir"], masks_dir=ds_cfg["masks_dir"], img_size=img_size)
    train_ds = SegmentationDataset(**kw, transform=get_train_augmentations(img_size)
                                   if ds_cfg.get("augment", False) else get_val_augmentations())
    base_ds  = SegmentationDataset(**kw, transform=get_val_augmentations())
    return train_ds, base_ds


# ── Public API ─────────────────────────────────────────────────────────────────

def get_splits(cfg):
    """
    Returns list of fold dicts:
      [{"fold": 1, "train_loader": ..., "val_loader": ..., "test_loader": ...}, ...]
    val_loader / test_loader may be None depending on eval_mode.

    Supported eval_mode values:
      train_val_test  : train + val + holdout test (test run once at end with best model)
      train_val       : train + val only; val also used as final test (no separate holdout)
      train_test      : train + test per epoch (no val, no holdout)
      training_only   : train only — no evaluation at all
    """
    fold_mode = cfg["training"].get("fold_mode", "single").lower()
    eval_mode = cfg["training"].get("eval_mode", "train_val_test").lower()

    valid_modes = {"train_val_test", "train_val", "train_test", "training_only"}
    if eval_mode not in valid_modes:
        raise ValueError(f"Unknown eval_mode: '{eval_mode}'. "
                         f"Options: {sorted(valid_modes)}")

    if fold_mode == "single":
        return _single_fold(cfg, eval_mode)
    elif fold_mode == "kfold":
        return _kfold(cfg, eval_mode)
    else:
        raise ValueError(f"Unknown fold_mode: '{fold_mode}'. Use 'single' or 'kfold'.")


def _single_fold(cfg, eval_mode):
    train_ds, base_ds = _build_datasets(cfg)
    kw  = _loader_kw(cfg)
    n   = len(base_ds)
    ds  = cfg["dataset"]

    if eval_mode == "training_only":
        # Entire dataset used for training — no splits at all
        return [{"fold": 1,
                 "train_loader": _make_loader(train_ds, list(range(n)), shuffle=True, **kw),
                 "val_loader":   None,
                 "test_loader":  None}]

    elif eval_mode == "train_val":
        # Train + val only. Val set doubles as the final test report.
        # No separate holdout — useful when dataset is too small to split 3 ways.
        val  = ds.get("val_split", 0.2)
        tr_i, val_i = _split_indices(n, [1 - val, val])
        val_loader  = _make_loader(base_ds, val_i, shuffle=False, **kw)
        return [{"fold": 1,
                 "train_loader": _make_loader(train_ds, tr_i, shuffle=True, **kw),
                 "val_loader":   val_loader,
                 "test_loader":  val_loader}]   # same loader → val acts as test

    elif eval_mode == "train_val_test":
        te   = ds.get("test_split", 0.2)
        val  = ds.get("val_split",  0.1)
        tr_i, val_i, te_i = _split_indices(n, [1 - te - val, val, te])
        return [{"fold": 1,
                 "train_loader": _make_loader(train_ds, tr_i,  shuffle=True,  **kw),
                 "val_loader":   _make_loader(base_ds,  val_i, shuffle=False, **kw),
                 "test_loader":  _make_loader(base_ds,  te_i,  shuffle=False, **kw)}]

    else:  # train_test
        te = ds.get("test_split", 0.2)
        tr_i, te_i = _split_indices(n, [1 - te, te])
        return [{"fold": 1,
                 "train_loader": _make_loader(train_ds, tr_i, shuffle=True,  **kw),
                 "val_loader":   None,
                 "test_loader":  _make_loader(base_ds,  te_i, shuffle=False, **kw)}]


def _kfold(cfg, eval_mode):
    from sklearn.model_selection import KFold
    train_ds, base_ds = _build_datasets(cfg)
    kw      = _loader_kw(cfg)
    n       = len(base_ds)
    n_folds = cfg["training"].get("n_folds", 5)
    ds      = cfg["dataset"]
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    if eval_mode == "training_only":
        # All data for training — no splits, same for every fold
        tr_i = list(range(n))
        return [{"fold": f + 1,
                 "train_loader": _make_loader(train_ds, tr_i, shuffle=True, **kw),
                 "val_loader":   None,
                 "test_loader":  None}
                for f in range(n_folds)]

    elif eval_mode == "train_val":
        # KFold: each fold's held-out slice = val AND test (no separate holdout)
        all_idx = list(range(n))
        splits  = []
        for fold_num, (tr_pos, val_pos) in enumerate(kf.split(all_idx), 1):
            val_loader = _make_loader(base_ds, list(val_pos), shuffle=False, **kw)
            splits.append({
                "fold":         fold_num,
                "train_loader": _make_loader(train_ds, list(tr_pos), shuffle=True,  **kw),
                "val_loader":   val_loader,
                "test_loader":  val_loader,   # val doubles as test
            })
        return splits

    elif eval_mode == "train_val_test":
        # Carve out a global holdout test set first, then KFold on remainder
        te              = ds.get("test_split", 0.2)
        tv_i, te_i      = _split_indices(n, [1 - te, te])
        tv_arr          = np.array(tv_i)
        splits = []
        for fold_num, (tr_pos, val_pos) in enumerate(kf.split(tv_arr), 1):
            splits.append({
                "fold":         fold_num,
                "train_loader": _make_loader(train_ds, tv_arr[tr_pos].tolist(),  shuffle=True,  **kw),
                "val_loader":   _make_loader(base_ds,  tv_arr[val_pos].tolist(), shuffle=False, **kw),
                "test_loader":  _make_loader(base_ds,  te_i,                     shuffle=False, **kw),
            })
        return splits

    else:  # train_test — each fold's held-out portion is test, no val
        all_idx = list(range(n))
        splits  = []
        for fold_num, (tr_pos, te_pos) in enumerate(kf.split(all_idx), 1):
            splits.append({
                "fold":         fold_num,
                "train_loader": _make_loader(train_ds, list(tr_pos), shuffle=True,  **kw),
                "val_loader":   None,
                "test_loader":  _make_loader(base_ds,  list(te_pos), shuffle=False, **kw),
            })
        return splits


# Legacy shim so existing imports don't break
def build_dataloaders(cfg):
    splits = get_splits(cfg)
    s = splits[0]
    return s["train_loader"], s.get("val_loader"), s.get("test_loader")
