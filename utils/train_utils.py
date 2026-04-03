"""
utils/train_utils.py
---------------------
Training process: loss functions, optimizer, scheduler, and the
ModelProcess class that handles train/eval/test loop, checkpointing,
and logging for ALL segmentation models.

Since every model (UNet, SegNet, nnU-Net, AttentionUNet, UNet++)
has the same training loop, a single class handles them all.
The model architecture lives in modules/ — this file only deals
with how to train it.

If a future model needs custom training logic, subclass ModelProcess
and override train_one_epoch() or evaluate().
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from utils.metrics import MetricTracker, pixel_accuracy, mean_iou, dice_score
from utils.logger  import get_logger, CSVLogger


# ── Loss functions ─────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, n_classes: int, eps: float = 1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.eps       = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        loss  = 0.0
        for cls in range(self.n_classes):
            p     = probs[:, cls]
            t     = (targets == cls).float()
            loss += 1 - (2 * (p * t).sum() + self.eps) / (p.sum() + t.sum() + self.eps)
        return loss / self.n_classes


def build_loss(name: str, n_classes: int) -> nn.Module:
    name = name.lower()
    if name == "cross_entropy": return nn.CrossEntropyLoss()
    if name == "dice":          return DiceLoss(n_classes)
    if name == "dice_ce":
        ce, dice = nn.CrossEntropyLoss(), DiceLoss(n_classes)
        return lambda logits, targets: ce(logits, targets) + dice(logits, targets)
    raise ValueError(f"Unknown loss '{name}'. Options: cross_entropy | dice | dice_ce")


def build_optimizer(cfg: dict, model: nn.Module) -> optim.Optimizer:
    name = cfg["optimizer"]["name"].lower()
    lr   = cfg["training"]["learning_rate"]
    wd   = cfg["training"].get("weight_decay", 0)
    if name == "adam":  return optim.Adam(model.parameters(),  lr=lr, weight_decay=wd)
    if name == "adamw": return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":   return optim.SGD(
        model.parameters(), lr=lr, weight_decay=wd,
        momentum=cfg["optimizer"].get("momentum", 0.9))
    raise ValueError(f"Unknown optimizer '{name}'. Options: adam | adamw | sgd")


def build_scheduler(cfg: dict, optimizer: optim.Optimizer):
    s    = cfg["scheduler"]
    name = s["name"].lower()
    if name == "cosine":  return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=s.get("min_lr", 1e-6))
    if name == "step":    return optim.lr_scheduler.StepLR(
        optimizer, step_size=s.get("step_size", 30), gamma=s.get("gamma", 0.1))
    if name == "plateau": return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=s.get("gamma", 0.1))
    raise ValueError(f"Unknown scheduler '{name}'. Options: cosine | step | plateau")


# ── Model process ──────────────────────────────────────────────────────────────

class ModelProcess:
    """
    Unified training / evaluation / inference handler for all models.

    Usage (from train.py):
        from utils.train_utils import ModelProcess
        process = ModelProcess(model, cfg, device,
                               checkpoint_dir=ckpt_dir, log_dir=fold_dir)
        process.train(train_loader, val_loader=val_loader, test_loader=test_loader)

    To add a model with custom training logic:
        class MyModelProcess(ModelProcess):
            def train_one_epoch(self, loader, epoch):
                ...  # custom logic
    """

    def __init__(self, model: nn.Module, cfg: dict, device: torch.device,
                 checkpoint_dir: str, log_dir: str):
        self.model     = model.to(device)
        self.cfg       = cfg
        self.device    = device
        self.n_classes = cfg["model"]["n_classes"]
        self.amp       = cfg["training"].get("amp", True)

        self.criterion = build_loss(cfg["training"].get("loss", "cross_entropy"),
                                    self.n_classes)
        self.optimizer = build_optimizer(cfg, model)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.scaler    = GradScaler('cuda', enabled=self.amp)

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir,        exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        model_name          = cfg["model"]["name"]
        self.logger         = get_logger(f"{model_name}_{os.path.basename(log_dir)}", log_dir)
        self.csv_logger     = CSVLogger(os.path.join(log_dir, "metrics.csv"))

        self.best_loss    = float("inf")
        self.patience_ctr = 0
        self.patience     = cfg["training"].get("early_stopping_patience", 15)

    # ── One training epoch ────────────────────────────────────────────────────

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        tracker = MetricTracker()

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device,  non_blocking=True)

            self.optimizer.zero_grad()
            with autocast('cuda', enabled=self.amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=1).detach().cpu()
            n     = images.size(0)
            tracker.update("train_loss", loss.item(),                                    n=n)
            tracker.update("train_acc",  pixel_accuracy(preds, masks.cpu()),             n=n)
            tracker.update("train_iou",  mean_iou(preds, masks.cpu(), self.n_classes),   n=n)
            tracker.update("train_dice", dice_score(preds, masks.cpu(), self.n_classes), n=n)

        return tracker.summary()

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "val") -> dict:
        """prefix = 'val' or 'test' — used as metric key prefix in CSV."""
        if loader is None or len(loader) == 0:
            return {}

        self.model.eval()
        tracker = MetricTracker()

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device,  non_blocking=True)

            with autocast('cuda', enabled=self.amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            preds = logits.argmax(dim=1).cpu()
            m_cpu = masks.cpu()
            n     = images.size(0)
            tracker.update(f"{prefix}_loss", loss.item(),                              n=n)
            tracker.update(f"{prefix}_acc",  pixel_accuracy(preds, m_cpu),             n=n)
            tracker.update(f"{prefix}_iou",  mean_iou(preds, m_cpu, self.n_classes),   n=n)
            tracker.update(f"{prefix}_dice", dice_score(preds, m_cpu, self.n_classes), n=n)

        return tracker.summary()

    # ── Full training loop ────────────────────────────────────────────────────

    def train(self, train_loader: DataLoader,
              val_loader=None, test_loader=None):
        epochs    = self.cfg["training"]["epochs"]
        eval_mode = self.cfg["training"].get("eval_mode", "train_val_test").lower()
        has_val   = val_loader  is not None and len(val_loader)  > 0
        has_test  = test_loader is not None and len(test_loader) > 0

        # In train_val mode val_loader and test_loader point to the same object.
        # Flag this so we don't run evaluate() twice per epoch.
        val_is_test = (eval_mode == "train_val")

        self.logger.info(
            f"Training {epochs} epochs | eval_mode={eval_mode} | "
            f"val={'yes' if has_val else 'no'} | "
            f"test={'yes (=val)' if val_is_test else ('yes' if has_test else 'no')}"
        )

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs} in progress ...")

            # ── Train ──────────────────────────────────────────────────
            train_metrics = self.train_one_epoch(train_loader, epoch)

            # ── Val ────────────────────────────────────────────────────
            val_metrics = self.evaluate(val_loader, prefix="val") if has_val else {}

            # ── Test (per-epoch only for train_test mode) ───────────────
            # train_val_test: test runs once at end (after loop)
            # train_val     : test=val, already computed above
            # train_test     : test runs every epoch
            # training_only  : no test
            if eval_mode == "train_test" and has_test:
                test_metrics = self.evaluate(test_loader, prefix="test")
            else:
                test_metrics = {}

            # ── Scheduler ──────────────────────────────────────────────
            sched_metric = val_metrics.get("val_loss", train_metrics["train_loss"])
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(sched_metric)
            else:
                self.scheduler.step()

            # ── CSV — one row per epoch ─────────────────────────────────
            row = {"epoch": epoch, **train_metrics, **val_metrics,
                   **test_metrics, "lr": self.optimizer.param_groups[0]["lr"]}
            self.csv_logger.log(row)

            # ── Console ────────────────────────────────────────────────
            msg = (f"Epoch {epoch:03d}/{epochs} | "
                   f"loss={train_metrics['train_loss']:.4f} "
                   f"iou={train_metrics['train_iou']:.4f}")
            if val_metrics:
                msg += (f" | val_loss={val_metrics['val_loss']:.4f} "
                        f"val_iou={val_metrics['val_iou']:.4f}")
            if test_metrics:
                msg += (f" | test_loss={test_metrics['test_loss']:.4f} "
                        f"test_iou={test_metrics['test_iou']:.4f}")
            self.logger.info(msg)

            # ── Checkpoints ─────────────────────────────────────────────
            monitor = val_metrics.get("val_loss",
                      test_metrics.get("test_loss", train_metrics["train_loss"]))
            self._save_checkpoints(epoch, monitor,
                                   {**train_metrics, **val_metrics, **test_metrics})

            # ── Early stopping ──────────────────────────────────────────
            if self.patience_ctr >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}.")
                break

        # ── Post-training test evaluation ──────────────────────────────
        test_csv = CSVLogger(os.path.join(
            os.path.dirname(self.csv_logger.path), "test_results.csv"))

        if eval_mode == "train_val_test" and has_test and has_val:
            # Separate holdout test — run once with best model
            self.logger.info("Running final test with best model ...")
            self.load_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pth"))
            test_metrics = self.evaluate(test_loader, prefix="test")
            self._log_final_test(test_metrics)
            test_csv.log({"epoch": "final_test", **test_metrics})

        elif eval_mode == "train_test" and has_test:
            # Re-evaluate test set with best model at end
            self.logger.info("Running final test with best model ...")
            self.load_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pth"))
            test_metrics = self.evaluate(test_loader, prefix="test")
            self._log_final_test(test_metrics)
            test_csv.log({"epoch": "final_test", **test_metrics})

        elif eval_mode == "train_val" and has_val:
            # Val set acts as test — run once with best model
            self.logger.info("Running final eval on val set (train_val mode) ...")
            self.load_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pth"))
            test_metrics = self.evaluate(val_loader, prefix="test")
            self._log_final_test(test_metrics)
            test_csv.log({"epoch": "final_val_as_test", **test_metrics})

        elif eval_mode == "training_only":
            self.logger.info("training_only mode — no evaluation performed.")

        self.logger.info("Training complete.")

    def _log_final_test(self, test_metrics: dict):
        self.logger.info(
            f"Final Test | loss={test_metrics['test_loss']:.4f} "
            f"acc={test_metrics['test_acc']:.4f} "
            f"iou={test_metrics['test_iou']:.4f} "
            f"dice={test_metrics['test_dice']:.4f}"
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with autocast('cuda', enabled=self.amp):
            logits = self.model(images.to(self.device))
        return logits.argmax(dim=1).cpu()

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoints(self, epoch: int, monitored_loss: float, metrics: dict):
        state = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics":         metrics,
        }
        torch.save(state, os.path.join(self.checkpoint_dir, "last_model.pth"))

        if monitored_loss < self.best_loss:
            self.best_loss    = monitored_loss
            self.patience_ctr = 0
            torch.save(state, os.path.join(self.checkpoint_dir, "best_model.pth"))
            self.logger.info(f"  ✓ Best model saved (loss={monitored_loss:.4f})")
        else:
            self.patience_ctr += 1

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.logger.info(f"Loaded checkpoint: {path}  (epoch {ckpt.get('epoch', '?')})")
