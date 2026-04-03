"""
utils/config.py
---------------
YAML config loader with dot-access, defaults, and simple validation.
"""
import os
import yaml
from typing import Any


class ConfigDict(dict):
    """Dict subclass that supports attribute (dot) access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            return ConfigDict(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __repr__(self):
        return f"ConfigDict({super().__repr__()})"


_DEFAULTS = {
    "training": {
        "amp": True,
        "early_stopping_patience": 15,
        "loss": "cross_entropy",
        "fold_mode": "single",
        "eval_mode": "train_val_test",
        "n_folds": 5,
    },
    "optimizer": {
        "name": "adam",
        "momentum": 0.9,
    },
    "scheduler": {
        "name": "cosine",
        "step_size": 30,
        "gamma": 0.1,
        "min_lr": 1e-6,
    },
    "logging": {
        "log_interval": 10,
    },
}

_REQUIRED_KEYS = {
    "model":    ["name", "n_channels", "n_classes"],
    "dataset":  ["images_dir", "masks_dir"],
    "training": ["epochs", "batch_size", "learning_rate"],
    "logging":  ["log_dir"],          # save_dir removed — managed by train.py
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str) -> ConfigDict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(_DEFAULTS, user_cfg)
    _validate(cfg)
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)  # only log_dir here
    return ConfigDict(cfg)


def _validate(cfg: dict):
    for section, keys in _REQUIRED_KEYS.items():
        if section not in cfg:
            raise ValueError(f"Config missing section: [{section}]")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Config [{section}] missing required key: '{key}'")


def print_config(cfg: ConfigDict):
    print("=" * 50)
    print(" Configuration")
    print("=" * 50)
    print(yaml.dump(dict(cfg), default_flow_style=False, sort_keys=False))
    print("=" * 50)