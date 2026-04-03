"""
Model Registry
--------------
To register a new model:
  1. Add its folder under modules/
  2. Import it here and add to MODEL_REGISTRY
  3. Add a matching config under configs/
  4. Add a process class under process/
"""

from modules.unet.unet_model     import UNet
from modules.segnet.segnet_model import SegNet
from modules.nnunet.nnunet_model  import NNUNet
from modules.attention_unet.attention_unet_model    import AttentionUNet
from modules.unetpp.unetpp_model                    import UNetPP
from modules.transunet.transunet_model              import TransUNet
from modules.swinunet.swinunet_model                import SwinUNet
from modules.segformer.segformer_model              import SegFormer

MODEL_REGISTRY = {
    "unet":   UNet,
    "segnet": SegNet,
    "nnunet": NNUNet,
    "attention_unet": AttentionUNet,
    "unetpp": UNetPP,
    "transunet": TransUNet,
    "swinunet": SwinUNet,
    "segformer": SegFormer,
}

def get_model(cfg: dict):
    """Instantiate a model from config dict."""
    name = cfg["model"]["name"].lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[name]
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    return model_cls(**model_cfg)
