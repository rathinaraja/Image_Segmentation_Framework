"""
utils/augmentations.py
-----------------------
Joint augmentations applied identically to both image and mask.
PIL-based so they integrate directly with SegmentationDataset.

Usage in dataset:
    from utils.augmentations import get_train_augmentations
    transform = get_train_augmentations(img_size=(512, 512))
    dataset   = SegmentationDataset(..., transform=transform)
"""

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class JointTransform:
    """Base class: apply the same random state to image and mask."""

    def __call__(self, image: Image.Image, mask: Image.Image):
        raise NotImplementedError


class JointRandomHorizontalFlip(JointTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        return image, mask


class JointRandomVerticalFlip(JointTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        return image, mask


class JointRandomRotation(JointTransform):
    """Rotate by a random angle within [-degrees, +degrees]."""

    def __init__(self, degrees: float = 15, p: float = 0.5):
        self.degrees = degrees
        self.p       = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)
        return image, mask


class JointRandomCrop(JointTransform):
    """Random crop then resize back to original size."""

    def __init__(self, img_size: Tuple[int, int], scale: Tuple[float, float] = (0.7, 1.0), p: float = 0.5):
        self.img_size = img_size
        self.scale    = scale
        self.p        = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            h, w     = self.img_size
            scale    = random.uniform(*self.scale)
            ch, cw   = int(h * scale), int(w * scale)
            i, j, _h, _w = T.RandomCrop.get_params(image, (ch, cw))
            image = TF.crop(image, i, j, _h, _w)
            mask  = TF.crop(mask,  i, j, _h, _w)
            image = TF.resize(image, self.img_size, interpolation=Image.BILINEAR)
            mask  = TF.resize(mask,  self.img_size, interpolation=Image.NEAREST)
        return image, mask


class JointColorJitter(JointTransform):
    """Color jitter applied only to the image (mask unchanged)."""

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p: float = 0.5):
        self.p      = p
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast,
                                     saturation=saturation, hue=hue)

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = self.jitter(image)
        return image, mask


class JointGaussianBlur(JointTransform):
    """Slight blur on image only."""

    def __init__(self, radius: float = 1.0, p: float = 0.2):
        self.radius = radius
        self.p      = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return image, mask


class JointCompose:
    """Chain multiple JointTransforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


# ── Factory functions ──────────────────────────────────────────────────────────

def get_train_augmentations(img_size: Tuple[int, int] = (512, 512)) -> JointCompose:
    """Standard augmentation pipeline for training."""
    return JointCompose([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.3),
        JointRandomRotation(degrees=15, p=0.4),
        JointRandomCrop(img_size=img_size, scale=(0.75, 1.0), p=0.4),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5),
        JointGaussianBlur(radius=1.0, p=0.2),
    ])


def get_val_augmentations() -> JointCompose:
    """No augmentation for validation — identity transform."""
    return JointCompose([])