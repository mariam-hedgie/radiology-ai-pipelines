from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch


@dataclass
class ImageTransformConfig:
    image_size: int = 224
    to_3ch: bool = True  # if input is grayscale, replicate to 3 channels


def load_image(path: str | Path) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing image: {p}")
    img = Image.open(p)

    # Ensure we don't crash on weird modes (L, LA, RGBA, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    PIL RGB -> torch float tensor in [0,1], shape [3,H,W]
    (No normalization here; keep it explicit.)
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))               # [3,H,W]
    return torch.from_numpy(arr)


def build_image_transform(cfg: ImageTransformConfig) -> Callable[[str | Path], torch.Tensor]:
    """
    Returns a callable that takes an image path and returns a tensor [3,S,S] float in [0,1].
    """
    def _tfm(path: str | Path) -> torch.Tensor:
        img = load_image(path)
        img = img.resize((cfg.image_size, cfg.image_size))
        x = pil_to_tensor(img)
        return x

    return _tfm