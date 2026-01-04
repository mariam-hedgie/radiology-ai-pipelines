# src/backbones/rad_dino_backbone.py
from __future__ import annotations

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from rad_dino import RadDino


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    img: [3,H,W] torch tensor.
    Accepts:
      - float in [0,1]  (common if ToTensor used)
      - uint8 in [0,255]
      - float in [0,255] (we'll handle it)
    Returns: PIL.Image RGB
    """
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(img.shape)}")

    img = img.detach().cpu()

    if img.dtype == torch.uint8:
        arr = img.permute(1, 2, 0).numpy()  # [H,W,3] uint8
        return Image.fromarray(arr, mode="RGB")

    # float / other: convert to float32 numpy
    arr = img.float().permute(1, 2, 0).numpy()  # [H,W,3] float

    # If values look like [0,255], scale down
    if arr.max() > 1.5:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class RadDinoBackbone(nn.Module):
    """
    RAD-DINO backbone wrapper for linear probing.

    Returns:
      forward_features(x) -> cls embeddings [B, 768]
    """
    def __init__(self, device: str | torch.device | None = None):
        super().__init__()
        self.encoder = RadDino()  # loads microsoft/rad-dino via HF internally
        self.embed_dim = 768

        if device is not None:
            self.encoder = self.encoder.to(device)

        # freeze
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch tensor [B,3,H,W]
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Expected [B,3,H,W], got {tuple(x.shape)}"

        # Convert batch to list[PIL.Image] because RadDino.extract_features is PIL-native.
        images = [tensor_to_pil(x[i]) for i in range(x.shape[0])]

        cls_embeddings, _patch_embeddings = self.encoder.extract_features(images)

        # move back to same device as input
        cls_embeddings = cls_embeddings.to(x.device)
        return cls_embeddings.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def build_rad_dino_backbone(device: str | torch.device | None = None) -> RadDinoBackbone:
    return RadDinoBackbone(device=device)