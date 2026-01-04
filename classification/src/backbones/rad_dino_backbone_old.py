# src/backbones/rad_dino_backbone.py

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

from rad_dino import RadDino


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
        x: torch tensor [B, 3, H, W], typically float in [0,1] or [0,255]
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Expected [B,3,H,W], got {tuple(x.shape)}"

        # Convert batch to list[PIL.Image] because RadDino.extract_features is PIL-native.
        images = []
        for i in range(x.shape[0]):
            img = x[i].detach().cpu()
            # to_pil_image expects float [0,1] or uint8 [0,255]; both are fine
            images.append(to_pil_image(img))

        cls_embeddings, _patch_embeddings = self.encoder.extract_features(images)
        
        cls_embeddings = cls_embeddings.to(x.device)
        return cls_embeddings.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def build_rad_dino_backbone(device: str | torch.device | None = None) -> RadDinoBackbone:
    return RadDinoBackbone(device=device)