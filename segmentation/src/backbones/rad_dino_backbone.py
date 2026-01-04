# src/backbones/rad_dino_backbone.py

import math
import torch
import torch.nn as nn
from rad_dino import RadDino


class RadDinoBackbone(nn.Module):
    """
    Frozen RAD-DINO (DINOv2 ViT-B/14) backbone that outputs 4 intermediate
    feature maps [B, 768, H, W] for UPerNet-style segmentation.
    """

    def __init__(self, out_indices=(2, 5, 8, 11)):
        super().__init__()

        self.rad = RadDino()
        self.backbone = self.rad.model  # Dinov2Model
        self.out_indices = set(out_indices)

        # Freeze encoder as per paper protocol
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embeddings = self.backbone.embeddings
        self.blocks = self.backbone.encoder.layer

    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, H, W] float tensor (image-like, roughly [0,1])
        returns: list of 4 feature maps [B, 768, h, w]
        """

        # Patch + position embeddings → tokens [B, 1+N, 768]
        hidden_states = self.embeddings(x)

        features = []
        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)

            if i in self.out_indices:
                features.append(self._tokens_to_map(hidden_states))

        return features

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, 1+N, C] (CLS + patch tokens)
        returns: [B, C, H, W]
        """
        # Remove CLS token
        tokens = tokens[:, 1:, :]  # [B, N, C]

        B, N, C = tokens.shape
        H = W = int(math.sqrt(N))
        if H * W != N:
            raise ValueError(
                f"Token count {N} is not a perfect square; cannot reshape."
            )

        return tokens.transpose(1, 2).reshape(B, C, H, W)