# src/backbones/rad_jepa_backbone.py

import math
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer


class RadJepaBackbone(nn.Module):
    """
    Frozen RAD-JEPA ViT-B/14 @224 backbone from a JEPA encoder checkpoint.
    Outputs 4 intermediate feature maps [B, 768, h, w] for UPerNet segmentation.
    """

    def __init__(self, ckpt_path: str, out_indices=(2, 5, 8, 11)):
        super().__init__()
        self.out_indices = set(out_indices)

        # building a ViT that matched checkpoint:
        # - embed_dim=768 (ViT-B)
        # - patch_size=14
        # - img_size=224 -> 16x16=256 patches
        # - NO CLS token (pos_embed is [1,256,768], not [1,257,768])
        self.backbone: VisionTransformer = VisionTransformer(
            img_size=224,
            patch_size=14,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            class_token=False,   # critical for pos_embed length = 256
            global_pool="avg",
            num_classes=0,
        )

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["encoder"] if isinstance(ckpt, dict) and "encoder" in ckpt else ckpt

        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        print(f"[RadJepaBackbone] loaded. missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            print("[RadJepaBackbone] unexpected (first 20):", unexpected[:20])
        if missing:
            print("[RadJepaBackbone] missing (first 20):", missing[:20])

        # Freeze (same protocol as your RadDINO backbone)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Handy handles
        self.patch_embed = self.backbone.patch_embed
        self.pos_embed = self.backbone.pos_embed
        self.pos_drop = self.backbone.pos_drop
        self.blocks = self.backbone.blocks
        self.norm = self.backbone.norm

    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, H, W]
        returns: list of 4 feature maps [B, 768, h, w]
        """
        # Tokens: [B, N, C] where N=256, C=768
        x = self.patch_embed(x)          # timm returns tokens for ViT
        x = x + self.pos_embed           # pos_embed is [1,256,768]
        x = self.pos_drop(x)

        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                features.append(self._tokens_to_map(x))

        x = self.norm(x)
        return features

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, C] with N=H*W
        returns: [B, C, H, W]
        """
        B, N, C = tokens.shape
        H = W = int(math.sqrt(N))
        if H * W != N:
            raise ValueError(f"Token count {N} is not a perfect square; cannot reshape.")
        return tokens.transpose(1, 2).reshape(B, C, H, W)