import torch
import torch.nn as nn

class ViTBackbone(nn.Module):
    """
    Wraps a ViT-like encoder and exposes:
      - forward_features(x) -> [B, C] (global embedding for classification)
      - embed_dim
    """
    def __init__(self, encoder: nn.Module, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a single vector per image for linear probing.

        Tries common conventions:
        - timm ViT: encoder.forward_features(x) -> [B, C] or [B, tokens, C]
        - dict outputs: tries 'x_norm_clstoken' or 'cls' keys
        """
        out = self.encoder.forward_features(x)

        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]  # [B, C]
            if "cls" in out:
                return out["cls"]
            raise KeyError(f"Unknown dict keys from encoder.forward_features: {list(out.keys())}")

        # If tokens returned: [B, T, C], use CLS at 0
        if out.ndim == 3:
            return out[:, 0, :]

        # If already pooled: [B, C]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)