import torch
import torch.nn as nn

class LinearProbeClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone

        embed_dim = getattr(backbone, "embed_dim", None) or getattr(backbone, "out_dim", None)
        if embed_dim is None:
            raise AttributeError("Backbone must expose embed_dim (or out_dim).")

        self.classifier = nn.Linear(embed_dim, num_classes)

        # freeze + set deterministic backbone behavior
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone.forward_features(x)  # [B, C]
        logits = self.classifier(feats)               # [B, K]
        return logits