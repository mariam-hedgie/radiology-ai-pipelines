import torch
import torch.nn as nn

class LinearProbeClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

        # freeze encoder
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)   # [B, C]
        logits = self.classifier(feats)             # [B, K]
        return logits