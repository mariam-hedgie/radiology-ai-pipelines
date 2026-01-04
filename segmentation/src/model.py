# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from src.backbones.rad_dino_backbone import RadDinoBackbone
# from src.decoders.upernet import UPerNetDecoder

# class RadDinoUPerNet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.backbone = RadDinoBackbone()

#         self.decoder = UPerNetDecoder(
#             in_channels=[768, 768, 768, 768],
#             num_classes=num_classes
#         )

#     def forward(self, x):
#         feats = self.backbone(x)
#         logits = self.decoder(feats)

#         logits = F.interpolate(
#             logits,
#             size=x.shape[2:],
#             mode="bilinear",
#             align_corners=False
#         )
#         return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbones.rad_dino_backbone import RadDinoBackbone
from src.backbones.rad_jepa_backbone import RadJepaBackbone
from src.decoders.upernet import UPerNetDecoder


class UPerNetSegModel(nn.Module):
    def __init__(self, num_classes, backbone="dino", jepa_ckpt=None):
        super().__init__()

        if backbone == "dino":
            self.backbone = RadDinoBackbone()
        elif backbone == "jepa":
            assert jepa_ckpt is not None, "Pass --jepa_ckpt for JEPA backbone"
            self.backbone = RadJepaBackbone(ckpt_path=jepa_ckpt)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.decoder = UPerNetDecoder(
            in_channels=[768, 768, 768, 768],
            num_classes=num_classes
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.decoder(feats)

        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        return logits