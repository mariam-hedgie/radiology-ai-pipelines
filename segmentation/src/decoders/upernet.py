import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
            )
            for scale in pool_scales
        ])

    def forward(self, x):
        h, w = x.shape[2:]
        ppm_outs = [x]

        for stage in self.stages:
            pooled = stage(x)
            pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            ppm_outs.append(pooled)

        return torch.cat(ppm_outs, dim=1)
    
class UPerNetDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        in_channels: list like [768, 768, 768, 768]
        """
        super().__init__()

        self.ppm = PyramidPoolingModule(
            in_channels=in_channels[-1],
            out_channels=256
        )

        ppm_out_channels = in_channels[-1] + 4 * 256

        self.ppm_bottleneck = ConvBNReLU(ppm_out_channels, 256)

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, 256, kernel_size=1)
            for c in in_channels[:-1]
        ])

        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(256, 256)
            for _ in in_channels[:-1]
        ])

        self.fpn_bottleneck = ConvBNReLU(256 * len(in_channels), 256)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, features):
        """
        features: list of feature maps
          [
            C1, C2, C3, C4
          ]
        """
        c1, c2, c3, c4 = features

        # PPM on top feature
        p4 = self.ppm(c4)
        p4 = self.ppm_bottleneck(p4)

        # Top-down FPN
        fpn_outs = [p4]

        for i in reversed(range(len(self.lateral_convs))):
            lateral = self.lateral_convs[i]([c1, c2, c3][i])
            p4 = F.interpolate(p4, size=lateral.shape[2:], mode="bilinear", align_corners=False)
            p4 = lateral + p4
            p4 = self.fpn_convs[i](p4)
            fpn_outs.insert(0, p4)

        # Fuse all FPN outputs
        for i in range(len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=False
            )

        fused = torch.cat(fpn_outs, dim=1)
        fused = self.fpn_bottleneck(fused)

        return self.classifier(fused)