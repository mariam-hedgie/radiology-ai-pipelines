import torch
from src.backbones.rad_dino_backbone import RadDinoBackbone

model = RadDinoBackbone()
model.eval()

dummy = torch.rand(1, 3, 518, 518) # random image-like tensor

with torch.no_grad():
    feats = model(dummy)

for i, f in enumerate(feats):
    print(f"Feature {i}: {f.shape}")