import torch
from src.backbones.rad_jepa_backbone import RadJepaBackbone

model = RadJepaBackbone(ckpt_path="/data1/mariam/anas_imp/best_jepa_weights.pth.tar")

model.eval()

dummy = torch.rand(1, 3, 224, 224) # random image-like tensor

with torch.no_grad():
    feats = model(dummy)

for i, f in enumerate(feats):
    print(f"Feature {i}: {f.shape}")