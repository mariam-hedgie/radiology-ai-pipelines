import torch
from src.backbones.ijepa_seg_backbone import IJEPABackbone

model = IJEPABackbone(
    model_id="facebook/ijepa_vith16_1k",
    out_indices=(7, 15, 23, 31),  # 4 stages for UPerNet
)

model.eval()

# iJEPA vith16 is trained at 448x448
dummy = torch.rand(1, 3, 448, 448)

with torch.no_grad():
    feats = model(dummy)

for i, f in enumerate(feats):
    print(f"Feature {i}: {f.shape}")