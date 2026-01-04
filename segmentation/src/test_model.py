import torch
#from src.model import RadDinoUPerNet
from src.model import UPerNetSegModel

#model = RadDinoUPerNet(num_classes=2)
model = UPerNetSegModel(
    num_classes=2,
    backbone=args.backbone,
    jepa_ckpt=args.jepa_ckpt
).to(device)
model.eval()

x = torch.rand(1, 3, 518, 518)

with torch.no_grad():
    y = model(x)

print("Output shape:", y.shape)