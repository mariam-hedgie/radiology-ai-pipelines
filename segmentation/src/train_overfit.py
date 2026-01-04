import torch
from torch.utils.data import DataLoader
from src.model import RadDinoUPerNet
from src.datasets.lung_dataset import LungSegDataset

# --------------------
# Config
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
LR = 1e-4
STEPS = 300

IMG_DIR = "data/lung_seg/debug/images"
MASK_DIR = "data/lung_seg/debug/masks"

# --------------------
# Dataset
# --------------------
dataset = LungSegDataset(
    img_dir=IMG_DIR,
    mask_dir=MASK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

# --------------------
# Model
# --------------------
model = RadDinoUPerNet(num_classes=NUM_CLASSES).to(DEVICE)

# Freeze RAD-DINO backbone
for p in model.backbone.parameters():
    p.requires_grad = False

# --------------------
# Loss & Optimizer
# --------------------
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.decoder.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# --------------------
# Training loop
# --------------------
model.train()

# Freeze BatchNorm layers (important for batch_size=1)
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()

print(f"Overfitting on {len(dataset)} images")

data_iter = iter(loader)

for step in range(STEPS):
    try:
        img, mask = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        img, mask = next(data_iter)

    img = img.to(DEVICE)
    mask = mask.to(DEVICE)

    logits = model(img)
    loss = criterion(logits, mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 25 == 0:
        print(f"Step {step:03d} | Loss: {loss.item():.4f}")

print("Overfit training complete.")