# src/infer_test_cls.py
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.models.linear_probe import LinearProbeClassifier
from src.backbones.rad_dino_backbone import build_rad_dino_backbone

# ------------- CONFIG -------------
DATA_ROOT = Path("data")
SPLIT = "test"                       # "train" | "val" | "test"
IMAGE_SIZE = 224
BATCH_SIZE = 1
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# choose one:
# CKPT_PATH = Path("checkpoints/epoch_10.pt")
# CKPT_PATH = Path("checkpoints/best/best_by_loss.pt")
CKPT_PATH = Path("checkpoints/best/best_by_acc.pt")

OUT_DIR = Path("outputs") / SPLIT / CKPT_PATH.stem
SAVE_MAX = 30
# ----------------------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] float -> uint8 [H,W,3]"""
    img = img_tensor.detach().cpu().clamp(0, 1)
    img = (img * 255).byte().permute(1, 2, 0).numpy()
    return img


def draw_banner(img_np: np.ndarray, text: str) -> np.ndarray:
    """
    Draw a top banner with text (GT/PRED info) on the image.
    Returns uint8 numpy image.
    """
    im = Image.fromarray(img_np)
    draw = ImageDraw.Draw(im)

    # banner
    W, H = im.size
    banner_h = max(28, H // 10)
    draw.rectangle([0, 0, W, banner_h], fill=(0, 0, 0))

    # font (default PIL font; portable on servers)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def main():
    split_dir = DATA_ROOT / SPLIT
    assert split_dir.exists(), f"Missing: {split_dir}"
    assert CKPT_PATH.exists(), f"Missing checkpoint: {CKPT_PATH}"

    ensure_dir(OUT_DIR)

    # --- data
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    ds = ImageFolder(root=str(split_dir), transform=tfm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    classes = ds.classes
    num_classes = len(classes)

    # --- model
    backbone = build_rad_dino_backbone(device=DEVICE)
    model = LinearProbeClassifier(backbone=backbone, num_classes=num_classes).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"Saving outputs to: {OUT_DIR}")
    print(f"Classes: {classes}")

    # --- csv
    csv_path = OUT_DIR / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["index", "filepath", "true_name", "pred_name"] + [f"prob_{c}" for c in classes]
        writer.writerow(header)

        saved = 0
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(loader):
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(imgs)            # [1,K]
                probs = F.softmax(logits, dim=1)[0]  # [K]

                pred_idx = int(torch.argmax(probs).item())
                true_idx = int(labels[0].item())

                pred_name = classes[pred_idx]
                true_name = classes[true_idx]
                pred_prob = float(probs[pred_idx].item())

                # path
                img_path, _ = ds.samples[i]

                # convert image to numpy
                img_np = to_uint8(imgs[0])

                # overlays (segmentation-style: gt overlay + pred overlay)
                gt_overlay = draw_banner(img_np, f"GT: {true_name}")
                pred_overlay = draw_banner(img_np, f"PRED: {pred_name} ({pred_prob:.3f})")

                # save outputs like segmentation
                Image.fromarray(img_np).save(OUT_DIR / f"{i:04d}_img.png")
                Image.fromarray(gt_overlay).save(OUT_DIR / f"{i:04d}_gt_overlay.png")
                Image.fromarray(pred_overlay).save(OUT_DIR / f"{i:04d}_pred_overlay.png")

                # write csv row
                row = [i, img_path, true_name, pred_name] + [f"{float(probs[k].item()):.6f}" for k in range(num_classes)]
                writer.writerow(row)

                saved += 1
                if saved >= SAVE_MAX:
                    break

    print(f"Done. Saved {saved} samples.")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()