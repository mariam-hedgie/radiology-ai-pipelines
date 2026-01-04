# src/infer_test.py
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from src.datasets.lung_dataset import LungSegDataset
#from src.model import RadDinoUPerNet
from src.model import UPerNetSegModel



# ------------- CONFIG -------------
DATA_ROOT = Path("data/lung_seg")
SPLIT = "test"                       # test split
NUM_CLASSES = 2
BATCH_SIZE = 1                       # inference = 1 is fine
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# choose one:
#CKPT_PATH = Path("checkpoints/epoch_10.pth")
CKPT_PATH = Path("checkpoints_jepa/best.pth")

OUT_DIR = Path("outputs_jepa") / SPLIT / CKPT_PATH.stem
SAVE_MAX = 30                        # save first N samples
# ----------------------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """img_tensor: [3,H,W] float in [0,1] -> uint8 [H,W,3]"""
    img = img_tensor.detach().cpu().clamp(0, 1)
    img = (img * 255).byte().permute(1, 2, 0).numpy()
    return img


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    mask: [H,W] values {0,1}
    return colored mask [H,W,3] for visualization
    """
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # class 1 = lung -> green
    color[mask == 1] = np.array([0, 255, 0], dtype=np.uint8)
    return color


def overlay(img: np.ndarray, colored_mask: np.ndarray, alpha=0.35) -> np.ndarray:
    """alpha blend mask on image"""
    return (img * (1 - alpha) + colored_mask * alpha).astype(np.uint8)


def main():
    # --- paths
    img_dir = DATA_ROOT / SPLIT / "images"
    mask_dir = DATA_ROOT / SPLIT / "masks"
    assert img_dir.exists(), f"Missing: {img_dir}"
    assert mask_dir.exists(), f"Missing: {mask_dir}"

    ensure_dir(OUT_DIR)

    # --- data
    ds = LungSegDataset(str(img_dir), str(mask_dir))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- model
    #model = RadDinoUPerNet(num_classes=NUM_CLASSES).to(DEVICE)
    JEPA_CKPT = "/data1/mariam/best_jepa_weights.pth.tar"  # <-- set your real path here

    model = UPerNetSegModel(
    num_classes=NUM_CLASSES,
    backbone="jepa",
    jepa_ckpt=JEPA_CKPT
    ).to(DEVICE)

    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"Saving visualizations to: {OUT_DIR}")

    saved = 0
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(DEVICE)               # [1,3,H,W]
            masks = masks.to(DEVICE)             # [1,H,W]

            logits = model(imgs)                 # [1,C,H,W]
            pred = torch.argmax(logits, dim=1)   # [1,H,W]

            # convert to numpy for saving
            img_np = to_uint8(imgs[0])
            gt_np = masks[0].detach().cpu().numpy().astype(np.uint8)
            pred_np = pred[0].detach().cpu().numpy().astype(np.uint8)

            # make overlays
            gt_col = mask_to_color(gt_np)
            pred_col = mask_to_color(pred_np)

            gt_overlay = overlay(img_np, gt_col, alpha=0.35)
            pred_overlay = overlay(img_np, pred_col, alpha=0.35)

            # save: original, gt overlay, pred overlay, raw pred mask
            Image.fromarray(img_np).save(OUT_DIR / f"{i:04d}_img.png")
            Image.fromarray(gt_overlay).save(OUT_DIR / f"{i:04d}_gt_overlay.png")
            Image.fromarray(pred_overlay).save(OUT_DIR / f"{i:04d}_pred_overlay.png")
            Image.fromarray((pred_np * 255).astype(np.uint8)).save(OUT_DIR / f"{i:04d}_pred_mask.png")

            saved += 1
            if saved >= SAVE_MAX:
                break

    print(f"Done. Saved {saved} samples.")


if __name__ == "__main__":
    main()