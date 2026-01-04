# src/infer_test_cls_fast.py
from pathlib import Path
import csv
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pil_to_tensor_resize(img: Image.Image, size: int) -> torch.Tensor:
    """
    Match training preprocessing:
    - Resize (H,W) to (size,size)
    - Convert to float tensor in [0,1], shape [3,H,W]
    """
    img = img.convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0        # [H,W,3] in [0,1]
    arr = np.transpose(arr, (2, 0, 1))                   # -> [3,H,W]
    return torch.from_numpy(arr)


def to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] float -> uint8 [H,W,3]"""
    img = img_tensor.detach().cpu().clamp(0, 1)
    img = (img * 255).byte().permute(1, 2, 0).numpy()
    return img


def draw_banner(img_np: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img_np)
    draw = ImageDraw.Draw(im)

    W, H = im.size
    banner_h = max(28, H // 10)
    draw.rectangle([0, 0, W, banner_h], fill=(0, 0, 0))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return np.array(im)


class FolderClassDataset(Dataset):
    """
    Expects:
      data/<split>/<class_name>/*.{jpg,png,...}
    Returns:
      tensor_image [3,H,W] float in [0,1], label int, path str
    """
    def __init__(self, root: Path, classes: list[str], image_size: int):
        self.root = root
        self.classes = classes
        self.image_size = image_size
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        samples = []
        for c in classes:
            class_dir = root / c
            if not class_dir.exists():
                continue
            for ext in IMG_EXTS:
                samples.extend([(p, self.class_to_idx[c]) for p in class_dir.glob(f"*{ext}")])

        # also catch uppercase extensions / mixed (glob above is case-sensitive)
        # fallback via glob module
        if len(samples) == 0:
            for c in classes:
                class_dir = root / c
                for p in glob.glob(str(class_dir / "**" / "*"), recursive=True):
                    pp = Path(p)
                    if pp.suffix.lower() in IMG_EXTS:
                        samples.append((pp, self.class_to_idx[c]))

        samples.sort(key=lambda x: str(x[0]))
        self.samples = samples

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {root} (classes={classes})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path)
        x = pil_to_tensor_resize(img, self.image_size)
        return x, label, str(path)


def collate_fn(batch):
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels, list(paths)


def main():
    split_dir = DATA_ROOT / SPLIT
    assert split_dir.exists(), f"Missing: {split_dir}"
    assert CKPT_PATH.exists(), f"Missing checkpoint: {CKPT_PATH}"

    # infer classes from folder names (like ImageFolder does)
    classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if len(classes) == 0:
        raise RuntimeError(f"No class subfolders found in: {split_dir}")

    ensure_dir(OUT_DIR)

    ds = FolderClassDataset(split_dir, classes=classes, image_size=IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)

    num_classes = len(classes)

    backbone = build_rad_dino_backbone(device=DEVICE)
    model = LinearProbeClassifier(backbone=backbone, num_classes=num_classes).to(DEVICE)

    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"Split: {SPLIT} | Classes: {classes}")
    print(f"Saving outputs to: {OUT_DIR}")
    print(f"Total images found: {len(ds)}")

    csv_path = OUT_DIR / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["index", "filepath", "true_name", "pred_name"] + [f"prob_{c}" for c in classes]
        writer.writerow(header)

        saved = 0
        with torch.no_grad():
            for i, (imgs, labels, paths) in enumerate(loader):
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(imgs)  # [B,K]
                probs = F.softmax(logits, dim=1)

                for b in range(imgs.size(0)):
                    idx = i * BATCH_SIZE + b
                    if idx >= SAVE_MAX:
                        break

                    p = probs[b]
                    pred_idx = int(torch.argmax(p).item())
                    true_idx = int(labels[b].item())

                    pred_name = classes[pred_idx]
                    true_name = classes[true_idx]
                    pred_prob = float(p[pred_idx].item())
                    path = paths[b]

                    img_np = to_uint8(imgs[b])
                    gt_overlay = draw_banner(img_np, f"GT: {true_name}")
                    pred_overlay = draw_banner(img_np, f"PRED: {pred_name} ({pred_prob:.3f})")

                    Image.fromarray(img_np).save(OUT_DIR / f"{idx:04d}_img.png")
                    Image.fromarray(gt_overlay).save(OUT_DIR / f"{idx:04d}_gt_overlay.png")
                    Image.fromarray(pred_overlay).save(OUT_DIR / f"{idx:04d}_pred_overlay.png")

                    row = [idx, path, true_name, pred_name] + [f"{float(p[k].item()):.6f}" for k in range(num_classes)]
                    writer.writerow(row)

                    saved += 1
                    if saved >= SAVE_MAX:
                        break

                if saved >= SAVE_MAX:
                    break

    print(f"Done. Saved {saved} samples.")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()